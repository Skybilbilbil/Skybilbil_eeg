# -*- coding: utf-8 -*-
"""对比学习表示:能否学出「身份纯度低、类别可分高」的共享特征

## 判据(本轮定义)
共享特征的理想形态:身份纯度(kNN 身份识别)显著低于 LE 的 1.000,而共享类别精度不掉。
当前事实:LE 253 身份纯度 1.000、共享线性 62.9、共享核 73.5。

## 四个臂
A 时间子窗对比(正样本=同一试次前半/后半窗的 LE) —— 学「时间位置不变」
B 加噪对比(正样本=同试次 + 高斯扰动) —— 学「噪声不变」
C SupCon(正样本=同类不同试次) —— 有监督类内紧凑
D 跨被试类中心对齐(创新:显式拉近同类不同被试的均值) + 对比
读出:编码器冻结 + 闭式 LW-LDA(避免 SGD 头损失 4.6pp)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import cov_tn, grp, fit_lda         # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'contrastive_repr'
TASK = ED.TASK
SUBS = ['S%02d' % i for i in range(1, 10)]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN_A = (375, 750)
WIN_B = (751, 1126)


def spdvec(M):
    iu = np.triu_indices(22)
    v = M[iu].copy()
    v[iu[0] != iu[1]] *= np.sqrt(2)
    return v


def le(A):
    w, V = np.linalg.eigh(A.astype(np.float64))
    w = np.maximum(w, 1e-12)
    return np.stack([spdvec((V[i] * np.log(w[i])) @ V[i].T) for i in range(len(A))])


class Enc(nn.Module):
    def __init__(self, d_in=253, d=256, z=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(),
                                 nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
                                 nn.Linear(d, z))

    def forward(self, x):
        return self.net(x)


def nt_xent(z1, z2, tau=0.2):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    B = z1.shape[0]
    z = torch.cat([z1, z2], 0)
    sim = z @ z.t() / tau
    sim.fill_diagonal_(-1e9)
    tgt = torch.arange(B, device=z.device)
    tgt = torch.cat([tgt + B, tgt], 0)
    return F.cross_entropy(sim, tgt)


def supcon(z, y, tau=0.2):
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau
    B = z.shape[0]
    mask = (y[:, None] == y[None, :]).float()
    mask.fill_diagonal_(0)
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    cnt = mask.sum(1).clamp(min=1)
    return -((log_prob * mask).sum(1) / cnt).mean()


def train_enc(X1, X2, y=None, mode='ntxent', epochs=120, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    enc = Enc(X1.shape[1]).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    X1t = torch.tensor(X1, dtype=torch.float32)
    X2t = torch.tensor(X2, dtype=torch.float32)
    yt = None if y is None else torch.tensor(y, dtype=torch.long).to(DEV)
    n = len(X1t)
    for ep in range(epochs):
        enc.train()
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            if len(idx) < 8:
                continue
            a = X1t[idx].to(DEV); b = X2t[idx].to(DEV)
            za, zb = enc(a), enc(b)
            if mode == 'supcon':
                z = torch.cat([za, zb], 0)
                yy = torch.cat([yt[idx], yt[idx]], 0)
                loss = supcon(z, yy)
            else:
                loss = nt_xent(za, zb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss)
    return enc


def embed(enc, X):
    enc.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(enc(torch.tensor(X[i:i + 1024], dtype=torch.float32).to(DEV)).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X = np.concatenate([s[0] for s in sd], 0).astype(np.float64)
    y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                          for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_ = ret[0], ret[1], ret[6]
    Xva, yva = ret[2], ret[3]
    Xte, yte, ste = ret[4], ret[5], ret[8]

    Ltr = le(cov_tn(Xtr[:, :, TASK[0]:TASK[1]]))
    Lva = le(cov_tn(Xva[:, :, TASK[0]:TASK[1]]))
    Lte = le(cov_tn(Xte[:, :, TASK[0]:TASK[1]]))
    A1 = le(cov_tn(Xtr[:, :, WIN_A[0]:WIN_A[1]]))
    A2 = le(cov_tn(Xtr[:, :, WIN_B[0]:WIN_B[1]]))

    rec = {'device': DEV, 'note': '读出=冻结编码器+闭式LW-LDA;身份纯度=kNN(9)', 'arms': {}}

    def idpurity(F, ids):
        kn = KNeighborsClassifier(n_neighbors=9).fit(F, ids)
        return float((kn.predict(F) == ids).mean())

    def rep(name, F, extra=None):
        ok = fit_lda(F[0], ytr).predict(F[2]) == yte
        per = {SUBS[i]: float(ok[ste == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        pur = idpurity(F[0], str_)
        print('%-18s TEST %5.1f (low %5.1f high %5.1f)  idpurity %.3f  %s'
              % (name, ok.mean() * 100, lo * 100, hi * 100, pur, extra or ''), flush=True)
        rec['arms'][name] = {'test': float(ok.mean()), 'low4': lo, 'high5': hi,
                             'id_purity': pur, 'per_subject': per}
        return np.asarray(ok)

    rep('base_LE253', (Ltr, Lva, Lte))
    m2 = float(np.median(((Ltr[np.random.default_rng(1).choice(len(Ltr), 800, replace=False)]
                           [:, None, :] - Ltr[np.random.default_rng(1).choice(len(Ltr), 800, replace=False)]
                           [None, :, :]) ** 2).sum(-1)[np.triu_indices(800, 1)]))
    okS = SVC(kernel='rbf', C=100., gamma=2.0 / (2 * m2)).fit(Ltr, ytr).predict(Lte) == yte
    print('base_LE253_svm    TEST %5.1f (low %5.1f high %5.1f)'
          % (okS.mean() * 100, np.mean([okS[ste == i].mean() for i in LOW]) * 100,
             np.mean([okS[ste == i].mean() for i in HIGH]) * 100), flush=True)
    rec['ref_svm'] = float(okS.mean())

    # A 时间子窗对比
    encA = train_enc(A1, A2, mode='ntxent', epochs=120, seed=0)
    FA = [embed(encA, Z) for Z in (Ltr, Lva, Lte)]
    rep('A_timewin_clr', FA)
    # B 加噪对比
    rng = np.random.default_rng(0)
    nz = lambda Z, s: Z + rng.normal(0, s * Z.std(0), Z.shape)
    encB = train_enc(nz(Ltr, 0.1), nz(Ltr, 0.1), mode='ntxent', epochs=120, seed=1)
    rep('B_noise_clr', [embed(encB, Z) for Z in (Ltr, Lva, Lte)])
    # C SupCon
    encC = train_enc(nz(Ltr, 0.05), nz(Ltr, 0.05), y=ytr, mode='supcon', epochs=120, seed=2)
    FC = [embed(encC, Z) for Z in (Ltr, Lva, Lte)]
    rep('C_supcon', FC)
    # D 跨被试类中心对齐(创新):拉近同类不同被试的均值
    encD = train_enc(nz(Ltr, 0.05), nz(Ltr, 0.05), y=ytr, mode='supcon', epochs=120, seed=3)
    for p in encD.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(encD.parameters(), lr=3e-4, weight_decay=1e-4)
    Xt = torch.tensor(Ltr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long).to(DEV)
    st = torch.tensor(str_, dtype=torch.long).to(DEV)
    for ep in range(40):
        encD.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), 512):
            idx = perm[i:i + 512]
            if len(idx) < 64:
                continue
            z = encD(Xt[idx].to(DEV))
            z = F.normalize(z, dim=1)
            yy = yt[idx]; ss = st[idx]
            # 每个 (类,被试) 的均值,拉近同类跨被试均值
            loss = 0.0; cnt = 0
            cents = {}
            for c in range(4):
                for s in range(9):
                    m = (yy == c) & (ss == s)
                    if m.sum() >= 3:
                        cents[(c, s)] = z[m].mean(0)
            for c in range(4):
                vs = [cents[(c, s)] for s in range(9) if (c, s) in cents]
                if len(vs) >= 2:
                    V_ = torch.stack(vs)
                    loss = loss + ((V_ - V_.mean(0)) ** 2).sum(1).mean()
                    cnt += 1
            if cnt == 0:
                continue
            loss = loss / cnt + supcon(z, yy)
            opt.zero_grad(); loss.backward(); opt.step()
    rep('D_cscenter_supcon', [embed(encD, Z) for Z in (Ltr, Lva, Lte)])

    (OUT / 'contrastive_repr.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/contrastive_repr.json' % OUT, flush=True)


if __name__ == '__main__':
    main()
