# -*- coding: utf-8 -*-
"""零训练教师蒸馏:用「逐人经典 LE+LDA」当教师,不需要训练任何逐人深度模型。

动机:上一版蒸馏需要 9 个逐人深度教师(每人 40 epoch),训练成本高。
经典 LE+LDA 是闭式解(秒级),跨 session 下 58.83——同样携带"个体信息"。
做法:
  1. 对每个被试,在 T 上做 5 折,得到每个 T 样本的「交叉预测概率」p_teacher(避免教师过拟合)
  2. 训练共享学生:L = CE + λ·KL(p_teacher || p_student)
  3. 测试 E(学生独立,不接收 ID)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'distill_classic_teacher'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN, STEP = 192, 96
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (C.shape[1] / np.trace(C, axis1=1, axis2=2)[:, None, None])


def logvec(C):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-12)
    L = np.einsum('nij,nj,nkj->nik', V, np.log(w), V)
    d = L.shape[-1]
    iu = np.triu_indices(d)
    v = L[:, iu[0], iu[1]].copy()
    v[:, iu[0] != iu[1]] *= np.sqrt(2)
    return v


def bp(X, lo=8.0, hi=40.0, fs=128, order=4):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, X, axis=-1)


def load_split():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        H = apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X))
        per.append((H[:288], ys[0], H[288:], ys[1]))
    return per


def grp(per):
    return (float(np.mean([per[SUBJ[i]] for i in LOW])),
            float(np.mean([per[SUBJ[i]] for i in HIGH])))


def sc(ok, sb):
    per = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    return float(ok.mean()), lo, hi


def teacher_probs(X, y, n_splits=5, seed=0):
    """每人的 5 折交叉预测概率(避免教师在自己训练集上过拟合)。"""
    F = logvec(cov_tn(bp(X)[:, :, 124:562]))
    p = np.zeros((len(y), 4))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in skf.split(F, y):
        mu, sg = F[tr].mean(0, keepdims=True), F[tr].std(0, keepdims=True) + 1e-6
        clf = LDA(solver='lsqr', shrinkage='auto').fit((F[tr] - mu) / sg, y[tr])
        p[va] = clf.predict_proba((F[va] - mu) / sg)
    return p


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    t0 = time.time()
    P = np.zeros((len(ytr), 4))
    for i in range(9):
        m = sb == i
        P[m] = teacher_probs(per[i][0], per[i][1])
    print('经典教师(9 人 × 5 折 LDA) %.1fs' % (time.time() - t0), flush=True)
    P = torch.tensor(P, dtype=torch.float32)
    print('教师在 T 上的交叉预测精度 %.2f%%' % (100 * (P.argmax(1).numpy() == ytr).mean()), flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]

    rec = {}
    for lam in (0.0, 0.3, 1.0):
        torch.manual_seed(42)
        net = build_matt_model(WIN, STEP, ARGS).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
        best = (1e10, None)
        for ep in range(60):
            net.train()
            pm = rng.permutation(itr)
            for i in range(0, len(pm) - 127, 128):
                j = pm[i:i + 128]
                logit = net(Xt[j].to(DEV).unsqueeze(1))
                loss = Fn.cross_entropy(logit, yt[j].to(DEV))
                if lam > 0:
                    logp_stu = Fn.log_softmax(logit / 2.0, 1)
                    loss = loss + lam * Fn.kl_div(logp_stu, P[j].to(DEV), reduction='batchmean')
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 4 or ep == 59:
                net.eval()
                with torch.no_grad():
                    vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
                if vl < best[0]:
                    best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
        net.load_state_dict(best[1]); net.eval()
        with torch.no_grad():
            pr = torch.cat([net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)).argmax(1).cpu()
                            for i in range(0, len(Xte), 128)]).numpy()
        ov, lo, hi = sc(pr == yte, sb)
        rec['lambda_%.1f' % lam] = {'pooled': ov, 'low4': lo, 'high5': hi}
        print('λ=%.1f  %.2f%%  (低四 %.1f / 高五 %.1f)' % (lam, 100 * ov, 100 * lo, 100 * hi), flush=True)

    (OUT / 'distill_classic_teacher.json').write_text(json.dumps(
        {'note': '零训练教师(逐人 LE+LDA,5 折交叉预测);跨 session;1 seed;60 epoch', **rec},
        indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
