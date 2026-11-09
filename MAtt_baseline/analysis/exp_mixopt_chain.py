# -*- coding: utf-8 -*-
"""用论文规定的 MixOptimizer 重跑整条链:基线 -> 集成 -> 蒸馏+集成。

之前的实验脚本误用 AdamW,绕过了 Stiefel 流形约束(论文 §3.3)。
本脚本:共享基线(MixOptimizer) + 零训练经典教师蒸馏(MixOptimizer) + 概率集成。
全部 1 seed,跨 session。
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
from mAtt.optimizer import MixOptimizer                            # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_chain'
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


def make_opt(net):
    base = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=5e-4, weight_decay=0.02)
    return MixOptimizer(base)


def train_student(Xtr, ytr, iva, itr, P, lam, epochs=60):
    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = make_opt(net)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            logit = net(Xt[j].to(DEV).unsqueeze(1))
            loss = Fn.cross_entropy(logit, yt[j].to(DEV))
            if lam > 0:
                lp = Fn.log_softmax(logit / 2.0, 1)
                loss = loss + lam * Fn.kl_div(lp, P[j].to(DEV), reduction='batchmean')
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return net


def teacher_probs(X, y, n_splits=5, seed=0):
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
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]

    # 经典读出(对照)
    Ltr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Lte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    mu, sg = Ltr[itr].mean(0, keepdims=True), Ltr[itr].std(0, keepdims=True) + 1e-6
    clf = LDA(solver='lsqr', shrinkage='auto').fit((Ltr[itr] - mu) / sg, ytr[itr])
    p_cls_va = clf.predict_proba((Ltr[iva] - mu) / sg)
    p_cls_te = clf.predict_proba((Lte - mu) / sg)
    ov, lo, hi = sc(p_cls_te.argmax(1) == yte, sb)
    print('经典单独            %.2f%% (低四 %.1f / 高五 %.1f)' % (100 * ov, 100 * lo, 100 * hi), flush=True)

    # 教师
    t0 = time.time()
    P = np.zeros((len(ytr), 4))
    for i in range(9):
        P[sb == i] = teacher_probs(per[i][0], per[i][1])
    P = torch.tensor(P, dtype=torch.float32)
    print('经典教师 %.1fs' % (time.time() - t0), flush=True)

    def logits(net, X):
        with torch.no_grad():
            return torch.cat([net(torch.tensor(X[i:i + 128]).to(DEV).unsqueeze(1)).cpu()
                              for i in range(0, len(X), 128)], 0).numpy()

    res = {}
    for lam, tag in ((0.0, 'MixOpt 基线'), (0.3, 'MixOpt + 经典蒸馏')):
        net = train_student(Xtr, ytr, iva, itr, P, lam)
        p_d_va = Fn.softmax(torch.tensor(logits(net, Xtr[iva])), 1).numpy()
        p_d_te = Fn.softmax(torch.tensor(logits(net, Xte)), 1).numpy()
        ov, lo, hi = sc(p_d_te.argmax(1) == yte, sb)
        print('%-22s %.2f%% (低四 %.1f / 高五 %.1f)  %.0fs'
              % (tag, 100 * ov, 100 * lo, 100 * hi, time.time() - t0), flush=True)
        bestw = None
        for w in np.arange(0.0, 0.95, 0.1):
            pv = w * p_d_va + (1 - w) * p_cls_va
            a = float((pv.argmax(1) == ytr[iva]).mean())
            if bestw is None or a > bestw[1]:
                bestw = (float(w), a)
        p = bestw[0] * p_d_te + (1 - bestw[0]) * p_cls_te
        ov2, lo2, hi2 = sc(p.argmax(1) == yte, sb)
        print('  ⊕ 经典 w=%.1f (val %.2f%%) -> %.2f%% (低四 %.1f / 高五 %.1f)'
              % (bestw[0], 100 * bestw[1], 100 * ov2, 100 * lo2, 100 * hi2), flush=True)
        res[tag] = {'alone': ov, 'ens': ov2, 'ens_low4': lo2, 'ens_high5': hi2, 'w': bestw[0]}
    (OUT / 'mixopt_chain.json').write_text(json.dumps(
        {'note': '跨 session;1 seed;MixOptimizer(论文 §3.3)', **res}, indent=2, ensure_ascii=False),
        encoding='utf-8')


if __name__ == '__main__':
    main()
