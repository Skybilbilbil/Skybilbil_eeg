# -*- coding: utf-8 -*-
"""跨 session 主协议:深度后端 + 经典协方差旁路的混合读出(闭式,零额外训练)。

动机:同 session 下「后端 180 维 ⊕ 协方差 253 维」+ 闭式 LW-LDA 达 63.2–63.5(§8.7)。
跨 session 尚未测——而跨 session 下经典 LE+LDA(58.76)与深度(58.29)持平,
两者若携带互补信息,拼接应超过任一单独。
只训练一次前端(约 2 分钟),读出全为闭式解。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'crosssession_hybrid'
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


def readout(A, ya, B, yb, sb):
    mu, sg = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
    As, Bs = (A - mu) / sg, (B - mu) / sg
    ok = LDA(solver='lsqr', shrinkage='auto').fit(As, ya).predict(Bs) == yb
    per = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    return float(ok.mean()), lo, hi, per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sub = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]
    torch.manual_seed(42)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV)
    yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(100):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 99:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    print('训练 %.0fs' % (time.time() - t0), flush=True)

    def backend(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                out.append(net._features(torch.tensor(X[i:i + 128]).to(DEV).unsqueeze(1)).cpu().numpy())
        return np.concatenate(out, 0)

    torch.save(net.state_dict(), OUT / 'net_s42.pt')
    t1 = time.time()
    Ltr = logvec(cov_tn(Xtr[:, :, 124:562]))
    Lte = logvec(cov_tn(Xte[:, :, 124:562]))
    from scipy import signal as _sg
    def bp(X, lo=8.0, hi=40.0, fs=128, order=4):
        b, a = _sg.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
        return _sg.filtfilt(b, a, X, axis=-1)
    Ptr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Pte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    Btr, Bte = backend(Xtr), backend(Xte)
    print('特征 %.0fs  LE %s  back %s' % (time.time() - t1, Ltr.shape, Btr.shape), flush=True)

    rec = {}
    for tag, A, B in (('LE_4to40', Ltr, Lte),
                      ('LE_8to40', Ptr, Pte),
                      ('backend684', Btr, Bte),
                      ('LE8to40+backend', np.concatenate([Ptr, Btr], 1), np.concatenate([Pte, Bte], 1)),
                      ('LE4to40+backend', np.concatenate([Ltr, Btr], 1), np.concatenate([Lte, Bte], 1))):
        ov, lo, hi, _ = readout(A, ytr, B, yte, sub)
        rec[tag] = {'pooled': ov, 'low4': lo, 'high5': hi, 'dim': int(A.shape[1])}
        print('  %-12s %6.2f%%  (低四 %.1f / 高五 %.1f)' % (tag, 100 * ov, 100 * lo, 100 * hi), flush=True)

    (OUT / 'crosssession_hybrid.json').write_text(
        json.dumps({'note': '跨 session,seed 42;闭式 LW-LDA;经典=58.76 深度=58.29', **rec},
                   indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/crosssession_hybrid.json' % OUT)


if __name__ == '__main__':
    main()
