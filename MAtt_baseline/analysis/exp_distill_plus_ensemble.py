# -*- coding: utf-8 -*-
"""蒸馏前端 + 经典集成:两个改进能否叠加。

已测:
  蒸馏(λ=0.1)      57.87 -> 59.57 (+1.70pp, 前端)
  概率集成(w=0.6)  58.5  -> 61.77 (+2.94pp, 头)
本脚本:训练 9 教师 -> 蒸馏学生(λ=0.1) -> 学生概率 ⊕ 经典 LDA 后验(验证集选 w)。
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

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'distill_plus_ensemble'
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


def feats(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(net._features(torch.tensor(np.asarray(X[i:i + 128], np.float32)).to(DEV).unsqueeze(1)).cpu())
    return torch.cat(out, 0)


def train_teacher(X, y, seed=42, epochs=40):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    nva = max(1, int(0.15 * len(X)))
    iva, itr = idx[:nva], idx[nva:]
    torch.manual_seed(seed)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(np.asarray(X, np.float32)), torch.tensor(y)
    best = (1e10, None)
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 31, 32):
            j = pm[i:i + 32]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(Fn.cross_entropy(net(Xt[iva].to(DEV).unsqueeze(1)), yt[iva].to(DEV)))
        if vl < best[0]:
            best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return net


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    t0 = time.time()
    Tf = torch.zeros(len(Xtr), 684)
    for i in range(9):
        net = train_teacher(per[i][0], per[i][1])
        Tf[sb == i] = feats(net, per[i][0])
    print('教师 %.0fs' % (time.time() - t0), flush=True)

    # 学生(λ=0.1)
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]
    torch.manual_seed(42)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
    Tn = float(Tf.std())
    best = (1e10, None)
    for ep in range(60):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            f = net._features(Xt[j].to(DEV).unsqueeze(1))
            loss = Fn.cross_entropy(net.linear(f), yt[j].to(DEV))
            loss = loss + 0.1 * ((f - Tf[j].to(DEV)) ** 2).mean() / (Tn ** 2)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 59:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    torch.save(net.state_dict(), OUT / 'student_distill_s42.pt')
    print('学生 %.0fs' % (time.time() - t0), flush=True)

    def logits(X):
        out = []
        with torch.no_grad():
            out = [net(torch.tensor(np.asarray(X[i:i + 128], np.float32)).to(DEV).unsqueeze(1)).cpu()
                   for i in range(0, len(X), 128)]
        return torch.cat(out, 0).numpy()

    p_deep_tr = Fn.softmax(torch.tensor(logits(Xtr[iva])), 1).numpy()
    p_deep_te = Fn.softmax(torch.tensor(logits(Xte)), 1).numpy()

    Ltr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Lte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    mu, sg = Ltr[itr].mean(0, keepdims=True), Ltr[itr].std(0, keepdims=True) + 1e-6
    clf = LDA(solver='lsqr', shrinkage='auto').fit((Ltr[itr] - mu) / sg, ytr[itr])
    p_cls_va = clf.predict_proba((Ltr[iva] - mu) / sg)
    p_cls_te = clf.predict_proba((Lte - mu) / sg)

    # 1) 蒸馏单独
    ov, lo, hi = sc(net(torch.tensor(Xte).to(DEV).unsqueeze(1)).argmax(1).cpu().numpy() == yte, sb) if False else (0, 0, 0)
    pr = logits(Xte).argmax(1)
    ov, lo, hi = sc(pr == yte, sb)
    print('蒸馏单独        %.2f%% (低四 %.1f / 高五 %.1f)' % (100 * ov, 100 * lo, 100 * hi), flush=True)

    bestw = None
    for w in np.arange(0.0, 0.95, 0.1):
        pv = w * p_deep_tr + (1 - w) * p_cls_va
        a = float((pv.argmax(1) == yv).mean())
        if bestw is None or a > bestw[1]:
            bestw = (float(w), a)
    w = bestw[0]
    p = w * p_deep_te + (1 - w) * p_cls_te
    ov, lo, hi = sc(p.argmax(1) == yte, sb)
    print('蒸馏 ⊕ 经典 w=%.1f (val %.2f%%) -> %.2f%% (低四 %.1f / 高五 %.1f)'
          % (w, 100 * bestw[1], 100 * ov, 100 * lo, 100 * hi), flush=True)
    (OUT / 'distill_plus_ensemble.json').write_text(json.dumps(
        {'w_val': w, 'val_acc': bestw[1], 'test_pooled': ov, 'test_low4': lo, 'test_high5': hi,
         'note': '1 seed;跨 session;教师 9 个逐人模型;λ=0.1'}, indent=2, ensure_ascii=False),
        encoding='utf-8')


if __name__ == '__main__':
    main()
