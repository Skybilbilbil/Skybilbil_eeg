# -*- coding: utf-8 -*-
"""加载已训的蒸馏学生,做与经典的集成(秒级,不重训)。"""
import json
import sys
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


def main():
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

    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    net.load_state_dict(torch.load(OUT / 'student_distill_s42.pt', map_location=DEV))
    net.eval()

    def logits(X):
        with torch.no_grad():
            out = [net(torch.tensor(np.asarray(X[i:i + 128], np.float32)).to(DEV).unsqueeze(1)).cpu()
                   for i in range(0, len(X), 128)]
        return torch.cat(out, 0).numpy()

    p_d_va = Fn.softmax(torch.tensor(logits(Xtr[iva])), 1).numpy()
    p_d_te = Fn.softmax(torch.tensor(logits(Xte)), 1).numpy()
    Ltr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Lte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    mu, sg = Ltr[itr].mean(0, keepdims=True), Ltr[itr].std(0, keepdims=True) + 1e-6
    clf = LDA(solver='lsqr', shrinkage='auto').fit((Ltr[itr] - mu) / sg, ytr[itr])
    p_c_va = clf.predict_proba((Ltr[iva] - mu) / sg)
    p_c_te = clf.predict_proba((Lte - mu) / sg)

    ov, lo, hi = sc(p_d_te.argmax(1) == yte, sb)
    print('蒸馏单独        %.2f%% (低四 %.1f / 高五 %.1f)' % (100 * ov, 100 * lo, 100 * hi))
    ov, lo, hi = sc(p_c_te.argmax(1) == yte, sb)
    print('经典单独        %.2f%% (低四 %.1f / 高五 %.1f)' % (100 * ov, 100 * lo, 100 * hi))

    bestw = None
    for w in np.arange(0.0, 0.95, 0.1):
        pv = w * p_d_va + (1 - w) * p_c_va
        a = float((pv.argmax(1) == ytr[iva]).mean())
        if bestw is None or a > bestw[1]:
            bestw = (float(w), a)
    w = bestw[0]
    p = w * p_d_te + (1 - w) * p_c_te
    ov, lo, hi = sc(p.argmax(1) == yte, sb)
    print('蒸馏 ⊕ 经典 w=%.1f (val %.2f%%) -> %.2f%% (低四 %.1f / 高五 %.1f)'
          % (w, 100 * bestw[1], 100 * ov, 100 * lo, 100 * hi))
    (OUT / 'distill_plus_ensemble.json').write_text(json.dumps(
        {'w_val': w, 'val_acc': bestw[1], 'test_pooled': ov, 'test_low4': lo, 'test_high5': hi,
         'note': '1 seed;跨 session;教师 9 个逐人模型;λ=0.1'}, indent=2, ensure_ascii=False),
        encoding='utf-8')


if __name__ == '__main__':
    main()
