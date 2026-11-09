# -*- coding: utf-8 -*-
"""经典读出的带通扫描（零训练）：跨 session 主协议下 LE+LDA 在不同频带上的最好读数。

现状：raw 4-40Hz = 57.06、bp 8-40Hz = 58.76（analyze_official_classic.py）。
本脚本把频带扩扫一圈，看 58.76 是否还能推高——经典读出在深度⊕经典集成里占大头
（w≈0.5-0.6），它每涨一点集成都跟着涨。零训练、纯 CPU。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import io, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'classic_bands'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
WIN_OFF = (124, 562)


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


def bp(X, lo, hi, fs=128, order=4):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, X, axis=-1)


def load_split():
    tr, te = [], []
    for s in range(1, 10):
        mT = io.loadmat(str(DATA / ('BCIC_S%02d_T.mat' % s)))
        mE = io.loadmat(str(DATA / ('BCIC_S%02d_E.mat' % s)))
        XT = np.asarray(mT['x_train'], dtype=np.float64)
        yT = np.asarray(mT['y_train']).reshape(-1).astype(np.int64)
        XE = np.asarray(mE['x_test'], dtype=np.float64)
        yE = np.asarray(mE['y_test']).reshape(-1).astype(np.int64)
        H = apply_ea_to_numpy(np.concatenate([XT, XE], 0),
                              compute_ea_matrix_from_numpy(np.concatenate([XT, XE], 0)))
        tr.append((H[:288], yT))
        te.append((H[288:], yE))
    return (np.concatenate([p[0] for p in tr], 0), np.concatenate([p[1] for p in tr], 0),
            np.concatenate([p[0] for p in te], 0), np.concatenate([p[1] for p in te], 0))


def grp(ps):
    return (float(np.mean([ps[SUBJ[i]] for i in LOW])),
            float(np.mean([ps[SUBJ[i]] for i in HIGH])))


BANDS = [(4, 40), (8, 40), (8, 30), (4, 30), (6, 40), (8, 45), (10, 40), (4, 45), (7, 35)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    Xtr, ytr, Xte, yte = load_split()
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    rec = {}
    print('%-10s %8s %7s %7s' % ('band', 'pooled', '低四', '高五'), flush=True)
    for lo, hi in BANDS:
        A = bp(Xtr, lo, hi) if lo > 4 else Xtr
        B = bp(Xte, lo, hi) if lo > 4 else Xte
        Fa = logvec(cov_tn(A[:, :, WIN_OFF[0]:WIN_OFF[1]]))
        Fb = logvec(cov_tn(B[:, :, WIN_OFF[0]:WIN_OFF[1]]))
        mu, sg = Fa.mean(0, keepdims=True), Fa.std(0, keepdims=True) + 1e-6
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit((Fa - mu) / sg, ytr)
        ok = clf.predict((Fb - mu) / sg) == yte
        ps = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
        l4, h5 = grp(ps)
        tag = '%d-%d' % (lo, hi)
        rec[tag] = {'pooled': float(ok.mean()), 'low4': l4, 'high5': h5, 'per_subject': ps}
        print('%-10s %7.2f%% %6.1f %6.1f' % (tag, 100 * ok.mean(), 100 * l4, 100 * h5), flush=True)
    best = max(rec.items(), key=lambda kv: kv[1]['pooled'])
    print('\n最好: %s = %.2f%% (低四 %.1f / 高五 %.1f)'
          % (best[0], 100 * best[1]['pooled'], 100 * best[1]['low4'], 100 * best[1]['high5']), flush=True)
    (OUT / 'classic_bands.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;零训练 LE+LDA;只变带通',
         'bands': rec, 'best': best[0]}, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
