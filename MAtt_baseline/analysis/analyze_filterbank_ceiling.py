# -*- coding: utf-8 -*-
"""滤波器组协方差:62.9 的上界还能不能抬(零训练,协议合法)。

## 依据
推进(十):任务窗协方差 + 共享 LDA = 62.9 是当前上界,但只用了**单一宽带表示**。
经典 Riemannian 文献的标配是滤波器组(各频带分别算协方差再拼接);镜像分析又显示
静息相对化对手二类公平性有益。全部零标签共享 ⇒ 协议合法。

## 臂(全部共享 shrinkage LDA,同一切分)
  A0 宽带任务窗协方差(253) = 现状 62.9
  A1 ⊕ μ 带(8-13)任务窗协方差(253) → 506
  A2 ⊕ β 带(15-25)任务窗协方差(253) → 506
  A3 ⊕ μ ⊕ β → 759
  A4 静息相对化:[logvec(C_task) − logvec(C_rest)] 宽带(253)
  A5 A3 ⊕ 静息相对化宽带 → 1012
  A6 窗敏感性(对最优臂):任务窗起点 250/375/500
只写 results/filterbank_ceiling/。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH            # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'filterbank_ceiling'
REST = (0, 250)


def sos(band):
    return butter(5, [band[0] / 125.0, band[1] / 125.0], btype='bandpass', output='sos')


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


def shared(Ftr, ytr, Fte, yte, sub_te):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(Ftr, ytr)
    ok = clf.predict(Fte) == yte
    return (float(ok.mean()),
            {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)})


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in sd], 0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in sd], 0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64) for i, s in enumerate(sd)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print('滤波器组协方差上界扫描(共享 LDA,低四/高五/全体):', flush=True)

    def bank(X, t0, t1):
        seg = X[:, :, t0:t1]
        parts = {'bb': logvec(cov_tn(seg))}
        for nm, band in (('mu', (8., 13.)), ('beta', (15., 25.))):
            Z = sosfiltfilt(sos(band), seg, axis=-1)
            parts[nm] = logvec(cov_tn(Z))
        rest = X[:, :, REST[0]:REST[1]]
        parts['rel'] = logvec(cov_tn(seg)) - logvec(cov_tn(rest))
        return parts
    P_tr, P_te = bank(X_tr, 375, 1126), bank(X_te, 375, 1126)
    ARMS = {'A0_bb': ['bb'], 'A1_+mu': ['bb', 'mu'], 'A2_+beta': ['bb', 'beta'],
            'A3_+mu+beta': ['bb', 'mu', 'beta'], 'A4_rel': ['rel'],
            'A5_fb+rel': ['bb', 'mu', 'beta', 'rel']}
    rec = {}
    for name, keys in ARMS.items():
        Ftr = np.concatenate([P_tr[k] for k in keys], 1)
        Fte = np.concatenate([P_te[k] for k in keys], 1)
        ov, po = shared(Ftr, y_tr, Fte, y_te, sub_te)
        lo = np.mean([po[SUBJ[i]] for i in LOW])
        hi = np.mean([po[SUBJ[i]] for i in HIGH])
        rec[name] = {'overall': ov, 'low4': float(lo), 'high5': float(hi), 'per': po}
        print('  %-12s 低四 %5.1f 高五 %5.1f 全体 %5.1f' % (name, lo * 100, hi * 100, ov * 100),
              flush=True)

    # 窗敏感性(取 A3 与 A0 各测三个起点)
    for name, keys in (('W250_fb', ['bb', 'mu', 'beta']), ('W500_fb', ['bb', 'mu', 'beta']),
                       ('W250_bb', ['bb']), ('W500_bb', ['bb'])):
        t0 = int(name[1:4])
        Ptr, Pte = bank(X_tr, t0, 1126), bank(X_te, t0, 1126)
        Ftr = np.concatenate([Ptr[k] for k in keys], 1)
        Fte = np.concatenate([Pte[k] for k in keys], 1)
        ov, po = shared(Ftr, y_tr, Fte, y_te, sub_te)
        lo = np.mean([po[SUBJ[i]] for i in LOW])
        hi = np.mean([po[SUBJ[i]] for i in HIGH])
        rec[name] = {'overall': ov, 'low4': float(lo), 'high5': float(hi), 'per': po}
        print('  %-12s 低四 %5.1f 高五 %5.1f 全体 %5.1f' % (name, lo * 100, hi * 100, ov * 100),
              flush=True)

    (OUT / 'filterbank_ceiling.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/filterbank_ceiling.json' % OUT)


if __name__ == '__main__':
    main()
