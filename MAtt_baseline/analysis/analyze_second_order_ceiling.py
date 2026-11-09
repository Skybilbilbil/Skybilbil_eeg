# -*- coding: utf-8 -*-
"""二阶(跨窗动力学)特征:62.9 之上还有没有可读的时间过程信息(零训练)。

## 依据(文献 × 本项目)
视觉领域 SoT(Second-order Transformer)证明分类头里的**二阶操作**能提升 ViT;
本项目 ①线确立「μ ERD 的时间过程」与精度同序,但 temporal_signal 只测过窗序与
相邻差分(动态净贡献 ~1pp),**从未测过跨窗方差/共变**——时间过程信息的线性
可读形式。

## 特征(任务窗 375:1126 切 4 个 188 步子窗,起点 375/563/751/939)
  每窗每通道对数功率 → P (4×22);宽带与 μ 带(逐人 IAF±1.5)两个版本。
  var22   逐通道跨窗方差(ERD 动态强度)
  slope22 逐通道跨窗线性斜率(增强/抑制走势)
  cov253  跨窗通道共变上三角(哪些通道一起动)
臂 = L0(62.9 参照) ⊕ 上述各组合。共享 shrinkage LDA,同一切分。
只写 results/second_order_ceiling/。
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
from analyze_mirror_commonality import PKL, TBL, SUBJ, LOW, HIGH, pow_band  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'second_order_ceiling'


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


def sos(band):
    return butter(5, [band[0] / 125.0, band[1] / 125.0], btype='bandpass', output='sos')


def dyn_feats(X, subs, iaf_of):
    """X (n,22,T) → {var,slope,cov} × {bb,mu}。任务窗切 4 子窗;μ 带按被试 IAF 分组滤波。"""
    out = {}
    starts = (375, 563, 751, 939)
    seg_mu = np.zeros_like(X)
    for s in np.unique(subs):
        m = subs == s
        seg_mu[m] = sosfiltfilt(sos((iaf_of(s) - 1.5, iaf_of(s) + 1.5)), X[m], axis=-1)
    for nm, seg in (('bb', X), ('mu', seg_mu)):
        P = []
        for s in starts:
            w = seg[:, :, s:s + 188]
            P.append(np.log((w ** 2).mean(-1) + 1e-24))      # (n,22)
        P = np.stack(P, 1)                                     # (n,4,22)
        Pv = P - P.mean(1, keepdims=True)
        out[nm + '_var'] = (Pv ** 2).mean(1)
        t = np.arange(4) - 1.5
        out[nm + '_slope'] = (Pv * t[None, :, None]).sum(1) / (t ** 2).sum()
        C = np.einsum('nwt,nws->nts', Pv, Pv) / 3.0
        d = C.shape[-1]
        iu = np.triu_indices(d)
        v = C[:, iu[0], iu[1]].copy()
        v[:, iu[0] != iu[1]] *= np.sqrt(2)
        out[nm + '_cov'] = v
    return out


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
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}

    def all_feats(X, subs):
        L0 = logvec(cov_tn(X[:, :, 375:1126]))
        return {'L0': L0, **dyn_feats(X, subs, lambda s: ref[SUBJ[s]]['iaf_hz'])}
    F_tr = all_feats(X_tr, sub_tr)
    F_te = all_feats(X_te, sub_te)

    ARMS = {
        'L0(参照)': ['L0'],
        'L0+var': ['L0', 'bb_var', 'mu_var'],
        'L0+slope': ['L0', 'bb_slope', 'mu_slope'],
        'L0+cov': ['L0', 'bb_cov', 'mu_cov'],
        'L0+var+slope': ['L0', 'bb_var', 'mu_var', 'bb_slope', 'mu_slope'],
        'L0+全二阶': ['L0', 'bb_var', 'mu_var', 'bb_slope', 'mu_slope', 'bb_cov', 'mu_cov'],
        '仅二阶': ['bb_var', 'mu_var', 'bb_slope', 'mu_slope', 'bb_cov', 'mu_cov'],
    }
    print('二阶(跨窗动力学)特征 × 共享 LDA(低四/高五/全体):')
    rec = {}
    for name, keys in ARMS.items():
        Ftr = np.concatenate([F_tr[k] for k in keys], 1)
        Fte = np.concatenate([F_te[k] for k in keys], 1)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Ftr, y_tr)
        ok = clf.predict(Fte) == y_te
        po = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        lo = np.mean([po[SUBJ[i]] for i in LOW])
        hi = np.mean([po[SUBJ[i]] for i in HIGH])
        rec[name] = {'overall': float(ok.mean()), 'low4': float(lo),
                     'high5': float(hi), 'per': po}
        print('  %-12s 低四 %5.1f 高五 %5.1f 全体 %5.1f' % (name, lo * 100, hi * 100, ok.mean() * 100))
    (OUT / 'second_order_ceiling.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/second_order_ceiling.json' % OUT)


if __name__ == '__main__':
    main()
