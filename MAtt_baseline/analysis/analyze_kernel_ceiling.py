# -*- coding: utf-8 -*-
"""核读出的上界收敛:73.5 还能不能推 —— 共享参数族在核层的全部出口(2026-10-25 推进十八)

## 问题
推进十七把「单切图 253 维 LE + RBF-SVM」推到 test 73.5(val 74.7),远超共享线性 62.9。
本轮问三件事:
  ① 73.5 是超参边界伪影吗(C/γ 网格只扫到 γ=2.0,val 本身就是上边界点)?
  ② 共享参数还能不能把它推高(度量/白化/多尺度/局部尺度/核形状/多切图)?
  ③ 73.5 与「逐人」(per-subject,仅诊断,主协议禁止 subject_id)差多远?

## 数学要点
- RBF 核 κ(x,y)=exp(−γ‖x−y‖²) 的 γ 决定「邻域半径」:γ 大 ⇒ 只有近邻有相似度。
- 度量学习:κ_A(x,y)=exp(−‖A(x−y)‖²/2σ²);A 若是线性白化,等价于用 Mahalanobis 距离。
- 多尺度:κ=Σ_m w_m κ_m(γ_m) —— 各被试簇离散度不同时的自适应带宽。
- 局部尺度(local scaling,Zelnik-Manor & Perona 2004):σ_i=第 k 近邻距离,κ=exp(−d²/(σ_iσ_j))。
- 多切图:SPD 曲率使不同参考点 R_k 的 log 坐标不等价,拼接后 253K 维。

## 口径
主协议 rs=42 同折;超参只在 val 选,test 只报一次;单折 <5pp 视为噪声,关键读数配 McNemar。
「逐人」臂使用 subject_id,**仅作诊断上界,不是主协议读数**(主协议硬约束:模型不得接收 subject_id)。
只写 results/kernel_ceiling/。纯 CPU。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import chi2
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import cov_tn, chart_feats, grp, fit_lda, eval_split  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'kernel_ceiling'
TASK = ED.TASK
SUBS = ['S%02d' % i for i in range(1, 10)]


def med2_of(F, n=None):
    """训练折上抽样估计的平方距离中位数(γ 的尺度锚)。"""
    n = n or min(len(F), 1000)
    r = np.random.default_rng(1).choice(len(F), n, replace=False)
    d2 = ((F[r][:, None, :] - F[r][None, :, :]) ** 2).sum(-1)
    return float(np.median(d2[np.triu_indices(len(r), 1)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_va, y_va, sub_va = ret[2], ret[3], ret[7]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]

    L = [chart_feats(cov_tn(X[:, :, TASK[0]:TASK[1]]), None)[0]
         for X in (X_tr, X_va, X_te)]
    Ltr, Lva, Lte = L
    M2 = med2_of(Ltr)
    rec = {'M2': M2, 'task': TASK, 'grid_note': 'gm/(2*med2) 的 gm 网格', 'arms': {}}
    ok_ref = None

    def rep(name, ok, note=''):
        nonlocal ok_ref
        per = {SUBS[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        ok = np.asarray(ok)
        if ok_ref is None:
            ok_ref = ok
        b = int(np.sum(ok_ref & ~ok)); c = int(np.sum(~ok_ref & ok))
        p = 1.0 if b + c == 0 else float(chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))
        print('%-30s TEST %5.1f (low %5.1f high %5.1f)  p=%.3f  %s'
              % (name, ok.mean() * 100, lo * 100, hi * 100, p, note), flush=True)
        rec['arms'][name] = {'test': float(ok.mean()), 'low4': lo, 'high5': hi,
                             'per_subject': per, 'p_vs_ref': p, 'note': note,
                             '_ok': [bool(v) for v in ok]}
        return ok

    # ---------- A. 参照 + γ/C 网格边界(诚实口径) ----------
    print('=== A. reference + hyperparameter grid border ===', flush=True)
    sv = SVC(kernel='rbf', C=100., gamma=2.0 / (2 * M2)).fit(Ltr, y_tr)
    okA = rep('A1_K1_C100_gm2', sv.predict(Lte) == y_te, '原网格 0.5/1/2 选出的 73.5')
    rep('A2_K1_lda_linear', fit_lda(Ltr, y_tr).predict(Lte) == y_te, '共享线性 62.9')
    rec['grid_val'] = {}
    best = None
    for C_ in (100., 300., 1000.):
        for gm in (0.5, 1., 2., 3., 5., 8.):
            s = SVC(kernel='rbf', C=C_, gamma=gm / (2 * M2)).fit(Ltr, y_tr)
            v = float((s.predict(Lva) == y_va).mean())
            rec['grid_val']['C%.0f_g%.1f' % (C_, gm)] = v
            if best is None or v > best[0]:
                best = (v, C_, gm, s)
    v, C_, gm, s = best
    rep('A3_gridbest_C%.0f_gm%.1f' % (C_, gm), s.predict(Lte) == y_te,
        'val 最优 %.1f;γ 扩网格后 test' % (v * 100))

    # ---------- B. 逐人(诊断上界,用 subject_id) ----------
    print('=== B. per-subject reference (DIAGNOSTIC ONLY, uses subject_id) ===', flush=True)
    for C_ in (10., 100.):
        oks = []
        for ss in range(9):
            m_tr, m_te = sub_tr == ss, sub_te == ss
            m2s = med2_of(Ltr[m_tr], n=min(int(m_tr.sum()), 400))
            clf = SVC(kernel='rbf', C=C_, gamma=2.0 / (2 * m2s)).fit(Ltr[m_tr], y_tr[m_tr])
            oks.append(clf.predict(Lte[m_te]) == y_te[m_te])
        rep('B1_per_subj_svm_C%.0f' % C_, np.concatenate(oks), '诊断:逐人核上界')
    oks = []
    for ss in range(9):
        m_tr, m_te = sub_tr == ss, sub_te == ss
        clf = LDA(solver='lsqr', shrinkage='auto').fit(Ltr[m_tr], y_tr[m_tr])
        oks.append(clf.predict(Lte[m_te]) == y_te[m_te])
    rep('B2_per_subj_lda', np.concatenate(oks), '诊断:逐人线性上界')

    # ---------- C. 去簇位置(逐人中心化,诊断) ----------
    print('=== C. subject-centered features (DIAGNOSTIC) ===', flush=True)
    mu = np.stack([Ltr[sub_tr == ss].mean(0) for ss in range(9)])
    Lc = [Ltr - mu[sub_tr], Lva - mu[sub_va], Lte - mu[sub_te]]
    M2c = med2_of(Lc[0])
    rep('C1_centered_lda', fit_lda(Lc[0], y_tr).predict(Lc[2]) == y_te, '去簇位置后线性')
    rep('C2_centered_svm_gm3', SVC(kernel='rbf', C=100., gamma=3.0 / (2 * M2c))
        .fit(Lc[0], y_tr).predict(Lc[2]) == y_te, '去簇位置后核:掉多少=身份锚定成分')

    # ---------- D. 距离结构 ----------
    print('=== D. distance structure ===', flush=True)
    D_te_tr = cdist(Lte, Ltr)
    order = np.argsort(D_te_tr, axis=1)
    for k in (1, 15, 30):
        same = (sub_tr[order[:, :k]] == sub_te[:, None]).mean()
        rec['arms']['D_nn_identity_k%d' % k] = float(same)
        print('  NN identity k=%2d: same-subject frac %.3f (chance 0.111)' % (k, same),
              flush=True)
    ds, dd = [], []
    for i in range(len(Lte)):
        d = D_te_tr[i]
        m_same = sub_tr == sub_te[i]
        ds.append(d[m_same].min()); dd.append(d[~m_same].min())
    ratio = float(np.median(ds) / np.median(dd))
    rec['arms']['D_dist_ratio'] = ratio
    print('  median(min-dist) same %.2f vs diff %.2f -> ratio %.3f' % (np.median(ds), np.median(dd), ratio), flush=True)
    for k in (1, 15, 30):
        rep('D_knn_k%d' % k, KNeighborsClassifier(n_neighbors=k).fit(Ltr, y_tr)
            .predict(Lte) == y_te, '纯局部投票')

    # ---------- E. 度量族(训练折拟合,无标签泄漏) ----------
    print('=== E. anisotropic metrics (fit on train fold) ===', flush=True)
    mu0 = Ltr.mean(0)
    sd0 = Ltr.std(0) + 1e-12
    Z = [(L - mu0) / sd0 for L in (Ltr, Lva, Lte)]
    m2z = med2_of(Z[0])
    rep('E1_zscore_svm', SVC(kernel='rbf', C=100., gamma=2.0 / (2 * m2z))
        .fit(Z[0], y_tr).predict(Z[2]) == y_te, '逐维标准化')
    Xc = Ltr - mu0
    Sig = (Xc.T @ Xc) / len(Xc)
    trS = np.trace(Sig) / Sig.shape[0]
    Sw = np.zeros_like(Sig)
    for c in range(4):
        Xq = Ltr[y_tr == c] - Ltr[y_tr == c].mean(0)
        Sw += Xq.T @ Xq
    Sw /= len(Ltr)
    trW = np.trace(Sw) / Sw.shape[0]

    def whiten(F, S, tr, lam, center):
        S_l = (1 - lam) * S + lam * tr * np.eye(S.shape[0])
        w, V = np.linalg.eigh(S_l)
        A = (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T
        return (F - center) @ A
    for lam in (0.2,):
        W_ = [whiten(L, Sig, trS, lam, mu0) for L in (Ltr, Lva, Lte)]
        m2w = med2_of(W_[0])
        rep('E2_covwhite_%.1f' % lam, SVC(kernel='rbf', C=100., gamma=3.0 / (2 * m2w))
            .fit(W_[0], y_tr).predict(W_[2]) == y_te, '协方差白化')
        W_ = [whiten(L, Sw, trW, lam, 0.0) for L in (Ltr, Lva, Lte)]
        m2w = med2_of(W_[0])
        rep('E3_withinwhite_%.1f' % lam, SVC(kernel='rbf', C=100., gamma=3.0 / (2 * m2w))
            .fit(W_[0], y_tr).predict(W_[2]) == y_te, '类内白化')

    # ---------- F. 多尺度核 ----------
    print('=== F. multi-scale kernel ===', flush=True)
    GMS = (0.5, 1.0, 2.0, 4.0, 8.0)
    D2 = {'tt': cdist(Ltr, Ltr, 'sqeuclidean'), 'vt': cdist(Lva, Ltr, 'sqeuclidean'),
          'et': cdist(Lte, Ltr, 'sqeuclidean')}
    K = {key: {g: np.exp(-g / (2 * M2) * D) for g in GMS} for key, D in D2.items()}

    def mix_eval(w, C_=100.):
        Ktr = sum(w[i] * K['tt'][g] for i, g in enumerate(GMS))
        Kva = sum(w[i] * K['vt'][g] for i, g in enumerate(GMS))
        Ket = sum(w[i] * K['et'][g] for i, g in enumerate(GMS))
        s = SVC(kernel='precomputed', C=C_).fit(Ktr, y_tr)
        return float((s.predict(Kva) == y_va).mean()), s.predict(Ket) == y_te
    w5 = np.ones(5) / 5
    v5, ok5 = mix_eval(w5)
    rep('F1_unif5', ok5, 'val %.1f' % (v5 * 100))
    bestF = None
    for i, j in ((0, 4), (1, 3), (0, 2), (2, 4)):
        for wf in (0.35, 0.5, 0.65):
            w = np.zeros(5); w[i] = wf; w[j] = 1 - wf
            v, ok = mix_eval(w)
            if bestF is None or v > bestF[0]:
                bestF = (v, i, j, wf, ok)
    v, i, j, wf, ok = bestF
    rep('F2_best2scale_g%.1f%.2f_g%.1f' % (GMS[i], wf, GMS[j]), ok, 'val %.1f(过拟合检验)' % (v * 100))
    del K, D2

    # ---------- G. 局部尺度核 + 核形状 ----------
    print('=== G. local scaling / other kernel shapes ===', flush=True)
    All = np.concatenate([Ltr, Lva, Lte], 0)
    ntr, nva = len(Ltr), len(Lva)
    Dall = cdist(All, All, 'sqeuclidean')
    np.fill_diagonal(Dall, np.inf)
    s = np.sqrt(np.sort(Dall, axis=1)[:, 6])
    Kfull = np.exp(-Dall / (s[:, None] * s[None, :]))
    Ktr = Kfull[:ntr, :ntr]
    Kva = Kfull[ntr:ntr + nva, :ntr]
    Kte = Kfull[ntr + nva:, :ntr]
    s2 = SVC(kernel='precomputed', C=10.).fit(Ktr, y_tr)
    rep('G1_localscale_k7', s2.predict(Kte) == y_te,
        'val %.1f' % ((s2.predict(Kva) == y_va).mean() * 100))
    del Kfull, Dall
    D1 = {'tt': cdist(Ltr, Ltr, 'cityblock'), 'vt': cdist(Lva, Ltr, 'cityblock'),
          'et': cdist(Lte, Ltr, 'cityblock')}
    m1 = float(np.median(D1['tt']))
    g = 2.0 / (2 * m1)
    s3 = SVC(kernel='precomputed', C=100.).fit(np.exp(-g * D1['tt']), y_tr)
    rep('G2_laplace_gm2', s3.predict(np.exp(-g * D1['et'])) == y_te,
        'val %.1f' % ((s3.predict(np.exp(-g * D1['vt'])) == y_va).mean() * 100))
    del D1

    # ---------- H. 多切图 + 核(线性增益与核增益可加否) ----------
    print('=== H. multi-chart + RBF kernel ===', flush=True)
    C_3 = [cov_tn(X[:, :, TASK[0]:TASK[1]]) for X in (X_tr, X_va, X_te)]
    for K_ in (4,):   # K=8 探索读数 73.0(test)/74.9(val) 已记录,不重复跑(逐样本 eigh 太慢)
        rng = np.random.default_rng(0)
        refs = [C_3[0][i] for i in rng.choice(len(y_tr), K_, replace=False)]
        F = [np.concatenate([chart_feats(C, r)[0] for r in refs], 1) for C in C_3]
        m2k = med2_of(F[0])
        bb = None
        for C_ in (100.,):
            for gmk in (0.5, 1., 2., 3.):
                sk = SVC(kernel='rbf', C=C_, gamma=gmk / (2 * m2k)).fit(F[0], y_tr)
                vv = float((sk.predict(F[1]) == y_va).mean())
                if bb is None or vv > bb[0]:
                    bb = (vv, gmk, sk)
        vv, gmk, sk = bb
        rep('H%d_multichart%d_svm_gm%.1f' % (K_, F[0].shape[1], gmk),
            sk.predict(F[2]) == y_te, 'val %.1f' % (vv * 100))
        rep('H%d_multichart%d_lda' % (K_, F[0].shape[1]),
            fit_lda(F[0], y_tr).predict(F[2]) == y_te, '同特征线性对照')

    (OUT / 'kernel_ceiling.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/kernel_ceiling.json' % OUT, flush=True)


if __name__ == '__main__':
    main()
