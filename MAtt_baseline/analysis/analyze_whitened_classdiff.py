# -*- coding: utf-8 -*-
"""§11.6 的白化检查:先在被试内做协方差白化,再看类差分还剩多少、还对不对齐。

## 要检验的命题(论文初稿 §10.2,现为 L4 推断)
「无监督个体化算子的估计量对类标签置换不变 ⇒ 只能作用于类共同统计量,
  而类共同统计量对判别在代数上不可见 ⇒ 无监督个体化在类差分上没有可对准的目标。」
其可检验形式:被试内协方差白化(EA 一族)是「无监督个体化」里最强的一类算子。
若命题成立,白化**不应**改变类差分方向的被试间异质度(它动不了 Δ 的方向);
若白化后 36 对余弦的均值显著上升 / 离散度显著下降(趋向共享主方向),
则 Δ 在白化后变得可共享 ⇒ 出口 2(把类差分估计做成共享对象)获得前置依据。

## 三臂(同一被试、同一批试次、同一频带与窗,均零训练)
  raw    : 非 EA 电极空间直接算 —— 复现论文表 11b 的输入(36 对余弦 -0.758~+0.811,
           sd 0.452 / PC1 0.499),作为可比性检查点
  eaW    : 信号空间白化。P_s = R_s^{-1/2}(R_s = 该被试全部试次、整个 epoch 的
           迹归一化协方差,**不含标签**,与管线 EA 同构造);X_w = P_s X 后重算全套
  featW  : 特征空间白化。对逐试次 22 维 log μ 带功率向量,按该被试跨试次协方差
           (不含标签)白化 —— 这是任何「按人二阶矩归一化」层(如 I-4)作用的上界形式

## 每臂读数
  方向几何:36 对 E_diff 单位向量余弦(min/mean/sd)、PC1 方差份额、
            留一模板余弦、低四/高五/跨组分组均值;对照随机正交零分布
            (22 维、9 个 iid 单位向量、20000 次)
  幅度与可分性:逐人 ||E_diff||(D_obs)、μ 功率层 5 折收缩 LDA、与 deep70_acc 的 Spearman

只写 results/whitened_classdiff/。约 20 分钟(GDF 重放是瓶颈)。纯 CPU。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))
import analyze_erd_decomposition as ED                     # noqa: E402
from alignment import compute_ea_matrix_from_numpy         # noqa: E402

OUT = BASE / 'results' / 'whitened_classdiff'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'
LOW = ('S02', 'S04', 'S05', 'S06')
ARMS = ('raw', 'eaW', 'featW')


def ch_erd(A, R):
    """逐通道 Pfurtscheller ERD(dB) —— 与 analyze_erd_class_diff.ch_erd 逐位同式。"""
    return 10 * np.log10(A.mean(0) / R.mean(0))


def lda5(F, yy):
    accs = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(F, yy):
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F[tr], yy[tr])
        accs.append(float((clf.predict(F[te]) == yy[te]).mean()))
    return float(np.mean(accs))


def dir_geometry(E, names):
    """9 个 22 维 E_diff 的单位向量几何 + 分组均值 + 留一模板余弦 + PC1 份额。"""
    U = E / np.linalg.norm(E, axis=1, keepdims=True)
    C = U @ U.T
    n = len(names)
    off = C[~np.eye(n, dtype=bool)]
    is_low = np.array([nm in LOW for nm in names])
    low_pairs = C[np.ix_(is_low, is_low)][~np.eye(int(is_low.sum()), dtype=bool)]
    high_pairs = C[np.ix_(~is_low, ~is_low)][~np.eye(int((~is_low).sum()), dtype=bool)]
    loo = [float(U[i] @ (U.sum(0) - U[i]) / np.linalg.norm(U.sum(0) - U[i])) for i in range(n)]
    return {'cos36_min': float(off.min()), 'cos36_max': float(off.max()),
            'cos36_mean': float(off.mean()), 'cos36_sd': float(off.std(ddof=1)),
            'pc1_share': float(PCA(1).fit(U).explained_variance_ratio_[0]),
            'group_mean_cosine': {'low4_within': float(low_pairs.mean()),
                                  'high5_within': float(high_pairs.mean()),
                                  'cross': float(C[np.ix_(is_low, ~is_low)].mean())},
            'loo_template_cosine': {nm: round(v, 4) for nm, v in zip(names, loo)}}


def null_orthogonal(rng, n_vec=9, dim=22, reps=20000):
    """随机正交零分布:dim 维 iid 单位向量组的 36 对余弦 sd 与 PC1 份额。"""
    sds, pc1s = [], []
    for _ in range(reps):
        G = rng.normal(size=(n_vec, dim))
        U = G / np.linalg.norm(G, axis=1, keepdims=True)
        C = U @ U.T
        sds.append(C[~np.eye(n_vec, dtype=bool)].std(ddof=1))
        pc1s.append(float(PCA(1).fit(U).explained_variance_ratio_[0]))
    q = lambda a: [round(float(np.percentile(a, p)), 4) for p in (2.5, 97.5)]
    return {'cos36_sd_mean': float(np.mean(sds)), 'cos36_sd_95': q(sds),
            'pc1_share_mean': float(np.mean(pc1s)), 'pc1_share_95': q(pc1s)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}
    from moabb_bci2a import load_subject_mi_trials, VALID_SUBJECTS

    per, edges = {}, {a: {} for a in ARMS}
    for i, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        name = 'S%02d' % (i + 1)
        iaf = ref[name]['iaf_hz']
        L, Rr, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, Rr, Fe, To], 0).astype(np.float64)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, Rr, Fe, To])])
        hL, hR = y == 0, y == 1

        # 信号空间白化(与管线 EA 同构造:全部试次、整个 epoch、迹归一化、不含标签)
        P = compute_ea_matrix_from_numpy(X)
        Xw = np.einsum('ij,bjt->bit', P, X)
        cov_w = np.einsum('bct,bet->ce', Xw, Xw)
        id_dev = float(np.abs(cov_w * 22 / np.trace(cov_w) - np.eye(22)).max())

        row = {'iaf_hz': round(float(iaf), 3), 'whitened_cov_dev_from_I': round(id_dev, 6)}
        F_raw = None
        for tag, Xi in (('raw', X), ('eaW', Xw)):
            A = ED.band_pow(Xi, iaf, ED.TASK[0], ED.TASK[1], 125)   # (n,22) 任务窗
            Rb = ED.band_pow(Xi, iaf, ED.REST[0], ED.REST[1], 0)    # (n,22) 静息窗
            E_diff = ch_erd(A[hL], Rb[hL]) - ch_erd(A[hR], Rb[hR])
            F = np.log(np.concatenate([A[hL], A[hR]], 0))
            yy = np.r_[np.zeros(hL.sum()), np.ones(hR.sum())]
            if tag == 'raw':
                F_raw = (F, yy)
            row[tag] = {'D_obs': round(float(np.linalg.norm(E_diff)), 4),
                        'acc_mu_topog': round(lda5(F, yy), 4),
                        'E_diff': [round(float(v), 4) for v in E_diff]}
            edges[tag][name] = E_diff
        # featW:对 raw 的 log μ 功率特征按被试跨试次协方差白化(不含标签)
        F0, yy0 = F_raw
        mu_f = F0.mean(0)
        Cf = np.cov((F0 - mu_f).T)
        ev, vec = np.linalg.eigh(Cf)
        Wf = vec @ np.diag(1.0 / np.sqrt(np.maximum(ev, 1e-12))) @ vec.T
        Fw = (F0 - mu_f) @ Wf
        Ew = Fw[yy0 == 0].mean(0) - Fw[yy0 == 1].mean(0)
        row['featW'] = {'D_obs': round(float(np.linalg.norm(Ew)), 4),
                        'acc_mu_topog': round(lda5(Fw, yy0), 4),
                        'E_diff': [round(float(v), 4) for v in Ew]}
        edges['featW'][name] = Ew
        per[name] = row
        print('  %s iaf=%.2f  D: raw %.3f / eaW %.3f / featW %.3f  LDA: %.1f%% / %.1f%% / %.1f%%'
              '  (cov_w−I dev %.1e)  (%.0fs)'
              % (name, iaf, row['raw']['D_obs'], row['eaW']['D_obs'], row['featW']['D_obs'],
                 row['raw']['acc_mu_topog'] * 100, row['eaW']['acc_mu_topog'] * 100,
                 row['featW']['acc_mu_topog'] * 100, id_dev, time.time() - t0), flush=True)

    names = list(per)
    acc = np.array([ref[n]['deep70_acc'] for n in names])
    geom, corr = {}, {}
    for a in ARMS:
        E = np.stack([edges[a][n] for n in names])
        geom[a] = dir_geometry(E, names)
        D = np.array([per[n][a]['D_obs'] for n in names])
        macc = np.array([per[n][a]['acc_mu_topog'] for n in names])
        corr[a] = {'D_vs_deep70': [round(float(v), 4) for v in spearmanr(D, acc)],
                   'muLDA_vs_deep70': [round(float(v), 4) for v in spearmanr(macc, acc)],
                   'D_per_subject': {n: per[n][a]['D_obs'] for n in names},
                   'muLDA_per_subject': {n: per[n][a]['acc_mu_topog'] for n in names}}
        print('\n=== 臂 %s ===' % a)
        print('  36 对余弦: %.3f ~ %.3f  均值 %+.3f  sd %.3f   PC1 份额 %.3f'
              % (geom[a]['cos36_min'], geom[a]['cos36_max'],
                 geom[a]['cos36_mean'], geom[a]['cos36_sd'], geom[a]['pc1_share']))
        print('  分组均值: 低四内 %+.3f  高五内 %+.3f  跨组 %+.3f'
              % (geom[a]['group_mean_cosine']['low4_within'],
                 geom[a]['group_mean_cosine']['high5_within'],
                 geom[a]['group_mean_cosine']['cross']))
        print('  ρ(D, deep70)=%+.3f (p=%.4f)   ρ(μLDA, deep70)=%+.3f (p=%.4f)'
              % (corr[a]['D_vs_deep70'][0], corr[a]['D_vs_deep70'][1],
                 corr[a]['muLDA_vs_deep70'][0], corr[a]['muLDA_vs_deep70'][1]))

    rng = np.random.default_rng(42)
    null = null_orthogonal(rng)
    print('\n=== 随机正交零分布(22 维、9 向量、20000 次) ===')
    print('  cos36 sd: %.3f (95%%: %s)   PC1 份额: %.3f (95%%: %s)'
          % (null['cos36_sd_mean'], null['cos36_sd_95'],
             null['pc1_share_mean'], null['pc1_share_95']))

    rec = {'purpose': 'paper §11.6 whitening check: does per-subject unsupervised whitening '
                      'align the class-differential directions (E_diff)?',
           'band': 'mu = per-subject IAF±1.5 (Welch Hann band integral), non-EA trials; '
                   'TASK=(375,1126), REST=(0,250), same as erd_class_diff',
           'arms': {'raw': 'non-EA electrode space (reproduces paper Tab 11b input)',
                    'eaW': 'signal-space whitening, P=R_s^{-1/2}, R_s = all-trial whole-epoch '
                           'trace-normalized covariance (label-free, same construction as pipeline EA)',
                    'featW': 'feature-space whitening of per-trial log mu power by per-subject '
                             'across-trial covariance (label-free)'},
           'low_group': list(LOW),
           'null_orthogonal': null, 'geometry': geom, 'corr': corr, 'per_subject': per}
    (OUT / 'whitened_classdiff.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/whitened_classdiff.json' % OUT)


if __name__ == '__main__':
    main()
