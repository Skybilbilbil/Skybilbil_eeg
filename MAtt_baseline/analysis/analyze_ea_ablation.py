# -*- coding: utf-8 -*-
"""EA / 非 EA 的读出层配对消融（零训练、纯 CPU）。

回答: EA 的效果在「表示 + 线性读出」这一层有多大,是否依赖深度模型。
三条协议、两个读出器、两臂:
  TS  = 22ch 任务窗协方差 -> Log-Euclidean 切空间(253d, 相对 ridge 1e-3) + LDA(shrinkage=auto)
  CSP = one-vs-rest k=2 + reg=0.05 -> log 方差(8d) + LDA(shrinkage=auto)
  ① within_5fold  逐人 576 试次分层 5 折
  ② cross_session 每类前 72=s0 / 后 72=s1 的单方向与双向平均
  ③ loso_9x9      9x9 逐对迁移(i 训 -> j 测),与 72.5%/30.5% 同口径
非 EA 臂: X_raw = P^{-1} X_EA, P 来自 subject_metadata[i]['ea_matrix']
(EA 可逆, 已实测 r>=0.99999985; 脚本内再做一次 L1 协方差恒等式自检)。

判据: EA-nonEA 的逐人配对差 >= 5pp 才算可测; 报 bootstrap CI 与 MDE80。
产物: results/ea_ablation/ea_ablation.{json,log}
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ttest_1samp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset  # noqa: E402
from analyze_individual_diff import KEYS, TASK, cov, csp_acc, fit_csp  # noqa: E402
from analyze_spd_separability import tangent, trial_cov  # noqa: E402

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'ea_ablation'
N = 9
N_BOOT = 10000


def ts_feats(X):
    """与 analyze_spd_separability 同款: 不去均值、协方差/段长 -> 切空间"""
    return tangent(trial_cov(X, TASK[0], TASK[1]))


def csp_feats(W, C):
    """与 csp_acc 内部同款: W^T C W 的对数方差"""
    return np.log(np.maximum(np.einsum('ia,nij,ja->na', W, C, W), 1e-12))


def lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


def paired(d):
    """逐人配对差 -> 均值/bootstrap CI/配对 t/MDE80"""
    d = np.asarray(d, dtype=float)
    n = len(d)
    mean, sd = float(d.mean()), float(d.std(ddof=1))
    se = sd / np.sqrt(n)
    t, p = ttest_1samp(d, 0.0)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
    ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    mde = float((2.306 + 0.8416) * sd / np.sqrt(n))   # n=9, two-sided .05, power .8
    return {'mean': mean, 'sd': sd, 'se': se, 't': float(t), 'p': float(p),
            'ci95': ci, 'mde80': mde, 'per_subject': [float(v) for v in d]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, md = load_preprocessed_dataset(str(PKL))
    assert len(sd) == N and len(md) == N

    arms = {'EA': [], 'nonEA': []}
    for i, (X, y) in enumerate(sd):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        P = np.asarray(md[i]['ea_matrix'], dtype=np.float64)
        Pinv = np.linalg.inv(P)
        assert np.allclose(P @ Pinv, np.eye(22), atol=1e-6), 'P 不可逆?'
        X_non = np.einsum('ij,njt->nit', Pinv, X)
        C_ea = cov(X[:, :, TASK[0]:TASK[1]])
        C_non = cov(X_non[:, :, TASK[0]:TASK[1]])
        C_chk = np.einsum('ij,njk,lk->nil', Pinv, C_ea, Pinv)
        rel = float(np.max(np.abs(C_non - C_chk)) / (np.max(np.abs(C_non)) + 1e-30))
        assert rel < 1e-6, f'S{i+1} 逆变换协方差不一致 rel={rel}'
        arms['EA'].append((X, y, C_ea, ts_feats(X)))
        arms['nonEA'].append((X_non, y, C_non, ts_feats(X_non)))
        print(f'  S{i+1:02d}  cond(P)={np.linalg.cond(P):.2e}  逆变换 rel={rel:.1e}', flush=True)

    # session 索引: 每类前 72=s0 / 后 72=s1(与 analyze_individual_diff 同约定)
    sess = np.concatenate([np.r_[np.zeros(144 // 2, int), np.ones(144 - 144 // 2, int)]
                           for _ in KEYS])
    m0, m1 = np.where(sess == 0)[0], np.where(sess == 1)[0]

    res = {'meta': {'pkl': str(PKL), 'n_subjects': N, 'task_window': list(TASK),
                    'readouts': {'TS': '22ch cov -> LE tangent 253d + LDA(shrinkage=auto)',
                                 'CSP': 'one-vs-rest k=2, reg=0.05, log-var 8d + LDA(shrinkage=auto)'},
                    'nonEA': "X_raw = P^{-1} X_EA, P from subject_metadata[i].ea_matrix"},
           'protocols': {}, 'paired': {}}

    # ---- ① within 5-fold ----
    within = {'TS': {'EA': [], 'nonEA': []}, 'CSP': {'EA': [], 'nonEA': []}}
    for arm in ('EA', 'nonEA'):
        for i, (X, y, C, F) in enumerate(arms[arm]):
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            a_ts = [lda().fit(F[tr], y[tr]).score(F[te], y[te]) for tr, te in skf.split(F, y)]
            a_csp = [csp_acc(C, y, tr, te) for tr, te in StratifiedKFold(
                5, shuffle=True, random_state=42).split(C, y)]
            within['TS'][arm].append(float(np.mean(a_ts)))
            within['CSP'][arm].append(float(np.mean(a_csp)))
            print(f'  [within/{arm}] S{i+1:02d} TS={np.mean(a_ts)*100:.1f} '
                  f'CSP={np.mean(a_csp)*100:.1f}', flush=True)
    for rd in ('TS', 'CSP'):
        res['protocols'].setdefault('within_5fold', {})[rd] = {
            'EA': within[rd]['EA'], 'nonEA': within[rd]['nonEA']}
        d = np.array(within[rd]['EA']) - np.array(within[rd]['nonEA'])
        res['paired'][f'within_5fold_{rd}'] = paired(d)

    # ---- ② cross session ----
    cross = {'TS': {'EA': {}, 'nonEA': {}}, 'CSP': {'EA': {}, 'nonEA': {}}}
    for arm in ('EA', 'nonEA'):
        for rd in ('TS', 'CSP'):
            cross[rd][arm] = {'B': [], 'C': []}
        for i, (X, y, C, F) in enumerate(arms[arm]):
            cross['TS'][arm]['B'].append(float(lda().fit(F[m0], y[m0]).score(F[m1], y[m1])))
            cross['TS'][arm]['C'].append(float(lda().fit(F[m1], y[m1]).score(F[m0], y[m0])))
            cross['CSP'][arm]['B'].append(float(csp_acc(C, y, m0, m1)))
            cross['CSP'][arm]['C'].append(float(csp_acc(C, y, m1, m0)))
            print(f'  [cross/{arm}] S{i+1:02d} TS {cross["TS"][arm]["B"][-1]*100:.1f}/'
                  f'{cross["TS"][arm]["C"][-1]*100:.1f}  CSP '
                  f'{cross["CSP"][arm]["B"][-1]*100:.1f}/{cross["CSP"][arm]["C"][-1]*100:.1f}',
                  flush=True)
    for rd in ('TS', 'CSP'):
        for arm in ('EA', 'nonEA'):
            b = cross[rd][arm]['B']; c = cross[rd][arm]['C']
            res['protocols'].setdefault('cross_session', {}).setdefault(rd, {})[arm] = {
                'B': b, 'C': c, 'BC': [(x + y) / 2 for x, y in zip(b, c)]}
        for key in ('B', 'C', 'BC'):
            d = (np.array(res['protocols']['cross_session'][rd]['EA'][key])
                 - np.array(res['protocols']['cross_session'][rd]['nonEA'][key]))
            res['paired'][f'cross_{key}_{rd}'] = paired(d)

    # ---- ③ 9x9 迁移 ----
    loso = {'TS': {'EA': None, 'nonEA': None}, 'CSP': {'EA': None, 'nonEA': None}}
    for arm in ('EA', 'nonEA'):
        M_ts = np.zeros((N, N)); M_csp = np.zeros((N, N))
        for i in range(N):
            Xi, yi, Ci, Fi = arms[arm][i]
            clf_ts = lda().fit(Fi, yi)
            W = fit_csp(Ci, yi)
            clf_csp = lda().fit(csp_feats(W, Ci), yi)
            for j in range(N):
                Xj, yj, Cj, Fj = arms[arm][j]
                M_ts[i, j] = clf_ts.score(Fj, yj)
                M_csp[i, j] = clf_csp.score(csp_feats(W, Cj), yj)
        loso['TS'][arm] = M_ts
        loso['CSP'][arm] = M_csp
        off_ts = (M_ts.sum() - np.trace(M_ts)) / (N * N - N)
        off_csp = (M_csp.sum() - np.trace(M_csp)) / (N * N - N)
        print(f'  [9x9/{arm}] TS offdiag={off_ts*100:.1f}%  CSP offdiag={off_csp*100:.1f}%',
              flush=True)
    for rd in ('TS', 'CSP'):
        for arm in ('EA', 'nonEA'):
            M = loso[rd][arm]
            col = (M.sum(0) - np.diag(M)) / (N - 1)
            res['protocols'].setdefault('loso_9x9', {}).setdefault(rd, {})[arm] = {
                'matrix_pct': np.round(M * 100, 2).tolist(),
                'offdiag_mean': float(((M.sum() - np.trace(M)) / (N * N - N)) * 100),
                'colmean_per_subject': (col * 100).tolist()}
        # colmean_per_subject 存的是百分数, 配对统计要换回比例
        d = (np.array(res['protocols']['loso_9x9'][rd]['EA']['colmean_per_subject'])
             - np.array(res['protocols']['loso_9x9'][rd]['nonEA']['colmean_per_subject'])) / 100.0
        res['paired'][f'loso_9x9_{rd}'] = paired(d)

    # ---- ④ LOSO: 8 人合并训练 -> 1 人测试(文献同款设定; 仅科学对照, 不对外报数) ----
    merged = {'TS': {'EA': [], 'nonEA': []}, 'CSP': {'EA': [], 'nonEA': []}}
    for arm in ('EA', 'nonEA'):
        F_all = np.concatenate([arms[arm][k][3] for k in range(N)])
        y_all = np.concatenate([arms[arm][k][1] for k in range(N)])
        C_all = np.concatenate([arms[arm][k][2] for k in range(N)])
        subj_idx = np.repeat(np.arange(N), 576)
        for j in range(N):
            te = np.where(subj_idx == j)[0]
            tr = np.setdiff1d(np.arange(len(y_all)), te)
            merged['TS'][arm].append(float(
                lda().fit(F_all[tr], y_all[tr]).score(F_all[te], y_all[te])))
            merged['CSP'][arm].append(float(csp_acc(C_all, y_all, tr, te)))
            print(f'  [loso_merged/{arm}] S{j+1:02d} TS={merged["TS"][arm][-1]*100:.1f} '
                  f'CSP={merged["CSP"][arm][-1]*100:.1f}', flush=True)
    for rd in ('TS', 'CSP'):
        res['protocols'].setdefault('loso_merged', {})[rd] = {
            'EA': merged[rd]['EA'], 'nonEA': merged[rd]['nonEA']}
        d = np.array(merged[rd]['EA']) - np.array(merged[rd]['nonEA'])
        res['paired'][f'loso_merged_{rd}'] = paired(d)

    print('\n=== 配对差 EA - nonEA (pp) ===')
    for k, v in res['paired'].items():
        print(f'  {k:24s} mean={v["mean"]*100:+6.2f}  sd={v["sd"]*100:5.2f}  '
              f'CI95=[{v["ci95"][0]*100:+6.2f},{v["ci95"][1]*100:+6.2f}]  '
              f'MDE80={v["mde80"]*100:5.2f}  p={v["p"]:.4f}')

    (OUT / 'ea_ablation.json').write_text(json.dumps(res, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
    print(f'\n[OK] {OUT}/ea_ablation.json')


if __name__ == '__main__':
    main()
