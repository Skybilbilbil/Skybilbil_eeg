"""给报告 §3.2/§3.3 的"范围读数"配上劈半噪声地板。

判据:同一被试的静息试次随机劈成两半,各算一次同一个量,两半之差 = 该量的估计波动;
      被试间离散度 ÷ 这个地板 = 有多少跨度是生理真差。这与报告 §3.6 的 M 谱判据同构,
      只是把对象从"M 谱"换成 IAF / α 功率 / 突出度 / 灶集中度 / 偏侧 / 有效秩。

实现要点:psd() 与 ch_psd() 与 cov() 都是"逐试次量的均值"(Welch 无重叠、静息窗正好一段),
      对试次集合线性 ⇒ 逐试次周期图/协方差只算一次,任意子集直接取均值,
      从而与 analyze_individual_diff.py::physiology() 在全体试次上逐位一致(脚本自带该校验)。

用法: cd MAtt_baseline && python analysis/analyze_split_half_floor.py [重复次数,默认30]
产物: results/split_half_floor/{split_half_floor.json, split_half_floor.log}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent        # = MAtt_baseline/
sys.path.insert(0, str(BASE / 'analysis'))
sys.path.insert(0, str(BASE.parent / 'preprocess'))

from analyze_individual_diff import (ALPHA_BAND, CH, REST, cov, eff_rank,        # noqa: E402
                                     physiology, FS)
from moabb_bci2a import VALID_SUBJECTS, load_subject_mi_trials                   # noqa: E402

OUT = BASE / 'results' / 'split_half_floor'
TABLE = BASE / 'results' / 'individual_diff' / 'subject_table.json'
NPZ = None  # 逐试次周期图太大不落盘,只在内存里用
METRICS = ['iaf_hz', 'alpha_pow_db', 'alpha_snr_db', 'mu_focus_rel',
           'c3c4_asym_db', 'rest_eff_rank']
FREQ_W = 1024   # nfft=2048 → 1025 个格点


def per_trial(seg):
    """(n,22,250) -> (P_tr (n,F,22), C_tr (n,22,22)),与 psd/ch_psd/cov 同参数同口径。"""
    from scipy.signal import welch
    nps = min(256, seg.shape[-1])
    f, P = welch(seg, fs=FS, nperseg=nps, noverlap=0, nfft=2048, axis=-1)   # (n,22,F)
    P = np.transpose(P, (0, 2, 1)).astype(np.float32)                       # (n,F,22)
    s = seg - seg.mean(axis=2, keepdims=True)
    C = np.einsum('nct,ndt->ncd', s, s) / seg.shape[-1]
    return np.asarray(f), P, C


def metrics_from(f, P_tr, C_tr, idx):
    """给定试次子集 idx,复现 physiology() 的 6 个静息读数。"""
    Pb = P_tr[idx].mean(axis=0).astype(np.float64)          # (F,22)  = ch_psd 的均值
    g = Pb.mean(axis=1)                                     # (F,)    = psd 的全脑均值

    m_a = (f >= ALPHA_BAND[0]) & (f <= ALPHA_BAND[1])
    gi = int(np.where(m_a)[0][np.argmax(g[m_a])])
    iaf = float(f[gi])
    if 0 < gi < len(g) - 1:
        a, b, c = (np.log(g[gi - 1] + 1e-30), np.log(g[gi] + 1e-30), np.log(g[gi + 1] + 1e-30))
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            iaf += float(0.5 * (a - c) / den * (f[1] - f[0]))

    lo, hi = iaf - 1.5, iaf + 1.5
    mm = (f >= lo) & (f <= hi)
    band = float(np.trapezoid(g[mm], f[mm])) if mm.sum() > 1 else float('nan')
    alpha_pow = 10 * np.log10(band + 1e-30)
    alpha_snr = 10 * np.log10((g[m_a].max() + 1e-30) /
                              (np.median(g[(f >= 5) & (f <= 30)]) + 1e-30))
    bpc = Pb[mm].mean(axis=0)                               # 逐通道带功率
    focus = float(bpc.max() / (bpc.mean() + 1e-30))
    asym = 10 * np.log10((bpc[CH['C4']] + 1e-30) / (bpc[CH['C3']] + 1e-30))
    rank = eff_rank(C_tr[idx].mean(axis=0).astype(np.float64))
    return dict(iaf_hz=iaf, alpha_pow_db=alpha_pow, alpha_snr_db=alpha_snr,
                mu_focus_rel=focus, c3c4_asym_db=asym, rest_eff_rank=rank)


def main(n_rep=30, seed=0):
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    full, sd_half, per_subj = {}, {}, {}
    t_all = time.time()

    for i, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        L, R, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, R, Fe, To], 0).astype(np.float64)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, R, Fe, To])])
        n = X.shape[0]
        f, P_tr, C_tr = per_trial(X[:, :, REST[0]:REST[1]])

        # ---- 校验 1:全体试次上我的实现必须复现 physiology() / subject_table ----
        mine = metrics_from(f, P_tr, C_tr, np.arange(n))
        ref = physiology(X, y)
        dev = {k: abs(mine[k] - ref[k]) for k in METRICS}
        full[sid] = {k: ref[k] for k in METRICS}          # 以 physiology() 为准(与报告表同源)
        # ---- 劈半 ----
        d = {k: [] for k in METRICS}
        for _ in range(n_rep):
            p = rng.permutation(n)
            A, B = p[:n // 2], p[n // 2:]
            ma, mb = metrics_from(f, P_tr, C_tr, A), metrics_from(f, P_tr, C_tr, B)
            for k in METRICS:
                d[k].append(ma[k] - mb[k])
        sd_half[sid] = {k: float(np.std(d[k], ddof=1)) for k in METRICS}
        per_subj[sid] = {k: [round(v, 4) for v in d[k]] for k in METRICS}
        print('  S%02d  n=%d  复现physiology最大偏差=%.2e  (%.0fs)'
              % (sid, n, max(dev.values()), time.time() - t0), flush=True)
        if max(dev.values()) > 2e-3:
            print('  *** 校验未通过: %s ***' % dev, flush=True)

    # ---- 汇总:地板(n) = SD(半半差)/2;跨度用 n=9 极差的噪声期望 ----
    rows = {k: {} for k in METRICS}
    vals = {k: [full[s][k] for s in VALID_SUBJECTS] for k in METRICS}
    for k in METRICS:
        v = np.asarray(vals[k], float)
        s_obs = float(np.std(v, ddof=1))
        s_noisy = float(np.sqrt(np.mean([sd_half[s][k] ** 2 for s in VALID_SUBJECTS])))
        # 独立正态下 SD(half-diff)=sqrt(2)*SD(半样本);SD(全样本)=SD(半样本)/sqrt(2) => /2
        s_floor = s_noisy / 2.0
        ratio = s_obs / s_floor if s_floor > 0 else float('inf')
        true_frac = float(np.sqrt(max(0.0, 1.0 - 1.0 / ratio ** 2))) if ratio > 1 else 0.0
        # n=9 的纯噪声极差期望 ≈ 3.06*SD (顺序统计量)
        span_obs = float(v.max() - v.min())
        span_noise = 3.0615 * s_floor
        rows[k] = dict(
            observed_sd=round(s_obs, 4), split_half_sd_diff=round(s_noisy, 4),
            noise_floor_sd_at_full_n=round(s_floor, 4),
            ratio_between_over_floor=round(ratio, 2),
            true_variance_fraction=round(true_frac, 3),
            observed_range=round(span_obs, 4),
            range_expected_from_noise_alone=round(span_noise, 4),
            range_excess_fraction=round(max(0.0, 1 - span_noise / span_obs), 3)
            if span_obs > 0 else None,
        )

    rep = dict(
        n_rep=n_rep, seed=seed, note=('地板按 SD(全样本)=SD(半半差)/2 外推,该式对线性估计器'
                                     '(带功率、协方差)精确;iaf_hz 是峰值位置、非均值型,'
                                     '外推只是近似,其比率应读作量级判断。'),
        per_subject_sd_of_half_diff=sd_half, per_subject_d=per_subj,
        full_sample_values=full, summary=rows,
    )
    with open(OUT / 'split_half_floor.json', 'w', encoding='utf-8') as fh:
        json.dump(rep, fh, ensure_ascii=False, indent=1)

    print('\n===== 范围读数的劈半地板(n=%d 被试, %d 次重复, seed=%d) =====' % (len(VALID_SUBJECTS), n_rep, seed))
    print('%-15s %8s %8s %8s %8s %9s %10s' % ('指标', '被试间SD', '劈半差SD', '地板SD', '比值', '真差占比', '噪声应得跨度'))
    for k in METRICS:
        r = rows[k]
        print('%-15s %8.3f %8.3f %8.3f %8.2f %8.1f%% %10.3f   (实测跨度 %.3f)'
              % (k, r['observed_sd'], r['split_half_sd_diff'], r['noise_floor_sd_at_full_n'],
                 r['ratio_between_over_floor'], 100 * r['true_variance_fraction'],
                 r['range_expected_from_noise_alone'], r['observed_range']))
    print('\n[OK] %s/split_half_floor.json  (总耗时 %.0fs)' % (OUT, time.time() - t_all))


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)
