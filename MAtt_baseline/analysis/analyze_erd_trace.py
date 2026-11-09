# -*- coding: utf-8 -*-
"""从【已确定的量】向下追溯 μ ERD 深度。

已确定的量(报告 §4 / 关键数据 §2 轴②):
    Q := brain_hand = 10*log10( mean_i A_i / mean_i R_i )    rho(Q, deep70_acc) = -0.950
    A = 手类 trial 任务窗 μ 带功率, R = 同一 trial 静息窗 μ 带功率,
    22 通道先平均功率, 带 = 各人 IAF ± 1.5 Hz(非 EA 数据, Welch 带积分)。

本脚本只做「向下」三问, 不加任何与 Q 同层的兄弟指标:

  L1 恒等式拆分: Q = T - B, T := 10log10(mean A), B := 10log10(mean R)。
      问: 与精度同序的那部分到底在 T(任务期功率) 还是在 B(静息功率)?
      判据: rho(T,acc)、rho(B,acc)、偏相关 rho(Q,acc | B)、rho(Q,acc | T)。
      附: Q 是两项之差 ⇒ 若 T、B 高度相关, Q 的跨被试方差是"大数相减"的残差,
      此时 rho(Q,acc) 讲的是"谁的任务-静息落差大", 不是"谁的绝对水平高"。

  L2 量得准不准(带捕获 / 估计噪声):
      (a) 带臂: IAF±1.5(已确定口径) vs 固定 8-13 vs IAF±3 vs 宽带 8-30。
          若换固定带后某人的 Q 大幅漂移, 那部分"ERD 深度"就是频带位置产物。
      (b) 跨被试劈半信度: 每个被试的 trial 随机劈半各算一次 Q/T/B,
          在 9 人上相关 + Spearman-Brown ⇒ 跨被试跨度里有多少是估计噪声。
          同一份带臂也复算, 与 L1 的结论交叉。

  L3 是"总是被抑制"还是"少数试次拉出来":
      逐 trial d_i = r_i - a_i(dB), 报 frac(d>0) 与效应量 d_z = mean(d)/sd(d)。

检查点: 本脚本 iaf15 臂的 Q 必须与 results/erd_decomposition/erd_decomposition.json
的 per_subject[*].brain_hand 逐位一致(同一估计量, 同一窗, 同一带)。
数据: 非 EA(moabb_bci2a.load_subject_mi_trials), 与 erd_decomposition 同源。
只写 results/erd_trace/。约 12 分钟(9 个被试 GDF 重放是瓶颈)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import pearsonr, rankdata, spearmanr

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))
from mAtt.mAtt import BCI2A_CHANNELS                      # noqa: E402
import analyze_erd_decomposition as ED                    # noqa: E402

OUT = BASE / 'results' / 'erd_trace'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'
REF = BASE / 'results' / 'erd_decomposition' / 'erd_decomposition.json'
FS = ED.FS
REST, TASK = ED.REST, ED.TASK
NSEG = ED.NPERSEG
ARMS = [('iaf15', -1.5, 1.5), ('fix8_13', None, None), ('iaf30', -3.0, 3.0),
        ('bb8_30', None, None)]
FIXED = {'fix8_13': (8.0, 13.0), 'bb8_30': (8.0, 30.0)}
PERF = ('deep70_acc', 'csp_within')


def bpow(X, lo, hi, a, b, noverlap):
    """与 ED.band_pow 同一实现, 只把频带写成 (lo, hi) 以便换带。"""
    f, P = welch(X[:, :, a:b], fs=FS, nperseg=min(NSEG, b - a),
                 noverlap=noverlap, axis=-1)
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(P[..., m], f[m], axis=-1)


def split_tb(A, R):
    """10log10(mean A) / 10log10(mean R) / 差 —— 三个量同一批 trial。"""
    T, B = 10 * np.log10(A.mean()), 10 * np.log10(R.mean())
    return float(T), float(B), float(T - B)


def boot_q(A, R, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        k = rng.integers(len(A), size=len(A))
        bs[i] = A[k].mean() / R[k].mean()
    return [float(np.percentile(10 * np.log10(bs), 2.5)),
            float(np.percentile(10 * np.log10(bs), 97.5))]


def sh_reliability(pairs, n_rep=200, seed=1):
    """跨被试劈半信度: 每人 trial 随机劈半各算 T/B/Q, 在 9 人上相关。

    pairs = [(A_i, R_i)]。返回 {T/B/Q: {r_mean, sb, r_sd}}; sb = Spearman-Brown
    矫正到全样本长度的信度 2r/(1+r)。
    """
    rng = np.random.default_rng(seed)
    cols = {k: [] for k in ('T', 'B', 'Q')}
    for _ in range(n_rep):
        h1, h2 = {k: [] for k in cols}, {k: [] for k in cols}
        for A, R in pairs:
            p = rng.permutation(len(A))
            for dst, k in ((h1, p[:len(A) // 2]), (h2, p[len(A) // 2:])):
                t, b, q = split_tb(A[k], R[k])
                dst['T'].append(t), dst['B'].append(b), dst['Q'].append(q)
        for k in cols:
            cols[k].append(float(pearsonr(h1[k], h2[k])[0]))
    out = {}
    for k, v in cols.items():
        r = float(np.mean(v))
        out[k] = {'r_mean': round(r, 4), 'r_sd': round(float(np.std(v)), 4),
                  'spearman_brown': round(2 * r / (1 + r), 4)}
    return out


def pcorr(x, y, z):
    """秩偏相关: 先转秩再线性剔除 z。返回 (r, p, df)。"""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = pearsonr(ex, ey)
    return round(float(r), 4), round(float(p), 5), len(rx) - 3


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}
    pub = json.loads(REF.read_text(encoding='utf-8'))['per_subject']
    from moabb_bci2a import load_subject_mi_trials, VALID_SUBJECTS

    per, pairs, cons = {}, {}, {}
    for i, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        name = 'S%02d' % (i + 1)
        iaf = ref[name]['iaf_hz']
        L, R_, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, R_, Fe, To], 0).astype(np.float64)
        n = len(X)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, R_, Fe, To])])
        h = np.isin(y, [0, 1])
        ci_idx = np.arange(n) % (n // 4)          # 类内序号(每类等长且按类连续)
        s0 = h & (ci_idx < n // 8)                # 每类前一半 = session0

        # 频带臂: 先对全 X 算带功率, 再取手类
        arms = {}
        for tag, dlo, dhi in ARMS:
            lo, hi = FIXED[tag] if tag in FIXED else (iaf + dlo, iaf + dhi)
            Rm = bpow(X, lo, hi, REST[0], REST[1], 0)
            Am = bpow(X, lo, hi, TASK[0], TASK[1], 125)
            arms[tag] = (Am, Rm, (lo, hi))
        # 检查点: iaf15 臂必须复现 ED.band_pow
        chk = np.abs(arms['iaf15'][0] - ED.band_pow(X, iaf, TASK[0], TASK[1], 125)).max()
        chk = max(chk, np.abs(arms['iaf15'][1] - ED.band_pow(X, iaf, REST[0], REST[1], 0)).max())

        row = {'iaf_hz': round(float(iaf), 3), 'n_hand_trials': int(h.sum()),
               'arm_reproduction_max_abs_diff': float(chk), 'arms': {}}
        for tag in [a[0] for a in ARMS]:
            Am, Rm, (lo, hi) = arms[tag]
            A, Rr = Am[h].mean(-1), Rm[h].mean(-1)          # 手类, 22 通道先平均功率
            T, B, Q = split_tb(A, Rr)
            row['arms'][tag] = {'band': [round(lo, 2), round(hi, 2)],
                                'T_db': round(T, 4), 'B_db': round(B, 4),
                                'Q_db': round(Q, 4)}
            if tag == 'iaf15':
                row['arms'][tag]['Q_ci95'] = [round(v, 4) for v in boot_q(A, Rr, seed=i)]
                pairs[name] = (A, Rr)
                a, r = 10 * np.log10(A), 10 * np.log10(Rr)
                d = r - a
                cons[name] = {'frac_pos': round(float((d > 0).mean()), 4),
                              'd_z': round(float(d.mean() / d.std(ddof=1)), 4),
                              'sd_a_db': round(float(a.std(ddof=1)), 3),
                              'sd_r_db': round(float(r.std(ddof=1)), 3),
                              'rho_a_r': round(float(spearmanr(a, r)[0]), 4)}
                # 逐 session(手类):
                for tag2, m in (('s0', h & s0), ('s1', h & ~s0)):
                    t2, b2, q2 = split_tb(Am[m].mean(-1), Rm[m].mean(-1))
                    row.setdefault('by_session', {})[tag2] = {
                        'T_db': round(t2, 4), 'B_db': round(b2, 4), 'Q_db': round(q2, 4)}
        per[name] = row
        print('  %s iaf=%.2f  检查点 %.2e  T=%+.3f B=%+.3f Q=%+.3f dB  (%.0fs)'
              % (name, iaf, chk, row['arms']['iaf15']['T_db'],
                 row['arms']['iaf15']['B_db'], row['arms']['iaf15']['Q_db'],
                 time.time() - t0), flush=True)

    # ---------- 检查点 ----------
    dif = {s: round(per[s]['arms']['iaf15']['Q_db'] - pub[s]['brain_hand'], 6) for s in per}
    chk_ok = max(abs(v) for v in dif.values()) < 1e-6
    print('\n=== 检查点: 本脚本 Q vs 已发布 brain_hand ===')
    print('最大偏差 %.2e dB  %s' % (max(abs(v) for v in dif.values()),
                                   'PASS' if chk_ok else '**FAIL**'))

    # ---------- L1 恒等式拆分 ----------
    S = list(per)
    T = np.array([per[s]['arms']['iaf15']['T_db'] for s in S])
    B = np.array([per[s]['arms']['iaf15']['B_db'] for s in S])
    Q = np.array([per[s]['arms']['iaf15']['Q_db'] for s in S])
    acc = np.array([ref[s]['deep70_acc'] for s in S])
    csp = np.array([ref[s]['csp_within'] for s in S])
    print('\n=== L1  Q = T - B 的跨被试构成(n=9, dB) ===')
    print('        %8s %8s %8s %8s' % ('sd', 'min', 'max', 'range'))
    for nm, v in (('T 任务', T), ('B 静息', B), ('Q=ERD', Q)):
        print('%-8s %8.3f %8.3f %8.3f %8.3f' % (nm, v.std(ddof=1), v.min(), v.max(),
                                                v.max() - v.min()))
    print('恒等式核对: var(Q) %.4f = var(T) %.4f + var(B) %.4f - 2cov %.4f -> %.4f'
          % (Q.var(ddof=1), T.var(ddof=1), B.var(ddof=1), 2 * np.cov(T, B)[0, 1],
             T.var(ddof=1) + B.var(ddof=1) - 2 * np.cov(T, B)[0, 1]))
    print('rho(T,B) = %+.3f  (T、B 共享的分量既不进 Q, 也不可能是 Q 的成因)'
          % spearmanr(T, B)[0])
    print('\n%-22s %14s %14s' % ('量', 'rho x deep70_acc', 'rho x csp_within'))
    corr = {}
    for nm, v in (('T 10log10(任务手类μ)', T), ('B 10log10(静息μ)', B),
                  ('Q ERD = T-B', Q)):
        r1, p1 = spearmanr(v, acc)
        r2, p2 = spearmanr(v, csp)
        corr[nm] = {'vs_deep70_acc': [round(float(r1), 4), round(float(p1), 6)],
                    'vs_csp_within': [round(float(r2), 4), round(float(p2), 6)]}
        print('%-22s %8.3f p=%.4f %8.3f p=%.4f' % (nm, r1, p1, r2, p2))
    print('\n偏相关(控制另一项后 Q 还剩下什么):')
    part = {}
    for nm, z in (('B 静息', B), ('T 任务', T)):
        r, p, df = pcorr(Q, acc, z)
        part[nm] = [r, p, df]
        print('  rho(Q, deep70_acc | %s) = %+.3f  p=%.4f  df=%d' % (nm, r, p, df))
    r, p, df = pcorr(T, acc, B)
    print('  (对照) rho(T, deep70_acc | B) = %+.3f  p=%.4f  df=%d' % (r, p, df))

    # ---------- L2a 带臂 ----------
    print('\n=== L2a  换带后 Q 的漂移(dB) ===')
    print('%-6s %10s %10s %10s %10s' % ('被试', 'IAF±1.5', '固定8-13', 'IAF±3', '宽带8-30'))
    band_tab = {}
    for s in S:
        v = [per[s]['arms'][k]['Q_db'] for k in ('iaf15', 'fix8_13', 'iaf30', 'bb8_30')]
        band_tab[s] = v
        print('%-6s %10.3f %10.3f %10.3f %10.3f' % (s, *v))
    V = np.array([band_tab[s] for s in S])
    print('跨被试 sd:  ' + ' '.join('%10.3f' % v for v in V.std(0, ddof=1)))
    d_fix = V[:, 1] - V[:, 0]
    r, p = spearmanr(d_fix, acc)
    print('rho(固定8-13 相对 IAF±1.5 的漂移, deep70_acc) = %+.3f p=%.4f  '
          '(漂移 span %.3f dB, 均值 %+.3f)'
          % (r, p, d_fix.max() - d_fix.min(), d_fix.mean()))
    for j, nm in ((1, '固定8-13'), (2, 'IAF±3'), (3, '宽带8-30')):
        rr, pp = spearmanr(V[:, j], acc)
        print('  rho(Q[%s], deep70_acc) = %+.3f p=%.4f' % (nm, rr, pp))

    # ---------- L2b 跨被试劈半信度 ----------
    print('\n=== L2b  跨被试劈半信度(200 次劈半, 9 人上的相关 + Spearman-Brown) ===')
    rel = {}
    for arm in ('iaf15', 'fix8_13', 'bb8_30'):
        pr = []
        for s in S:
            A, Rr = pairs[s]
            if arm != 'iaf15':   # 用主臂的 trial 结构重算其它带代价高, 只报主臂
                break
            pr.append((A, Rr))
        if pr:
            rel[arm] = sh_reliability(pr)
            print('%s:' % arm)
            for k, d in rel[arm].items():
                print('   %-8s r=%+.3f (sd %.3f)  SB=%.3f' % (k, d['r_mean'],
                                                              d['r_sd'], d['spearman_brown']))
    print('参考: 逐 trial 估计量本身的单试次信度 rho≈0.19(X-0 噪声预算)')

    # ---------- L3 抑制的一致性 ----------
    print('\n=== L3  逐 trial 抑制的一致性(手类 trial) ===')
    print('%-6s %10s %8s %9s %9s %9s' % ('被试', 'frac(d>0)', 'd_z', 'sd_a', 'sd_r', 'rho(a,r)'))
    for s in S:
        c = cons[s]
        print('%-6s %10.3f %8.2f %9.3f %9.3f %9.3f'
              % (s, c['frac_pos'], c['d_z'], c['sd_a_db'], c['sd_r_db'], c['rho_a_r']))
    fp = np.array([cons[s]['frac_pos'] for s in S])
    dz = np.array([cons[s]['d_z'] for s in S])
    for nm, v in (('frac(d>0)', fp), ('d_z', dz)):
        r1, p1 = spearmanr(v, acc)
        r2, p2 = spearmanr(v, Q)
        print('%-10s x deep70_acc rho=%+.3f p=%.4f | x Q rho=%+.3f p=%.4f'
              % (nm, r1, p1, r2, p2))

    # ---------- session 稳定性 ----------
    print('\n=== 附带: T/B/Q 的 session 内稳定性(每类前/后半) ===')
    print('%-6s %8s %8s %8s %8s' % ('被试', 'dT', 'dB', 'dQ', '|dQ|'))
    dT = np.array([per[s]['by_session']['s1']['T_db'] - per[s]['by_session']['s0']['T_db']
                   for s in S])
    dB = np.array([per[s]['by_session']['s1']['B_db'] - per[s]['by_session']['s0']['B_db']
                   for s in S])
    dQ = np.array([per[s]['by_session']['s1']['Q_db'] - per[s]['by_session']['s0']['Q_db']
                   for s in S])
    for j, s in enumerate(S):
        print('%-6s %+8.3f %+8.3f %+8.3f %8.3f' % (s, dT[j], dB[j], dQ[j], abs(dQ[j])))
    for nm, v in (('dT', dT), ('dB', dB), ('dQ', dQ)):
        print('%-4s 跨被试 sd %.3f  与 deep70_acc 的 rho %+.3f p=%.4f'
              % (nm, v.std(ddof=1), *spearmanr(v, acc)))

    rec = {'quantity': 'Q = brain_hand = 10log10(mean_i A_i / mean_i R_i), hand classes, '
                       '22ch mean power, per-subject IAF±1.5, non-EA, Welch(Hann)',
           'checkpoint_vs_published_brain_hand': {'max_abs_diff_db':
                                                  max(abs(v) for v in dif.values()),
                                                  'pass': bool(chk_ok)},
           'L1': {'per_subject_T_B_Q': {s: per[s]['arms']['iaf15'] for s in S},
                  'sd_db': {'T': round(float(T.std(ddof=1)), 4),
                            'B': round(float(B.std(ddof=1)), 4),
                            'Q': round(float(Q.std(ddof=1)), 4)},
                  'rho_T_B': round(float(spearmanr(T, B)[0]), 4),
                  'corr_with_perf': corr, 'partial_corr': part},
           'L2a_band_arms': {'per_subject_Q': band_tab,
                             'sd_by_arm': [round(float(v), 4) for v in V.std(0, ddof=1)],
                             'drift_fix_vs_iaf': [round(float(v), 4) for v in d_fix],
                             'rho_drift_acc': [round(float(r), 4), round(float(p), 6)]},
           'L2b_split_half_reliability': rel,
           'L3_consistency': cons,
           'session_within': {'per_subject_dT': [round(float(v), 4) for v in dT],
                              'per_subject_dB': [round(float(v), 4) for v in dB],
                              'per_subject_dQ': [round(float(v), 4) for v in dQ]},
           'per_subject': per}
    (OUT / 'erd_trace.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                        encoding='utf-8')
    print('\n[OK] %s/erd_trace.json' % OUT)


if __name__ == '__main__':
    main()
