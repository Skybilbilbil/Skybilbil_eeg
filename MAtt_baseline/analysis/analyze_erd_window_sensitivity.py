# -*- coding: utf-8 -*-
"""轴 2(手类 μ ERD 深度)的时间窗敏感性：epoch 从 cue−1.0 s 起切，会不会让 ERD 读数偏浅？

两个独立的可测问题(都用现有非 EA 数据，不重放 GDF、不训练)：

Q1 基线污染。REST=(0,250) 就是 cue 前那 1 秒，整段贴着 cue。运动前 1 s 常有准备性
   去同步 ⇒ 基线本身可能已被压低 ⇒ ERD 被系统性估浅，且压低量因人而异(污染轴 2 的排序)。
   判据：把基线切成前/后两个 0.5 s，看 μ 带功率之比 R_late/R_early 是否 < 1(后段更低=有漂移)，
   并比较"以早半段为基线"与"以晚半段为基线"两种 ERD。

Q2 任务窗稀释。TASK=(375,1126)=+0.5~+3.5 s，而时间过程显示 +2.0 s 起回升、+2.2 s 后与基线
   不可区分 ⇒ 平台段与回升段被平均在一起 ⇒ 读数被稀释，且回升快的人被稀释得多。
   判据：换 6 个任务窗重算同一口径的 ERD，各自与 deep70_acc 做 Spearman，看 ρ 与逐人跨度是否
   随窗口变化——若 ρ 稳，轴 2 的结论对窗口不敏感；若平台窗更强，则该换。

运行: python analysis/analyze_erd_window_sensitivity.py
产物: results/erd_window_sensitivity/{erd_window_sensitivity.json, .log}
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
sys.path.insert(0, str(BASE.parent / 'preprocess'))
from analyze_erd_decomposition import band_pow, CH, LAT, BCI2A_CHANNELS, FS   # noqa: E402
from moabb_bci2a import load_subject_mi_trials, VALID_SUBJECTS  # noqa: E402

SUBJ = ["S%02d" % k for k in VALID_SUBJECTS]

OUT = BASE / 'results' / 'erd_window_sensitivity'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'
HAND = [0, 1]                                   # 左/右手类
# 任务窗(样本; cue 在 250)。noverlap=125 需窗长 >=375 才有 2 段以上
TASKW = {'+0.5~+3.5 现行': (375, 1126), '+0.5~+2.5': (375, 875), '+0.5~+2.0 平台': (375, 750),
         '+1.0~+2.0': (500, 750), '+0.5~+1.5': (375, 625), '+1.0~+2.5': (500, 875)}
ALLCH = BCI2A_CHANNELS
def NOVL(a, b):
    """Welch 段重叠数。关键：nperseg=min(250,L) 且 noverlap=0 时，scipy 会【静默丢掉】
    第一个段之后的余数样本 —— 375 样本的窗会退化成只测前 250 点(实测与 250 窗给出同一
    数字)。故 L>=375 必须用 noverlap=125，让段与段首尾相接覆盖整个窗。"""
    return 125 if (b - a) >= 375 else 0


def covered(a, b):
    """断言 Welch 真的覆盖了 [a,b)，不覆盖就当场报错，不再让结果静默失真。"""
    L, npg = b - a, min(250, b - a)
    nseg = 1 + (L - npg) // NOVL(a, b) if NOVL(a, b) else max(1, L // npg)
    end = (nseg - 1) * (NOVL(a, b) if NOVL(a, b) else npg) + npg
    assert L - end <= 1, ('Welch 未覆盖整个窗', a, b, L, nseg, end)
    return nseg


def band_pow_half(X, iaf, a, b):
    """0.5 s 半窗专用带功率。必须补零到 nfft=250 拿回 1 Hz 格点：
    nperseg=125 时格点是 2 Hz，而 IAF±1.5 Hz 只有 3 Hz 宽 ⇒ 常只落进 1 个 bin，
    而 np.trapezoid 对单点积分恒为 0(实测 5/9 人被试直接 divide by zero)。
    两个半窗用同一估计器，故彼此可比；其绝对 dB 不可与 1 s 窗口径混用。"""
    from scipy.signal import welch
    f, P = welch(X[:, :, a:b], fs=FS, nperseg=b - a, nfft=250, noverlap=0, axis=-1)
    m = (f >= iaf - 1.5) & (f <= iaf + 1.5)
    assert m.sum() >= 2, ('带内 bin 不足，带积分会退化为 0', int(m.sum()))
    return np.trapezoid(P[..., m], f[m], axis=-1)


def erd(A, R):
    return 10.0 * np.log10(A.mean() / R.mean())


def main():
    ref = {r['subject']: r for r in json.load(open(TBL, encoding='utf-8'))}
    rows = {}
    for sid, name in zip(VALID_SUBJECTS, SUBJ):
        L, R, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, R, Fe, To], 0).astype(np.float64)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, R, Fe, To])])
        m = np.isin(y, HAND)
        Xh, yh = X[m], y[m]                     # 只留手类：口径与 brain_hand/contra_elec 一致
        iaf = ref[name]['iaf_hz']
        rec = {}
        rest_full = band_pow(Xh, iaf, 0, 250, 0)          # 现行基线：cue 前整 1 s，单段
        idx = [CH[c] for c in ALLCH]
        # Q1 基线漂移：两个 0.5 s 半窗(同一估计器，nperseg=125 ⇒ 2 Hz 格点，两侧同受影响)
        pe = band_pow_half(Xh, iaf, 0, 125)
        pl = band_pow_half(Xh, iaf, 125, 250)
        covered(0, 250); covered(0, 125); covered(125, 250)
        rec['base_late_over_early_db'] = erd(pl, pe)
        # Q2 任务窗 × 两种基线(现行整 1 s；以及早/晚半段)的 ERD
        for tag, (a, b) in TASKW.items():
            nseg = covered(a, b)
            rec['nseg__' + tag] = nseg
            task = band_pow(Xh, iaf, a, b, NOVL(a, b))
            rec['brain_hand__' + tag] = erd(task[:, idx].mean(-1), rest_full[:, idx].mean(-1))
            # contra_elec: 左 cue→C4、右 cue→C3。必须"先跨 trial 平均功率、再取比值"
            # —— 逐 trial 先取比值 = E[A/R] = E[A]·E[1/R]，静息窗只 1 段 ≈6 自由度，
            #    倒数矩偏差会把 ERD 抬高约 +2 dB(报告 §3.4 已实测并弃用该口径)。
            A_c, R_c = [], []
            for c, side in ((0, 'left'), (1, 'right')):
                mm = yh == c
                ch = CH[LAT[side][0]]
                A_c.append(task[mm][:, ch])
                R_c.append(rest_full[mm][:, ch])
            rec['contra_elec__' + tag] = erd(np.concatenate(A_c), np.concatenate(R_c))
            # 基线取早半段 / 晚半段(任务窗固定为现行)
            if tag == '+0.5~+3.5 现行':
                for bn, bw in (('early_half', (0, 125)), ('late_half', (125, 250))):
                    rr = band_pow_half(Xh, iaf, bw[0], bw[1])
                    rec['brain_hand__baseref_' + bn] = erd(task[:, idx].mean(-1), rr[:, idx].mean(-1))
        rows[name] = rec
        print('%s iaf=%.2f  基线漂移=%+.3f dB  brain_hand(现行)=%+.3f  平台窗=%+.3f' % (
            name, iaf, rec['base_late_over_early_db'], rec['brain_hand__+0.5~+3.5 现行'],
            rec['brain_hand__+0.5~+2.0 平台']), flush=True)

    d70 = np.array([ref[s]['deep70_acc'] for s in SUBJ])
    res = {'per_subject': rows, 'window_corr': {}, 'baseline': {}}
    print('\n=== Q2 任务窗敏感性：各窗的 brain_hand / contra_elec 与 deep70_acc ===')
    print('%-18s %4s %9s %9s %9s %9s %9s' % ('任务窗', '段数', 'BH均值', 'BH跨度', 'ρ(BH,d70)', 'CE均值', 'ρ(CE,d70)'))
    for tag in TASKW:
        bh = np.array([rows[s]['brain_hand__' + tag] for s in SUBJ])
        ce = np.array([rows[s]['contra_elec__' + tag] for s in SUBJ])
        rb, pb = spearmanr(bh, d70)
        rc, pc = spearmanr(ce, d70)
        res['window_corr'][tag] = {'brain_hand_mean': float(bh.mean()),
                                   'brain_hand_span': float(bh.max() - bh.min()),
                                   'rho_brain_hand': float(rb), 'p_brain_hand': float(pb),
                                   'contra_mean': float(ce.mean()),
                                   'contra_span': float(ce.max() - ce.min()),
                                   'rho_contra': float(rc), 'p_contra': float(pc)}
        print('%-18s %4d %+9.3f %9.3f %+9.3f(p=%.4f) %+9.3f %+9.3f(p=%.4f)' % (
            tag, rows[SUBJ[0]]['nseg__' + tag], bh.mean(), bh.max() - bh.min(), rb, pb,
            ce.mean(), rc, pc))
    print('\n=== Q1 基线：cue 前 1 s 内部有无去同步漂移 ===')
    drift = np.array([rows[s]['base_late_over_early_db'] for s in SUBJ])
    e = np.array([rows[s]['brain_hand__baseref_early_half'] for s in SUBJ])
    l = np.array([rows[s]['brain_hand__baseref_late_half'] for s in SUBJ])
    f = np.array([rows[s]['brain_hand__+0.5~+3.5 现行'] for s in SUBJ])
    print('R_late/R_early (dB, 负=后段功率更低=基线在漂移): 均值 %+.3f  范围 %+.3f~%+.3f  %d/%d 为负' % (
        drift.mean(), drift.min(), drift.max(), int((drift < 0).sum()), len(drift)))
    print('brain_hand 以早半段为基线 %+.3f | 现行整 1 s %+.3f | 以晚半段为基线 %+.3f' % (
        e.mean(), f.mean(), l.mean()))
    print('  ⇒ 早半段基线比现行更深 %+.3f dB(若为正，说明越靠近 cue 的基线越被压低 ⇒ ERD 被估浅)' % (
        f.mean() - e.mean()))
    for k, v in (('early_half', e), ('full_1s', f), ('late_half', l)):
        rr, pp = spearmanr(v, d70)
        res['baseline'][k] = {'mean': float(v.mean()), 'span': float(v.max() - v.min()),
                              'rho': float(rr), 'p': float(pp)}
        print('  基线=%-9s 均值%+.3f 跨度%.3f  ρ(BH,d70)=%+.3f p=%.4f' % (k, v.mean(),
              v.max() - v.min(), rr, pp))
    res['note'] = ('0.5 s 半窗用 nperseg=125(2 Hz 格点)，两个半窗同受影响 ⇒ 只可用于半窗之间的'
                   '相对比较，其绝对 dB 不可与 1 s 基线口径混用。')
    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(OUT / 'erd_window_sensitivity.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n[OK]', OUT / 'erd_window_sensitivity.json')


if __name__ == '__main__':
    main()
