# -*- coding: utf-8 -*-
"""逐步验证 analyze_erd_decomposition.py 的每个环节【真的实现了它声明的功能】。

设计原则: 每个检查都用【解析真值已知】的合成信号。期望值由公式独立算出, 不拿被测
代码自己的输出当基准。合成信号一律 = 白噪声 + IAF 正弦, 且【幅度调制是唯一变量】,
所以任何非零 ERD 都只能来自被检验的那一步(映射/选类/选窗/带掩膜/比值口径)。

覆盖的环节:
  1 通道名->索引 CH / ROI / LAT 的解剖正确性
  2 band_pow: 带掩膜的频率选择性与有效带宽(1 Hz bin 量化)
  3 erd_db: 点估计口径、尺度不变性、n_boot=0、bootstrap CI 语义
  4 roi_pow: 选类、先跨通道平均功率(而非先取比)、形状
  5 subject_erd 偏侧化映射: 左 cue->对侧 C4 / 右 cue->对侧 C3(反向写会立刻暴露)
  6 M1/M2/M3 三个机制方向与解析值
  7 旧 erd_c3/c4/min 口径的缺陷被定量复现(证明修正确有必要)
  8 时间过程: TC_STARTS/TC_LEN/窗心时刻/基线全窗共用/逐人曲线形状
  9 组统计: SEM / Wilcoxon 单侧方向语义 / 跨被试 t 区间 / 留一法
 10 产物一致性: JSON 内部自洽 + 报告引用的数字与 JSON 逐项对上
 11 (--real) 真数据端到端: 2x2 解剖反对称 + 非 EA 确认

用法(WSL, eeg_analysis):
    python MAtt_baseline/analysis/verify_erd_pipeline.py            # 1-10, 秒级
    python MAtt_baseline/analysis/verify_erd_pipeline.py --real     # 追加 11, 约 2 分钟
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr, t as tdist, wilcoxon

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
sys.path.insert(0, str(BASE))
from analyze_erd_decomposition import (  # noqa: E402
    BCI2A_CHANNELS, CH, FS, LAT, MIDLINE, NPERSEG, REST, ROI_L, ROI_R, TASK,
    TC_LEN, TC_STARTS, TC_STEP, band_pow, erd_db, roi_pow, subject_erd)

IAF = 10.0
T = 1126
N_PER_CLS = 60
S_DROP = 0.6                                  # 幅度乘子 -> 功率乘子 0.36
DB_DROP = 20.0 * np.log10(S_DROP)             # 解析真值 -4.4370 dB

FAIL = []
NCHK = [0]


def chk(name, ok, detail=''):
    NCHK[0] += 1
    if not ok:
        FAIL.append(name)
    print('%-4s %s  %s' % ('' if ok else 'FAIL', 'PASS' if ok else '!!!!', name))
    if detail:
        print('       %s' % detail)


def close(a, b, tol, unit='dB'):
    return abs(a - b) <= tol, 'meas %+.4f %s vs 解析 %+.4f %s (tol %.4f)' % (a, unit, b, unit, tol)


def synth(seed=0, noise=1.0, tone=2.0):
    """(4*n, 22, T) 白噪声 + 全通道 IAF 正弦; y = 0 left / 1 right / 2 feet / 3 tongue"""
    rng = np.random.default_rng(seed)
    n = 4 * N_PER_CLS
    X = rng.standard_normal((n, 22, T)) * noise
    t = np.arange(T) / FS
    ph = rng.uniform(0, 2 * np.pi, 22)[:, None]
    X += (tone * np.sin(2 * np.pi * IAF * t + ph))[None]
    y = np.repeat(np.arange(4), N_PER_CLS)
    return X, y


def mod(X, y, chs, cls, a, b, s):
    """把 chs x cls 试次在样本 [a,b) 的幅度乘 s(功率乘 s^2)"""
    idx = np.where(np.isin(y, cls))[0]
    ci = [CH[c] for c in chs] if chs is not None else list(range(22))
    X[np.ix_(idx, ci, np.arange(a, b))] *= s
    return X


ALL = BCI2A_CHANNELS


def scalar(e, k):
    return erd_db(*e[k])[0]


# ---------------------------------------------------------------- 1 通道映射
print('\n=== 1 通道名 -> 索引 / ROI / LAT 的解剖正确性 ===')
chk('C3 与 C4 都在 22 通道表里且互异', CH['C3'] != CH['C4'],
    'CH[C3]=%d CH[C4]=%d len=%d' % (CH['C3'], CH['C4'], len(BCI2A_CHANNELS)))
chk('对侧 ROI(右)含 C4 不含 C3', 'C4' in ROI_R and 'C3' not in ROI_R)
chk('同侧 ROI(左)含 C3 不含 C4', 'C3' in ROI_L and 'C4' not in ROI_L)
chk('两侧 ROI 不相交且各 7 通道', not set(ROI_R) & set(ROI_L) and len(set(ROI_R)) == 7 == len(set(ROI_L)))
chk('中线 ROI 不含 C3/C4', not {'C3', 'C4'} & set(MIDLINE))
chk('LAT: 左 cue 对侧=C4 / 右 cue 对侧=C3',
    LAT['left'] == ('C4', 'C3') and LAT['right'] == ('C3', 'C4'), str(LAT))
chk('窗定义: 静息在 cue 前, 任务在 cue 后', REST == (0, 250) and TASK[0] >= 250,
    'REST=%s TASK=%s(cue 在样本 250)' % (str(REST), str(TASK)))

# ---------------------------------------------------------------- 2 band_pow
print('\n=== 2 band_pow: 频率选择性 / 有效带宽 ===')
rng = np.random.default_rng(1)
w = rng.standard_normal((400, 1, T))                      # 单位方差白噪声
f, P = welch(w[:, :, REST[0]:REST[1]], fs=FS, nperseg=NPERSEG, noverlap=0, axis=-1)
chk('Welch 频率分辨率为 1 Hz(nperseg=250 @250Hz)', np.allclose(np.diff(f), 1.0),
    'df=%.4f Hz, bins=%d' % (np.diff(f)[0], len(f)))
# 单位方差白噪声单边 PSD 期望 = 2/fs; 带积分 = 带宽 x 2/fs
bp_noise = band_pow(w, IAF, REST[0], REST[1], 0)[:, 0].mean()
eff_bw = bp_noise / (2.0 / FS)
nb = int(((f >= IAF - 1.5) & (f <= IAF + 1.5)).sum())
chk('带掩膜确实生效(有效带宽 = 名义 3 Hz 的量化结果)', 1.0 <= eff_bw <= 3.2,
    '实测有效带宽 %.3f Hz; 名义 IAF±1.5 = 3 Hz, 因 1 Hz bin 量化实际只取 %d 个 bin'
    % (eff_bw, nb))
tt = np.arange(T) / FS
ref = band_pow(np.broadcast_to(3.0 * np.sin(2 * np.pi * IAF * tt), (200, 1, T)).copy(),
               IAF, REST[0], REST[1], 0)[:, 0].mean()
for off, tag in ((0.0, '带内'), (8.0, '带外 +8Hz')):
    x = np.zeros((200, 1, T))
    x[:, 0] = 3.0 * np.sin(2 * np.pi * (IAF + off) * tt)
    v = band_pow(x, IAF, REST[0], REST[1], 0)[:, 0].mean()
    if off == 0:
        chk('带内正弦被计入(功率 ~ A^2/2)', close(v, 4.5, 1.0, 'power')[0],
            'meas %.3f vs A^2/2=4.5' % v)
    else:
        chk('带外正弦被排除', v < 0.02 * ref, '带外/带内 = %.5f' % (v / ref))
# 多段平均只降方差、不改期望(任务窗 751 样本: noverlap 0 -> 3 段, 125 -> 5 段)
nseg = lambda L, ov: (L - NPERSEG) // (NPERSEG - ov) + 1
chk('段数与 docstring 声明一致(静息 1 段 / 任务 5 段)',
    nseg(REST[1] - REST[0], 0) == 1 and nseg(TASK[1] - TASK[0], 125) == 5,
    'rest=%d task=%d' % (nseg(250, 0), nseg(751, 125)))
v0 = band_pow(w, IAF, TASK[0], TASK[1], 0)[:, 0]
v5 = band_pow(w, IAF, TASK[0], TASK[1], 125)[:, 0]
chk('noverlap 多段平均: 期望不变、方差下降',
    abs(v5.mean() - v0.mean()) < 0.05 * v0.mean() and v5.var() < 0.8 * v0.var(),
    'mean %.5f->%.5f  var %.3e->%.3e(3 段->5 段, 比值 %.2f)'
    % (v0.mean(), v5.mean(), v0.var(), v5.var(), v5.var() / v0.var()))

# ---------------------------------------------------------------- 3 erd_db
print('\n=== 3 erd_db: 口径与统计性质 ===')
A = np.array([0.36, 0.72, 0.18, 0.54]); R = np.array([1.0, 1.3, 0.6, 1.9])   # 不相关, 避免退化
hand = 10 * np.log10(A.mean() / R.mean())
pt, ci = erd_db(A, R, n_boot=500)
chk('点估计 == 10log10(mean A / mean R) 手算值', close(pt, hand, 1e-9)[0], close(pt, hand, 1e-9)[1])
chk('n_boot=0 返回 (point, None)', erd_db(A, R, n_boot=0)[1] is None)
chk('CI 是升序二元组且夹住点估计', ci[0] <= pt <= ci[1], 'CI=[%.3f,%.3f] pt=%.3f' % (ci[0], ci[1], pt))
big = erd_db(np.tile(A, 40), np.tile(R, 40), n_boot=500)
chk('样本量增大 -> CI 变窄', big[1][1] - big[1][0] < ci[1] - ci[0],
    'CI 宽度 %.3f -> %.3f' % (ci[1] - ci[0], big[1][1] - big[1][0]))
X, y = synth()
chk('A、R 同乘常数 -> ERD 不变(dB 是比值)',
    close(scalar(subject_erd(X * 7.3, y, IAF), 'brain_all'),
          scalar(subject_erd(X, y, IAF), 'brain_all'), 1e-9)[0])

# ---------------------------------------------------------------- 4 roi_pow
print('\n=== 4 roi_pow: 选类 / 先跨通道平均功率 ===')
X, y = synth()
X = mod(X, y, ['C3', 'C4'], [0, 1], TASK[0], TASK[1], S_DROP)
A2, R2 = roi_pow(band_pow(X, IAF, TASK[0], TASK[1], 125), band_pow(X, IAF, *REST, 0),
                 y, [0, 1], ['C3', 'C4'])
chk('选类正确: 只取手类, 行数 = 2*每类试次', len(A2) == 2 * N_PER_CLS, 'n=%d' % len(A2))
got = 10 * np.log10(A2.mean() / R2.mean())
chk('ROI 选类后 ERD == 解析真值', close(got, DB_DROP, 0.05)[0], close(got, DB_DROP, 0.05)[1])
# 先平均功率 vs 先取比再平均 —— 不等价, 证明代码走前者
Xh = synth(seed=3)[0]
task = band_pow(Xh, IAF, TASK[0], TASK[1], 125); rest = band_pow(Xh, IAF, *REST, 0)
m = np.isin(y, [0, 1])
task[m, CH['C3']] *= 0.2 ** 0.5; task[m, CH['C4']] *= 3.0 ** 0.5   # 通道间极不均
pooled = 10 * np.log10(task[m][:, [CH['C3'], CH['C4']]].mean(-1).mean()
                       / rest[m][:, [CH['C3'], CH['C4']]].mean(-1).mean())
perchan = np.mean([10 * np.log10(task[m, k].mean() / rest[m, k].mean())
                   for k in (CH['C3'], CH['C4'])])
Aa, Ra = roi_pow(task, rest, y, [0, 1], ['C3', 'C4'])
chk('roi_pow = 先跨通道平均功率再求比(非逐通道取比再平均)',
    close(10 * np.log10(Aa.mean() / Ra.mean()), pooled, 1e-9)[0] and abs(pooled - perchan) > 0.5,
    'pooled %+.3f == roi_pow %+.3f; 逐通道口径 %+.3f(相差 %.2f dB)'
    % (pooled, 10 * np.log10(Aa.mean() / Ra.mean()), perchan, abs(pooled - perchan)))

# ---------------------------------------------------------------- 5 偏侧化映射
print('\n=== 5 对侧/同侧映射(反向写会立刻暴露) ===')
X, y = synth()
X = mod(X, y, ['C4'], [0], TASK[0], TASK[1], S_DROP)     # 左手 cue -> C4 降
X = mod(X, y, ['C3'], [1], TASK[0], TASK[1], S_DROP)     # 右手 cue -> C3 降
e = subject_erd(X, y, IAF)
c, cc = scalar(e, 'contra_elec'), erd_db(*e['contra_elec'], n_boot=500)[1]
i, ci = scalar(e, 'ipsi_elec'), erd_db(*e['ipsi_elec'], n_boot=500)[1]
chk('contra_elec == 解析真值 %+.4f dB' % DB_DROP, close(c, DB_DROP, 0.06)[0], close(c, DB_DROP, 0.06)[1])
chk('ipsi_elec == 0(同侧未被调制)', close(i, 0.0, 0.06)[0], close(i, 0.0, 0.06)[1])
chk('contra 与 ipsi 之差 == 真值且 CI 不重叠',
    c < i and (c - i) - DB_DROP < 0.12 and cc[1] < ci[0],
    'contra CI [%.2f,%.2f] vs ipsi CI [%.2f,%.2f]' % (cc[0], cc[1], ci[0], ci[1]))
X2 = mod(mod(synth(seed=5)[0], y, ['C3'], [0], TASK[0], TASK[1], S_DROP),
         y, ['C4'], [1], TASK[0], TASK[1], S_DROP)       # 故意把映射写反
e2 = subject_erd(X2, y, IAF)
chk('鉴别力: 映射写反时 contra 不再给出深抑制',
    abs(scalar(e2, 'contra_elec') - 0.0) < 0.06 and abs(scalar(e2, 'ipsi_elec') - DB_DROP) < 0.06,
    '反写 -> contra %+.3f / ipsi %+.3f(应互换)' % (scalar(e2, 'contra_elec'), scalar(e2, 'ipsi_elec')))

# ---------------------------------------------------------------- 6 三机制
print('\n=== 6 M1/M2/M3 方向与解析值 ===')
X, y = synth(seed=7)
X = mod(X, y, ALL, [0, 1], TASK[0], TASK[1], S_DROP)
X = mod(X, y, ALL, [2, 3], TASK[0], TASK[1], 1.25)       # feet/tongue 净增强
e = subject_erd(X, y, IAF)
ba, bh = scalar(e, 'brain_all'), scalar(e, 'brain_hand')
exp_ba = 10 * np.log10((2 * S_DROP ** 2 + 2 * 1.25 ** 2) / 4)
chk('M3 类别抵消: brain_all 被冲淡到解析值', close(ba, exp_ba, 0.06)[0],
    close(ba, exp_ba, 0.06)[1])
chk('M3: brain_hand 完整保留真值', close(bh, DB_DROP, 0.06)[0], close(bh, DB_DROP, 0.06)[1])
# M2/M1: 按解剖正确的双侧调制(左 cue -> 右半球 ROI_R, 右 cue -> 左半球 ROI_L)
X2 = mod(mod(synth(seed=9)[0], y, ROI_R, [0], TASK[0], TASK[1], S_DROP),
         y, ROI_L, [1], TASK[0], TASK[1], S_DROP)
e2 = subject_erd(X2, y, IAF)
exp_brain = 10 * np.log10((7 * S_DROP ** 2 + 15) / 22)     # 22 通道里只有 7 个降
exp_cen = 10 * np.log10((7 * S_DROP ** 2 + 7) / 14)        # 中央 14 通道里 7 个降
exp_roi = DB_DROP                                          # 对侧 ROI 全部降
got = [scalar(e2, k) for k in ('brain_hand', 'central_hand', 'contra_roi', 'ipsi_roi')]
exp = (exp_brain, exp_cen, exp_roi, 0.0)
chk('M2 空间稀释逐级符合解析值(全脑 < 中央 < 对侧)',
    all(abs(g - x) < 0.06 for g, x in zip(got[:3], exp[:3])),
    '\n       '.join('%-12s meas %+.3f exp %+.3f' % (k, g, x) for k, g, x in
                     zip(('brain_hand', 'central_hand', 'contra_roi'), got[:3], exp[:3])))
chk('M1 同侧 ROI 未被调制 -> ~0(证明 contra/ipsi 取的是不同 ROI)',
    abs(got[3]) < 0.06, 'ipsi_roi meas %+.3f exp 0' % got[3])
chk('M2 幅度: 灶限于 7/22 通道时全脑稀释 ~3.5 dB(真数据只有 0.06 dB)',
    abs((got[0] - got[2]) - (exp_brain - exp_roi)) < 0.08,
    '本合成 %+.3f dB vs 真数据 %+.3f dB' % (got[0] - got[2], -0.062))

# ---------------------------------------------------------------- 7 旧口径缺陷
print('\n=== 7 定量复现旧 erd_c3/c4/min 口径的缺陷 ===')
X, y = synth(seed=11)
X = mod(X, y, ['C4'], [0], TASK[0], TASK[1], S_DROP)
X = mod(X, y, ['C3'], [1], TASK[0], TASK[1], S_DROP)
task = band_pow(X, IAF, TASK[0], TASK[1], 125); rest = band_pow(X, IAF, *REST, 0)
old = {k: 10 * np.log10(task[:, CH[k]].mean() / rest[:, CH[k]].mean()) for k in ('C3', 'C4')}
e = subject_erd(X, y, IAF)
chk('旧口径跨类平均把 %+.2f dB 的真信号抹到 ~%+.2f dB' % (DB_DROP, 10 * np.log10((S_DROP ** 2 + 3) / 4)),
    abs(old['C3'] - 10 * np.log10((S_DROP ** 2 + 3) / 4)) < 0.06
    and abs(old['C4'] - 10 * np.log10((S_DROP ** 2 + 3) / 4)) < 0.06,
    'erd_c3 %+.3f / erd_c4 %+.3f(两者都 ~-0.76, 偏侧化被抵消) vs 新 contra %+.3f'
    % (old['C3'], old['C4'], scalar(e, 'contra_elec')))
chk('旧 erd_contra_db = min(C3,C4) 严重低估',
    min(old.values()) - DB_DROP > 3.0,
    'min = %+.3f vs 真值 %+.3f(差 %.2f dB)' % (min(old.values()), DB_DROP, min(old.values()) - DB_DROP))

# ---------------------------------------------------------------- 8 时间过程
print('\n=== 8 时间过程: 窗栅格 / 窗心时刻 / 基线共用 ===')
chk('所有窗完整落在任务窗内', TC_STARTS.min() >= TASK[0] and TC_STARTS.max() + TC_LEN <= TASK[1],
    'starts %d..%d len=%d step=%d' % (TC_STARTS.min(), TC_STARTS.max(), TC_LEN, TC_STEP))
chk('窗数 = 11 且步长 0.2 s', len(TC_STARTS) == 11 and TC_STEP == 50)
t_axis = (TC_STARTS + TC_LEN / 2 - REST[1]) / FS
chk('窗心时刻 = cue 后 1.0..3.0 s(cue 在样本 250)',
    np.allclose(t_axis, np.arange(1.0, 3.01, 0.2)), str(np.round(t_axis, 2)))
X, y = synth(seed=13)
X = mod(X, y, ALL, [0, 1], TASK[0], TASK[0] + 250, S_DROP)   # 只调制第 1 个窗
e = subject_erd(X, y, IAF)
Ac, Rc = e['tc_contra']
chk('逐人时间过程形状 = (n_hand, 11) 且 A/R 行对齐',
    Ac.shape == (2 * N_PER_CLS, 11) and Rc.shape == Ac.shape, str(Ac.shape))
chk('基线 R 全窗共用(每行同一 trial 的 R 相同)',
    all(np.allclose(Rc[:, j], Rc[:, 0]) for j in range(11)))
meas = np.array([10 * np.log10(Ac[:, j].mean() / Rc[:, j].mean()) for j in range(11)])
# 解析预测: Welch 对每段乘 Hann 窗并按窗能量归一, 故窗内各样本的权重是 w^2,
# 段两端被强烈压零。朴素"重叠率"预测(等权)会错, 必须带 Hann 权重。
w2 = np.hanning(TC_LEN) ** 2
mod_end = TASK[0] + 250
exp_hann = np.array([10 * np.log10((w2 * np.where(np.arange(s, s + TC_LEN) < mod_end,
                                                   S_DROP ** 2, 1.0)).sum() / w2.sum())
                     for s in TC_STARTS])
ov = [max(0, min(s + TC_LEN, mod_end) - max(s, TASK[0])) / TC_LEN for s in TC_STARTS]
exp_naive = np.array([10 * np.log10(o * S_DROP ** 2 + (1 - o)) for o in ov])
chk('逐窗 ERD 与 Hann 加权解析值同量级(二阶谱泄漏误差内)',
    np.max(np.abs(meas - exp_hann)) < 0.25,
    'max|dev| = %.4f dB(幅度阶跃产生边带, 固定带积分捕获比例随之变, 属二阶效应)'
    % np.max(np.abs(meas - exp_hann)))
chk('已记录: Hann 压零使各窗有效支撑 < 名义 1.0 s(等权预测会偏达 %.2f dB)'
    % np.max(np.abs(exp_naive - exp_hann)),
    np.max(np.abs(exp_naive - exp_hann)) > 0.5,
    '等权 vs Hann 最大差 %.3f dB(窗1: 等权 %+.3f vs Hann %+.3f)'
    % (np.max(np.abs(exp_naive - exp_hann)), exp_naive[1], exp_hann[1]))
chk('调制只在首窗 -> 后续窗回到基线(具备时间定位能力)',
    meas[0] < meas[3] and np.max(np.abs(meas[5:])) < 0.3,
    'profile = %s' % np.round(meas, 3))
# 不依赖谱估计二阶效应的定位测试: 把调制整体挪到第 9 个窗, 峰值必须精确移动 8 格
X3 = mod(synth(seed=15)[0], y, ALL, [0, 1], 775, 1025, S_DROP)
Ac3, Rc3 = subject_erd(X3, y, IAF)['tc_contra']
meas3 = np.array([10 * np.log10(Ac3[:, j].mean() / Rc3[:, j].mean()) for j in range(11)])
chk('窗索引无偏移: 调制移到第 9 窗, 最深窗精确从 0 移到 8',
    int(np.argmin(meas)) == 0 and int(np.argmin(meas3)) == 8,
    'argmin %d -> %d; 峰值 %+.3f -> %+.3f'
    % (int(np.argmin(meas)), int(np.argmin(meas3)), meas.min(), meas3.min()))
chk('不与调制区重叠(或只落在 Hann 压零尾)的窗回到基线',
    np.max(np.abs(meas3[:5])) < 0.3,
    'meas3[:5] = %s(窗 5 起与调制区实质重叠, 不该为 0)' % np.round(meas3[:5], 3))

# ---------------------------------------------------------------- 9 组统计
print('\n=== 9 组统计: SEM / Wilcoxon 方向 / t 区间 / 留一法 ===')
# 逐人数值一律从产物 JSON 读, 不手抄(手抄曾把 brain_all 误当 brain_hand)
JP = BASE / 'results' / 'erd_decomposition' / 'erd_decomposition.json'
chk('JSON 存在', JP.exists(), str(JP))
J = json.loads(JP.read_text(encoding='utf-8')) if JP.exists() else None
SP = json.loads((BASE / 'results' / 'individual_diff' / 'subject_table.json')
                .read_text(encoding='utf-8')) if JP.exists() else []
if J:
    ce = np.array(J['group']['contra_elec']['per_subject_db'])
    chk('SEM == std(ddof=1)/sqrt(n)',
        abs(ce.std(ddof=1) / np.sqrt(9) - J['group']['contra_elec']['sem_across_subj']) < 1e-3,
        '实测 %.4f vs JSON %.4f' % (ce.std(ddof=1) / np.sqrt(9),
                                    J['group']['contra_elec']['sem_across_subj']))
d_neg = np.array([-2.734, -1.185, -0.640, -0.438, -0.421, -0.119, -0.05, -0.02, -0.01])
d_pos = -d_neg
p_less_neg = wilcoxon(d_neg, alternative='less').pvalue
p_less_pos = wilcoxon(d_pos, alternative='less').pvalue
chk('Wilcoxon(alternative=less) 检验的是 contra-ipsi < 0(方向语义正确)',
    p_less_neg < 0.01 < 0.9 < p_less_pos,
    '全负 p=%.5f / 全正 p=%.5f' % (p_less_neg, p_less_pos))
chk('跨被试 t 区间用 t.ppf(0.975, 8) = 2.306', abs(tdist.ppf(0.975, 8) - 2.3060) < 1e-3,
    'tcrit = %.4f' % tdist.ppf(0.975, 8))
if J:
    subs = [r['subject'] for r in SP]
    bh = np.array([J['per_subject'][s]['brain_hand'] for s in subs])
    d70 = np.array([r['deep70_acc'] for r in SP])
    rho_all = spearmanr(bh, d70)
    loo = [spearmanr(np.delete(bh, i), np.delete(d70, i))[0] for i in range(9)]
    chk('留一法复现 JSON 记录的 brain_hand × deep70_acc 相关',
        abs(rho_all[0] - J['perf_corr_spearman']['brain_hand']['deep70_acc'][0]) < 1e-3,
        '重算 rho=%+.4f (p=%.5f) vs JSON %+.4f'
        % (rho_all[0], rho_all[1], J['perf_corr_spearman']['brain_hand']['deep70_acc'][0]))
    chk('留一法: 去掉任一人后 |rho| 仍 > 0.9(非单人驱动)',
        min(abs(np.array(loo))) > 0.9,
        'rho 区间 [%+.3f, %+.3f]' % (max(loo), min(loo)))

# ---------------------------------------------------------------- 10 产物
print('\n=== 10 产物一致性(JSON 自洽 + 报告正文引用的数字) ===')
if J:
    tc, grp = J['timecourse'], J['group']
    PC = np.array(tc['contra_per_subject_db'])
    m = PC.mean(0); h = tdist.ppf(0.975, PC.shape[0] - 1) * PC.std(0, ddof=1) / np.sqrt(PC.shape[0])
    chk('JSON: contra_db == 逐人曲线均值', np.allclose(tc['contra_db'], m, atol=1e-3),
        'max|dev| %.2e' % np.max(np.abs(tc['contra_db'] - m)))
    chk('JSON: contra_ci95 == mean ± t·SEM',
        np.allclose([c[0] for c in tc['contra_ci95']], m - h, atol=1e-3)
        and np.allclose([c[1] for c in tc['contra_ci95']], m + h, atol=1e-3))
    chk('JSON: 曲线均值与标量同口径(差 < 0.1 dB)',
        abs(tc['consistency_curve_mean_vs_scalar_db']) < 0.1,
        '差 %+.4f dB' % tc['consistency_curve_mean_vs_scalar_db'])
    chk('JSON: 逐人曲线行数 == 9 且标量逐人数 == 9',
        PC.shape[0] == 9 == len(grp['contra_elec']['per_subject_db']))
    chk('JSON: 偏侧化 n_neg 与逐人差值一致',
        J['lateralization']['n_neg_elec'] == int((np.array(grp['contra_elec']['per_subject_db'])
                                                 - np.array(grp['ipsi_elec']['per_subject_db']) < 0).sum()))

    def dig(path):
        o = J
        for k in path:
            o = o[k]
        return float(o)

    quoted = {('group', 'contra_elec', 'mean_db'): -1.330,
              ('group', 'ipsi_elec', 'mean_db'): -0.770,
              ('group', 'brain_hand', 'mean_db'): -0.865,
              ('group', 'brain_all', 'mean_db'): -0.110,
              ('group', 'central_hand', 'mean_db'): -0.927,
              ('group', 'contra_roi', 'mean_db'): -1.148,
              ('group', 'ipsi_roi', 'mean_db'): -0.727,
              ('group', 'mid_feet', 'mean_db'): -0.037,
              ('group', 'mid_tongue', 'mean_db'): 0.674,
              ('lateralization', 'diff_elec_db'): -0.561,
              ('lateralization', 'wilcoxon_p_elec_one_sided'): 0.0371,
              ('lateralization', 'wilcoxon_p_roi_one_sided'): 0.0098,
              ('timecourse', 'latency_s'): 1.60,
              ('timecourse', 'consistency_curve_mean_vs_scalar_db'): -0.0324,
              ('perf_corr_spearman', 'brain_hand', 'deep70_acc', 0): -0.950,
              ('perf_corr_spearman', 'central_hand', 'deep70_acc', 0): -0.950,
              ('perf_corr_spearman', 'ipsi_roi', 'deep70_acc', 0): -0.933,
              ('perf_corr_spearman', 'mid_tongue', 'deep70_acc', 0): 0.100,
              ('perf_corr_spearman', 'brain_all', 'deep70_acc', 0): -0.583}
    bad = ['%s: JSON %s vs 报告 %s' % ('.'.join(map(str, p)), dig(p), w)
           for p, w in quoted.items() if abs(dig(p) - w) > 0.0051]
    chk('报告引用的 %d 个数字逐项对得上 JSON' % len(quoted), not bad,
        '\n       '.join(bad) or '全部一致')

    md = (Path(__file__).resolve().parent / '差异报告' / '个体化差异研究报告.md').read_text(encoding='utf-8')
    miss = [s for s in ('1.330', '0.770', '0.865', '0.110', '0.927', '1.148', '0.727',
                        '0.674', '0.561', '0.0371', '0.0098', '0.950', '0.933', '0.583',
                        '1.60', '0.032', '0.0155', '4.4370', '4.4786', '0.0557',
                        '6/9', '7/9')
            if s not in md]
    chk('这些数字确实出现在报告正文里(闭环: 代码 -> JSON -> 报告)', not miss,
        '缺失: %s' % miss if miss else '全部命中')

# ---------------------------------------------------------------- 11 真数据
if '--real' in sys.argv:
    print('\n=== 11 真数据端到端(2 人, 需 GDF 缓存) ===')
    sys.path.insert(0, str(BASE.parent / 'preprocess'))
    from moabb_bci2a import load_subject_mi_trials
    tbl = {r['subject']: r for r in json.loads(
        (BASE / 'results' / 'individual_diff' / 'subject_table.json').read_text(encoding='utf-8'))}
    hits = 0
    for i in (0, 2):
        name = 'S%02d' % (i + 1)
        L, Rr, F, Tg = load_subject_mi_trials(i + 1)
        X = np.concatenate([L, Rr, F, Tg], 0).astype(np.float64)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, Rr, F, Tg])])
        rest = band_pow(X, tbl[name]['iaf_hz'], *REST, 0)
        task = band_pow(X, tbl[name]['iaf_hz'], TASK[0], TASK[1], 125)
        d = {}
        for c, lab in ((0, 'left'), (1, 'right')):
            m = y == c
            d[lab] = 10 * np.log10(task[m, CH['C4']].mean() / rest[m, CH['C4']].mean()) \
                - 10 * np.log10(task[m, CH['C3']].mean() / rest[m, CH['C3']].mean())
        ok = d['left'] < 0 < d['right']
        hits += ok
        chk('%s 2x2 解剖反对称成立(左 cue C4<C3, 右 cue C3<C4)' % name, ok,
            'left(C4-C3)=%+.3f right(C4-C3)=%+.3f dB' % (d['left'], d['right']))
        S = X[:, :, REST[0]:REST[1]]
        S = S - S.mean(-1, keepdims=True)
        C = np.einsum('nct,ndt->cd', S, S) / (S.shape[-1] * S.shape[0])
        er = C.trace() ** 2 / (C ** 2).sum()
        chk('%s 数据确为非 EA(静息协方差有效秩 %.2f ≪ 22;报告该列 1.28–2.15)'
            % (name, er), er < 6, '表内 rest_eff_rank = %.2f' % tbl[name]['rest_eff_rank'])
        if J:
            e3 = subject_erd(X, y, tbl[name]['iaf_hz'])
            for k in ('contra_elec', 'ipsi_elec', 'brain_hand'):
                pub, now = J['per_subject'][name][k], scalar(e3, k)
                chk('%s %s 重算 == 已发布 JSON 值' % (name, k), abs(pub - now) < 1e-3,
                    'JSON %+.4f vs 重算 %+.4f' % (pub, now))
    chk('真数据端到端: 2/2 被试通过', hits == 2)

print('\n' + '=' * 62)
print('检查 %d 项, 失败 %d 项' % (NCHK[0], len(FAIL)))
if FAIL:
    for f in FAIL:
        print('  FAIL:', f)
sys.exit(1 if FAIL else 0)
