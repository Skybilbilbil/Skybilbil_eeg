# -*- coding: utf-8 -*-
"""§3.4 ERD 的【逐 trial 基线归一化 + 不确定性 + 对侧/同侧 ROI 分解】。

报告 §3.4 现有两处明示缺口, 导致它不敢下结论:
  (1)「未作逐 trial 基线归一化或不确定性估计, 因此仅作探索性的时间过程描述,
      不能直接称为 ERD 潜伏期」
  (2)「没有按左/右手类别和对侧/同侧中央 ROI 分解, 不能据此证明对侧 ERD 与
      同侧变化的相互抵消, 也不能单独定位判别信息来源」

现有 subject_table 里的 ERD 列还有两个方法缺陷, 本脚本一并修正:
  A. `erd_c3_db` / `erd_c4_db` 是【跨 4 类平均】的功率比 —— 左手 cue 的对侧是 C4、
     右手 cue 的对侧是 C3, 两类一平均就把偏侧化信号抵消掉了。必须按 cue 侧分解。
  B. `erd_contra_db = min(erd_c3_db, erd_c4_db)` 只是「取更被抑制的那个电极」,
     与 cue 侧无关, 因此它【不是】对侧指标(附录 A 的定义描述也因此不成立)。
  C. 口径本身: 采用经典 Pfurtscheller ERD =【跨 trial 平均功率之比】
     10log10(mean_i A_i / mean_i R_i)。两种"逐 trial 先归一化"的口径都实测有系统
     偏差(见 erd_db 的 docstring), 已弃用。

「全脑 μ 无净下降」的三个可能成因, 本脚本分别量化(不要混为一谈):
  M1 偏侧化抵消 : 对侧 ERD(负) 与 同侧 ERS(正) 相互抵消
  M2 空间稀释   : 中央灶的 ERD 被 22 通道平均稀释
  M3 类别抵消   : 手类 ERD 被 feet/tongue 的 ERS 冲掉
                  (已知线索: S01 的 erd_cls_left/right = −1.42/−1.41 而
                   feet/tongue = +1.33/+1.84)

数据: 【非 EA】电极空间(moabb_bci2a.load_subject_mi_trials, GDF 缓存在
~/mne_data/MNE-bnci-data)。EA 是 22×22 全通道白化, 白化后第 i 通道不是第 i 电极,
故 C3/C4 与对侧/同侧只能在非 EA 数据上定义。
频带: 每被试自己的 IAF ± 1.5 Hz(取自 subject_table.json)。⚠ Welch bin 是 1 Hz, 掩膜实际
只落 3 个 bin ⇒ 有效带宽 2.004 Hz 而非名义 3 Hz(实测见 verify_erd_pipeline.py;对 ERD
比值无影响, 但绝对带功率不可与别的带宽定义比较)。
窗: 静息 [0,250)(cue 前 1s), 任务 [375,1126)(cue 后 0.5–3.5s)。

每一步实现由同目录 verify_erd_pipeline.py 用【解析真值已知的合成信号】逐项验证
(默认档 55 项 + --real 追加 11 项 = 66 项检查), 含 --real 的真数据端到端(2x2 解剖反对称 + 重算值对齐已发布 JSON)。

⚠ 带功率一律用【Welch 带积分】(Hann 窗), 与 analyze_individual_diff.psd()/bp() 同族,
  故本脚本的 dB 值可与 subject_table 的 erd_* 列直接比较。
  【不要用 sosfiltfilt 逐 epoch 滤波】: epoch 首尾没有邻接样本, 零相位滤波的边缘
  暂态无法补偿。实测(合成恒幅 10Hz 正弦 + 白噪声, 无任何生理变化): 同一套
  sosfiltfilt + 0.5s 滑窗在 epoch 末窗给出 **−2.43 dB 的假性下降**, 且从 sample 937
  起单调漂移。真实数据上表现为 +2.85s 之后 ERD 从 −1.19 发散到 −4.85 dB。
  该伪影只依赖样本位置, 对侧/同侧共用同一批 trial 与窗 ⇒ **对比量(contra−ipsi)
  不受影响, 但绝对 dB 值会被污染**, 故必须换估计器。

纯 CPU; 约 10 分钟(9 个被试需从 GDF 缓存重放)。只写 results/erd_decomposition/。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import welch
from scipy.stats import spearmanr, t as tdist, wilcoxon

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))   # moabb_bci2a(2026-09-17 起在 preprocess/)
from mAtt.mAtt import BCI2A_CHANNELS  # noqa: E402

OUT = BASE / 'results' / 'erd_decomposition'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'

FS = 250
REST = (0, 250)
TASK = (375, 1126)
LABELS = ['left', 'right', 'feet', 'tongue']
CH = {n: i for i, n in enumerate(BCI2A_CHANNELS)}
# cue 侧 -> (对侧单电极, 同侧单电极)
LAT = {'left': ('C4', 'C3'), 'right': ('C3', 'C4')}
# 镜像中央 ROI(7 电极/侧): 额中央 + 中央 + 顶中央
ROI_R = ['FC4', 'FC2', 'C6', 'C4', 'C2', 'CP4', 'CP2']
ROI_L = ['FC3', 'FC1', 'C5', 'C3', 'C1', 'CP3', 'CP1']
MIDLINE = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz']
NPERSEG = 250          # 1 Hz 频率分辨率, 与静息窗长度一致
# 时间过程: 1.0s 窗、0.2s 步, 窗完整落在任务窗内(故最晚窗心 = +3.0s)
TC_LEN, TC_STEP = 250, 50
TC_STARTS = np.arange(TASK[0], TASK[1] - TC_LEN + 1, TC_STEP)


def band_pow(X, iaf, a, b, noverlap):
    """X (n,ch,T) 的 [a,b) 窗 -> 每 trial 每通道的 IAF±1.5Hz 带功率 (n,ch)。

    Welch(Hann) 带积分: 逐段加窗, 无 filtfilt 边缘暂态。noverlap>0 时多段平均,
    频率分辨率不变但方差更低(任务窗用 125, 静息窗与时间过程窗用 0)。
    """
    f, P = welch(X[:, :, a:b], fs=FS, nperseg=min(NPERSEG, b - a),
                 noverlap=noverlap, axis=-1)
    m = (f >= iaf - 1.5) & (f <= iaf + 1.5)
    return np.trapezoid(P[..., m], f[m], axis=-1)


def erd_db(A, R, n_boot=2000, seed=0):
    """经典 ERD(Pfurtscheller): 先跨 trial 平均功率, 再取比值与对数。

        ERD_dB = 10·log10( mean_i A_i / mean_i R_i )

    A = 逐 trial 任务窗带功率, R = 逐 trial 静息窗带功率(同一 trial)。
    不确定性用 trial 级 bootstrap(A/R 同一索引 jointly 重采样)。

    ⚠ 为什么不用两种"逐 trial 先归一化"的口径(都实测过, 都有系统偏差):
      (a) 逐 trial 先取 dB 再平均 = E[log R]。log 是凹函数, Jensen 给 E[log R] ≤ log E[R],
          偏差随方差增大; 而方差取决于 Welch 段数(静息窗 250 样本 = 1 段, 任务窗
          751 样本 = 5 段, 时间过程每窗 1 段)⇒ 实测标量与时间过程相差 ~1.2 dB,
          绝对值互不可比。
      (b) 逐 trial 取比值再平均 = E[A/R] = E[A]·E[1/R]。静息窗只有 1 段 ≈ 6 个自由度,
          R 高度右偏 ⇒ E[1/R] ≫ 1/E[R], 实测把 ERD 抬高约 +2 dB(9 人全脑四类
          给到 +2.34 dB, 而本口径与 subject_table.erd_mu_db 的 ~+0.3 dB 一致)。
      **纯正弦自检测不出 (b)**: 无噪声时比值是确定量, 倒数矩偏差恒为 0。必须用
      【带噪声的恒幅信号】做自检(期望 0 dB), 见文件末的验证记录。
    """
    point = 10.0 * np.log10(A.mean() / R.mean())
    if n_boot == 0:
        return float(point), None
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for b in range(n_boot):
        k = rng.integers(len(A), size=len(A))
        bs[b] = A[k].mean() / R[k].mean()
    return float(point), [float(10 * np.log10(np.percentile(bs, 2.5))),
                          float(10 * np.log10(np.percentile(bs, 97.5)))]


def roi_pow(task, rest, y, cls, chs):
    """给定 cue 类别集合 cls, 在 chs 上【先跨通道平均功率】-> (A, R) 逐 trial 数组。

    先平均功率再求比, 是"该 ROI 的总能量变化"(与"全脑平均 μ 功率"口径一致);
    若先逐通道取比再平均则各通道等权, 两者不等价。
    """
    idx = [CH[c] for c in chs]
    m = np.isin(y, cls)
    return task[m][:, idx].mean(-1), rest[m][:, idx].mean(-1)


def subject_erd(X, y, iaf):
    """-> dict: 该被试各口径的 (A, R) 逐 trial 功率数组(取对数在 main 里统一做)"""
    rest = band_pow(X, iaf, REST[0], REST[1], 0)      # (n,ch) 单段
    task = band_pow(X, iaf, TASK[0], TASK[1], 125)    # (n,ch) 5 段平均, 方差更低
    out = {}

    # M2/M3 基线: 全脑、四类 vs 仅手类
    out['brain_all'] = roi_pow(task, rest, y, [0, 1, 2, 3], BCI2A_CHANNELS)
    out['brain_hand'] = roi_pow(task, rest, y, [0, 1], BCI2A_CHANNELS)
    out['central_all'] = roi_pow(task, rest, y, [0, 1, 2, 3], ROI_L + ROI_R)
    out['central_hand'] = roi_pow(task, rest, y, [0, 1], ROI_L + ROI_R)

    # M1 偏侧化: 按 cue 侧取对侧/同侧(单电极 + 镜像 ROI)
    ce, ie, cr, ir = [], [], [], []
    for side, (cc, ci) in LAT.items():
        c = 0 if side == 'left' else 1
        m = (y == c)
        for arr, ch in ((ce, cc), (ie, ci)):
            k = CH[ch]
            arr.append((task[m, k], rest[m, k]))
        cr.append(roi_pow(task, rest, y, [c], ROI_R if side == 'left' else ROI_L))
        ir.append(roi_pow(task, rest, y, [c], ROI_L if side == 'left' else ROI_R))
    out['contra_elec'] = (np.concatenate([a for a, _ in ce]),
                          np.concatenate([r for _, r in ce]))
    out['ipsi_elec'] = (np.concatenate([a for a, _ in ie]),
                        np.concatenate([r for _, r in ie]))
    out['contra_roi'] = (np.concatenate([a for a, _ in cr]),
                         np.concatenate([r for _, r in cr]))
    out['ipsi_roi'] = (np.concatenate([a for a, _ in ir]),
                       np.concatenate([r for _, r in ir]))

    # feet / tongue 在中线上的反应(无偏侧化可言)
    out['mid_feet'] = roi_pow(task, rest, y, [2], MIDLINE)
    out['mid_tongue'] = roi_pow(task, rest, y, [3], MIDLINE)

    # 时间过程(对侧/同侧单电极, 手类): 每窗一组逐 trial 功率, 基线 R 全窗共用
    tc_pow = [band_pow(X, iaf, s, s + TC_LEN, 0) for s in TC_STARTS]
    for tag in ('contra', 'ipsi'):
        A_blocks, R_blocks = [], []
        for side, (cc, ci) in LAT.items():
            k = CH[cc if tag == 'contra' else ci]
            m = (y == (0 if side == 'left' else 1))
            r = rest[m, k]
            A_blocks.append(np.stack([w[m, k] for w in tc_pow], 1))   # (n_trial, n_win)
            R_blocks.append(np.tile(r, (len(TC_STARTS), 1)).T)
        out['tc_' + tag] = (np.concatenate(A_blocks, 0), np.concatenate(R_blocks, 0))
    return out


METRICS = ['brain_all', 'brain_hand', 'central_all', 'central_hand',
           'contra_elec', 'ipsi_elec', 'contra_roi', 'ipsi_roi',
           'mid_feet', 'mid_tongue']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}
    from moabb_bci2a import load_subject_mi_trials, VALID_SUBJECTS

    per_subj, agg, ntrial = {}, {k: [] for k in METRICS}, {}
    tc_c, tc_i = [], []
    for i, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        name = 'S%02d' % (i + 1)
        L, R, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, R, Fe, To], 0).astype(np.float64)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, R, Fe, To])])
        e = subject_erd(X, y, ref[name]['iaf_hz'])
        per_subj[name] = {'iaf_hz': ref[name]['iaf_hz']}
        for k in METRICS:
            pt, ci = erd_db(*e[k])
            per_subj[name][k] = round(pt, 4)
            per_subj[name][k + '_ci95'] = [round(x, 4) for x in ci]
            agg[k].append(pt)
            ntrial.setdefault(k, len(e[k][0]))
        # 逐人时间过程(dB 曲线), 与标量同口径: 先每人算 ERD, 再跨被试平均
        for src, dst in ((e['tc_contra'], tc_c), (e['tc_ipsi'], tc_i)):
            A, R = src
            dst.append([erd_db(A[:, j], R[:, j], n_boot=0)[0] for j in range(A.shape[1])])
        print('  %s 完成 (%.0fs)  对侧电极 %.3f / 同侧电极 %.3f dB'
              % (name, time.time() - t0, per_subj[name]['contra_elec'],
                 per_subj[name]['ipsi_elec']), flush=True)

    print('\n=== 逐口径 ERD = 10log10(mean_i A_i / mean_i R_i)(9 人均值, dB; 负 = 抑制) ===')
    print('%-14s %8s %8s %8s %8s' % ('口径', '均值', '跨被试SEM', '最小', '最大'))
    rows = {}
    for k in METRICS:
        v = np.array(agg[k])
        rows[k] = {'mean_db': round(float(v.mean()), 4),
                   'sem_across_subj': round(float(v.std(ddof=1) / np.sqrt(len(v))), 4),
                   'per_subject_db': [round(float(x), 4) for x in v],
                   'n_trial_per_subj': ntrial[k]}
        print('%-14s %8.3f %8.3f %8.3f %8.3f'
              % (k, v.mean(), rows[k]['sem_across_subj'], v.min(), v.max()))

    # M1 偏侧化: 对侧 vs 同侧 配对检验
    c_e, i_e = np.array(agg['contra_elec']), np.array(agg['ipsi_elec'])
    c_r, i_r = np.array(agg['contra_roi']), np.array(agg['ipsi_roi'])
    lat_e, lat_r = c_e - i_e, c_r - i_r
    w_e = wilcoxon(lat_e, alternative='less')
    w_r = wilcoxon(lat_r, alternative='less')
    print('\n=== M1 偏侧化抵消检验(配对, n=9) ===')
    print('单电极  对侧 %.3f vs 同侧 %.3f  差 %.3f ± %.3f  Wilcoxon(单侧,对侧更低) p=%.4f  %d/9 为负'
          % (c_e.mean(), i_e.mean(), lat_e.mean(), lat_e.std(ddof=1) / 3,
             w_e.pvalue, int((lat_e < 0).sum())))
    print('镜像ROI 对侧 %.3f vs 同侧 %.3f  差 %.3f ± %.3f  Wilcoxon p=%.4f  %d/9 为负'
          % (c_r.mean(), i_r.mean(), lat_r.mean(), lat_r.std(ddof=1) / 3,
             w_r.pvalue, int((lat_r < 0).sum())))

    print('\n=== 三个成因的量化(为何"全脑无净下降") ===')
    b_all, b_hand = np.array(agg['brain_all']), np.array(agg['brain_hand'])
    cen_hand = np.array(agg['central_hand'])
    print('M3 类别抵消 : 全脑四类 %.3f -> 全脑仅手类 %.3f   (差 %+.3f)'
          % (b_all.mean(), b_hand.mean(), b_hand.mean() - b_all.mean()))
    print('M2 空间稀释 : 全脑手类 %.3f -> 中央ROI手类 %.3f  (差 %+.3f)'
          % (b_hand.mean(), cen_hand.mean(), cen_hand.mean() - b_hand.mean()))
    print('M1 偏侧化   : 中央ROI手类 %.3f -> 对侧ROI %.3f / 同侧ROI %.3f'
          % (cen_hand.mean(), c_r.mean(), i_r.mean()))
    print('feet 中线 %.3f / tongue 中线 %.3f  (手类对侧 %.3f)'
          % (np.array(agg['mid_feet']).mean(), np.array(agg['mid_tongue']).mean(),
             c_r.mean()))

    # 各口径 ERD 与判别性能的秩相关(+ 最强项的留一法稳健性)
    PERF = ('deep70_acc', 'csp_within', 'tangent_lda_within')
    pv = {k: np.array([ref[s][k] for s in per_subj]) for k in PERF}
    print('\n=== ERD 口径 × 判别性能(Spearman, n=9, 未校正) ===')
    print('%-13s' % '口径' + ''.join('%24s' % k for k in PERF))
    corr = {}
    for k in METRICS + ['lat_elec', 'lat_roi']:
        v = (c_e - i_e) if k == 'lat_elec' else ((c_r - i_r) if k == 'lat_roi'
                                                 else np.array(agg[k]))
        row = {}
        line = '%-13s' % k
        for pk, y in pv.items():
            r, p = spearmanr(v, y)
            row[pk] = [round(float(r), 4), round(float(p), 6)]
            line += '  rho=%+.3f p=%.4f' % (r, p)
        corr[k] = row
        print(line)
    key = 'brain_hand'
    bh = np.array(agg[key])
    loo = {pk: [round(float(spearmanr(np.delete(bh, i), np.delete(y, i))[0]), 4)
                for i in range(len(bh))] for pk, y in pv.items()}
    print('\n留一法(去掉单个被试后重算 rho), 口径 = %s:' % key)
    for pk in PERF:
        print('  %-18s 全样本 %+.3f -> 留一区间 [%+.3f, %+.3f]'
              % (pk, corr[key][pk][0], min(loo[pk]), max(loo[pk])))
    print('  与既有生理列的冗余: brain_hand × alpha_pow_db rho=%+.3f, × iaf_hz rho=%+.3f, '
          '× erd_mu_db rho=%+.3f'
          % tuple(spearmanr(bh, np.array([ref[s][c] for s in per_subj]))[0]
                  for c in ('alpha_pow_db', 'iaf_hz', 'erd_mu_db')))

    # 时间过程: 逐人 dB 曲线 -> 跨被试平均; 不确定性 = 跨被试 t 区间(n=9)
    TCc, TCi = np.array(tc_c), np.array(tc_i)          # (9, n_win)
    t_axis = (TC_STARTS + TC_LEN / 2 - REST[1]) / FS   # 事件相对秒(cue 在样本 250)
    tcrit = tdist.ppf(0.975, TCc.shape[0] - 1)

    def group_curve(M):
        m = M.mean(0)
        h = tcrit * M.std(0, ddof=1) / np.sqrt(M.shape[0])
        return m, m - h, m + h

    mc, lo_c, hi_c = group_curve(TCc)
    mi, lo_i, hi_i = group_curve(TCi)
    k = int(np.argmin(mc))
    print('\n=== 时间过程(对侧单电极: 左手cue取C4/右手cue取C3; 9 人各自算 ERD 后平均) ===')
    print('最低点 = %+.2f s (事件相对), 深度 %.3f dB, 跨被试95%%CI [%.3f, %.3f]'
          % (t_axis[k], mc[k], lo_c[k], hi_c[k]))
    print('同侧同一时刻 = %.3f dB, CI [%.3f, %.3f]' % (mi[k], lo_i[k], hi_i[k]))
    print('口径一致性检查: 曲线 11 窗均值 %.3f dB vs 标量 contra_elec %.3f dB (差 %+.3f)'
          % (mc.mean(), c_e.mean(), mc.mean() - c_e.mean()))
    print('%-8s %9s %18s %9s %18s' % ('t(s)', '对侧', 'CI95', '同侧', 'CI95'))
    for j in range(len(t_axis)):
        print('%-8.2f %9.3f [%7.3f,%7.3f] %9.3f [%7.3f,%7.3f]'
              % (t_axis[j], mc[j], lo_c[j], hi_c[j], mi[j], lo_i[j], hi_i[j]))

    rec = {'band': 'per-subject IAF ± 1.5 Hz, Welch(Hann) band integral, nperseg=250 (1 Hz)',
           'estimator': 'ERD_dB = 10*log10( mean_trials(A_i) / mean_trials(R_i) )  '
                        '(classic Pfurtscheller ERD: ratio of trial-averaged powers)',
           'estimator_note':
               '(1) 未用 sosfiltfilt: 逐 epoch 零相位滤波在首尾产生边缘暂态(合成恒幅正弦'
               '实测末窗 −2.43 dB 假性下降)。(2) 未用「逐 trial 先取 dB 再平均」: Jensen '
               '不等式使其偏差随 Welch 段数变化(静息窗 1 段 / 任务窗 5 段 / 时间过程每窗 1 段), '
               '实测标量与时间过程相差 ~1.2 dB。(3) 未用「逐 trial 取比值再平均」: 静息窗仅 '
               '~6 个自由度, E[1/R] ≫ 1/E[R], 实测把 ERD 抬高约 +2 dB。'
               ' 自检见文件末 VALIDATION 注释。',
           'windows': {'rest': list(REST), 'task': list(TASK)},
           'roi_r': ROI_R, 'roi_l': ROI_L, 'midline': MIDLINE,
           'lateral_map': {k: list(v) for k, v in LAT.items()},
           'group': rows,
           'lateralization': {
               'contra_elec_mean_db': round(float(c_e.mean()), 4),
               'ipsi_elec_mean_db': round(float(i_e.mean()), 4),
               'diff_elec_db': round(float(lat_e.mean()), 4),
               'wilcoxon_p_elec_one_sided': round(float(w_e.pvalue), 6),
               'n_neg_elec': int((lat_e < 0).sum()),
               'contra_roi_mean_db': round(float(c_r.mean()), 4),
               'ipsi_roi_mean_db': round(float(i_r.mean()), 4),
               'diff_roi_db': round(float(lat_r.mean()), 4),
               'wilcoxon_p_roi_one_sided': round(float(w_r.pvalue), 6),
               'n_neg_roi': int((lat_r < 0).sum())},
           'mechanisms': {
               'M3_brain_all_classes': round(float(b_all.mean()), 4),
               'M3_brain_hand_only': round(float(b_hand.mean()), 4),
               'M2_brain_hand': round(float(b_hand.mean()), 4),
               'M2_central_hand': round(float(cen_hand.mean()), 4),
               'M1_contra_roi': round(float(c_r.mean()), 4),
               'M1_ipsi_roi': round(float(i_r.mean()), 4),
               'feet_midline': round(float(np.array(agg['mid_feet']).mean()), 4),
               'tongue_midline': round(float(np.array(agg['mid_tongue']).mean()), 4)},
           'perf_corr_spearman': corr,
           'perf_corr_leave_one_out': {'metric': key, 'rho_range': loo},
           'timecourse': {'roi': 'single electrode: contra = C4 (left cue) / C3 (right cue)',
                          'aggregation': 'per-subject ERD curve, then mean across 9 subjects; '
                                         'CI = across-subject t interval (n=9). Same estimand '
                                         'as the scalar contra_elec (NOT trial-pooled).',
                          'win_len_samples': int(TC_LEN), 'step_samples': int(TC_STEP),
                          't_event_s': [round(float(t), 3) for t in t_axis],
                          'contra_db': [round(float(v), 4) for v in mc],
                          'contra_ci95': [[round(float(a), 4), round(float(b), 4)]
                                          for a, b in zip(lo_c, hi_c)],
                          'ipsi_db': [round(float(v), 4) for v in mi],
                          'ipsi_ci95': [[round(float(a), 4), round(float(b), 4)]
                                        for a, b in zip(lo_i, hi_i)],
                          'contra_per_subject_db': [[round(float(v), 4) for v in row] for row in TCc],
                          'ipsi_per_subject_db': [[round(float(v), 4) for v in row] for row in TCi],
                          'latency_s': round(float(t_axis[k]), 3),
                          'consistency_curve_mean_vs_scalar_db':
                              round(float(mc.mean() - c_e.mean()), 4)},
           'per_subject': per_subj}
    (OUT / 'erd_decomposition.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/erd_decomposition.json' % OUT)


if __name__ == '__main__':
    main()
