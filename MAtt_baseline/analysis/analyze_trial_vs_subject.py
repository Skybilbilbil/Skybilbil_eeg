# -*- coding: utf-8 -*-
"""
试次间差异 vs 个体间差异:量级分解(纯 CPU、零训练)

问题:同一个共享模型,换一批试次几乎不掉分,换一个人立刻掉几十个百分点。
本脚本把「试次间变异」与「个体间变异」放到同一把尺子上量,给出三组数:

  A 协方差层(SPD 流形,仿射不变距离):within(试次间) vs between(个体间)
    的典型距离尺度,以及试次平均的收缩率 q(n) —— 回答"要多少试次才能把
    试次噪声压到个体间距的一半"。
  B 切空间方差分解:SS = 身份 + (被试内)类别 + 残差,给出三者 η²。
  C 判别层三级阶梯:跨试次(同 session) → 跨 session(同被试) → 跨被试,
    再加一条「个体校准曲线」:给共享分类器补 k 个目标被试试次,精度怎么走。
  D 三方向试次间一致性(相位 / 幅值 / 波形)+ ERD,逐人对照深度模型精度。

全部度量都在同一份数据上: saved_data/preprocessed_data_bci2a_raw_ica.pkl
(9 人 × 4 类 × 144 试次,22 通道,250 Hz,8-30 Hz 带通 + ICA + 独立 EA)。
注意:该 pkl 已经做过 EA,即"每个被试的二阶统计已被线性、无标签地对齐到
同一坐标系"。因此下面测到的个体间距离是**EA 之后仍然剩下**的那部分。

公式对应(报告里同编号):
  (A1) 仿射不变黎曼距离 d(A,B) = || log( A^{-1/2} B A^{-1/2} ) ||_F
  (A2) 中心 C̄_s = argmin Σ_t d²(C_st, M)   (黎曼均值)
  (A3) within_s² = mean_t d²(C_st, C̄_s) ; between² = mean_s d²(C̄_s, C̄)
  (B1) 切空间 v = vec(log C) ; SS_tot = SS_身份 + SS_类别 + SS_残差
  (D1) ITC(t) = | (1/N) Σ_k z_k(t) / |z_k(t)| |,  零假设期望 0.886/√N
  (D2) R_pair / R_hold = 单试次包络与「两两 / 留一均值」包络的 Pearson r
  (D3) EVF = Var_t(试次平均波形) / mean_k Var_t(单试次波形)
"""
import sys, json, time, warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann

BASE = Path(__file__).resolve().parent.parent          # = MAtt_baseline/
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))
sys.path.insert(0, str(BASE / 'analysis'))
# 全仓库唯一的 tangent():RIDGE 相对 ridge 的 log(矩阵),即 log-Euclidean 坐标。
from analyze_spd_separability import tangent          # noqa: E402

FS = 250
CUE = 250                       # cue 在 epoch 中的采样点(epoch = [-1.0, 3.5] s)
BASE_WIN = (0, 250)             # 基线 -1.0 ~ 0 s
EARLY = (250, 375)              # 0 ~ 0.5 s(cue 后即时窗)
TASK = (375, 1126)              # 0.5 ~ 3.5 s(与 analyze_individual_diff 一致)
MU, BETA = (8, 13), (15, 25)
RIDGE = 1e-7
PKL = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'trial_vs_subject'
CLASSES = ['左手', '右手', '脚', '舌头']
CH = {'C3': 7, 'C4': 11}        # 22 通道顺序里的 C3 / C4
NS_Q = [1, 2, 4, 8, 16, 32, 64, 96, 144]


def sos(lo, hi):
    return butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band', output='sos')


SOS_MU, SOS_BETA = sos(*MU), sos(*BETA)


# ---------------------------------------------------------------- 数据
def load():
    import pickle
    with open(PKL, 'rb') as f:
        d = pickle.load(f)
    keys = ['aligned_left_epochs_list', 'aligned_right_epochs_list',
            'aligned_feet_epochs_list', 'aligned_hands_epochs_list']
    Xs, Ys = [], []
    for i in range(len(d['subject_metadata'])):
        Xs.append(np.concatenate([d[k][i] for k in keys], 0).astype(np.float64))
        Ys.append(np.concatenate([np.full(len(d[k][i]), c) for c, k in enumerate(keys)]))
    return Xs, Ys


def sess_of(y):
    """BCI-2a 每人每类 144 = 2 个 session × 72;按类内前 72 / 后 72 分(沿用
    analyze_individual_diff.py 的口径)。"""
    idx = np.arange(len(y))
    pos = idx - y * 144                       # 类内序号 0..143
    return (pos >= 72).astype(int)


def filter_win(X, band, win):
    S = sosfiltfilt(sos(*band) if band != MU else SOS_MU, X, axis=-1)[:, :, win[0]:win[1]]
    return S - S.mean(axis=-1, keepdims=True)


def cov_of(S, norm=True, ridge=RIDGE):
    C = np.einsum('nct,ndt->ncd', S, S) / S.shape[-1]
    if norm:                                              # 规一化尺度(EA 后仍有的带宽内残差)
        C = C / np.trace(C, axis1=1, axis2=2)[:, None, None]
    # ridge 必须相对(frobenius 量级随单位变:pkl 是 Volt,不加权的协方差不量级 ~1e-10)
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    return C + ridge * tr * np.eye(C.shape[1])


def trial_cov(X, band, win, norm=True):
    return cov_of(filter_win(X, band, win), norm=norm)


# ---------------------------------------------------------------- A 协方差层
def rmean(C, init=None):
    """(A2) 黎曼均值:argmin Σ d²(C_k, M)。pyriemann 默认 50 次迭代常不收敛,放宽。"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return mean_riemann(C, tol=1e-9, maxiter=300, init=init)


def d_many(Cs, M):
    return np.array([distance_riemann(C, M) for C in Cs])


def block_A(Cs_list, Ys):
    print('\n' + '=' * 72)
    print('A  协方差层:试次间 (within) vs 个体间 (between)  —— SPD 仿射不变距离')
    print('=' * 72)
    Ms = [rmean(C) for C in Cs_list]
    Mg = rmean(np.stack(Ms))
    per = []
    for s, C in enumerate(Cs_list):
        y = Ys[s]
        w = float(np.sqrt(np.mean(d_many(C, Ms[s]) ** 2)))
        b = float(distance_riemann(Ms[s], Mg))
        # 被试内类间距离(信号),用于与个体间距离比
        Mc = [rmean(C[y == c]) for c in range(4)]
        cls = float(np.mean([distance_riemann(Mc[a], Mc[b_])
                             for a in range(4) for b_ in range(a + 1, 4)]))
        per.append({'subject': f'S{s+1:02d}', 'within_trial': w,
                    'between_subject': b, 'within_own_class_dist': cls})
        print(f'  S{s+1:02d}  试次间 within={w:.3f}   个体间 between={b:.3f}   '
              f'自身类间(信号)={cls:.4f}   between/within={b/w:.1f}x   '
              f'between/类间={b/cls:.1f}x')
    wr = float(np.sqrt(np.mean([p['within_trial'] ** 2 for p in per])))
    br = float(np.sqrt(np.mean([p['between_subject'] ** 2 for p in per])))
    crm = float(np.mean([p['within_own_class_dist'] for p in per]))
    print(f'\n  汇总: within_rms={wr:.3f}  between_rms={br:.3f}  '
          f'比值 R=between²/within²={ (br/wr)**2 :.1f}')
    print(f'  类间(判别信号)={crm:.4f}  ⇒ 个体间距 / 类间距 = {br/crm:.1f}x')
    print(f'  三个尺度排序: 个体间距 {br:.2f}  >  试次散布 {wr:.2f}  >  类间信号 {crm:.2f}'
          f'   (试次散布 / 类间信号 = {wr/crm:.1f}x ⇒ 单试次判别信噪比只有 {crm/wr:.2f})')

    print('\n  试次平均收缩率(每被试随机抽 n 个试次求黎曼均值):')
    print('     n    抽样误差 e(n)    e(n)/within     e(n)/between')
    rng = np.random.default_rng(0)
    curve = []
    for n in NS_Q:
        reps = 24 if n <= 32 else 8
        es = []
        for s in range(len(Cs_list)):
            C = Cs_list[s]
            for _ in range(reps):
                idx = rng.choice(len(C), n, replace=False)
                es.append(distance_riemann(rmean(C[idx], init=Ms[s]), Ms[s]))
        e = float(np.sqrt(np.mean(np.square(es))))
        curve.append({'n': n, 'e': e, 'e_over_within': e / wr, 'e_over_between': e / br})
        print(f'    {n:4d}      {e:.4f}          {e/wr:.3f}          {e/br:.4f}')
    o = np.array([c['e_over_between'] for c in curve])
    n_need = {}
    for thr in (0.5, 0.2, 0.1, 0.05):
        val = None
        if o[0] <= thr:
            val = 1.0
        else:
            for i in range(1, len(curve)):
                if o[i - 1] > thr >= o[i]:
                    f = (o[i - 1] - thr) / (o[i - 1] - o[i])
                    val = float(np.exp(np.log(curve[i - 1]['n']) +
                                       f * (np.log(curve[i]['n']) - np.log(curve[i - 1]['n']))))
                    break
        n_need[f'{thr:g}'] = val
    print('  ⇒ 把试次抽样误差压到个体间距的若干比例,所需试次数 n:')
    for thr, val in n_need.items():
        s = f'{val:.0f}' if val else '>144(本数据不可达)'
        print(f'       e(n)/between ≤ {thr:>4}  →  n ≈ {s}')
    print(f'  ⇒ 等价说法:试次噪声的**方差**降到个体间方差水平只需 n≈{1/(wr/br)**2:.1f} 个试次')
    return {'per_subject': per, 'within_rms': wr, 'between_rms': br,
            'class_dist_mean': crm, 'ratio_sq': (br / wr) ** 2,
            'n_for_ratio': n_need, 'subsample_curve': curve}


def block_A_ridge(Xs, Ys):
    """A 块的 ridge 敏感性:between/within 对 ridge 有多敏感(必须交代,否则数字不可引用)。"""
    print('\n' + '=' * 72)
    print('A2 ridge 敏感性:同一批数据、只改相对 ridge 的扫描')
    print('=' * 72)
    segs = [filter_win(X, MU, TASK) for X in Xs]
    print('   ridge        within   between   类间信号   between/within  between/类间')
    rows = []
    for rg in [1e-9, 1e-8, 1e-7, 1e-6]:
        Cs = [cov_of(S, norm=True, ridge=rg) for S in segs]
        Ms = [rmean(C) for C in Cs]
        Mg = rmean(np.stack(Ms))
        wr = float(np.sqrt(np.mean([np.mean(d_many(C, Ms[s]) ** 2)
                                    for s, C in enumerate(Cs)])))
        br = float(np.sqrt(np.mean([distance_riemann(M, Mg) ** 2 for M in Ms])))
        cls = []
        for s, C in enumerate(Cs):
            y = Ys[s]
            cls.append(float(np.mean([distance_riemann(rmean(C[y == c]), Ms[s])
                                      for c in range(4)])))
        cr = float(np.mean(cls))
        rows.append({'ridge': rg, 'within_rms': wr, 'between_rms': br, 'class_dist': cr})
        print(f'   {rg:.0e}   {wr:8.3f}  {br:8.3f}   {cr:9.4f}   {br/wr:8.2f}x     {br/cr:8.2f}x')
    lo = min(r['between_rms'] for r in rows)
    hi = max(r['between_rms'] for r in rows)
    print(f'  ⇒ between_rms 随 ridge 在 {lo:.2f}–{hi:.2f} 之间变({hi/lo:.2f}x),'
          f'within_rms 在 '
          f'{min(r["within_rms"] for r in rows):.2f}–{max(r["within_rms"] for r in rows):.2f} 之间')
    print('  ⇒ 口径:主口径取 1e-7(与仓库 analyze_spd_separability 的 RIDGE 同量级);'
          '量级与方向稳健,精确倍数不稳健。')
    return rows


# ---------------------------------------------------------------- B 方差分解
def block_B(Cs_list, Ys):
    print('\n' + '=' * 72)
    print('B  切空间方差分解: 身份 vs (被试内)类别 vs 残差')
    print('=' * 72)
    V = [tangent(C) for C in Cs_list]
    allV = np.concatenate(V, 0)
    g = allV.mean(0)
    SSt = float(((allV - g) ** 2).sum())
    SSs = float(sum(len(v) * ((v.mean(0) - g) ** 2).sum() for v in V))
    SSw = SSt - SSs
    SSc = 0.0
    for v, y in zip(V, Ys):
        m = v.mean(0)
        for c in range(4):
            SSc += float((y == c).sum() * ((v[y == c].mean(0) - m) ** 2).sum())
    SSr = SSw - SSc
    print(f'  SS_tot={SSt:.4g}  →  身份(被试)={SSs:.4g} ({SSs/SSt:.4f})   '
          f'类别(被试内)={SSc:.4g} ({SSc/SSt:.4f})   残差={SSr:.4g} ({SSr/SSt:.4f})')
    print(f'  ⇒ 切空间里「这人是谁」解释 {100*SSs/SSt:.2f}% 的方差,'
          f'「想的是哪一类」只解释 {100*SSc/SSt:.2f}%,比值 {(SSs/SSc):.1f}x')
    # 逐人 Fisher 式信噪比:类间散度 / 类内散度(同为切空间欧氏)
    per = []
    for v, y in zip(V, Ys):
        pb = []
        for c in range(4):
            pb.append(np.linalg.norm(v[y == c].mean(0) - v.mean(0)))
        within = np.sqrt(np.mean([((v[y == c] - v[y == c].mean(0)) ** 2).sum(1).mean()
                                 for c in range(4)]))
        per.append({'subject': f'S{len(per)+1:02d}',
                    'class_sep': float(np.mean(pb)), 'class_within': float(within),
                    'snr': float(np.mean(pb) / within)})
    print('  逐被试类别信噪比(类间散度 / 类内散度):')
    for p in per:
        print(f'    {p["subject"]}  类间={p["class_sep"]:.3f}  类内={p["class_within"]:.3f}'
              f'   SNR={p["snr"]:.3f}')
    return {'SS_tot': SSt, 'eta2_subject': SSs / SSt, 'eta2_class_within': SSc / SSt,
            'eta2_resid': SSr / SSt, 'per_subject_snr': per}


# ---------------------------------------------------------------- C 三级阶梯
def acc(Ftr, ytr, Fte, yte):
    m = LDA(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
    return float(m.score(Fte, yte))


def block_C(F, Ys, tag='mu 带 + trace 归一化'):
    print('\n' + '=' * 72)
    print(f'C  判别层三级阶梯(零训练,切空间 + 收缩 LDA)  口径: {tag}')
    print('=' * 72)
    S = [sess_of(y) for y in Ys]

    # C1 跨试次(同被试 5 折)
    accs = [float(np.mean([acc(f[tr], y[tr], f[te], y[te]) for tr, te in
                           StratifiedKFold(5, shuffle=True, random_state=0).split(f, y)]))
            for f, y in zip(F, Ys)]
    # C2 跨 session(同被试,同 session 内 5 折作为对照)
    a_cross = [acc(f[s == 0], y[s == 0], f[s == 1], y[s == 1]) for f, y, s in zip(F, Ys, S)]
    a_within = [float(np.mean([acc(f[(s == k)][tr], y[(s == k)][tr], f[(s == k)][te],
                                    y[(s == k)][te])
                               for tr, te in StratifiedKFold(5, shuffle=True,
                                                             random_state=0).split(
                                   f[s == k], y[s == k])]))
                for f, y, s in zip(F, Ys, S) for k in [0]]
    # C3 跨被试 LOSO
    loso = []
    for i in range(len(F)):
        tr = np.concatenate([F[j] for j in range(len(F)) if j != i], 0)
        ytr = np.concatenate([Ys[j] for j in range(len(F)) if j != i], 0)
        loso.append(acc(tr, ytr, F[i], Ys[i]))
    print(f'  C1 跨试次(同被试同 session 5 折)   被试均值={np.mean(a_within):.3f}  '
          f'逐人 {np.min(a_within):.3f}–{np.max(a_within):.3f}')
    print(f'  C2 跨 session(同被试 sess0→sess1)  被试均值={np.mean(a_cross):.3f}')
    print(f'  C3 跨被试(LOSO,训 8 人测 1 人)     被试均值={np.mean(loso):.3f}  '
          f'逐人 {np.min(loso):.3f}–{np.max(loso):.3f}   (随机 25%)')
    print(f'  ⇒ 阶梯落差: C1−C2={np.mean(a_within)-np.mean(a_cross):+.3f}  '
          f'C2−C3={np.mean(a_cross)-np.mean(loso):+.3f}   ({tag})')
    per = [{'subject': f'S{i+1:02d}', 'within': a_within[i], 'cross_session': a_cross[i],
            'loso': loso[i]} for i in range(len(F))]

    # C4 试次平均曲线(最近类均值):n 个试次平均后,试次噪声被压掉多少
    print('\n  C4 试次平均曲线(每类抽 n 个试次求类均值,最近类均值分类,20 次重复):')
    rng = np.random.default_rng(1)
    avg_curve = []
    for n in [1, 2, 4, 8, 16, 32, 72]:
        a = []
        for f, y in zip(F, Ys):
            for _ in range(20):
                tr_idx, te_idx = [], []
                for c in range(4):
                    ci = np.where(y == c)[0]
                    pick = rng.choice(ci, n, replace=False)
                    tr_idx += list(pick)
                    te_idx += list(np.setdiff1d(ci, pick))
                cen = np.stack([f[[i for i in tr_idx if y[i] == c]].mean(0)
                                for c in range(4)])
                d = ((f[te_idx][:, None, :] - cen[None]) ** 2).sum(-1)
                a.append(float((d.argmin(1) == y[te_idx]).mean()))
        avg_curve.append({'n': n, 'acc': float(np.mean(a))})
        print(f'    n={n:3d}  acc={np.mean(a):.3f}')
    return {'per_subject': per, 'within_mean': float(np.mean(a_within)),
            'cross_session_mean': float(np.mean(a_cross)),
            'loso_mean': float(np.mean(loso)), 'loso_per_subject': loso,
            'avg_curve': avg_curve}


# ---------------------------------------------------------------- C5 个体校准
def block_C5(Cs_list, Ys):
    print('\n' + '=' * 72)
    print('C5 个体校准曲线:给共享分类器补 k 个目标被试试次(其余 8 人固定)')
    print('   对照:同样 k 个试次,但完全不给他人数据(纯目标自己训练)')
    print('=' * 72)
    F = [tangent(C) for C in Cs_list]
    KS = [0, 4, 8, 16, 32, 72, 144, 288]
    rng = np.random.default_rng(2)
    rows = []
    for k in KS:
        a, solo = [], []
        for i in range(len(F)):
            tr = np.concatenate([F[j] for j in range(len(F)) if j != i], 0)
            ytr = np.concatenate([Ys[j] for j in range(len(F)) if j != i], 0)
            ci = np.arange(len(Ys[i]))
            pick = np.array([], dtype=int)
            if k > 0:
                pick = np.concatenate([rng.choice(np.where(Ys[i] == c)[0],
                                                  max(1, k // 4), replace=False)
                                       for c in range(4)])
            te = np.setdiff1d(ci, pick)
            a.append(acc(np.concatenate([tr, F[i][pick]], 0),
                         np.concatenate([ytr, Ys[i][pick]], 0), F[i][te], Ys[i][te]))
            if len(pick) > 4:      # LDA 要求 n_samples > n_classes,1 个/类时跳过
                solo.append(acc(F[i][pick], Ys[i][pick], F[i][te], Ys[i][te]))
        sl = f'  | 纯目标自己={np.mean(solo):.3f}' if solo else ''
        print(f'  k={k:4d} 个目标被试试次 → 8人+k: acc={np.mean(a):.3f}  '
              f'(逐人 {np.min(a):.3f}–{np.max(a):.3f}){sl}')
        rows.append({'k': k, 'acc': float(np.mean(a)),
                     'solo_acc': float(np.mean(solo)) if solo else None,
                     'per_subject': a})
    print('  ⇒ 校准曲线若在 k 很小时就贴近 C1,说明"个体适应余量"很小;'
          '若必须 k 上百才抬头,说明共享读法与个体读法之间是结构性错配。')
    return {'calibration': rows}


# ---------------------------------------------------------------- D 三方向
def band_analytic(x, sosf):
    return hilbert(sosfiltfilt(sosf, x, axis=-1), axis=-1)


def itc_of(Z):
    """Z: (n_trials, T) 复解析信号 → ITC(t) (T,)"""
    U = Z / np.maximum(np.abs(Z), 1e-20)
    return np.abs(U.mean(0))


def block_D(Xs, Ys, subj_table):
    print('\n' + '=' * 72)
    print('D  三方向试次间一致性(BCI-2a;C3/C4,与 ITC 材料同一套方向)')
    print('=' * 72)
    rng = np.random.default_rng(3)
    rows = []
    for s, (X, y) in enumerate(zip(Xs, Ys)):
        for c, chn in [(0, 'C4'), (1, 'C3')]:      # 左手取对侧 C4,右手取 C3
            ci = CH[chn]
            x = X[y == c, ci, :]
            zmu = band_analytic(x, SOS_MU)
            itc = itc_of(zmu)
            # 经验零假设:每试次独立循环平移
            nulls = []
            for _ in range(24):
                sh = rng.integers(60, len(x[0]) - 60, size=len(x))
                zz = np.stack([np.roll(z, k) for z, k in zip(zmu, sh)])
                nulls.append(itc_of(zz))
            nulls = np.array(nulls)
            itc_e = float(itc[EARLY[0]:EARLY[1]].mean())
            itc_l = float(itc[TASK[0]:TASK[1] - 25].mean())
            itc_a = float(itc.mean())
            # 劈半可重复性(奇偶两半在 0-3.5s 上的相关)
            h1, h2 = itc_of(zmu[0::2]), itc_of(zmu[1::2])
            sl = slice(CUE, TASK[1] - 25)
            r_half = float(np.corrcoef(h1[sl], h2[sl])[0, 1])
            # 幅值方向:mu 包络
            env = np.abs(zmu)[:, TASK[0]:TASK[1] - 25]
            E = env - env.mean(1, keepdims=True)
            E /= np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-20)
            R_pair = float((E @ E.T).sum() - len(E)) / (len(E) * (len(E) - 1))
            loo = (env.sum(0)[None, :] - env) / (len(env) - 1)
            L = loo - loo.mean(1, keepdims=True)
            L /= np.maximum(np.linalg.norm(L, axis=1, keepdims=True), 1e-20)
            R_hold = float((E * L).sum(1).mean())
            # 波形方向:宽带单试次与留一均值的相关 + EVF
            wf = x[:, TASK[0]:TASK[1] - 25]
            wloo = (wf.sum(0)[None, :] - wf) / (len(wf) - 1)
            W = wf - wf.mean(1, keepdims=True)
            W /= np.maximum(np.linalg.norm(W, axis=1, keepdims=True), 1e-20)
            Wl = wloo - wloo.mean(1, keepdims=True)
            Wl /= np.maximum(np.linalg.norm(Wl, axis=1, keepdims=True), 1e-20)
            r_hold = float((W * Wl).sum(1).mean())
            evf = float(wf.mean(0).var() / np.mean(wf.var(1)))
            # ERD%(mu 带功率,任务窗 vs cue 前基线)
            pm = float((np.abs(zmu[:, TASK[0]:TASK[1]]) ** 2).mean())
            pb = float((np.abs(zmu[:, BASE_WIN[0]:BASE_WIN[1]]) ** 2).mean())
            erd = float(10 * np.log10(pm / pb))
            rows.append({'subject': f'S{s+1:02d}', 'cond': CLASSES[c], 'ch': chn,
                         'itc_full': itc_a, 'itc_early': itc_e, 'itc_late': itc_l,
                         'itc_null_mean': float(nulls.mean()),
                         'itc_null_sd': float(nulls.std()),
                         'itc_early_z': float((itc_e - nulls.mean()) / nulls.std()),
                         'itc_late_z': float((itc_l - nulls.mean()) / nulls.std()),
                         'half_split_r': r_half, 'R_pair': R_pair, 'R_hold': R_hold,
                         'wf_r_hold': r_hold, 'EVF': evf, 'ERD_mu_dB': erd})
            print(f'  {rows[-1]["subject"]} {CLASSES[c]:>2} {chn}  '
                  f'ITC 早={itc_e:.3f} 晚={itc_l:.3f} (零假设 {nulls.mean():.3f}±{nulls.std():.3f}) '
                  f'劈半r={r_half:+.2f} | 包络 R_pair={R_pair:+.2f} R_hold={R_hold:+.2f} | '
                  f'波形 r={r_hold:+.2f} EVF={evf:.3f} | ERD={erd:+.2f}dB')
    # 逐人汇总(两条件平均)后与既有精度读数求 Spearman
    acc = {r['subject']: r['deep70_acc'] for r in subj_table}
    csp = {r['subject']: r['csp_within'] for r in subj_table}
    tan = {r['subject']: r['tangent_lda_within'] for r in subj_table}
    subj = [f'S{i+1:02d}' for i in range(9)]
    def agg(key):
        return np.array([np.mean([r[key] for r in rows if r['subject'] == s]) for s in subj])
    print('\n  逐人(S01-S09) 汇总与相关性(n=9,Spearman):')
    corr = {}
    for key in ['itc_early', 'itc_late', 'half_split_r', 'R_hold', 'wf_r_hold',
                'EVF', 'ERD_mu_dB']:
        v = agg(key)
        line = f'    {key:<14}'
        for nm, ref in [('deep70', [acc[s] for s in subj]),
                        ('csp_within', [csp[s] for s in subj]),
                        ('tangent_lda', [tan[s] for s in subj])]:
            rho, p = spearmanr(v, ref)
            corr[f'{key}_vs_{nm}'] = {'rho': float(rho), 'p': float(p)}
            line += f'  ρ({nm})={rho:+.3f}(p={p:.3f})'
        print(line)
        print(f'                   逐人值: ' + ' '.join(f'{x:6.3f}' for x in v))
    print('    deep70 逐人:        ' + ' '.join(f'{acc[s]:6.3f}' for s in subj))
    print('    csp_within 逐人:    ' + ' '.join(f'{csp[s]:6.3f}' for s in subj))
    return {'rows': rows, 'corr': corr}


# ---------------------------------------------------------------- 图
def plots(A, C, C5, D, subj_table):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    per = A['per_subject']
    ax[0].bar(np.arange(9) - 0.2, [p['within_trial'] for p in per], 0.4, label='within (trial-to-trial)')
    ax[0].bar(np.arange(9) + 0.2, [p['between_subject'] for p in per], 0.4, label='between (subject)')
    ax[0].set_yscale('log'); ax[0].set_xticks(range(9))
    ax[0].set_xticklabels([p['subject'] for p in per], rotation=45)
    ax[0].set_ylabel('affine-invariant distance (log)')
    ax[0].set_title('A  trial vs subject spread (SPD)'); ax[0].legend(fontsize=8)
    n = [c['n'] for c in A['subsample_curve']]
    ax[1].plot(n, [c['e_over_between'] for c in A['subsample_curve']], 'o-', label='e(n)/between')
    ax[1].plot(n, [c['e_over_within'] for c in A['subsample_curve']], 's--', label='e(n)/within')
    ax[1].axhline(0.5, color='r', lw=1, ls=':')
    ax[1].set_xscale('log'); ax[1].set_yscale('log')
    ax[1].set_xlabel('n trials averaged'); ax[1].set_title('A2  shrinkage of trial noise')
    ax[1].legend(fontsize=8)
    ac = C['avg_curve']
    ax[2].plot([c['n'] for c in ac], [c['acc'] for c in ac], 'o-')
    ax[2].axhline(C['within_mean'], color='g', ls='--', label='within-session CV')
    ax[2].axhline(C['loso_mean'], color='r', ls=':', label='LOSO (no target data)')
    ax[2].set_xscale('log'); ax[2].set_xlabel('n trials averaged per class')
    ax[2].set_ylabel('accuracy'); ax[2].set_title('C4  averaging over trials')
    ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / 'fig_A_C.png', dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    names = ['within-session\n(trial-to-trial)', 'cross-session\n(same subject)',
             'cross-subject\n(8 train -> 1 test)']
    vals = [C['within_mean'], C['cross_session_mean'], C['loso_mean']]
    ax[0].bar(names, vals, color=['#3b7', '#39c', '#c33'])
    for i, v in enumerate(vals):
        ax[0].text(i, v + 0.01, f'{v:.3f}', ha='center')
    ax[0].axhline(0.25, color='k', ls=':', lw=1); ax[0].set_ylim(0, 1)
    ax[0].set_ylabel('4-class accuracy'); ax[0].set_title('C  generalization ladder')
    ks = [r['k'] for r in C5['calibration']]
    ax[1].plot(ks, [r['acc'] for r in C5['calibration']], 'o-', label='8 others + k')
    solo = [(r['k'], r['solo_acc']) for r in C5['calibration'] if r.get('solo_acc')]
    if solo:
        ax[1].plot([s[0] for s in solo], [s[1] for s in solo], 's--',
                   label='target alone (k trials)')
    ax[1].axhline(C['loso_mean'], color='r', ls=':', label='k=0 (LOSO)')
    ax[1].axhline(C['within_mean'], color='g', ls='--', label='within-subject ceiling')
    ax[1].set_xscale('symlog'); ax[1].set_xlabel('k target-subject trials added')
    ax[1].set_ylabel('4-class accuracy'); ax[1].set_title('C5  individual calibration')
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / 'fig_ladder.png', dpi=130); plt.close()

    rows = D['rows']
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sub = [f'S{i+1:02d}' for i in range(9)]
    erd = [np.mean([r['ERD_mu_dB'] for r in rows if r['subject'] == s]) for s in sub]
    accs = [r['deep70_acc'] for r in subj_table]
    ax[0].scatter(erd, accs)
    for s, x, yy in zip(sub, erd, accs):
        ax[0].annotate(s, (x, yy), fontsize=7)
    rho, p = spearmanr(erd, accs)
    ax[0].set_xlabel('ERD mu (dB, task vs baseline)'); ax[0].set_ylabel('deep70 accuracy')
    ax[0].set_title(f'D  ERD vs accuracy  rho={rho:+.3f} (p={p:.3f})')
    itc_l = [np.mean([r['itc_late'] for r in rows if r['subject'] == s]) for s in sub]
    itc_e = [np.mean([r['itc_early'] for r in rows if r['subject'] == s]) for s in sub]
    ax[1].plot(sub, itc_e, 'o-', label='ITC mu early 0-0.5s')
    ax[1].plot(sub, itc_l, 's-', label='ITC mu late 0.5-3.5s')
    ax[1].axhline(np.mean([r['itc_null_mean'] for r in rows]), color='r', ls=':',
                  label='empirical null 0.886/sqrt(N)')
    ax[1].set_ylabel('ITC'); ax[1].set_title('D  phase consistency per subject')
    ax[1].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(OUT / 'fig_D.png', dpi=130); plt.close()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('=' * 72)
    print('试次间差异 vs 个体间差异 —— 量级分解(零训练)')
    print('=' * 72)
    Xs, Ys = load()
    print(f'数据: {len(Xs)} 人 × {len(Xs[0])} 试次 × {Xs[0].shape[1]} 通道  '
          f'epoch={Xs[0].shape[2]} 采样({Xs[0].shape[2]/FS:.2f}s)  窗口基线{BASE_WIN} 任务{TASK}')
    Cs = [trial_cov(X, MU, TASK) for X in Xs]
    Csb = [trial_cov(X, (8, 30), TASK, norm=False) for X in Xs]   # 与仓库 cov() 同口径
    out = {'config': {'pkl': str(PKL), 'n_subjects': len(Xs), 'n_trials': len(Xs[0]),
                      'window_task': TASK, 'band_mu': MU, 'fs': FS}}
    out['A'] = block_A(Cs, Ys)
    out['A_ridge'] = block_A_ridge(Xs, Ys)
    out['B'] = block_B(Cs, Ys)
    out['C'] = block_C([tangent(C) for C in Cs], Ys,
                       'mu 带 8-13Hz + 逐试次 trace 归一化(本脚本主口径)')
    out['C_broadband'] = block_C([tangent(C) for C in Csb], Ys,
                                 '全带 8-30Hz、不归一化(与 analyze_individual_diff 的 cov() 同口径)')
    out['C5'] = block_C5(Cs, Ys)
    st = json.load(open(BASE / 'results' / 'individual_diff' / 'subject_table.json'))
    out['D'] = block_D(Xs, Ys, st)
    with open(OUT / 'trial_vs_subject.json', 'w') as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    try:
        plots(out['A'], out['C'], out['C5'], out['D'], st)
        print(f'\n图已落盘: {OUT}/fig_A_C.png / fig_ladder.png / fig_D.png')
    except Exception as e:
        print(f'[warn] 画图失败: {e}')
    print(f'\n总耗时 {time.time()-t0:.1f}s   json -> {OUT}/trial_vs_subject.json')


if __name__ == '__main__':
    main()
