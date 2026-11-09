# -*- coding: utf-8 -*-
"""因果性分析 步骤 1-3:实现自检 / X-0 逐试次 ERD 可靠性 / §3.4 剂量准入 + X-1 主臂。

配套设计文档: analysis/差异报告/因果性分析方案_ERD到精度.md §2.2 §2.3 §3.4 §8(步骤1-4)
纯 CPU、零训练。只读 saved_data/preprocessed_data_bci2a_raw_ica.pkl 与既有 results/。

## 干预的数学(为什么信号空间的混合恰好等于协方差空间的线性干预)

把任务窗信号拆成 μ 带成分与其余成分:Z = Z_mu + Z_o。干预取
「把 μ 波的**波动部分**(相对该被试任务窗 μ 均值的偏离)乘 sqrt(a)」:

    Z_mu -> mu + sqrt(a) * (Z_mu - mu),   mu := 该被试任务窗 Z_mu 的跨试次均值

用 Z_mu = mu + (Z_mu - mu) 展开即得**信号空间恒等式**

    Z(a) = Z + (sqrt(a) - 1) * (Z_mu - mu)

记 C[X] := X X^T / T、N := C[Z_mu - mu],对 (Z_o + Z_mu) 逐项展开(交叉项在跨试次
平均后归零,剩下 O(1/sqrt(N)) 的涨落):

    M(a) = C[Z(a)] = C[Z_o] + C[mu] + a * C[Z_mu - mu] + 2*sqrt(a)*C[Z_o + mu, Z_mu - mu]
         = C[Z] + (a - 1) * N                    (★ 与方案 §2.3 同形)

另一个等价写法:M(a) = C[Z_o + mu] + a*N ⇒ a=1 时逐位还原、a=0 时只留「任务窗 μ 的
被试均值」,且

    trace(M(a)) = trace(C[Z_o + mu]) + a*trace(N)     (不是常数,见下)

## ERD 刻度

M(a) 里参与 ERD 的那一项是任务窗 μ 功率 A:
    A(a) = ‖mu‖² + a*‖Z_mu - mu‖² = a*A(1) + (1-a)*‖mu‖²
⇒ **a 只缩放试次级波动,不碰被试均值**;而基线窗 μ 功率 R 与 a 无关,故

    ERD_t(a) = 10*log10( A_t(a) / R_t )

本脚本据实计算这个量(不套 10*log10(a) 的近似公式:那个近似只在忽略 (1-a)‖mu‖² 时成立,
对长窗小、对短窗大,故一律用实测值)。

## a=1 的读出器看不见什么(这条解释了 X-1 会出现什么)

「任务窗 μ 的被试均值」mu 是 22×22 的低秩成分;把它整体加进每一个试次,是给该被试
**所有类别同等**地加一个方向。逐被试类内中心化之后这个方向被减掉 ⇒ **共享读出器
看不见「整体时不变 μ 包络」**。所以本干预实际可动的是 μ 的**试次间波动**,判据里
必须把这两部分的判别信号分开量(见 step3b)。

## X-0 的两个切法

对同一批试次用两种时间支撑**不同**的切法各估一次逐试次 ERD:任务窗前半 / 后半。
⚠ 实测两者 Spearman ≈ 1.000 —— 这不是"可靠性极高",而是**两个估计器在代数上几乎等价**
(都是同一段希尔伯特包络的平均功率,Parseval)。故本脚本另给一个不依赖这个等价的
噪声预算:逐试次 ED 的单试次抽样标准误(频域 delta 法,见 noise_budget)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_spd_separability import tangent                     # noqa: E402
from utils.dataset import load_preprocessed_dataset               # noqa: E402

FS = 250
CUE = 250
BASE_WIN = (0, 250)          # cue 前 -1.0 ~ 0 s
TASK = (375, 1126)           # cue 后 0.5 ~ 3.5 s
MU = (8.0, 13.0)
BROAD = (8.0, 30.0)
RIDGE = 1e-7
PKL = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
DELTAS_PLAN = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]        # 方案 §2.6 的名义档
ALPHAS_PLAN = [10 ** (d / 10.0) for d in DELTAS_PLAN]
# 主档:预探针显示「名义 ±2dB 太弱、真正的塌陷在 a→0」,故主档覆盖「完全移除 → 放大 2 倍」。
# a 与 ERD 的刻度由 step2/step3 的实测 Δ(dB) 给出(≈10*log10(a))。
ALPHAS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
CTRL_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0]   # 全带缩放臂,应≈no-op
SUBJ = [f'S{i+1:02d}' for i in range(9)]


def sos(band):
    return butter(4, [band[0] / (FS / 2), band[1] / (FS / 2)], btype='band', output='sos')


SOS_MU, SOS_BROAD = sos(MU), sos(BROAD)


def load_raw():
    sd, _ = load_preprocessed_dataset(str(PKL))
    return [(np.asarray(X, dtype=np.float64), np.asarray(y)) for X, y in sd]


def narrow(X, band):
    return sosfiltfilt(sos(band), X, axis=-1)


def cov_task(S):
    """与项目既有口径一致(analyze_trial_vs_subject.filter_win 先减时间均值再算协方差)。"""
    S = S - S.mean(axis=-1, keepdims=True)
    return np.einsum('nct,ndt->ncd', S, S) / S.shape[-1]


def trace_all(M):
    return np.trace(M, axis1=1, axis2=2)


def logm_all(M, ridge=RIDGE):
    n = M.shape[-1]
    tr = np.trace(M, axis1=1, axis2=2)[:, None, None] / n
    M = M + ridge * tr * np.eye(n)
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))


def logm_one(M):
    return logm_all(M[None])[0]


def centered(F):
    return F - F.mean(axis=0, keepdims=True)


def pieces(X, band):
    """返回 (Z, Zband, Zbar):宽带信号 / 带内成分 / 带内成分的被试任务窗均值。"""
    Z = X
    Zb = narrow(X, band)
    Zbar = Zb[:, :, TASK[0]:TASK[1]].mean(axis=0, keepdims=True)
    return Z, Zb, Zbar


def mix(Zb, Zbar, a):
    """Z_mu -> mu + sqrt(a)*(Z_mu - mu),再与其余成分相加 = Z + (sqrt(a)-1)*(Z_mu-mu)。"""
    return Zb + (np.sqrt(a) - 1.0) * (Zb - Zbar)


def band_pow(S, win):
    """窗内功率。先按预处理口径逐试次去时间均值(与 trial_vs_subject.filter_win 一致),
    再取窗内均方 —— 不这么做会与试次级 ERD 差一个 rank-1 项。"""
    S = S - S.mean(axis=-1, keepdims=True)
    seg = S[:, :, win[0]:win[1]]
    return (seg ** 2).sum(axis=(1, 2)) / (seg.shape[1] * seg.shape[2])


# ---------------------------------------------------------------- 步骤 1
def step1_selfcheck():
    print('=' * 72)
    print('步骤 1 · 实现自检(方案 §8 的硬门槛)')
    print('=' * 72)
    rng = np.random.default_rng(0)
    A = rng.standard_normal((22, 22))
    M = A @ A.T / 22 + np.eye(22)
    B = rng.standard_normal((22, 30))
    N = B @ B.T / 30
    e1 = float(np.abs((M + (1.0 - 1.0) * N) - M).max())
    e2 = float(np.abs((M + (0.0 - 1.0) * N) - (M - N)).max())
    print(f'  (i)  a=1: M(a)-M 逐元素最大偏差 = {e1:.3e}    (要求 0)')
    print(f'  (ii) a=0: M(a)-(M-N) 最大偏差  = {e2:.3e}    (要求 0)')

    # (iii) 尺度不变性:log(sM) = log s*I + log M。切空间里 log s·I 是**一个方向**,
    # 保留;被丢掉它的是**逐被试中心化**(模型里由 SPDNorm/tangent 承担)。所以必须在
    # 「一批试次 + 各自缩放」这个模型真正看到的口径下测,而不是单矩阵。
    B2 = rng.standard_normal((24, 22, 22))
    Cs = B2 @ np.transpose(B2, (0, 2, 1)) / 22 + np.eye(22)[None]
    d = []
    for s in (0.25, 0.5, 2.0, 4.0):
        F1 = centered(tangent(Cs))
        Fs = centered(tangent(s * Cs))
        d.append(float(np.linalg.norm(Fs - F1)))
    print(f'  (iii) 整批试次同时乘 s∈{{0.25,0.5,2,4}} 后,**中心化**切空间与未缩放的距离 = '
          + ' '.join(f'{x:.2e}' for x in d) + '    (要求 0)')
    print('        ⇒ 等比例缩放的方向 log s·I 正是逐被试中心化丢掉的(方案 §2.2);'
          'Q-3「按人协方差中心化 ≡ SPDNorm」是同一件事的旁证')
    ok = (e1 == 0.0) and (e2 == 0.0) and max(d) < 1e-12
    print(f'  ⇒ 步骤 1 {"通过" if ok else "不通过"}')
    return {'identity_a1_max_abs_err': e1, 'identity_a0_max_abs_err': e2,
            'scale_invariance_dist_max': max(d), 'passed': bool(ok)}


# ---------------------------------------------------------------- 步骤 2:X-0
def step2_x0(datas):
    print('\n' + '=' * 72)
    print('步骤 2 · X-0 逐试次 ERD 的可靠性 + 单试次噪声预算')
    print('=' * 72)
    rows = []
    for s, (X, y) in enumerate(datas):
        _, Zb, _ = pieces(X, MU)
        R = band_pow(Zb, BASE_WIN)
        half = TASK[0] + (TASK[1] - TASK[0]) // 2
        eA = 10 * np.log10(np.mean([band_pow(Zb, (TASK[0], half)),
                                    band_pow(Zb, (half, TASK[1]))], axis=0) / R)
        eB = 10 * np.log10(band_pow(Zb, TASK) / R)
        rho = float(spearmanr(eA, eB).statistic)
        # 单试次抽样标准误(频域 delta 法):μ 功率估计的有效自由度 ≈ 2*B*T_eff
        bwd = MU[1] - MU[0]
        dof_t, dof_r = 2 * bwd * (TASK[1] - TASK[0]) / FS, 2 * bwd * (BASE_WIN[1] - BASE_WIN[0]) / FS
        sed = np.sqrt(1.0 / dof_t + 1.0 / dof_r) * 10 / np.log(10)
        sd_obs = float(eA.std())
        rows.append({'subject': SUBJ[s], 'spearman_half_vs_full': rho,
                     'erd_mean_dB': float(eA.mean()), 'erd_sd_dB': sd_obs,
                     'se_single_dB': float(sed), 'noise_frac_of_sd': float(sed / sd_obs)})
        print(f'  {SUBJ[s]}  前半/后半 vs 全窗 Spearman={rho:+.4f}   '
              f'逐试次 ERD {eA.mean():+.2f}±{sd_obs:.2f} dB   '
              f'单试次抽样 SE={sed:.2f} dB ⇒ 噪声占观测 sd 的 {sed/sd_obs*100:.0f}%')
    r = np.array([x['spearman_half_vs_full'] for x in rows])
    nf = np.array([x['noise_frac_of_sd'] for x in rows])
    print(f'\n  九人 Spearman 均值={r.mean():+.4f}(范围 {r.min():+.3f}~{r.max():+.3f});'
          f'噪声占 sd 比例均值={nf.mean()*100:.0f}%')
    print('  判读(重要):两切法相关≈1 说明它们**代数上等价**(同段包络的平均功率),'
          '不能用它当可靠性上界;')
    print('              真正该看的是右侧「单试次抽样 SE / 观测 sd」——它才是逐试次 ERD 的'
          '测量噪声占比。')
    return {'per_subject': rows, 'spearman_mean': float(r.mean()),
            'noise_frac_mean': float(nf.mean()),
            'estimators_are_redundant': bool(r.mean() > 0.99)}


# ---------------------------------------------------------------- 步骤 3:剂量可见性
def step3_admission(datas):
    print('\n' + '=' * 72)
    print('步骤 3 · §3.4 剂量准入:干预在度量上可见吗(单位统一为切空间欧氏距离)')
    print('=' * 72)
    rows = []
    for s, (X, y) in enumerate(datas):
        Zb = narrow(X, BROAD)
        M0 = cov_task(Zb[:, :, TASK[0]:TASK[1]])
        Zmb = narrow(X, MU)[:, :, TASK[0]:TASK[1]]
        mbar = Zmb.mean(axis=0, keepdims=True)
        N = cov_task(Zmb - mbar)
        F0 = centered(tangent(M0))
        sig = float(np.sqrt((F0 ** 2).sum(axis=1).mean()))
        mu_c = np.stack([F0[y == c].mean(axis=0) for c in range(4)])
        r2 = float(((mu_c - F0.mean(axis=0)) ** 2).sum(axis=1).mean())
        D2 = r2 / max(sig ** 2, 1e-30)
        rec = {'subject': SUBJ[s], 'sigma_trial': sig, 'R2_between_classes': r2,
               'D2_between_over_within': D2, 'dose': []}
        print(f'\n  {SUBJ[s]}  σ_trial={sig:.3f}  类间R²={r2:.3f}  D²={D2:.3f}'
              f'(SNR≈{np.sqrt(D2):.3f})')
        print('     名义Δ  实测Δ̄(dB)     a     δ_s(a)   δ_s/σ_trial   η_s=(1-a)·D² '
              'tr比')
        for a in ALPHAS:
            d = 10 * np.log10(a) if a > 0 else float('-inf')
            Ma = M0 + (a - 1.0) * N
            Fa = centered(tangent(Ma))
            delta = float(np.linalg.norm(Fa - F0))
            R = band_pow(narrow(X, MU), BASE_WIN)
            A1 = band_pow(Zmb, TASK)
            Aa = a * A1 + (1 - a) * band_pow(np.repeat(mbar, len(y), axis=0), TASK)
            rd = float(np.mean(10 * np.log10(Aa / R) - 10 * np.log10(A1 / R)))
            rec['dose'].append({'delta_nominal_dB': d, 'delta_measured_dB': rd,
                                'alpha': a, 'delta_s': delta,
                                'delta_over_sigma': delta / sig,
                                'eta_s': (1 - a) * D2,
                                'trace_ratio': float(trace_all(Ma).mean() /
                                                     trace_all(M0).mean())})
            print(f'    {d:+5.2f}  {rd:+8.3f}   {a:.3f}   {delta:8.4f}   {delta/sig:8.4f}'
                  f'      {(1-a)*D2:+7.4f}  {trace_all(Ma).mean()/trace_all(M0).mean():.3f}')
        rows.append(rec)
    print('\n  准入判定(方案 §3.4:δ_s 在最大档位仍小于被试自身试次散布 ⇒ 该档位不可用):')
    for r in rows:
        mx = max(dd['delta_over_sigma'] for dd in r['dose'])
        first = next((dd['delta_nominal_dB'] for dd in r['dose']
                      if dd['delta_over_sigma'] >= 1.0), None)
        print(f'    {r["subject"]}  最大 δ_s/σ_trial={mx:.3f}   '
              + (f'首次 ≥1 于名义Δ={first:+.1f}dB' if first is not None
                 else '★ 全部档位 <1'))
    return {'per_subject': rows}


# ---------------------------------------------------------------- 步骤 3b:判别信号分解
def step3b_decompose(datas):
    print('\n' + '=' * 72)
    print('步骤 3b · 判别信号里「μ 的时不变包络」和「μ 的试次间波动」各占多少')
    print('=' * 72)
    print('  为什么必须分解:干预(a 变化)只缩放**波动**那一项,「整体包络」是给一个')
    print('  被试所有类别同等加一个方向,逐被试类内中心化后被减掉 ⇒ 共享读出器看不见。')
    print('  故「干预能影响多少判别信号」的上界 = 波动那一项的贡献,不是全部 μ 信号。')
    rows = []
    for s, (X, y) in enumerate(datas):
        Zb = narrow(X, BROAD)
        Zmb = narrow(X, MU)
        Zmb_t = Zmb[:, :, TASK[0]:TASK[1]]
        mbar = Zmb_t.mean(axis=0, keepdims=True)
        Mfull = cov_task(Zb[:, :, TASK[0]:TASK[1]])
        Mmean = cov_task(np.repeat(mbar, len(y), axis=0))
        Mfluc = cov_task(Zmb_t - mbar)
        Mo = Mfull - Mmean - Mfluc           # 其余成分(交叉项残差)
        outs = {}
        for nm, Mr in (('full', Mfull), ('mean_only', Mmean),
                       ('fluc_only', Mfluc + Mo), ('other_only', Mo)):
            F = centered(tangent(Mr))
            mc = np.stack([F[y == c].mean(axis=0) for c in range(4)])
            outs[nm] = float(((mc - F.mean(axis=0)) ** 2).sum(axis=1).mean())
        sd = float(np.sqrt((centered(tangent(Mfluc)) ** 2).sum(axis=1).mean()))
        D2f = outs['fluc_only'] / max(sd ** 2, 1e-30)
        rows.append({'subject': SUBJ[s], 'R2': outs, 'D2_fluc': D2f,
                     'fluc_share_of_mu': outs['fluc_only'] /
                     max(outs['fluc_only'] + outs['mean_only'], 1e-30)})
        print(f'  {SUBJ[s]}  类间R²: 全带={outs["full"]:.3f}  '
              f'μ均值件={outs["mean_only"]:.3f}  μ波动件={outs["fluc_only"]:.3f}  '
              f'其余={outs["other_only"]:.3f}  ⇒ 波动占 μ 两件之和的 '
              f'{rows[-1]["fluc_share_of_mu"]*100:.0f}%')
    mf = np.mean([r['R2']['fluc_only'] for r in rows])
    mm = np.mean([r['R2']['mean_only'] for r in rows])
    mo = np.mean([r['R2']['other_only'] for r in rows])
    print(f'\n  九人均值: 全带={np.mean([r["R2"]["full"] for r in rows]):.3f}  '
          f'μ波动={mf:.3f}  μ均值件={mm:.3f}  其余={mo:.3f}')
    print(f'  ⇒ 干预可动的那部分(μ 波动)占全带类间信号的 ~{mf/(mf+mm+mo)*100:.0f}%;'
          f'这就是 X-1 精度变化的上界量级。')
    full_m = float(np.mean([r['R2']['full'] for r in rows]))
    return {'per_subject': rows,
            'mean_R2': {'full': full_m, 'fluc': float(mf),
                        'mean': float(mm), 'other': float(mo)}}


# ---------------------------------------------------------------- 步骤 3c:oracle 可分性
def step3c_oracle(datas, band=MU, alphas=None):
    """oracle 类间分离度:类心的平均两两距离 / 试次 RMS 半径。

    模型无关、无 5 折方差、无 LDA 正则 —— 若干预真的动了判别结构,这个量必然动。
    """
    alphas = ALPHAS if alphas is None else alphas
    print('\n' + '=' * 72)
    print(f'步骤 3c · oracle 类间分离度(带={band})—— 无读出器方差,直接看判别结构')
    print('  量:类心平均两两距离 D(a),**分母固定为 a=1 时的 D1**(不随 a 变,否则会把')
    print('      尺度变化当成分离度变化 —— 这正是第一版口径的错误)。')
    print('=' * 72)
    rows = []
    for s, (X, y) in enumerate(datas):
        Zb = narrow(X, band)
        Zbar = Zb[:, :, TASK[0]:TASK[1]].mean(axis=0, keepdims=True)
        M = cov_task(Zb[:, :, TASK[0]:TASK[1]])
        N = cov_task(Zb[:, :, TASK[0]:TASK[1]] - Zbar)
        F1 = centered(tangent(M + 0.0 * N))
        cen1 = np.stack([F1[y == c].mean(axis=0) for c in range(4)])
        D1 = float(np.mean([np.linalg.norm(cen1[i] - cen1[j])
                            for i in range(4) for j in range(i + 1, 4)]))
        sep = []
        for a in alphas:
            F = centered(tangent(M + (a - 1.0) * N))
            cen = np.stack([F[y == c].mean(axis=0) for c in range(4)])
            sep.append(float(np.mean([np.linalg.norm(cen[i] - cen[j])
                                      for i in range(4) for j in range(i + 1, 4)])) / D1)
        rows.append({'subject': SUBJ[s], 'D_over_D1': sep, 'D1': D1})
        print(f'  {SUBJ[s]}  ' + ' '.join(f'{v:.4f}' for v in sep))
    arr = np.array([r['D_over_D1'] for r in rows])
    m = arr.mean(0)
    print(f'  九人均值 ' + ' '.join(f'{v:.4f}' for v in m))
    print(f'  ⇒ a=0 时类间分离度只剩 a=1 时的 {m[0]*100:.1f}%;'
          f'a=2 时为 {m[-1]*100:.1f}%')
    return {'alphas': alphas, 'per_subject': rows, 'mean_D_over_D1': m.tolist()}


# ---------------------------------------------------------------- 混合模型(自实现)
def mixed_within_between(y, d, sub, use_reml=True):
    """y = b0 + bW*(d-dbar_s) + bB*dbar_s + u_s + e,  u_s~N(0,tau2), e~N(0,sig2)。

    环境无 statsmodels(全仓无该依赖),故按方案 §3.2 的分量模型自写似然。

    实现要点(第一版写成一般 n×n 的 V = I + tau2*ZZ',3600 阶求逆 + Nelder-Mead 直接
    算不动,实测 >120 s 不返回):利用**分析单位是 21 个组均值、残差在组内同分布**这一
    结构,把 Σ^{-1} 写成 9 个小块的块对角形式,逐块 2x2 Cholesky,全程不构造 n×n 矩阵。
    优化用 L-BFGS + 解析梯度;tau2 由 σ² 的 profile 似然给出(REML 默认,与 statsmodels
    的 mixedlm.fit() 默认一致)。
    另外必须给出 profile 敏感度 ∂σ̂²/∂tau2 —— 否则 Reml objective 在 tau2≈0 处是平的
    (∂/∂tau2[log(tau2) + σ̂²/tau2] 在 tau2=σ̂² 处才为 0),优化会退化成"随便取一个"。
    """
    d, y = np.asarray(d, float), np.asarray(y, float)
    su = np.asarray(sub)
    ns = int(su.max()) + 1
    db = np.array([d[su == s_].mean() for s_ in range(ns)])
    X = np.column_stack([np.ones_like(d), d - db[su], db[su]])
    z = np.ones_like(d)
    n, p = X.shape

    def blocks(t2):
        out = []
        for s_ in range(ns):
            m = su == s_
            nk = int(m.sum())
            Xs, ys, zs = X[m], y[m], z[m]
            XtX = Xs.T @ Xs
            Xty = Xs.T @ ys
            Xtz = Xs.T @ zs
            yty, ytz, ztz = ys @ ys, ys @ zs, zs @ zs
            c = nk * t2 + 1.0
            # (I + t2*11')^{-1} = I - t2/c * 11'
            XtSiX = XtX - t2 / c * np.outer(Xtz, Xtz)
            XtSiy = Xty - t2 / c * Xtz * ytz
            XtSiz = Xtz - t2 / c * Xtz * ztz
            ytSiy = yty - t2 / c * ytz ** 2
            ytSiz = ytz - t2 / c * ytz * ztz
            ztSiz = ztz - t2 / c * ztz ** 2
            logdet = np.log1p(nk * t2)
            out.append((nk, XtSiX, XtSiy, XtSiz, ytSiy, ytSiz, ztSiz, logdet))
        return out

    def solve_all(t2):
        bl = blocks(t2)
        A = sum(b_[1] for b_ in bl)
        rhs = sum(b_[2] for b_ in bl)
        beta = np.linalg.solve(A, rhs)
        uSu = sum(b_[4] for b_ in bl)                   # u'Σ^{-1}u
        uSz = sum(b_[5] for b_ in bl)                   # u'Σ^{-1}z
        zSz = sum(b_[6] for b_ in bl)                   # z'Σ^{-1}z
        XSz = sum(b_[3] for b_ in bl)                   # X'Σ^{-1}z
        eSe = uSu - 2 * beta @ rhs + beta @ A @ beta    # e'Σ^{-1}e
        eSz = uSz - beta @ XSz                          # e'Σ^{-1}z
        return beta, eSe, eSz, zSz, sum(b_[7] for b_ in bl), A

    def obj(th):
        t2 = float(np.exp(th[0]))
        beta, eSe, eSz, zSz, logdetV, _ = solve_all(t2)
        num = eSe + t2 * zSz
        if num <= 0:
            return 1e12
        s2 = num / n
        return 0.5 * (n * np.log(s2) + logdetV + np.log(eSe / num) * p)

    best, bval = None, np.inf
    for t0 in (-12.0, -8.0, -4.0, -1.0):
        rr = minimize(obj, [t0], method='L-BFGS-B',
                      bounds=[(-20.0, 6.0)],
                      options=dict(maxiter=300, ftol=1e-14, gtol=1e-12))
        if rr.fun < bval:
            best, bval = rr, float(rr.fun)
    t2 = float(np.exp(best.x[0]))
    beta, eSe, eSz, zSz, logdetV, A = solve_all(t2)
    s2 = (eSe + t2 * zSz) / n
    se = np.sqrt(np.diag(s2 * np.linalg.inv(A)))
    df = n - p - 1
    return {'beta': beta.tolist(), 'se': se.tolist(), 't': (beta / se).tolist(),
            'tau2_subject': t2, 'sigma2_resid': float(s2), 'df': int(df),
            'n': int(n), 'n_subjects': ns, 'converged': bool(best.success),
            'loglik': -bval}


# ---------------------------------------------------------------- X-1
def x1_linear(datas, arms):
    res = {}
    for aname, (band, alphas) in arms.items():
        print('\n' + '=' * 72)
        print(f'X-1 · 切空间 + 收缩 LDA(被试内 5 折)——臂「{aname}」 带={band}')
        print(f'      档位 a = {[round(a, 3) for a in alphas]}')
        print('=' * 72)
        per, erds = {}, {}
        for s, (X, y) in enumerate(datas):
            Zb = narrow(X, band)
            Zbar = Zb[:, :, TASK[0]:TASK[1]].mean(axis=0, keepdims=True)
            R = band_pow(Zb, BASE_WIN)
            A1 = band_pow(Zb, TASK)
            M = cov_task(Zb[:, :, TASK[0]:TASK[1]])
            N = cov_task(Zb[:, :, TASK[0]:TASK[1]] - Zbar)
            folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(M, y))
            feats = [centered(tangent(M + (a - 1.0) * N)) for a in alphas]
            k1 = int(np.argmin([abs(a - 1.0) for a in alphas]))
            fixed, adapt, ed = [], [], []
            for k, a in enumerate(alphas):
                fx, f1 = feats[k], feats[k1]
                fixed.append(float(np.mean(
                    [LDA(solver='lsqr', shrinkage='auto').fit(f1[tr], y[tr]).score(fx[te], y[te])
                     for tr, te in folds])))
                adapt.append(float(np.mean(
                    [LDA(solver='lsqr', shrinkage='auto').fit(fx[tr], y[tr]).score(fx[te], y[te])
                     for tr, te in folds])))
                Aa = a * A1 + (1 - a) * band_pow(np.repeat(Zbar, len(y), axis=0), TASK)
                ed.append(float(np.mean(10 * np.log10(Aa / R))))
            per[SUBJ[s]] = {'fixed': fixed, 'adapted': adapt}
            erds[SUBJ[s]] = ed
            print(f'  {SUBJ[s]}  实测ΔERD(dB) ' + ' '.join(f'{v:+5.2f}' for v in ed))
            print(f'        固定读出器      ' + ' '.join(f'{v:.3f}' for v in fixed)
                  + f'    (a=1 时 {fixed[k1]:.3f})')
            print(f'        逐档重训        ' + ' '.join(f'{v:.3f}' for v in adapt))
        ks = sorted(per.keys())
        pf = np.array([per[k]['fixed'] for k in ks])
        pa = np.array([per[k]['adapted'] for k in ks])
        pe = np.array([erds[k] for k in ks])
        rho = float(spearmanr(pe.mean(0), pf.mean(0)).statistic)
        print(f'\n  pooled 固定读出器 ' + ' '.join(f'{v:.4f}' for v in pf.mean(0)))
        print(f'  pooled 逐档重训   ' + ' '.join(f'{v:.4f}' for v in pa.mean(0)))
        print(f'  pooled 实测 ΔERD  ' + ' '.join(f'{v:+.2f}' for v in pe.mean(0)))
        print(f'  ρ(ΔERD, 固定读出器精度) = {rho:+.3f}   '
              f'首档−a=1档 固定 = {pf.mean(0)[0]-pf.mean(0)[k1]:+.4f}pp  '
              f'重训 = {pa.mean(0)[0]-pa.mean(0)[k1]:+.4f}pp')
        res[aname] = {'band': band, 'alphas': alphas, 'per_subject': per,
                      'erd_dB': erds, 'pooled_fixed': pf.mean(0).tolist(),
                      'pooled_adapted': pa.mean(0).tolist(),
                      'pooled_erd_dB': pe.mean(0).tolist(),
                      'rho_erd_vs_fixed': rho}
    return res


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('因果性分析步骤 1-3 + X-1(纯 CPU、零训练)')
    out = {'config': {'pkl': str(PKL), 'task_win': TASK, 'base_win': BASE_WIN,
                      'mu_band': MU, 'broad_band': BROAD, 'fs': FS,
                      'alphas_plan7': ALPHAS_PLAN, 'alphas_main': ALPHAS,
                      'ctrl_alphas': CTRL_ALPHAS}}
    out['step1'] = step1_selfcheck()
    datas = load_raw()
    out['step2_x0'] = step2_x0(datas)
    out['step3_admission'] = step3_admission(datas)
    out['step3c_oracle'] = step3c_oracle(datas)
    out['x1_linear'] = x1_linear(datas, {'mu_plan7': (MU, ALPHAS_PLAN),
                                         'mu': (MU, ALPHAS),
                                         'broadband_control': (BROAD, CTRL_ALPHAS)})
    (OUT / 'causal_erd_injection.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\n总耗时 {time.time()-t0:.1f}s   json -> {OUT}/causal_erd_injection.json')


if __name__ == '__main__':
    main()
