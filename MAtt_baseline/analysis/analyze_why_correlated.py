# -*- coding: utf-8 -*-
"""「为什么会这么相关」的判决性检验:控制「数据本身好不好」之后 ERD 还剩多少?

## 逻辑
精度必然由「类间可分性 / 类内噪声」这个比值决定(任何分类器都如此)。所以**任何**按
这个比值给被试排序的量都必然与精度相关 —— μ ERD 若是这种量的代理,它就会表现出
ρ≈−0.95,而完全不必是原因。

## 做法(零训练)
① 造「与被试类别标签无关」的数据质量量与「与标签有关」的可分性量:
   · 静息段 μ 带能量占比 / 峰显著度 / 峰锐度 CI(逐试次周期图,Hann,1 s 段)
   · (20–45 Hz)/(8–30 Hz) 比值 —— 肌电与宽带噪声代理
   · **oracle 可分性 D²** = 类间R²/类内R²(切空间,模型无关、确定性)
② 偏相关:控制住这些量之后,ρ(μ ERD, 精度) 还剩多少。

## 口径
被试数量只有 9 ⇒ 偏相关只作**方向性**判读,不作显著性结论(n=9 上一次自由度消耗
就掉到 7)。所以下面同时给「控制前 / 控制后」两个 ρ,让幅度比较说话。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_spd_separability import tangent                          # noqa: E402
from analyze_causal_deep_reader import ROI_SYM, _CH                    # noqa: E402
from utils.dataset import load_preprocessed_dataset                    # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'why_correlated'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'
ERDJ = BASE / 'results' / 'erd_decomposition' / 'erd_decomposition.json'
FS = 250
TASK = (375, 1126)
BASE_WIN = (0, 250)
NPER = 250                                    # 1 s 段 ⇒ 1 Hz 分辨率


def periodogram(x):
    """x:(N, C, T) → (N*C, nfreq) 的 Hann 周期图(nperseg=NPER, 50% 重叠平均)。"""
    n, c, t = x.shape
    flat = x.reshape(n * c, t)
    step = NPER // 2
    starts = np.arange(0, t - NPER + 1, step) if t > NPER else np.array([0])
    seg = np.stack([flat[:, s:s + NPER] for s in starts], 1)      # (NC, nseg, NPER)
    w = np.hanning(NPER)[None, None, :]
    f = np.fft.rfftfreq(NPER, 1 / FS)
    P = np.abs(np.fft.rfft(seg * w, axis=-1)) ** 2
    P = P.mean(1) / (w ** 2).sum()                                 # (NC, nfreq)
    return P, f


def band_mean(P, f, lo, hi):
    return float(P[:, (f >= lo) & (f <= hi)].mean())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    tbl = json.load(open(TBL))
    erdj = json.load(open(ERDJ))
    SUBJ = [f'S{i+1:02d}' for i in range(9)]
    acc = np.array([r['deep70_acc'] for r in tbl])
    csp = np.array([r['csp_within'] for r in tbl])
    iaf = np.array([r['iaf_hz'] for r in tbl])
    erd = np.array([erdj['per_subject'][s]['brain_hand'] for s in SUBJ], dtype=float)

    rows, D2 = [], []
    for s, (X, y) in enumerate(sd):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        ci = np.array([_CH[n] for n in ROI_SYM])
        Xr = X[:, ci, :]
        Xr = Xr - Xr.mean(axis=-1, keepdims=True)
        # --- oracle 可分性(全 22 通道、任务窗、切空间) ---
        seg = X[:, :, TASK[0]:TASK[1]]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        C = np.einsum('nct,ndt->ncd', seg, seg) / seg.shape[-1]
        F = tangent(C)
        F = F - F.mean(0, keepdims=True)
        cen = np.stack([F[y == c].mean(0) for c in range(4)])
        r2 = float(((cen - F.mean(0)) ** 2).sum(1).mean())
        sig2 = float((F ** 2).sum(1).mean())
        # --- 静息/基线段谱质量(ROI + 全脑)---
        Prest, f = periodogram(Xr[:, :, BASE_WIN[0]:BASE_WIN[1]])
        Pw, _ = periodogram(X[:, :, BASE_WIN[0]:BASE_WIN[1]])
        i0 = iaf[s]
        mu = band_mean(Prest, f, i0 - 1.5, i0 + 1.5)
        wide = band_mean(Prest, f, 8, 30)
        lo = band_mean(Prest, f, i0 - 4.5, i0 - 1.5)
        hi = band_mean(Prest, f, i0 + 1.5, i0 + 4.5)
        side = 0.5 * (lo + hi)
        hf = band_mean(Pw, f, 20, 45)
        wmu = band_mean(Pw, f, i0 - 1.5, i0 + 1.5)
        # --- 任务窗 μ 谱质量(用来看"任务期节律是否同样干净") ---
        Ptask, _ = periodogram(Xr[:, :, TASK[0]:TASK[1]])
        mu_t = band_mean(Ptask, f, i0 - 1.5, i0 + 1.5)
        rows.append({'subject': SUBJ[s], 'iaf': float(i0),
                     'mu_share': mu / max(wide, 1e-30),
                     'peak_ratio': mu / max(side, 1e-30),
                     'ci': (mu - side) / max(mu, 1e-30),
                     'hf_over_mu': hf / max(wmu, 1e-30),
                     'mu_rest': mu, 'mu_task': mu_t,
                     'R2_between': r2, 'oracle_D2': r2 / sig2,
                     'acc': float(acc[s]), 'erd': float(erd[s]),
                     'csp_within': float(csp[s])})
        D2.append(r2 / sig2)
        print(f'  {SUBJ[s]}  D²={r2/sig2:.4f}  μ占比={mu/max(wide,1e-30):.3f}  '
              f'峰比={mu/max(side,1e-30):.2f}  CI={(mu-side)/max(mu,1e-30):+.3f}  '
              f'高频/μ={hf/max(wmu,1e-30):+.4f}  精度={acc[s]*100:.1f}%  '
              f'ERD={erd[s]:+.2f}dB', flush=True)
    D2 = np.array(D2)

    def sp(x, y):
        r = spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)

    def partial(x, y, z):
        xx = x - np.polyval(np.polyfit(z, x, 1), z)
        yy = y - np.polyval(np.polyfit(z, y, 1), z)
        return sp(xx, yy)

    cands = {'oracle_D2': D2, 'csp_within': csp,
             'mu_share': np.array([r['mu_share'] for r in rows]),
             'peak_ratio': np.array([r['peak_ratio'] for r in rows]),
             'ci': np.array([r['ci'] for r in rows]),
             'hf_over_mu': np.array([r['hf_over_mu'] for r in rows]),
             'mu_task': np.array([r['mu_task'] for r in rows])}
    print('\n' + '=' * 74)
    print('① 各量与精度的相关(n=9,Spearman)')
    print('=' * 74)
    for nm, v in cands.items():
        r1, p1 = sp(v, acc)
        r2_, p2 = sp(v, erd)
        print(f'  {nm:14s} ρ(acc)={r1:+.3f}(p={p1:.3f})   ρ(ERD)={r2_:+.3f}(p={p2:.3f})')

    print('\n' + '=' * 74)
    print('② 偏相关:控制「数据好不好」之后,μ ERD 与精度的相关还剩多少')
    print('=' * 74)
    rf, pf = sp(erd, acc)
    print(f'  未控制                                  ρ = {rf:+.3f} (p={pf:.4f})')
    for nm, v in cands.items():
        if np.std(v) == 0:
            continue
        r_p, p_p = partial(erd, acc, v)
        tag = '★' if abs(r_p) < abs(rf) * 0.5 else ' '
        print(f'  {tag} 控制 {nm:14s} 后                ρ = {r_p:+.3f} (p={p_p:.4f})'
              f'   衰减 {100*(1-abs(r_p)/abs(rf)):.0f}%')
    (OUT / 'why_correlated.json').write_text(json.dumps(
        {'per_subject': rows, 'oracle_D2': D2.tolist(), 'acc': acc.tolist(),
         'erd': erd.tolist(), 'csp_within': csp.tolist(),
         'partial': {nm: list(partial(erd, acc, v)) for nm, v in cands.items()}},
        ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/why_correlated.json')


if __name__ == '__main__':
    main()
