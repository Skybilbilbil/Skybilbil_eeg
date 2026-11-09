# -*- coding: utf-8 -*-
"""被试内逐量扫描:session0→session1 的**精度变化**跟着哪个量一起变?

## 为什么这是「找真正因果」的正确形式
在被试内比较两个 session,**任何不随 session 变的东西自动被消掉**(电极、容积导体、
颅皮、阻抗、这个人的节律基线)。剩下的差异只可能来自**随 session 变的东西**。
所以:哪个量的 session 间变化能预测精度的 session 间变化,哪个量就是候选因果通道。

## 口径
n = 9 个被试(每个贡献一个配对差)。所以一次只测一个量 ⇒ 9 个自由度。
本脚本对每个量做同样的比较,用 p 值排序,并明确标注**未校正多重比较**。
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'results' / 'session_within_causal'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
CANDS = ['erd_hand_dB', 'iaf_hz', 'alpha_pow', 'peak_ratio', 'ci',
         'hf_over_mu', 'eff_rank', 'oracle_D2']


def main():
    rec = json.load(open(OUT / 'session_panel.json'))
    R = {(r['subject'], r['session']): r for r in rec}
    print('=' * 78)
    print('① 先把「精度到底有没有真的变」标定清楚(各 session 的测试试次数不等)')
    print('=' * 78)
    print(f"{'被试':>5s}{'n0':>5s}{'n1':>5s}{'deep0':>8s}{'deep1':>8s}{'Δdeep':>8s}"
          f"{'二项SE':>8s}{'CSP0':>7s}{'CSP1':>7s}{'ΔCSP':>8s}")
    d_deep, d_csp = [], []
    for s in SUBJ:
        a, b = R[(s, 0)], R[(s, 1)]
        n0, n1 = a['n_test'], b['n_test']
        dd = (b['deep_acc'] - a['deep_acc']) * 100
        dc = (b['csp_acc'] - a['csp_acc']) * 100
        se = 100 * np.sqrt(a['deep_acc'] * (1 - a['deep_acc']) / max(n0, 1) +
                           b['deep_acc'] * (1 - b['deep_acc']) / max(n1, 1))
        d_deep.append(dd); d_csp.append(dc)
        print(f'{s:>5s}{n0:>5d}{n1:>5d}{a["deep_acc"]*100:>7.1f}%'
              f'{b["deep_acc"]*100:>7.1f}%{dd:>+8.1f}{se:>8.1f}'
              f'{a["csp_acc"]*100:>6.1f}%{b["csp_acc"]*100:>6.1f}%{dc:>+8.1f}')
    d_deep, d_csp = np.array(d_deep), np.array(d_csp)
    print(f'\n  Δdeep:均值 {d_deep.mean():+.1f}pp  范围 {d_deep.min():+.1f}~'
          f'{d_deep.max():+.1f}pp  |Δ|>10pp 的被试数 = {int((np.abs(d_deep)>10).sum())}/9')
    print(f'  ΔCSP :均值 {d_csp.mean():+.1f}pp  范围 {d_csp.min():+.1f}~'
          f'{d_csp.max():+.1f}pp  |Δ|>10pp 的被试数 = {int((np.abs(d_csp)>10).sum())}/9')
    print(f'  ⇒ Δdeep 与 ΔCSP 的相关:r = {pearsonr(d_deep, d_csp)[0]:+.3f} '
          f'(p={pearsonr(d_deep, d_csp)[1]:.3f}) '
          f'—— 两个读出器在「谁变了」这件事上一致吗')

    print('\n' + '=' * 78)
    print('② ★ 逐量扫描:量的 session 间变化 能否预测 精度的 session 间变化')
    print('   (dbc = Δ 量 与 Δ 精度的 Pearson r;n = 9,未校正多重比较)')
    print('=' * 78)
    print(f"{'量':>14s}{'r(Δ,Δdeep)':>13s}{'p':>8s}{'ρ(Δ,Δdeep)':>13s}"
          f"{'r(Δ,ΔCSP)':>12s}{'p':>8s}")
    rows = []
    for c in CANDS:
        dv = np.array([R[(s, 1)][c] - R[(s, 0)][c] for s in SUBJ], float)
        if dv.std() == 0 or not np.isfinite(dv).all():
            print(f'{c:>14s}    (该量在 session 间无变异或含 NaN,跳过)')
            continue
        r1, p1 = pearsonr(dv, d_deep)
        sp2 = spearmanr(dv, d_deep)
        r2 = float(sp2.correlation)
        r3, p3 = pearsonr(dv, d_csp)
        rows.append((c, r1, p1, r3, p3))
        print(f'{c:>14s}{r1:>+13.3f}{p1:>8.3f}{r2:>+13.3f}'
              f'{r3:>+12.3f}{p3:>8.3f}')
    print('\n  按 |r(Δ量, Δdeep)| 排序:')
    for c, r1, p1, r3, p3 in sorted(rows, key=lambda x: -abs(x[1])):
        star = ' ★' if p1 < 0.05 else ''
        print(f'    {c:>14s}  r={r1:+.3f} (p={p1:.3f})  r_CSP={r3:+.3f} (p={p3:.3f}){star}')

    print('\n' + '=' * 78)
    print('③ ★ 关键对照:μ ERD 在这一层上到底行不行')
    print('=' * 78)
    dv = np.array([R[(s, 1)]['erd_hand_dB'] - R[(s, 0)]['erd_hand_dB'] for s in SUBJ])
    b = float((dv @ d_deep) / (dv @ dv))
    rr = d_deep - b * dv
    seb = float(np.sqrt((rr @ rr) / (len(rr) - 2) / (dv @ dv)))
    print(f'  Δ ERD(手类)的 sd = {dv.std():.3f} dB')
    print(f'  β = Δ精度/ΔERD = {b:+.3f} pp/dB (SE {seb:.3f})')
    print(f'  同口径的 CSP 臂:β = '
          f'{float((dv @ d_csp) / (dv @ dv)):+.3f} pp/dB')
    (OUT / 'session_scan.json').write_text(json.dumps(
        {'delta_deep': d_deep.tolist(), 'delta_csp': d_csp.tolist(),
         'scan': [{'var': c, 'r_deep': r1, 'p_deep': p1, 'r_csp': r3, 'p_csp': p3}
                  for c, r1, p1, r3, p3 in rows],
         'beta_erd_deep': b, 'beta_erd_csp': float((dv @ d_csp) / (dv @ dv))},
        ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/session_scan.json')


if __name__ == '__main__':
    main()
