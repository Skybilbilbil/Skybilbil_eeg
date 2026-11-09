# -*- coding: utf-8 -*-
"""因果性分析 步骤 1-3 + X-1 的图(读 json,不重算)。

产出两套图:
  fig_dose_response.png  剂量-响应主图(4 面板)
  fig_diagnostics.png    准入与自检诊断(3 面板)
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'results' / 'causal_erd_injection'
SUBJ = [f'S{i+1:02d}' for i in range(9)]


def main():
    j = json.load(open(OUT / 'causal_erd_injection.json'))
    x1 = j['x1_linear']
    fig, ax = plt.subplots(2, 2, figsize=(12.5, 8.4))

    # (1) 主臂 pooled 剂量-响应
    for nm, lbl, col in (('mu', 'mu band (main)', '#c33'),
                         ('broadband_control', 'broadband 8-30Hz (control)', '#888')):
        a = np.array(x1[nm]['alphas'])
        e = np.array(x1[nm]['pooled_erd_dB'])
        f = np.array(x1[nm]['pooled_fixed']) * 100
        k = np.argsort(e)
        ax[0, 0].plot(e[k], f[k], 'o-', color=col, label=lbl + ' fixed reader')
    a = np.array(x1['mu_plan7']['alphas'])
    e = np.array(x1['mu_plan7']['pooled_erd_dB'])
    f = np.array(x1['mu_plan7']['pooled_fixed']) * 100
    k = np.argsort(e)
    ax[0, 0].plot(e[k], f[k], 's--', color='#36c',
                  label='mu band, plan 7 doses (+/-2 dB)')
    ax[0, 0].axhline(np.array(x1['mu']['pooled_fixed'])[5] * 100, color='k', lw=0.8, ls=':')
    ax[0, 0].set_xlabel('measured ERD change (dB, log scale inset)')
    ax[0, 0].set_ylabel('pooled accuracy (%)')
    ax[0, 0].set_title('X-1  dose-response (linear reader, within-subject 5-fold)')
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=.3)

    # (2) 逐人轨迹
    for s in SUBJ:
        f = np.array(x1['mu']['per_subject'][s]['fixed']) * 100
        ax[0, 1].plot(np.arange(len(f)), f, 'o-', lw=1, ms=3)
    ax[0, 1].set_xticks(np.arange(len(x1['mu']['alphas'])))
    ax[0, 1].set_xticklabels([f'{v:.2f}' for v in x1['mu']['alphas']])
    ax[0, 1].set_xlabel('$a$  (a=1 is unmodified; a=0 removes trial-level mu fluctuation)')
    ax[0, 1].set_ylabel('accuracy (%)')
    ax[0, 1].set_title('per-subject dose-response (mu band)')
    ax[0, 1].grid(alpha=.3)

    # (3) oracle 类间分离度 + 读出器(同一横轴 a)
    o = np.array(j['step3c_oracle']['mean_D_over_D1']) * 100
    ax[1, 0].plot(np.arange(len(o)), o, 'D-', color='#c33', label='oracle class separation D(a)/D(1)')
    pf = np.array(x1['mu']['pooled_fixed']) * 100
    ax[1, 0].plot(np.arange(len(pf)), pf, 'o-', color='#36c', label='pooled accuracy')
    ax[1, 0].axhline(100, color='k', lw=0.8, ls=':')
    ax[1, 0].set_xticks(np.arange(len(x1['mu']['alphas'])))
    ax[1, 0].set_xticklabels([f'{v:.2f}' for v in x1['mu']['alphas']])
    ax[1, 0].set_xlabel('$a$')
    ax[1, 0].set_title('oracle separability vs reader accuracy (same axis)')
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=.3)

    # (4) 实测 ERD 刻度 vs 名义
    ax[1, 1].plot(np.arange(len(x1['mu']['alphas'])), x1['mu']['pooled_erd_dB'], 'o-',
                  color='#c33', label='mu band')
    ax[1, 1].plot(np.arange(len(x1['broadband_control']['alphas'])),
                  x1['broadband_control']['pooled_erd_dB'], 's-', color='#888',
                  label='broadband')
    pl = x1['mu_plan7']
    ax[1, 1].plot(np.arange(len(pl['alphas'])), pl['pooled_erd_dB'], '^--', color='#36c',
                  label='mu, plan 7 doses')
    ax[1, 1].set_xlabel('dose index (see log for a values)')
    ax[1, 1].set_ylabel('measured ERD change (dB)')
    ax[1, 1].set_title('realised ERD scale (pooled mean)')
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT / 'fig_dose_response.png', dpi=130)
    plt.close()

    # ---- 诊断图
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ad = j['step3_admission']['per_subject']
    A = np.array([d['dose'] for d in ad], dtype=object)
    alphas = [d['alpha'] for d in ad[0]['dose']]
    M = np.array([[x['delta_over_sigma'] for x in d['dose']] for d in ad])
    for i in range(M.shape[0]):
        ax[0].semilogy(np.arange(len(alphas)), M[i] + 1e-9, 'o-', lw=1, ms=3)
    ax[0].axhline(1.0, color='r', ls='--', label='trial scatter $\\sigma_{trial}$')
    ax[0].set_xticks(np.arange(len(alphas)))
    ax[0].set_xticklabels([('0' if a == 0 else f'{a:g}') for a in alphas], fontsize=8)
    ax[0].set_xlabel('$a$')
    ax[0].set_ylabel('$\\delta_s(a)/\\sigma_{trial}$')
    ax[0].set_title('admission: is the intervention visible?')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=.3)

    x0 = j['step2_x0']['per_subject']
    sd = np.array([r['erd_sd_dB'] for r in x0])
    se = np.array([r['se_single_dB'] for r in x0])
    ax[1].bar(np.arange(9) - 0.2, sd, 0.4, label='observed sd of per-trial ERD')
    ax[1].bar(np.arange(9) + 0.2, se, 0.4, label='single-trial sampling SE')
    ax[1].set_xticks(range(9))
    ax[1].set_xticklabels(SUBJ, fontsize=8)
    ax[1].set_ylabel('dB')
    ax[1].set_title('X-0  per-trial ERD noise budget')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=.3, axis='y')

    sc = j['step1']
    txt = (f"(i)  M(a)-M at a=1 max err = {sc['identity_a1_max_abs_err']:.1e}\n"
           f"(ii) M(a)-(M-N) at a=0 max err = {sc['identity_a0_max_abs_err']:.1e}\n"
           f"(iii) centred tangent shift under global scaling s: max = "
           f"{sc['scale_invariance_dist_max']:.1e}\n"
           f"passed = {sc['passed']}\n\n"
           f"X-0: half-vs-full Spearman = {j['step2_x0']['spearman_mean']:.4f} "
           f"(estimators algebraically redundant)\n"
           f"     noise/sd ratio = {j['step2_x0']['noise_frac_mean']*100:.0f}%\n\n"
           f"oracle D(a=0)/D(1) = {j['step3c_oracle']['mean_D_over_D1'][0]:.3f}")
    ax[2].axis('off')
    ax[2].text(0.0, 0.95, txt, va='top', family='monospace', fontsize=9)
    ax[2].set_title('step-1 self-check & summary')
    plt.tight_layout()
    plt.savefig(OUT / 'fig_diagnostics.png', dpi=130)
    plt.close()
    print('figs ->', OUT / 'fig_dose_response.png', OUT / 'fig_diagnostics.png')


if __name__ == '__main__':
    main()
