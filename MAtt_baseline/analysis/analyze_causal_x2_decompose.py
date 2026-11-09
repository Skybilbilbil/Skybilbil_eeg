# -*- coding: utf-8 -*-
"""X-2 收尾:β_B(被试间斜率)与 β_W(被试内斜率)的并排分解 + 逐人汇总效应。

回答「ρ=−0.950 到底是什么」:把同一个预测子(逐试次 μ ERD)拆成
  dbar_s(被试均值)与 (d−dbar_s)(被试内偏差)两个正交分量,分别对「该试次对否」回归。
只有在两者都进模型时,β_W 才是被试内(免于被试层混杂)的那个量。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
import analyze_causal_erd_injection as M                                  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
import torch                                                              # noqa: E402
from analyze_causal_x2 import per_trial_erd, predict_ok, RUNS, ARGS, WIN, STEP, SUBJ  # noqa: E402
from utils.functions import load_matt_model                               # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'


def main():
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = [np.asarray(d[0], dtype=np.float64) for d in sd]
    y_all = [np.asarray(d[1]) for d in sd]
    erd = [per_trial_erd(X) for X in X_all]
    nets = [load_matt_model(str(BASE / 'results' / d / f'{t}_cross_mix_best_acc.pt'),
                            WIN, STEP, ARGS) for (d, t) in RUNS.values()]
    ok = [predict_ok(nets, X_all[s], y_all[s])[0] for s in range(9)]

    dbar = np.array([e.mean() for e in erd])
    accs = np.array([o.mean() for o in ok])
    print('被试  μ ERD 均值(dB)   该人精度(%)')
    for s in range(9):
        print(f'  {SUBJ[s]}   {dbar[s]:+8.3f}        {accs[s]*100:6.2f}')
    rho, p = spearmanr(dbar, accs)
    pr = float(np.corrcoef(dbar, accs)[0, 1])
    print(f'\n跨被试(n=9):Spearman ρ(ERD, 精度) = {rho:+.3f} (p={p:.4f});'
          f'  Pearson r = {pr:+.3f}')
    print(f'  逐人 ERD 均值极差 = {dbar.max()-dbar.min():.2f} dB;'
          f' 逐人精度极差 = {(accs.max()-accs.min())*100:.1f} pp')

    # 线性概率:同时放 dbar_s 与中心化 d,得到分解后的两个斜率
    d = np.concatenate(erd)
    o = np.concatenate([ok[s].astype(float) for s in range(9)])
    su = np.concatenate([np.full(len(erd[s]), s) for s in range(9)])
    db = dbar[su]
    X = np.column_stack([np.ones_like(d), db, d - db])
    beta, *_ = np.linalg.lstsq(X, o, rcond=None)
    r = o - X @ beta
    s2 = (r @ r) / (len(r) - X.shape[1])
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    print(f'\n线性概率回归(逐试次,n={len(d)}):')
    print(f'  β_B(被试间,每 dB 的判对概率) = {beta[1]*100:+.4f} pp/dB '
          f'(SE {se[1]*100:.4f}, t={beta[1]/se[1]:+.2f})')
    print(f'  β_W(被试内)                  = {beta[2]*100:+.4f} pp/dB '
          f'(SE {se[2]*100:.4f}, t={beta[2]/se[2]:+.2f})')
    out = {'per_subject': [{'subject': SUBJ[s], 'erd_mean_dB': float(dbar[s]),
                            'acc': float(accs[s])} for s in range(9)],
           'between_subject_spearman': float(rho), 'between_subject_p': float(p),
           'between_subject_pearson': pr,
           'beta_B_pp_per_dB': float(beta[1] * 100), 'beta_B_se': float(se[1] * 100),
           'beta_W_pp_per_dB': float(beta[2] * 100), 'beta_W_se': float(se[2] * 100),
           'n_trials': int(len(d))}
    (OUT / 'causal_erd_x2_decompose.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/causal_erd_x2_decompose.json')


if __name__ == '__main__':
    main()
