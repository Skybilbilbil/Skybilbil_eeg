# -*- coding: utf-8 -*-
"""PhysionetMI 62ch 逐人 CSP+LDA 5 折 CV(与 22ch 的 physionet_adapt.json::csp_within 同尺)。
产物: results/physionet_62ch_s42/csp62.json。纯 CPU,零训练。"""
import sys, json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset
from analyze_individual_diff import fit_csp, cov

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica_62ch.pkl'
OUT = BASE / 'results' / 'physionet_62ch_s42' / 'csp62.json'


def main():
    sd, _ = load_preprocessed_dataset(str(PKL))
    accs = []
    for X, y in sd:
        C = cov(X[:, :, 160:721].astype(np.float64))   # 任务期 [0,3.5)s, 同 22ch 口径
        cv = []
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(C, y):
            W = fit_csp(C[tr], y[tr])
            F = lambda Cs: np.log(np.maximum(np.einsum('ia,nij,ja->na', W, Cs, W), 1e-12))
            cv.append(LDA(solver='lsqr', shrinkage='auto')
                      .fit(F(C[tr]), y[tr]).score(F(C[te]), y[te]))
        accs.append(float(np.mean(cv)))
    a = np.array(accs) * 100
    ref = json.load(open(BASE / 'results' / 'physionet_adapt' / 'physionet_adapt.json'))
    a22 = np.array(ref['csp_within']['acc_by_subject']) * 100
    from scipy.stats import spearmanr
    rho, p = spearmanr(accs, ref['csp_within']['acc_by_subject'])
    print(f'62ch CSP 5折CV: min={a.min():.1f} 中位={np.median(a):.1f} max={a.max():.1f}%'
          f' | 22ch 对照: min={a22.min():.1f} 中位={np.median(a22):.1f} max={a22.max():.1f}%')
    print(f'逐人 62ch × 22ch CSP Spearman ρ={rho:.3f} (p={p:.1e})')
    json.dump({'acc_by_subject': accs,
               'summary': {'min': float(a.min()), 'median': float(np.median(a)),
                           'max': float(a.max()), 'mean': float(a.mean())},
               'spearman_vs_22ch': {'rho': float(rho), 'p': float(p)}},
              open(OUT, 'w'), indent=2)
    print(f'[OK] {OUT}')


if __name__ == '__main__':
    main()
