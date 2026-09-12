# -*- coding: utf-8 -*-
"""3-seed 被试内平均（官方协议）——验证多种子平滑后是否超原文 74.71%"""
import sys
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'MAtt_baseline'))
from _reproduce_official import train_eval, load_official, DEVICE

SEEDS = [42, 2024, 7]


def main():
    print(f'Device: {DEVICE}  | 3-seed 平均（350iter 无Hann 原文打分）')
    per_subj = []
    for s in range(9):
        X_tr, y_tr, X_te, y_te = load_official(s)
        accs = []
        for sd in SEEDS:
            a = train_eval(X_tr, y_tr, X_te, y_te, f'S{s+1:02d} seed{sd}', seed=sd)
            accs.append(a)
        mean = float(np.mean(accs))
        per_subj.append(mean)
        print(f'S{s+1:02d}: seeds={[f"{x*100:.1f}" for x in accs]} 平均={mean*100:.2f}%', flush=True)
    print('\n==== 3-seed 平均汇总 ====')
    for s, m in enumerate(per_subj):
        print(f'S{s+1:02d}: {m*100:.2f}%')
    print(f'\n全体平均: {np.mean(per_subj)*100:.2f}%  （原文 74.71% ± 5.01）')


if __name__ == '__main__':
    main()