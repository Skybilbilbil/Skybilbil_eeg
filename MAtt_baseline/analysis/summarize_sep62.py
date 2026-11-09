# -*- coding: utf-8 -*-
"""汇总 sep62(62.5 Hz + 深度可分离)三项主协议读数,供汇报稿 §3.1 / 图1 使用。
来源: results/sep62_full_s{42,2024,7}/pair_training.json(混合 + 被试内)
     + results/sep62_eval/sep62_eval.json(best_acc 权重复评 = 验证集选模 + 逐人)
只写 results/sep62_full_summary/。
"""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'results' / 'sep62_full_summary'
SEEDS = ['42', '2024', '7']


def main():
    runs = {}
    for s in SEEDS:
        d = json.load(open(BASE / 'results' / f'sep62_full_s{s}' / 'pair_training.json'))
        runs[s] = {k: (v.get('test_acc') or [None])[-1] * 100
                   for k, v in d['models'].items() if v.get('test_acc')}
    ev = json.load(open(BASE / 'results' / 'sep62_eval' / 'sep62_eval.json'))

    summ = {'config': {'resample_hz': 62.5, 'conv2_separable': True, 'T': 282, 'M': 5,
                       'fs_src': 250, 'win': 94, 'step': 47},
            'old_baseline': {'cross_mix_final': 55.4, 'per_seed': [56.17, 54.24, 55.91],
                             'within_S3': 75.9, 'within_S9': 76.6, 'val_selected': 54.6},
            'per_seed': runs}
    for k in ['cross_mix', 'within_S3', 'within_S9']:
        v = [runs[s][k] for s in SEEDS]
        summ[k] = {'mean': float(np.mean(v)), 'range': [float(min(v)), float(max(v))],
                   'per_seed': v}
    summ['val_selected'] = {'mean': ev['pooled_mean'] * 100,
                            'range': [ev['pooled_min'] * 100, ev['pooled_max'] * 100],
                            'note': 'best_acc 权重在同一 778 试次 test 上复评'}
    summ['per_subject'] = {k: v['mean'] * 100 for k, v in ev['per_subject'].items()}
    summ['low4_mean'] = ev['low4_mean'] * 100
    summ['high5_mean'] = ev['high5_mean'] * 100
    summ['per_subject_range'] = [min(summ['per_subject'].values()),
                                 max(summ['per_subject'].values())]

    OUT.mkdir(parents=True, exist_ok=True)
    json.dump(summ, open(OUT / 'sep62_full_summary.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print(json.dumps({k: summ[k] for k in ['cross_mix', 'within_S3', 'within_S9',
                                            'val_selected', 'low4_mean', 'high5_mean',
                                            'per_subject_range']},
                     ensure_ascii=False, indent=1))
    print(f'[落盘] {OUT}/sep62_full_summary.json')


if __name__ == '__main__':
    main()
