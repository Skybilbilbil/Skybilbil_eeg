# -*- coding: utf-8 -*-
"""镜像翻转增广实验的离线评估:1 臂 × 3 seed,与基线(无增广)按 seed 配对。

## 实验问题
手二类的类信息几乎完全落在镜像反对称子空间(被试内 8 维 57.1/80.6 ≈ 全 22 维
57.4/82.2,`results/mirror_commonality/`),且偏侧化解码是共享解剖结构(X-3 对侧/同侧
反号、9/9 同向;Nam 2011;Hoshino 2024 通道+标签翻转增广)。本实验训练折加倍
(左右通道互换 + 左右手标签互换,脚/舌与中线不动),把该等变约束写进共享模型,
问三件事(判据与 eval_conv2_widen 相同):
  ① 全体 pooled 是否上升;② 低四人是否被【单独】抬起(Δ低四−Δ高五);
  ③ 逐人离散度是否缩小(I-7);MDE = 2×seed-sd(I-8)。

切分 = split_train_valid_test(random_state=42),与基线同一测试集;基线 =
verify70_persubject.json。只写 results/mirror_aug_eval/。纯推理,分钟级。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'mirror_aug_eval'
BASELINE = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
RUNS = {42: ('mirror_aug_s42', 'mas42'), 2024: ('mirror_aug_s2024', 'mas2024'),
        7: ('mirror_aug_s7', 'mas7')}
SEEDS = (42, 2024, 7)
LOW = ('S02', 'S04', 'S05', 'S06')
SUBS = ['S%02d' % i for i in range(1, 10)]


def group_stats(p):
    low = np.array([p[s] for s in LOW])
    high = np.array([p[s] for s in SUBS if s not in LOW])
    return {'low4_mean': round(float(low.mean()), 4), 'high5_mean': round(float(high.mean()), 4),
            'range': round(float(max(p.values()) - min(p.values())), 4),
            'between_over_within_var': round(
                float(((high.mean() - low.mean()) ** 2 * 20 / 9)
                      / (low.var(ddof=1) * 3 / 8 + high.var(ddof=1) * 4 / 9 + 1e-12)), 4)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    x_t = torch.Tensor(X_te).unsqueeze(1)
    y_np = np.asarray(y_te)
    base = json.loads(BASELINE.read_text(encoding='utf-8'))
    base_per_seed = {s: base['seed%d' % s]['best_acc'] for s in SEEDS}

    per_seed, per_subj_seed = {}, {}
    for seed in SEEDS:
        dirname, tag = RUNS[seed]
        ck = BASE / 'results' / dirname / ('%s_cross_mix_best_acc.pt' % tag)
        if not ck.exists():
            print('skip seed%d: %s 不存在' % (seed, ck.name))
            continue
        net = load_matt_model(str(ck), 375, 188,
                              dict(dataset='bci2a', n_classes=4, d_attn_in=8,
                                   d_attn_out=8, local_radius=6, anchor_stride=None,
                                   n_heads=1, dropout=0.3))
        net.eval()
        x = x_t.to(next(net.parameters()).device)
        with torch.no_grad():
            pred = net(x).argmax(dim=1).cpu().numpy()
        ok = pred == y_np
        per_subj_seed[seed] = {SUBS[i]: round(float(ok[sub_te == i].mean()), 4)
                               for i in range(9)}
        per_seed[seed] = round(float(ok.mean()), 4)
        print('seed%d: pooled %.1f%%' % (seed, 100 * per_seed[seed]))
    present = [s for s in SEEDS if s in per_seed]

    ps_mean = {s: round(float(np.mean([per_subj_seed[sd][s] for sd in present])), 4)
               for s in SUBS}
    pooled = np.array([per_seed[s] for s in present])
    b_pooled = np.array([base_per_seed[s]['overall'] for s in present])
    d_pooled = pooled - b_pooled
    dlow, dhigh = [], []
    for seed in present:
        bp = base_per_seed[seed]['per_subject']
        dlow.append(np.mean([per_subj_seed[seed][s] - bp[s][1] for s in LOW]))
        dhigh.append(np.mean([per_subj_seed[seed][s] - bp[s][1]
                              for s in SUBS if s not in LOW]))
    rec = {'per_seed': per_seed,
           'pooled_mean': round(float(pooled.mean()), 4),
           'pooled_seed_sd': round(float(pooled.std(ddof=1)), 4),
           'per_subject_mean': ps_mean, 'per_subject_per_seed': per_subj_seed,
           'group': group_stats(ps_mean),
           'paired_vs_baseline': {
               'd_pooled_per_seed': [round(float(v), 4) for v in d_pooled],
               'd_pooled_mean': round(float(d_pooled.mean()), 4),
               'd_low4_mean': round(float(np.mean(dlow)), 4),
               'd_high5_mean': round(float(np.mean(dhigh)), 4),
               'd_low4_minus_high5': round(float(np.mean(dlow) - np.mean(dhigh)), 4),
               'mde_2seedsd': round(float(2 * max(np.std(dlow, ddof=1),
                                                  np.std(dhigh, ddof=1))), 4)}}
    (OUT / 'mirror_aug_eval.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== 镜像翻转增广 ===')
    print('pooled %.1f%% ±%.2f  配对 Δ=%+.2fpp (逐 seed %s)'
          % (100 * rec['pooled_mean'], 100 * rec['pooled_seed_sd'],
             100 * rec['paired_vs_baseline']['d_pooled_mean'],
             '/'.join('%+.1f' % (100 * v) for v in d_pooled)))
    print('低四 %.1f%% 高五 %.1f%% 极差 %.1fpp  Δ低四 %+.2f  Δ高五 %+.2f  (Δ低−Δ高 %+.2f, MDE %.2f)'
          % (100 * rec['group']['low4_mean'], 100 * rec['group']['high5_mean'],
             100 * rec['group']['range'], 100 * rec['paired_vs_baseline']['d_low4_mean'],
             100 * rec['paired_vs_baseline']['d_high5_mean'],
             100 * rec['paired_vs_baseline']['d_low4_minus_high5'],
             100 * rec['paired_vs_baseline']['mde_2seedsd']))
    print('逐人:', ' '.join('%s %.1f' % (s, 100 * ps_mean[s]) for s in SUBS))
    print('[OK] %s/mirror_aug_eval.json' % OUT)


if __name__ == '__main__':
    main()
