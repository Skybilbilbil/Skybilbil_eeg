# -*- coding: utf-8 -*-
"""conv2 扩维实验的离线评估:16/22 两臂 × 3 seed,与基线(8 维)按 seed 配对。

## 实验问题(论文初稿 §11 第 8 条 + 整改方案 §5 决策二行 1)
逐级追踪显示共享读出下唯一的低点是 conv2 降秩(低四 41.0→34.4,−6.6pp;高五 −3.1pp,
表 14),但 4.8pp(全体)略低于 5pp 地板,不能当结论。本实验把 conv2 输出从 8 维放宽到
16/22 维重训(d_attn_out=8 不变 ⇒ 切空间每窗 36 维与分类头不变,单一变量 = 共享降秩),
问三件事:
  ① 全体 pooled 是否上升(表示容量是否是约束);
  ② 低四人是否被【单独】抬起(分组响应判据:Δlow4 − Δhigh5 > 0 且不同步平移);
  ③ 逐人离散度(极差、组间/组内方差比)是否缩小(方差齐性,I-7 并行判据)。

## 口径
切分 = split_train_valid_test(random_state=42),与 eval70_persubject 逐位一致 ⇒
与基线同一测试集,**按 seed 配对**;单折差异 <5pp 按噪声处理(I-8:附 seed 间 sd 作 MDE)。
基线 = verify70_persubject.json 的 best_acc 复评(全体 54.5%、逐人 33.3–71.8)。

只写 results/conv2_widen_eval/。纯推理,分钟级。
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
OUT = BASE / 'results' / 'conv2_widen_eval'
BASELINE = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
RUNS = {                                                    # (arm, seed) → (目录, tag)
    (16, 42): ('conv2_out16_s42', 'c16s42'),
    (16, 2024): ('conv2_out16_s2024', 'c16s2024'),
    (16, 7): ('conv2_out16_s7', 'c16s7'),
    (22, 42): ('conv2_out22_s42', 'c22s42'),
    (22, 2024): ('conv2_out22_s2024', 'c22s2024'),
    (22, 7): ('conv2_out22_s7', 'c22s7'),
}
SEEDS = (42, 2024, 7)
LOW = ('S02', 'S04', 'S05', 'S06')
SUBS = ['S%02d' % i for i in range(1, 10)]


def group_stats(per_subject_mean):
    low = np.array([per_subject_mean[s] for s in LOW])
    high = np.array([per_subject_mean[s] for s in SUBS if s not in LOW])
    return {'low4_mean': round(float(low.mean()), 4), 'high5_mean': round(float(high.mean()), 4),
            'range': round(float(per_subject_mean[max(per_subject_mean, key=per_subject_mean.get)]
                               - per_subject_mean[min(per_subject_mean, key=per_subject_mean.get)]), 4),
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
    print('test %d trials' % len(y_te))

    base = json.loads(BASELINE.read_text(encoding='utf-8'))
    base_per_seed = {s: base['seed%d' % s]['best_acc'] for s in SEEDS}

    rec = {'purpose': 'conv2 widening 8->16/22, paired vs baseline per seed, same test split',
           'judgment': ['pooled lift', 'grouped response: dlow4-dhigh5', 'variance homogeneity (I-7)',
                        'MDE = 2*seed-sd (I-8)'],
           'arms': {}}
    for arm in (16, 22):
        per_seed, per_subj_seed = {}, {}
        for seed in SEEDS:
            dirname, tag = RUNS[(arm, seed)]
            ck = BASE / 'results' / dirname / ('%s_cross_mix_best_acc.pt' % tag)
            if not ck.exists():
                print('skip arm%d seed%d: %s 不存在' % (arm, seed, ck.name))
                continue
            net = load_matt_model(str(ck), 375, 188,
                                  dict(dataset='bci2a', n_classes=4, d_attn_in=arm,
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
        present = [sd for sd in SEEDS if sd in per_seed]
        if not present:
            print('=== 臂 conv2_out=%d: 无可用 ckpt,跳过 ===' % arm)
            continue
        ps_mean = {s: round(float(np.mean([per_subj_seed[sd][s] for sd in present])), 4)
                   for s in SUBS}
        pooled = np.array([per_seed[s] for s in present])
        rec['arms'][str(arm)] = {
            'pooled_per_seed': per_seed,
            'pooled_mean': round(float(pooled.mean()), 4),
            'pooled_seed_sd': round(float(pooled.std(ddof=1)), 4),
            'per_subject_mean': ps_mean, 'per_subject_per_seed': per_subj_seed,
            'group': group_stats(ps_mean)}
        # 与基线按 seed 配对
        b_pooled = np.array([base_per_seed[s]['overall'] for s in present])
        d_pooled = pooled - b_pooled
        dlow, dhigh = [], []
        for seed in present:
            bp = base_per_seed[seed]['per_subject']
            dlow.append(np.mean([per_subj_seed[seed][s] - bp[s][1] for s in LOW]))
            dhigh.append(np.mean([per_subj_seed[seed][s] - bp[s][1]
                                  for s in SUBS if s not in LOW]))
        rec['arms'][str(arm)]['paired_vs_baseline'] = {
            'd_pooled_per_seed': [round(float(v), 4) for v in d_pooled],
            'd_pooled_mean': round(float(d_pooled.mean()), 4),
            'd_low4_mean': round(float(np.mean(dlow)), 4),
            'd_high5_mean': round(float(np.mean(dhigh)), 4),
            'd_low4_minus_high5': round(float(np.mean(dlow) - np.mean(dhigh)), 4),
            'mde_2seedsd': round(float(2 * max(np.std(dlow, ddof=1), np.std(dhigh, ddof=1))), 4)}
        a = rec['arms'][str(arm)]
        print('\n=== 臂 conv2_out=%d ===' % arm)
        print('pooled %.1f%% ±%.2f (逐 seed %s)  配对 Δ=%+.2fpp'
              % (100 * a['pooled_mean'], 100 * a['pooled_seed_sd'],
                 '/'.join('%.1f' % (100 * v) for v in a['pooled_per_seed'].values()),
                 100 * a['paired_vs_baseline']['d_pooled_mean']))
        print('低四 %.1f%% 高五 %.1f%% 极差 %.1fpp  Δ低四 %+.2f  Δ高五 %+.2f  (Δ低−Δ高 %+.2f, MDE %.2f)'
              % (100 * a['group']['low4_mean'], 100 * a['group']['high5_mean'],
                 100 * a['group']['range'], 100 * a['paired_vs_baseline']['d_low4_mean'],
                 100 * a['paired_vs_baseline']['d_high5_mean'],
                 100 * a['paired_vs_baseline']['d_low4_minus_high5'],
                 100 * a['paired_vs_baseline']['mde_2seedsd']))
        print('逐人:', ' '.join('%s %.1f' % (s, 100 * ps_mean[s]) for s in SUBS))

    # 基线同表(便于对照)
    bps = {s: float(np.mean([base_per_seed[sd]['per_subject'][s][1] for sd in SEEDS]))
           for s in SUBS}
    rec['baseline'] = {'per_subject_mean': {s: round(bps[s], 4) for s in SUBS},
                       'group': group_stats({s: round(bps[s], 4) for s in SUBS})}
    print('\n基线逐人:', ' '.join('%s %.1f' % (s, 100 * bps[s]) for s in SUBS))

    (OUT / 'conv2_widen_eval.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/conv2_widen_eval.json' % OUT)


if __name__ == '__main__':
    main()
