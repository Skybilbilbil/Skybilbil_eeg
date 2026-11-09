# -*- coding: utf-8 -*-
"""原型对齐(I-5)实验的离线评估:K=2/3 两臂 × 3 seed,与基线按 seed 配对。

## 实验问题(整改方案 §4.2 第二档唯一残留路径)
白化预检只盖住 K=1 按人形式;按【试次】路由时不同类试次的协方差不同、可能
路由到不同原型,理论上不被论文 §10.2 覆盖。本实验在输入端实现登记的数学形式
X <- P_k^{-1/2} X(原型 = 训练折整段协方差 LE 度量 K-means 的重心;路由 = 最近原型;
标签自由、不接收 subject_id),重训共享模型,判据 = eval_conv2_widen 同三件套。

切分 = split_train_valid_test(random_state=42),与基线同一测试集;基线 =
verify70_persubject.json。只写 results/proto_align_eval/。纯推理,分钟级。
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
OUT = BASE / 'results' / 'proto_align_eval'
BASELINE = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
RUNS = {2: {42: ('proto2_s42', 'p2s42'), 2024: ('proto2_s2024', 'p2s2024'),
            7: ('proto2_s7', 'p2s7')},
        3: {42: ('proto3_s42', 'p3s42'), 2024: ('proto3_s2024', 'p3s2024'),
            7: ('proto3_s7', 'p3s7')}}
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

    rec = {'arms': {}}
    for arm, runs in RUNS.items():
        per_seed, per_subj_seed = {}, {}
        for seed in SEEDS:
            dirname, tag = runs[seed]
            ck = BASE / 'results' / dirname / ('%s_cross_mix_best_acc.pt' % tag)
            if not ck.exists():
                print('skip K=%d seed%d: %s 不存在' % (arm, seed, ck.name))
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
        present = [s for s in SEEDS if s in per_seed]
        if not present:
            print('=== K=%d: 无可用 ckpt ===' % arm)
            continue
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
        rec['arms'][str(arm)] = {
            'per_seed': per_seed, 'pooled_mean': round(float(pooled.mean()), 4),
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
        a = rec['arms'][str(arm)]
        print('\n=== 原型对齐 K=%d ===' % arm)
        print('pooled %.1f%% ±%.2f  配对 Δ=%+.2fpp (逐 seed %s)'
              % (100 * a['pooled_mean'], 100 * a['pooled_seed_sd'],
                 100 * a['paired_vs_baseline']['d_pooled_mean'],
                 '/'.join('%+.1f' % (100 * v) for v in d_pooled)))
        print('低四 %.1f%% 高五 %.1f%% 极差 %.1fpp  Δ低四 %+.2f  Δ高五 %+.2f  (Δ低−Δ高 %+.2f, MDE %.2f)'
              % (100 * a['group']['low4_mean'], 100 * a['group']['high5_mean'],
                 100 * a['group']['range'], 100 * a['paired_vs_baseline']['d_low4_mean'],
                 100 * a['paired_vs_baseline']['d_high5_mean'],
                 100 * a['paired_vs_baseline']['d_low4_minus_high5'],
                 100 * a['paired_vs_baseline']['mde_2seedsd']))
        print('逐人:', ' '.join('%s %.1f' % (s, 100 * ps_mean[s]) for s in SUBS))

    (OUT / 'proto_align_eval.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/proto_align_eval.json' % OUT)


if __name__ == '__main__':
    main()
