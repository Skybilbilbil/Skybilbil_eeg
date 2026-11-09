# -*- coding: utf-8 -*-
"""改进项批量验证的配对评估（零训练，只前向）。

读 results/improve_sweep/<arm>_<seed>/ 的 best_acc / best_loss ckpt，
与现有 EA 3-seed 基线在同一 778 测试试次上配对，输出总体/低四/高五/逐人差。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import ttest_1samp

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model  # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'improve_sweep'
WIN, STEP = 375, 188
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
SEEDS = [int(x) for x in sys.argv[1:]] or [42]   # 单 seed 筛查默认 42
SEED_MAP = {42: ('verify_samepool_70_15_15', 'verify70'),
            2024: ('verify70_seed2024', 'v70s2024'),
            7: ('verify70_seed7', 'v70s7')}
BASE_RUNS = [SEED_MAP[s] for s in SEEDS]
ARMS = ['le_diff1', 'le_diff2', 'attn_ref', 'attn_diff', 'attn_prior',
        'conv2_ms', 'nsf10', 'nsf15', 'chgauss', 'ortho']
# 每个 arm 的结构参数必须与训练时一致，否则 load_state_dict 对不上
ARM_ARGS = {
    'le_diff1': {'manifold_change': 'le_diff1'},
    'le_diff2': {'manifold_change': 'le_diff2'},
    'attn_ref': {'attn_change': 'ref'},
    'attn_diff': {'attn_change': 'diff'},
    'attn_prior': {'attn_prior': (0.0, 0.6, 0.6, 0.6, 0.6)},
    'conv2_ms': {'conv2_multiscale': True},
    'nsf10': {'n_spatial_filters': 10},
    'nsf15': {'n_spatial_filters': 15},
    'chgauss': {'channel_gaussian': True},
    'ortho': {},
}
KINDS = ['best_acc', 'best_loss']
LOW4 = ['S02', 'S04', 'S05', 'S06']
NAMES = [f'S{i:02d}' for i in range(1, 10)]


def accs(net, x_t, y_np, sub_te):
    net.eval()
    with torch.no_grad():
        pred = net(x_t).argmax(1).cpu().numpy()
    ok = pred == y_np
    per = {f'S{s+1:02d}': round(float(ok[sub_te == s].mean()), 4)
           for s in range(9) if (sub_te == s).sum() > 0}
    return round(float(ok.mean()), 4), per


def paired(d):
    d = np.asarray(d, float)
    n = len(d)
    if n == 0:
        return {}
    if n > 1:
        t, p = ttest_1samp(d, 0.0)
        rng = np.random.default_rng(0)
        boots = d[rng.integers(0, n, (10000, n))].mean(1)
        ci = [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]
    else:
        t = p = float('nan'); ci = [float(d[0]), float(d[0])]
    return {'mean': float(d.mean()), 'sd': float(d.std(ddof=1)) if n > 1 else 0.0,
            'ci95': ci, 't': float(t), 'p': float(p), 'n': n}


def eval_runs(kind, runs, arm_args=None):
    cfg = {**ARGS, **(arm_args or {})}
    res = []
    for dirname, tag in runs:
        ck = BASE / 'results' / dirname / f'{tag}_cross_mix_{kind}.pt'
        if not ck.exists():
            print('missing', ck)
            return None
        net = load_matt_model(str(ck), WIN, STEP, cfg)
        x_t = X_TE
        overall, per = accs(net, x_t, Y_TE, SUB_TE)
        res.append({'overall': overall, 'per_subject': per})
        del net
    return res


def main():
    global X_TE, Y_TE, SUB_TE
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([np.asarray(X) for X, _ in sd], 0)
    y_all = np.concatenate([np.asarray(y) for _, y in sd], 0)
    sub_all = np.concatenate([np.full(len(y), i, np.int64) for i, (_, y) in enumerate(sd)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_TE = torch.Tensor(ret[4]).unsqueeze(1).cuda()
    Y_TE, SUB_TE = ret[5], ret[8]
    print(f'test {len(Y_TE)} trials')

    base_all = {}
    for kind in KINDS:
        base_all[kind] = eval_runs(kind, BASE_RUNS)
    out = {'meta': {'test_n': int(len(Y_TE)), 'seeds': SEEDS, 'arms': ARMS},
           'base': {}, 'arms': {}}
    for kind in KINDS:
        out['base'][kind] = base_all[kind]

    for arm in ARMS:
        out['arms'][arm] = {}
        for kind in KINDS:
            runs = [(f'improve_sweep/{arm}_{s}', f'{arm}_{s}') for s in SEEDS]
            arm_res = eval_runs(kind, runs, ARM_ARGS.get(arm, {}))
            if arm_res is None:
                out['arms'][arm][kind] = {'missing': True}
                continue
            d_over = [a['overall'] - b['overall'] for a, b in zip(arm_res, base_all[kind])]
            d_per = [float(np.mean([a['per_subject'][s] - b['per_subject'][s]
                                    for a, b in zip(arm_res, base_all[kind])])) for s in NAMES]
            lo = float(np.mean([d_per[NAMES.index(s)] for s in LOW4]))
            hi = float(np.mean([d_per[NAMES.index(s)] for s in NAMES if s not in LOW4]))
            out['arms'][arm][kind] = {
                'overall_3seed': [a['overall'] for a in arm_res],
                'diff': paired(d_over),
                'per_subject_diff': d_per,
                'low4_mean': lo, 'high5_mean': hi, 'low_minus_high': lo - hi}
            print(f'[{arm}/{kind}] arm={np.mean([a["overall"] for a in arm_res])*100:.2f} '
                  f'diff={np.mean(d_over)*100:+.2f} low4={lo*100:+.2f} '
                  f'high5={hi*100:+.2f} low-high={(lo-hi)*100:+.2f}', flush=True)

    fn = OUT / f'improve_sweep_{len(SEEDS)}seed.json'
    fn.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {fn}')


if __name__ == '__main__':
    main()
