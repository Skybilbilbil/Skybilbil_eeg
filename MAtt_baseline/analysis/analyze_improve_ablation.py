# -*- coding: utf-8 -*-
"""le_win0（任务相对静息）在主协议 3-seed 上的配对验证（零训练评估）。

基线 = 现有 EA 3-seed（verify_samepool_70_15_15 / verify70_seed2024 / verify70_seed7）；
实验臂 = results/improve_ablation/lewin0_{42,2024,7}。
同一切分（samepool 70/15/15 rs=42），同一 778 测试试次，逐人配对。
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
OUT = BASE / 'results' / 'improve_ablation'
WIN, STEP = 375, 188
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
BASE_RUNS = [('verify_samepool_70_15_15', 'verify70'),
             ('verify70_seed2024', 'v70s2024'),
             ('verify70_seed7', 'v70s7')]
LEW_RUNS = [(f'improve_ablation/lewin0_{s}', f'lewin0_{s}') for s in (42, 2024, 7)]
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
    t, p = (ttest_1samp(d, 0.0) if n > 1 else (float('nan'), float('nan')))
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, n, (10000, n))].mean(1) if n > 1 else d
    return {'mean': float(d.mean()),
            'sd': float(d.std(ddof=1)) if n > 1 else 0.0,
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            't': float(t), 'p': float(p), 'n': n,
            'per_subject': [float(v) for v in d]}


def main():
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([np.asarray(X) for X, _ in sd], 0)
    y_all = np.concatenate([np.asarray(y) for _, y in sd], 0)
    sub_all = np.concatenate([np.full(len(y), i, np.int64) for i, (_, y) in enumerate(sd)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    x_t = torch.Tensor(X_te).unsqueeze(1).cuda()
    print(f'test {len(y_te)} trials, {len(set(sub_te.tolist()))} subjects')

    out = {'meta': {'test_n': int(len(y_te)),
                    'base': [f'{d}/{t}' for d, t in BASE_RUNS],
                    'arm': [f'{d}/{t}' for d, t in LEW_RUNS]},
           'runs': {}, 'paired': {}}
    for kind in KINDS:
        out['runs'][kind] = {}
        for label, runs in (('base', BASE_RUNS), ('lewin0', LEW_RUNS)):
            for dirname, tag in runs:
                ck = BASE / 'results' / dirname / f'{tag}_cross_mix_{kind}.pt'
                if not ck.exists():
                    print('missing', ck)
                    continue
                net = load_matt_model(str(ck), WIN, STEP, ARGS)
                overall, per = accs(net, x_t, y_te, sub_te)
                out['runs'][kind][f'{label}_{tag}'] = {'overall': overall, 'per_subject': per}
                print(f'[{kind}] {label}_{tag}: {overall*100:.2f}', flush=True)
                del net
        base = [out['runs'][kind].get(f'base_{t}') for _, t in BASE_RUNS]
        arm = [out['runs'][kind].get(f'lewin0_{t}') for _, t in LEW_RUNS]
        pairs = [(b, a) for b, a in zip(base, arm) if b and a]
        if pairs:
            d_over = [a['overall'] - b['overall'] for b, a in pairs]
            out['paired'][f'overall_{kind}'] = paired(d_over)
            d_per = [float(np.mean([a['per_subject'][s] - b['per_subject'][s]
                                    for b, a in pairs])) for s in NAMES]
            out['paired'][f'per_subject_{kind}'] = paired(d_per)
            lo = float(np.mean([d_per[NAMES.index(s)] for s in LOW4]))
            hi = float(np.mean([d_per[NAMES.index(s)] for s in NAMES if s not in LOW4]))
            out['paired'][f'per_subject_{kind}']['low4_mean'] = lo
            out['paired'][f'per_subject_{kind}']['high5_mean'] = hi
            out['paired'][f'per_subject_{kind}']['low_minus_high'] = lo - hi
            print(f'[{kind}] overall {np.mean(d_over)*100:+.2f} pp; '
                  f'low4 {lo*100:+.2f}, high5 {hi*100:+.2f}, '
                  f'low-high {(lo-hi)*100:+.2f}', flush=True)

    (OUT / 'improve_ablation.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/improve_ablation.json')


if __name__ == '__main__':
    main()
