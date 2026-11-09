# -*- coding: utf-8 -*-
"""EA / 非 EA 的模型层配对评估（零训练，只前向）。

读 results/ea_model_ablation/{ea,raw}_{42,2024,7}/ 的 best_acc / best_loss ckpt，
用与训练入口逐位一致的 samepool 70/15/15 切分评估 test 精度，并报逐人配对差。
⚠ 非 EA 臂的测试集也必须做 P^{-1} 逆变换（否则与训练输入不一致）。
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
OUT = BASE / 'results' / 'ea_model_ablation'
WIN, STEP = 375, 188
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
SEEDS = [42, 2024, 7]
ARMS = ['ea', 'raw']
KINDS = ['best_acc', 'best_loss']


def test_split(X_all):
    y_all = np.concatenate([np.asarray(y) for _, y in SD], 0)
    sub_all = np.concatenate([np.full(len(y), i, np.int64) for i, (_, y) in enumerate(SD)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    return ret[4], ret[5], ret[8]


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
    t, p = ttest_1samp(d, 0.0)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, n, (10000, n))].mean(1)
    return {'mean': float(d.mean()), 'sd': float(d.std(ddof=1)),
            'ci95': [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            't': float(t), 'p': float(p),
            'mde80': float((2.306 + 0.8416) * d.std(ddof=1) / np.sqrt(n)),
            'per_subject': [float(v) for v in d]}


def main():
    global SD
    SD, md = load_preprocessed_dataset(str(DATA_PATH))
    X_ea = np.concatenate([np.asarray(X, np.float64) for X, _ in SD], 0)
    X_raw = np.concatenate([
        np.einsum('ij,njt->nit', np.linalg.inv(np.asarray(md[i]['ea_matrix'], np.float64)),
                  np.asarray(SD[i][0], np.float64)) for i in range(len(SD))], 0)
    Xte_ea, Yte, Ste = test_split(X_ea)
    Xte_raw, Yte2, Ste2 = test_split(X_raw)
    assert np.array_equal(Yte, Yte2) and np.array_equal(Ste, Ste2), '两臂切分不一致'
    print(f'test {len(Yte)} trials, {len(set(Ste.tolist()))} subjects')

    out = {'meta': {'test_n': int(len(Yte)), 'seeds': SEEDS,
                    'split': 'samepool 70/15/15 rs=42',
                    'note': '每臂测试输入分别为 EA / P^{-1} 还原的非 EA'}, 'runs': {}, 'paired': {}}
    for kind in KINDS:
        out['runs'][kind] = {}
        for arm in ARMS:
            for seed in SEEDS:
                d = OUT / f'{arm}_{seed}'
                ck = d / f'{arm}_{seed}_cross_mix_{kind}.pt'
                if not ck.exists():
                    print(f'missing {ck}')
                    continue
                jp = d / 'pair_training.json'
                if not jp.exists():
                    print(f'skip {arm}_{seed}: pair_training.json 缺失（中断 run）')
                    continue
                net = load_matt_model(str(ck), WIN, STEP, ARGS)
                x_t = torch.Tensor(Xte_ea if arm == 'ea' else Xte_raw).unsqueeze(1)
                x_t = x_t.to(next(net.parameters()).device)
                overall, per = accs(net, x_t, Yte, Ste)
                j = json.loads((d / 'pair_training.json').read_text(encoding='utf-8'))
                log_last = float(j['models']['cross_mix']['test_acc'][-1])
                out['runs'][kind][f'{arm}_{seed}'] = {
                    'overall': overall, 'per_subject': per, 'log_test_acc_last': log_last}
                print(f'[{kind}] {arm}_{seed}: overall={overall*100:.2f}  '
                      f'log_last={log_last*100:.2f}', flush=True)
                del net
        pairs = [(s, out['runs'][kind].get(f'ea_{s}'), out['runs'][kind].get(f'raw_{s}'))
                 for s in SEEDS]
        pairs = [(s, e, r) for s, e, r in pairs if e and r]
        if pairs:
            ea = [e for _, e, _ in pairs]
            raw = [r for _, _, r in pairs]
            d_over = [e['overall'] - r['overall'] for e, r in zip(ea, raw)]
            out['paired'][f'overall_{kind}'] = paired(d_over)
            out['paired'][f'overall_{kind}']['n_seeds'] = len(pairs)
            subs = sorted(ea[0]['per_subject'])
            d_per = [float(np.mean([e['per_subject'][s] - r['per_subject'][s]
                                    for e, r in zip(ea, raw)])) for s in subs]
            out['paired'][f'per_subject_{kind}'] = paired(d_per)
            out['paired'][f'per_subject_{kind}']['n_seeds'] = len(pairs)
            print(f'[{kind}] seeds={[s for s, _, _ in pairs]} '
                  f'overall EA-raw = {np.mean(d_over)*100:+.2f} pp; '
                  f'per-subject mean = {np.mean(d_per)*100:+.2f} pp', flush=True)

    (OUT / 'ea_model_ablation.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/ea_model_ablation.json')


if __name__ == '__main__':
    main()
