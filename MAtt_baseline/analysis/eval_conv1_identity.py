# -*- coding: utf-8 -*-
"""conv1 恒等冻结实验的评估(seed42,与基线同折配对)。

## 实验问题
conv1 机制分析(results/conv1_mechanism/)判明:共享 conv1 的病态尾部(cond≈1e3)
删掉镜像对称+中线子空间 = 低四人「手 vs 非手」边界的唯一载体。本实验把 conv1
冻结为单位阵重训(不学空间混合),问:
  ① 低四人是否被单独抬起(主判据,配对 Δ低四−Δ高五);
  ② 逐类精度:低四的「手↔非手」跨界混淆是否收窄;
  ③ pooled 不塌(高五不被牺牲)。
单 seed(用户指示);与 verify70 seed42 同一测试折。只写 results/conv1id_eval/。
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
OUT = BASE / 'results' / 'conv1id_eval'
BASELINE = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
RUNS = {42: ('conv1id_s42', 'c1id42'), 2024: ('conv1id_s2024', 'c1id2024'),
        7: ('conv1id_s7', 'c1id7')}
SEEDS = (42, 2024, 7)
OUTNAME = 'conv1id_eval.json'
NORM = 'in'
RCS = False
if len(sys.argv) == 3:      # 可选:别的 run 前缀,如 cwid cw → cwid_s42/cw42
    RUNS = {s: (f'{sys.argv[1]}_s{s}', f'{sys.argv[2]}{s}') for s in SEEDS}
    OUTNAME = f'{sys.argv[2]}_eval.json'
if len(sys.argv) == 4:      # 再传 bn 则按 BatchNorm 建模(BN ckpt 含 running stats)
    RUNS = {s: (f'{sys.argv[1]}_s{s}', f'{sys.argv[2]}{s}') for s in SEEDS}
    OUTNAME = f'{sys.argv[2]}_eval.json'
    NORM = sys.argv[3]
if len(sys.argv) == 5:      # 再再传 rcs 则并联流建模(特征 433 维)
    RUNS = {s: (f'{sys.argv[1]}_s{s}', f'{sys.argv[2]}{s}') for s in SEEDS}
    OUTNAME = f'{sys.argv[2]}_eval.json'
    NORM = sys.argv[3]
    RCS = sys.argv[4] == 'rcs'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
CLS = ['左手', '右手', '脚', '舌']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    y_np = np.asarray(y_te)
    base_all = json.loads(BASELINE.read_text(encoding='utf-8'))
    rec = {'per_seed': {}}
    pooled, dlo_list, dhi_list = [], [], []
    per_mean = {s: [] for s in SUBJ}
    for seed in SEEDS:
        dirname, tag = RUNS[seed]
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), 375, 188,
                              dict(dataset='bci2a', n_classes=4, d_attn_in=8,
                                   d_attn_out=8, local_radius=6, anchor_stride=None,
                                   n_heads=1, dropout=0.3, norm_type=NORM,
                                   raw_cov_stream=RCS))
        net.eval()
        with torch.no_grad():
            pred = net(torch.Tensor(X_te).unsqueeze(1).to(
                next(net.parameters()).device)).argmax(1).cpu().numpy()
        ok = pred == y_np
        per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        percls = {SUBJ[i]: {CLS[c]: float(ok[sub_te == i][y_np[sub_te == i] == c].mean())
                            for c in range(4)} for i in range(9)}
        base = base_all['seed%d' % seed]['best_acc']
        bp = {s: base['per_subject'][s][1] for s in SUBJ}
        dlo = float(np.mean([per[SUBJ[i]] - bp[SUBJ[i]] for i in LOW]))
        dhi = float(np.mean([per[SUBJ[i]] - bp[SUBJ[i]] for i in HIGH]))
        rec['per_seed'][str(seed)] = {
            'pooled': float(ok.mean()), 'per_subject': per, 'per_class': percls,
            'd_low4': dlo, 'd_high5': dhi,
            'd_per_subject': {s: per[s] - bp[s] for s in SUBJ}}
        pooled.append(float(ok.mean()))
        dlo_list.append(dlo)
        dhi_list.append(dhi)
        for s in SUBJ:
            per_mean[s].append(per[s])
        print('seed%d: pooled %.1f%%(基线 %.1f) 低四Δ%+.2f 高五Δ%+.2f'
              % (seed, ok.mean() * 100, base['overall'] * 100, dlo * 100, dhi * 100))
    per_mean = {s: float(np.mean(v)) for s, v in per_mean.items()}
    mde = float(2 * max(np.std(dlo_list, ddof=1), np.std(dhi_list, ddof=1)))
    rec['summary'] = {
        'pooled_mean': float(np.mean(pooled)), 'pooled_per_seed': pooled,
        'd_pooled_mean': float(np.mean(pooled) - np.mean(
            [base_all['seed%d' % s]['best_acc']['overall'] for s in SEEDS])),
        'd_low4_mean': float(np.mean(dlo_list)), 'd_high5_mean': float(np.mean(dhi_list)),
        'd_low4_minus_high5': float(np.mean(dlo_list) - np.mean(dhi_list)),
        'mde_2seedsd': mde, 'per_subject_mean': per_mean,
        'low4_mean': float(np.mean([per_mean[SUBJ[i]] for i in LOW])),
        'high5_mean': float(np.mean([per_mean[SUBJ[i]] for i in HIGH]))}
    (OUT / OUTNAME).write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    s_ = rec['summary']
    print('\n=== conv1 恒等 × 3 seed ===')
    print('pooled %.1f%% ±%.2f 配对 Δ=%+.2f | 低四 Δ%+.2f 高五 Δ%+.2f (Δ低−Δ高 %+.2f, MDE %.2f)'
          % (100 * s_['pooled_mean'], 100 * np.std(pooled, ddof=1),
             100 * s_['d_pooled_mean'], 100 * s_['d_low4_mean'],
             100 * s_['d_high5_mean'], 100 * s_['d_low4_minus_high5'],
             100 * s_['mde_2seedsd']))
    print('低四 %.1f 高五 %.1f | 逐人均值:' % (100 * s_['low4_mean'], 100 * s_['high5_mean']),
          ' '.join('%s%.1f' % (s, 100 * per_mean[s]) for s in SUBJ))
    print('[OK] %s/conv1id_eval.json' % OUT)


if __name__ == '__main__':
    main()
