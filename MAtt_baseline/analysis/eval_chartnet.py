# -*- coding: utf-8 -*-
"""可学习多切图流两臂(chartfix/chartlrn × 3 seed)的闭式 LDA 头改装评估。

写法照 retrofit_lda_head.py(_features 取法 + LDA)与 analyze_multichart_ceiling.py
(eval_split/grp/低四高五)。每个 run:载入 best_acc ckpt、冻结,_features 在主协议
train/test 折(rs=42 同一切分)→ LinearDiscriminantAnalysis(solver=lsqr, shrinkage=auto)
拟合训练折 → 报 test 总体+逐人+低四{S02,S04,S05,S06}/高五;另报模型自身 CE 头
(训练时的 linear)的 test 总体+逐人。
汇总:逐 seed 配对差(chartlrn − chartfix,LDA 头 test)、两臂各自与 rand8 家族
(64.8±0.86,= multichart_ceiling.json stability.summary_K8)的差。
产物 results/chartnet_eval/chartnet_eval.json;stdout 重定向到同目录 chartnet_eval.log。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'chartnet_eval'
RUNS = {('chartfix', 42): ('chartfix_s42', 'chartfix42'),
        ('chartfix', 2024): ('chartfix_s2024', 'chartfix2024'),
        ('chartfix', 7): ('chartfix_s7', 'chartfix7'),
        ('chartlrn', 42): ('chartlrn_s42', 'chartlrn42'),
        ('chartlrn', 2024): ('chartlrn_s2024', 'chartlrn2024'),
        ('chartlrn', 7): ('chartlrn_s7', 'chartlrn7')}
# raw_cov_stream=False(与 n_charts 互斥,构造处 assert);其余与 rcs 系 ARGS 一致
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            n_charts=8)
RAND8 = {'test_mean': 0.6480719794344474, 'test_sd': 0.008641509672993125,
         'source': 'results/multichart_ceiling/multichart_ceiling.json stability.summary_K8'}
SUBJ = ['S%02d' % (i + 1) for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]   # 低四 {S02,S04,S05,S06} / 高五


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print('train %d test %d (主协议 rs=42)' % (len(y_tr), len(y_te)))

    rec = {'rand8_family': RAND8, 'per_run': {}}
    acc = {}          # (arm, seed) -> {'lda':…, 'ce':…}
    for (arm, seed), (dirname, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), 375, 188, ARGS)
        net.eval()                     # 冻结:只前向,不训练
        dev = next(net.parameters()).device

        def feats(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    out.append(net._features(
                        torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy())
            return np.concatenate(out, 0)
        Ftr, Fte = feats(X_tr), feats(X_te)
        assert Ftr.shape[1] == 180 + 8 * 253, f'特征维 {Ftr.shape[1]} != 2204'
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Ftr, y_tr)
        ok_lda = clf.predict(Fte) == y_te

        # 模型自身 CE 头(与 LDA 头共用同一冻结特征管道的 forward)
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 128):
                preds.append(net(torch.Tensor(X_te[i:i + 128]).unsqueeze(1).to(
                    dev)).argmax(1).cpu().numpy())
        ok_ce = np.concatenate(preds) == y_te

        def split_rep(ok):
            per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
            return {'test': float(ok.mean()), 'per_subject': per,
                    'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
                    'high5': float(np.mean([per[SUBJ[i]] for i in HIGH]))}
        rep = {'lda_head': split_rep(ok_lda), 'ce_head': split_rep(ok_ce)}
        rec['per_run'][dirname] = {'seed': seed, 'arm': arm, **rep}
        acc[(arm, seed)] = rep
        print('%-15s LDA头 %.1f%% (低%.1f 高%.1f) | CE头 %.1f%% (低%.1f 高%.1f)'
              % (dirname, 100 * rep['lda_head']['test'],
                 100 * rep['lda_head']['low4'], 100 * rep['lda_head']['high5'],
                 100 * rep['ce_head']['test'],
                 100 * rep['ce_head']['low4'], 100 * rep['ce_head']['high5']))

    # ---- 汇总:逐 seed 配对差 + 两臂 vs rand8 ----
    summary = {'paired_diff_chartlrn_minus_chartfix_lda': {},
               'arm_means_lda': {}, 'arm_vs_rand8_lda': {}}
    for seed in (42, 2024, 7):
        d = acc[('chartlrn', seed)]['lda_head']['test'] - \
            acc[('chartfix', seed)]['lda_head']['test']
        summary['paired_diff_chartlrn_minus_chartfix_lda'][str(seed)] = d
        print('seed%-5d 配对差(chartlrn−chartfix) LDA头 %+.2fpp' % (seed, 100 * d))
    for arm in ('chartfix', 'chartlrn'):
        ms = [acc[(arm, s)]['lda_head']['test'] for s in (42, 2024, 7)]
        mean = float(np.mean(ms))
        summary['arm_means_lda'][arm] = {
            'mean': mean, 'sd': float(np.std(ms, ddof=1)),
            'low4_mean': float(np.mean([acc[(arm, s)]['lda_head']['low4']
                                        for s in (42, 2024, 7)])),
            'high5_mean': float(np.mean([acc[(arm, s)]['lda_head']['high5']
                                         for s in (42, 2024, 7)]))}
        summary['arm_vs_rand8_lda'][arm] = mean - RAND8['test_mean']
        print('%s LDA头 3-seed %.2f±%.2f (vs rand8 家族 %+.2fpp)'
              % (arm, 100 * mean, 100 * np.std(ms, ddof=1),
                 100 * (mean - RAND8['test_mean'])))
    rec['summary'] = summary
    (OUT / 'chartnet_eval.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/chartnet_eval.json' % OUT)


if __name__ == '__main__':
    main()
