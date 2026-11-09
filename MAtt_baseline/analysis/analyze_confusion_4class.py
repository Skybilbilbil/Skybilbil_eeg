# -*- coding: utf-8 -*-
"""主协议内的细化评价:4 类混淆结构、逐人逐类、置信区间。

## 口径(严格遵守)
主协议:subject-dependent,九人合并按类别分层 70/15/15(rs=42),模型不接收 subject_id。
本脚本**不做 LOSO**、不做任何留出被试实验。

## 两个读出(同一 test 折,778 试次)
  A 经典零训练:LE 253 + 闭式 LW-LDA(=62.9,锚点)
  B 深度 3 seed:SGD+CE 训练头(62.5Hz + 深度可分离),逐 seed 报后再平均

## 输出
  混淆矩阵(行=真值,列=预测)、每类 precision/recall/F1、macro-F1、
  逐人 × 逐类 recall 矩阵、逐人 acc 的 Wilson 95% CI。
产物 results/confusion_4class/
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

OUT = BASE / 'results' / 'confusion_4class'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
TASK0 = 375
CLS = ['left', 'right', 'feet', 'tongue']
SUBJ = ['S%02d' % i for i in range(1, 10)]
RUNS = {42: ('sep62_full_s42', 'sep62f42'), 2024: ('sep62_full_s2024', 'sep62f2024'),
        7: ('sep62_full_s7', 'sep62f7')}
WIN, STEP, DOWN = 94, 47, 4
ARGS = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=True)


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    den = 1 + z * z / n
    c = p + z * z / (2 * n)
    d = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (float((c - d) / den), float((c + d) / den))


def full_report(pred, yte, sub_te, tag):
    K = 4
    cm = np.zeros((K, K), dtype=int)
    for t, p in zip(yte, pred):
        cm[t, p] += 1
    rec, prec, f1 = [], [], []
    for c in range(K):
        tp = cm[c, c]
        rec.append(float(tp / cm[c].sum()) if cm[c].sum() else 0.0)
        col = cm[:, c].sum()
        prec.append(float(tp / col) if col else 0.0)
        f1.append(2 * prec[-1] * rec[-1] / (prec[-1] + rec[-1]) if (prec[-1] + rec[-1]) else 0.0)
    macro_f1 = float(np.mean(f1))
    per_sub, per_sub_cls = {}, {}
    for i in range(9):
        m = sub_te == i
        k, n = int((pred[m] == yte[m]).sum()), int(m.sum())
        per_sub[SUBJ[i]] = {'acc': k / n, 'n': n, 'ci95': wilson(k, n)}
        per_sub_cls[SUBJ[i]] = [float((pred[m & (yte == c)] == c).mean())
                                for c in range(K)]
    acc = float((pred == yte).mean())
    print('[%s] acc %.1f%% | macro-F1 %.3f | 逐类 recall %s'
          % (tag, 100 * acc, macro_f1,
             ' '.join('%s%.1f' % (CLS[c], 100 * rec[c]) for c in range(K))), flush=True)
    print('        混淆矩阵(行真/列预测):', flush=True)
    for c in range(K):
        print('          %-7s %s' % (CLS[c], ' '.join('%4d' % v for v in cm[c])), flush=True)
    return {'acc': acc, 'confusion': cm.tolist(), 'recall': rec, 'precision': prec,
            'f1': f1, 'macro_f1': macro_f1, 'per_subject': per_sub,
            'per_subject_per_class': per_sub_cls}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X250 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    from scipy.signal import resample_poly
    X62 = np.concatenate([resample_poly(np.asarray(s[0], dtype=np.float32), 1, DOWN,
                                        axis=-1) for s in subject_data], 0)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X250, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, ite = ret[9], ret[11]
    ytr, yte, sub_te = ret[1], ret[5], ret[8]
    print('[主协议] train %d / test %d | 逐人 test 约 %d 试次'
          % (len(itr), len(ite), int(len(ite) / 9)), flush=True)

    rec = {'protocol': 'subject-dependent 70/15/15 rs=42,shared,no subject_id',
           'n_test': int(len(ite)), 'readouts': {}}

    F = raw_cov_vec(X250, TASK0)
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F[itr], ytr)
    rec['readouts']['classic_LE_LDA'] = full_report(clf.predict(F[ite]), yte, sub_te,
                                                   '经典 LE+LDA')

    for seed, (d, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / d / ('%s_cross_mix_best_acc.pt' % tag)),
                              WIN, STEP, ARGS)
        net.eval()
        dev = next(net.parameters()).device
        pr = []
        with torch.no_grad():
            for i in range(0, len(X62[ite]), 128):
                pr.append(net(torch.Tensor(X62[ite][i:i + 128]).unsqueeze(1).to(dev)
                              ).argmax(1).cpu().numpy())
        rec['readouts']['deep_s%d' % seed] = full_report(np.concatenate(pr), yte, sub_te,
                                                         '深度 seed%d' % seed)

    (OUT / 'confusion_4class.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')

    print('')
    print('=== 逐人类别 recall(经典 LE+LDA) ===')
    c = rec['readouts']['classic_LE_LDA']
    print('%-6s %6s %7s %7s %7s %7s   %s' % ('被试', 'acc', 'left', 'right', 'feet',
                                           'tongue', '95%CI'))
    for s in SUBJ:
        v = c['per_subject'][s]
        pc = c['per_subject_per_class'][s]
        print('%-6s %5.1f%% %6.1f%% %6.1f%% %6.1f%% %6.1f%%   [%.1f, %.1f]'
              % (s, 100 * v['acc'], 100 * pc[0], 100 * pc[1], 100 * pc[2], 100 * pc[3],
                 100 * v['ci95'][0], 100 * v['ci95'][1]))
    print('')
    print('=== 两读出对比 ===')
    for k, v in rec['readouts'].items():
        print('%-16s acc %5.1f%% | macro-F1 %.3f | recall %s'
              % (k, 100 * v['acc'], v['macro_f1'],
                 ' '.join('%.1f' % (100 * r) for r in v['recall'])))
    print('[OK] %s/confusion_4class.json' % OUT)


if __name__ == '__main__':
    main()
