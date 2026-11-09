# -*- coding: utf-8 -*-
"""口径核查:同一批数据、同一管线,4 类 vs 2 类(左/右)差多少?

## 为什么
用户问:文献「完全跨被试 60–70%」为什么能超过我们「不完全跨被试」的 62.9?
首要怀疑 = **类数**。BCI-2a 的 4 类(左/右/脚/舌)随机水平 25%;文献大量「跨被试 70%+」
其实是 **2 类(左 vs 右),随机 50%**。同一个数字在两种任务里不是一回事。

## 臂(全部零训练、rs=42 同折、LE 253 + 闭式 LW-LDA)
  A 4 类(全部)          —— 锚点,须复现 62.9
  B 2 类(左 vs 右)      —— 与文献 2 类口径对齐
  C 2 类(硬类对:脚 vs 舌)  —— 同难度的另一个类对(仅参考)
 每臂同时报 随机水平 / 超出随机多少

产物 results/two_class_check/
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'two_class_check'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
TASK0 = 375
SUBJ = ['S%02d' % i for i in range(1, 10)]
PAIRS = {'4class': (0, 1, 2, 3), 'LR': (0, 1), 'FT': (2, 3)}


def pack(ok, sub):
    per = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
    return {'pooled': float(ok.mean()), 'per_subject': per}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X250 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    print('[data] 全量 %s' % (X250.shape,), flush=True)

    rec = {'note': 'LE253 + 闭式 LW-LDA;rs=42 同折;零训练', 'arms': {}}
    for name, classes in PAIRS.items():
        keep = np.isin(y_all, classes)
        X, y, sub = X250[keep], y_all[keep], sub_all[keep]
        y = np.searchsorted(classes, y)
        F = raw_cov_vec(X, TASK0)
        idx = np.arange(len(y), dtype=np.int64)
        ret = split_train_valid_test(F, y, 0.70, 0.15, random_state=42,
                                     others=[sub, idx])
        Ftr, Fte = ret[0], ret[4]
        ytr, yte, sub_te = ret[1], ret[5], ret[8]
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
        ok = clf.predict(Fte) == yte
        r = pack(ok, sub_te)
        chance = 1.0 / len(classes)
        r.update({'n_class': len(classes), 'chance': chance,
                  'n_train': int(len(ytr)), 'n_test': int(len(yte)),
                  'excess_over_chance': r['pooled'] - chance})
        rec['arms'][name] = r
        print('[%s] %d 类 | n_test %d | pooled %.1f%% | 随机 %.0f%% | 超出随机 %+.1fpp'
              % (name, len(classes), len(yte), 100 * r['pooled'], 100 * chance,
                 100 * (r['pooled'] - chance)), flush=True)
        print('        逐人 %s' % ' '.join('%s%.1f' % (s, 100 * r['per_subject'][s])
                                          for s in SUBJ), flush=True)

    (OUT / 'two_class_check.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== 同一管线、同一份数据:类数的影响 ===')
    for k, v in rec['arms'].items():
        print('%-8s %d类 pooled %5.1f%% (随机 %2.0f%%, 超出 %+5.1fpp)'
              % (k, v['n_class'], 100 * v['pooled'], 100 * v['chance'],
                 100 * v['excess_over_chance']))
    print('[OK] %s/two_class_check.json' % OUT)


if __name__ == '__main__':
    main()
