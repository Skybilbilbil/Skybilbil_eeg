# -*- coding: utf-8 -*-
"""算法水平评价矩阵:同一读出(LE 253 + 闭式 LW-LDA,零训练)跑遍所有口径。

## 为什么这样做
绝对精度跨口径不可比:4 类随机 25%、2 类随机 50%;被试内、逐人、共享混合、跨 session、
LOSO 的难度完全不同。故每个格子同时报
  acc / chance / **excess over chance** / **Cohen's kappa** / 逐人分布,
kappa = (acc − 1/K) / (1 − 1/K),对平衡类与 excess 单调等价,是文献里可跨界比较的尺度。

## 口径(每个都跑 4 类 与 2 类 左/右)
  S1 被试内 5 折(每人自己,文献最常见)
  S2 逐人 70/15/15(每人一个模型,诊断上界)
  S3 共享混合(主协议:九人合并 70/15/15,一套参数)
  S4 跨 session(session0 训练 → session1 测试,同人不同时段)
  S5 LOSO(留一被试:8 人训练 → 1 人测试,文献「跨被试」口径)

## 说明
全部零训练(闭式解),无 seed 方差;同一特征构造(250Hz 任务窗协方差 log 切向量 253 维)。
产物 results/benchmark_matrix/
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'benchmark_matrix'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
TASK0 = 375
SUBJ = ['S%02d' % i for i in range(1, 10)]
TASKS = {'4class': (0, 1, 2, 3), '2class_LR': (0, 1)}


def lda():
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


def metrics(ok, chance):
    acc = float(np.mean(ok))
    return {'acc': acc, 'chance': chance, 'excess': acc - chance,
            'kappa': (acc - chance) / (1 - chance)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    print('[data] %d 人' % len(subject_data), flush=True)
    # 每人一份 LE 特征(只算一次)
    FEAT, Y, SUB, SES = [], [], [], []
    for i, (X, y) in enumerate(subject_data):
        X = np.asarray(X, dtype=np.float32)
        F = raw_cov_vec(X, TASK0)
        n = len(y)
        # session 划分:pkl 组织为每类连续 144 试次,前 72 = session0,后 72 = session1
        s = np.zeros(n, dtype=np.int64)
        for c in np.unique(y):
            idx = np.where(y == c)[0]
            s[idx[len(idx) // 2:]] = 1
        FEAT.append(F)
        Y.append(y.astype(np.int64))
        SUB.append(np.full(n, i, dtype=np.int64))
        SES.append(s)
    FEAT = np.concatenate(FEAT)
    Y = np.concatenate(Y)
    SUB = np.concatenate(SUB)
    SES = np.concatenate(SES)
    print('[data] 全量 %d 试次 | session1 占比 %.2f'
          % (len(Y), float(SES.mean())), flush=True)

    rec = {'note': 'LE253 + 闭式 LW-LDA;零训练;kappa=(acc-1/K)/(1-1/K)', 'grid': {}}
    for tname, classes in TASKS.items():
        keep = np.isin(Y, classes)
        F, y, sub, ses = FEAT[keep], Y[keep], SUB[keep], SES[keep]
        y = np.searchsorted(classes, y)
        K = len(classes)
        chance = 1.0 / K
        rec['grid'][tname] = {'n_class': K, 'chance': chance, 'protocols': {}}
        print('\n=== %s (%d 类, chance %.0f%%) ===' % (tname, K, 100 * chance), flush=True)

        # S1 被试内 5 折
        accs = []
        for i in range(9):
            m = sub == i
            for tr, te in StratifiedKFold(5, shuffle=True,
                                          random_state=42).split(F[m], y[m]):
                accs.append(lda().fit(F[m][tr], y[m][tr]).score(F[m][te], y[m][te]))
        r = {'acc': float(np.mean(accs)), 'std_across_folds': float(np.std(accs)),
             'n_eval': len(accs)}
        r.update({k: v for k, v in metrics(np.array([r['acc']]), chance).items()
                  if k != 'acc'})
        rec['grid'][tname]['protocols']['S1_within_5fold'] = r
        print('  S1 被试内 5 折      %.1f%% | kappa %.3f' % (100 * r['acc'], r['kappa']),
              flush=True)

        # S2 逐人 70/15/15
        accs, pers = [], []
        for i in range(9):
            m = sub == i
            ret = split_train_valid_test(F[m], y[m], 0.70, 0.15, random_state=42)
            pr = lda().fit(ret[0], ret[1]).predict(ret[4])
            accs.append(float((pr == ret[5]).mean()))
        r = {'acc': float(np.mean(accs)), 'per_subject': dict(zip(SUBJ, accs))}
        r.update({k: v for k, v in metrics(np.array([r['acc']]), chance).items()
                  if k != 'acc'})
        rec['grid'][tname]['protocols']['S2_per_subject'] = r
        print('  S2 逐人 70/15/15    %.1f%% | kappa %.3f' % (100 * r['acc'], r['kappa']),
              flush=True)

        # S3 共享混合(主协议)
        idx = np.arange(len(y), dtype=np.int64)
        ret = split_train_valid_test(F, y, 0.70, 0.15, random_state=42,
                                     others=[sub, idx])
        pr = lda().fit(ret[0], ret[1]).predict(ret[4])
        ok = pr == ret[5]
        sub_te = ret[8]
        per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        r = metrics(ok, chance)
        r.update({'n_test': int(len(ok)), 'per_subject': per})
        rec['grid'][tname]['protocols']['S3_shared_mixed'] = r
        print('  S3 共享混合         %.1f%% | kappa %.3f | 逐人 %s'
              % (100 * r['acc'], r['kappa'],
                 ' '.join('%s%.0f' % (s, 100 * v) for s, v in per.items())), flush=True)

        # S4 跨 session
        tr = ses == 0
        te = ses == 1
        pr = lda().fit(F[tr], y[tr]).predict(F[te])
        ok = pr == y[te]
        r = metrics(ok, chance)
        r['n_test'] = int(len(ok))
        r['per_subject'] = {SUBJ[i]: float(ok[sub[te] == i].mean()) for i in range(9)}
        rec['grid'][tname]['protocols']['S4_cross_session'] = r
        print('  S4 跨 session       %.1f%% | kappa %.3f' % (100 * r['acc'], r['kappa']),
              flush=True)

        # S5 LOSO
        ok_all = np.zeros(len(y), dtype=bool)
        accs = []
        for i in range(9):
            tr, te = sub != i, sub == i
            pr = lda().fit(F[tr], y[tr]).predict(F[te])
            ok_all[te] = pr == y[te]
            accs.append(float(ok_all[te].mean()))
        r = metrics(ok_all, chance)
        r['mean_per_subject'] = float(np.mean(accs))
        r['per_subject'] = dict(zip(SUBJ, accs))
        rec['grid'][tname]['protocols']['S5_LOSO'] = r
        print('  S5 LOSO             %.1f%% (逐人均值 %.1f%%) | kappa %.3f'
              % (100 * r['acc'], 100 * r['mean_per_subject'], r['kappa']), flush=True)

    (OUT / 'benchmark_matrix.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== 汇总:acc / kappa ===')
    for tname, t in rec['grid'].items():
        print('--- %s (chance %.0f%%) ---' % (tname, 100 * t['chance']))
        for pname, p in t['protocols'].items():
            print('  %-20s acc %5.1f%% | excess %+5.1fpp | kappa %.3f'
                  % (pname, 100 * p['acc'], 100 * p['excess'], p['kappa']))
    print('[OK] %s/benchmark_matrix.json' % OUT)


if __name__ == '__main__':
    main()
