# -*- coding: utf-8 -*-
"""主协议下的 CSP+LDA 基线(BCI 领域最标准的经典对照)。

## 为什么
要"体现优势",唯一站得住的方式是**同一口径下打败标准基线**。项目此前只有被试内 CSP
(45–60%),从未在主协议(共享混合 70/15/15、一套参数读九人)下测 CSP+LDA。
本脚本补这一格:若我们的 LE+LDA(62.9)显著高于 CSP+LDA,则「表征选择」的优势可量化。

## 口径
主协议 rs=42 同折(训练 3628 / test 778),模型不接收 subject_id,不做 LOSO。
CSP 仅在**训练折**拟合(one-vs-rest,每类取两端各 per_class 个滤波器),
特征 = log(var(滤波后)),LDA 收缩拟合,test 只读一次。

产物 results/csp_baseline/
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'csp_baseline'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
TASK = (375, 1126)
NCLS = 4
SUBJ = ['S%02d' % i for i in range(1, 10)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


def trial_covs(X):
    X = X - X.mean(-1, keepdims=True)
    C = np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)
    return C * (C.shape[1] / tr[:, None, None])


def fit_csp(C, y, per_class=2, eps=1e-6):
    covs = [C[y == c].mean(0) for c in range(NCLS)]
    d = covs[0].shape[0]
    reg = eps * np.trace(sum(covs) / NCLS) / d * np.eye(d)
    W = []
    for c in range(NCLS):
        Crest = sum(covs[i] for i in range(NCLS) if i != c) / (NCLS - 1)
        _, V = eigh(covs[c], Crest + reg)
        W.append(V[:, -per_class:])
        W.append(V[:, :per_class])
    return np.linalg.qr(np.concatenate(W, axis=1))[0]


def csp_feats(C, W):
    Z = np.einsum('ncd,dk->nck', C, W)       # C @ W
    v = np.einsum('nck,ck->nc', Z, W)        # w_c^T C w_c
    return np.log(np.maximum(v, 1e-12))


def pack(ok, sub_te):
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    acc = float(ok.mean())
    return {'acc': acc, 'kappa': (acc - 0.25) / 0.75, 'per_subject': per,
            'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([per[SUBJ[i]] for i in HIGH]))}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X250 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X250, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, ite = ret[9], ret[11]
    ytr, yte, sub_te = ret[1], ret[5], ret[8]
    print('[主协议] train %d / test %d' % (len(itr), len(ite)), flush=True)

    Ctr = trial_covs(X250[itr][:, :, TASK[0]:TASK[1]])
    Cte = trial_covs(X250[ite][:, :, TASK[0]:TASK[1]])
    rec = {'protocol': 'subject-dependent 70/15/15 rs=42, shared, CSP fit on train only',
           'arms': {}}
    for pc in (1, 2, 3, 5):
        W = fit_csp(Ctr, ytr, per_class=pc)
        Ftr, Fte = csp_feats(Ctr, W), csp_feats(Cte, W)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
        ok = clf.predict(Fte) == yte
        r = pack(ok, sub_te)
        r['dim'] = int(W.shape[1])
        rec['arms']['CSP_pc%d' % pc] = r
        print('[CSP per_class=%d] %2d维 | acc %.1f%% | kappa %.3f | 低四 %.1f 高五 %.1f'
              % (pc, r['dim'], 100 * r['acc'], r['kappa'], 100 * r['low4'],
                 100 * r['high5']), flush=True)

    # 同折的 LE+LDA(我们的表征)
    F = raw_cov_vec(X250, TASK[0])
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F[itr], ytr)
    ok = clf.predict(F[ite]) == yte
    r = pack(ok, sub_te)
    r['dim'] = 253
    rec['arms']['LE253_LDA(ours)'] = r
    print('[LE253+LDA(我们的)] %d维 | acc %.1f%% | kappa %.3f | 低四 %.1f 高五 %.1f'
          % (r['dim'], 100 * r['acc'], r['kappa'], 100 * r['low4'], 100 * r['high5']),
          flush=True)

    best_csp = max((k for k in rec['arms'] if k.startswith('CSP')),
                   key=lambda k: rec['arms'][k]['acc'])
    gap = rec['arms']['LE253_LDA(ours)']['acc'] - rec['arms'][best_csp]['acc']
    rec['summary'] = {'best_csp_arm': best_csp,
                      'best_csp_acc': rec['arms'][best_csp]['acc'],
                      'ours_acc': rec['arms']['LE253_LDA(ours)']['acc'],
                      'gap_pp': 100 * gap}
    (OUT / 'csp_baseline.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== 主协议:标准基线 vs 我们的表征 ===')
    for k, v in rec['arms'].items():
        print('%-18s %2d维 acc %5.1f%% | kappa %.3f | 低四 %5.1f 高五 %5.1f'
              % (k, v['dim'], 100 * v['acc'], v['kappa'], 100 * v['low4'],
                 100 * v['high5']))
    print('最佳 CSP 基线 %.1f%% ; 我们 %.1f%% ; 差 %+.1fpp'
          % (100 * rec['summary']['best_csp_acc'], 100 * rec['summary']['ours_acc'],
             rec['summary']['gap_pp']))
    print('[OK] %s/csp_baseline.json' % OUT)


if __name__ == '__main__':
    main()
