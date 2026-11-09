# -*- coding: utf-8 -*-
"""RFF(随机傅里叶特征)头:把 73.5 的 RBF 核读出变成可部署的「固定非线性层+闭式线性头」。

SVM(73.5)证明信号在切空间特征的非线性结构里,但 SVM 头不是闭式层。RFF 近似
exp(−γ‖x−y‖²) 核:z_i(x)=√(2/D)·cos(w_iᵀx+b_i),w~N(0,2γI),b~U(0,2π);
RFF + LW-LDA = 确定性前向 + 闭式头,可直接写进模型 linear 头之前。
问:① D要多大才贴 SVM(1024/4096/16384);② RFF ⊕ 原始 LE(核+线性)是否更高;
③ 读数稳定性(3 个 RFF 种子)。
口径:主协议 rs=42 同折;γ=1/median²(与 SVM 选定等价);D 与种子在 val 选,报 test。
只写 results/rff_head/。零训练(固定层+闭式头)。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import cov_tn, fit_lda, eval_split, grp  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'rff_head'
TASK = ED.TASK
SUBS = ['S%02d' % i for i in range(1, 10)]


def spdvec(L):
    iu = np.triu_indices(22)
    v = L[iu].copy()
    v[iu[0] != iu[1]] *= np.sqrt(2)
    return v


def le_feats(C):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-12)
    return np.stack([spdvec((V[i] * np.log(w[i])) @ V[i].T) for i in range(len(C))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr = ret[0], ret[1]
    X_va, y_va, sub_va = ret[2], ret[3], ret[7]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    F = [le_feats(cov_tn(X[:, :, TASK[0]:TASK[1]]))
         for X in (X_tr, X_va, X_te)]

    sub_ = np.random.default_rng(1).choice(len(y_tr), 1000, replace=False)
    d2 = ((F[0][sub_][:, None, :] - F[0][sub_][None, :]) ** 2).sum(-1)
    med2 = float(np.median(d2[np.triu_indices(len(sub_), 1)]))
    gamma = 1.0 / med2  # SVM 选定 γ=2/(2med²)
    print('γ=%.3e  (median²=%.3e)' % (gamma, med2))

    rec = {'gamma': gamma, 'arms': {}}
    best = None
    for D in (1024, 4096, 16384):
        for tag, with_le in (('rff', False), ('rff_plus_le', True)):
            accs = []
            for rff_seed in (0, 1, 2):
                rng = np.random.default_rng(rff_seed)
                W = rng.normal(0, np.sqrt(2 * gamma), size=(D, F[0].shape[1]))
                b = rng.uniform(0, 2 * np.pi, size=D)
                Z = [np.sqrt(2 / D) * np.cos(F[i] @ W.T + b) for i in range(3)]
                if with_le:
                    Z = [np.concatenate([F[i], Z[i]], 1) for i in range(3)]
                clf = fit_lda(Z[0], y_tr)
                ovv, _, _ = eval_split(clf, Z[1], y_va, sub_va)
                ov, per, ok = eval_split(clf, Z[2], y_te, sub_te)
                accs.append((ovv, ov, per, ok))
            ovv = float(np.mean([a[0] for a in accs]))
            ov = float(np.mean([a[1] for a in accs]))
            sd = float(np.std([a[1] for a in accs], ddof=1))
            lo = float(np.mean([grp(a[2])[0] for a in accs]))
            hi = float(np.mean([grp(a[2])[1] for a in accs]))
            key = '%s_D%d' % (tag, D)
            rec['arms'][key] = {'val': ovv, 'test': ov, 'test_sd': sd,
                                'low4': lo, 'high5': hi,
                                'per_subject_seed0': accs[0][2]}
            print('%-16s val %.1f  test %.1f±%.2f (低%.1f 高%.1f)'
                  % (key, ovv * 100, ov * 100, sd * 100, lo * 100, hi * 100))
            if best is None or ovv > best[0]:
                best = (ovv, key, accs)
    print('val 最优臂 = %s' % best[1])
    rec['chosen'] = best[1]

    (OUT / 'rff_head.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/rff_head.json' % OUT)


if __name__ == '__main__':
    main()
