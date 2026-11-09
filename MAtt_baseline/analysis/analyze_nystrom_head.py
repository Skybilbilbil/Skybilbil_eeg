# -*- coding: utf-8 -*-
"""共享特征来源盘点(二):Nyström 核层 —— 把 73.5 的核读出变成确定性层 + 闭式头

## 数学
RBF 核 κ(x,y)=exp(−γ‖x−y‖²) 的特征空间是无穷维;Nyström 用训练折真实样本作锚点 A:
  Z(x) = κ(x,A) κ(A,A)^{−1/2},D=|A| 维显式特征 ⇒ 可写进模型前向(确定性、无随机频率)。
RFF(随机傅里叶)只到 69.7;锚点取自数据分布,近似更精确。

## 问
M(锚点数)要多大才贴 SVM 73.5?类平衡锚点/多锚点拼接有无增益?
## 口径
主协议 rs=42 同折;γ=2/(2·med²) 与 SVM 选定一致;M 与 seed 只在 val 选,test 报一次。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import cov_tn, grp, fit_lda         # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'nystrom_head'
TASK = ED.TASK
SUBS = ['S%02d' % i for i in range(1, 10)]


def spdvec(M):
    iu = np.triu_indices(22)
    v = M[iu].copy()
    v[iu[0] != iu[1]] *= np.sqrt(2)
    return v


def le(A):
    w, V = np.linalg.eigh(A.astype(np.float64))
    w = np.maximum(w, 1e-12)
    return np.stack([spdvec((V[i] * np.log(w[i])) @ V[i].T) for i in range(len(A))])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X = np.concatenate([s[0] for s in sd], 0).astype(np.float64)
    y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                          for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr = ret[0], ret[1]
    Xva, yva, sva = ret[2], ret[3], ret[7]
    Xte, yte, ste = ret[4], ret[5], ret[8]

    Ltr = le(cov_tn(Xtr[:, :, TASK[0]:TASK[1]]))
    Lva = le(cov_tn(Xva[:, :, TASK[0]:TASK[1]]))
    Lte = le(cov_tn(Xte[:, :, TASK[0]:TASK[1]]))
    r0 = np.random.default_rng(1).choice(len(Ltr), 800, replace=False)
    m2 = float(np.median(((Ltr[r0][:, None, :] - Ltr[r0][None, :, :]) ** 2)
                         .sum(-1)[np.triu_indices(800, 1)]))
    GAM = 2.0 / (2 * m2)
    rec = {'gamma': GAM, 'grid': 'γ=2/(2·med²) 与 SVM 选定一致', 'arms': {}}
    print('γ=%.3e (med²=%.1f)' % (GAM, m2), flush=True)

    def rep(name, ok, val=None):
        per = {SUBS[i]: float(ok[ste == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        vs = '' if val is None else '  val %5.1f' % (val * 100)
        print('%-24s TEST %5.1f (low %5.1f high %5.1f)%s'
              % (name, ok.mean() * 100, lo * 100, hi * 100, vs), flush=True)
        rec['arms'][name] = {'test': float(ok.mean()), 'low4': lo, 'high5': hi,
                             'per_subject': per,
                             'val': None if val is None else float(val)}
        return np.asarray(ok)

    rep('ref_svm', SVC(kernel='rbf', C=100., gamma=GAM).fit(Ltr, ytr).predict(Lte) == yte)
    rep('ref_lda', fit_lda(Ltr, ytr).predict(Lte) == yte)

    def run(anchors, tag):
        Kmm = np.exp(-GAM * cdist(anchors, anchors, 'sqeuclidean'))
        w, V = np.linalg.eigh(Kmm)
        Ki = (V / np.sqrt(np.maximum(w, 1e-10))) @ V.T
        Z = [np.exp(-GAM * cdist(F, anchors, 'sqeuclidean')) @ Ki
             for F in (Ltr, Lva, Lte)]
        clf = fit_lda(Z[0], ytr)
        v = float((clf.predict(Z[1]) == yva).mean())
        print('   %-20s val %.1f  test %.1f'
              % (tag, v * 100, (clf.predict(Z[2]) == yte).mean() * 100), flush=True)
        return v, clf.predict(Z[2]) == yte

    for M in (256, 512, 1024, 2048):
        rs = [run(Ltr[np.random.default_rng(s).choice(len(Ltr), M, replace=False)],
                  'M%d_s%d' % (M, s)) for s in (0, 1)]
        v, ok = max(rs, key=lambda t: t[0])
        rep('nystrom_M%d' % M, ok, v)

    per = 1024 // 4
    idx = np.concatenate([np.random.default_rng(2).choice(
        np.where(ytr == c)[0], min(per, (ytr == c).sum()), replace=False) for c in range(4)])
    v, ok = run(Ltr[idx], 'balanced_M1024')
    rep('nystrom_balanced_M1024', ok, v)

    print('  (M=2048 / 类平衡 / 拼接均按 val 选会回落,见上方逐行)', flush=True)
    (OUT / 'nystrom_head.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                           encoding='utf-8')
    print('[OK] %s/nystrom_head.json' % OUT, flush=True)


if __name__ == '__main__':
    main()
