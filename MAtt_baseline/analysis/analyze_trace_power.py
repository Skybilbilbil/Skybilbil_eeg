# -*- coding: utf-8 -*-
"""零训练验证:把 trace 归一化删掉的功率信息拼回切空间特征,看能否提升共享读出。

动机:`mAtt.py:219` 的 signal2spd 做了 trace 归一化(C/tr(C)),静息与任务的**绝对功率**
被除掉;而 ERD 的定义恰好是两个绝对量之比 (P_task-P_rest)/P_rest。本脚本在 253 维 LE
切空间特征后拼功率特征,检同一份**共享** shrinkage LDA 是否提升——尤其低四人。

臂(全部同一 70/15/15 切分、同一 778 test、同一 LDA):
  A0  [LE253]                                     基线 62.9
  A1  [LE253, log tr(task)]                       全局功率 +1 维
  A2  [LE253, log diag(task)]                     逐通道功率 +22 维
  A3  [LE253, log diag(task) - log diag(rest)]    逐通道 ERD +22 维
  A4  [LE253, log diag(rest)]                     逐通道静息功率 +22 维

每组同时报 pooled / 低四 / 高五 / 逐人 / 身份纯度(kNN k=9)。
零训练、凸解;只写 results/trace_power/。
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'trace_power'
TASK, REST = ED.TASK, ED.REST
D = 22
IU = np.triu_indices(D)


def cov_raw(x):
    """未归一化协方差(幂为原始量纲):x (n,C,T) → (n,C,C)。"""
    xc = x - x.mean(-1, keepdims=True)
    return np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]


def le_feats(C):
    """trace 归一化后的 SPD → 253 维 LE 切向量(与 svm_readout/ceiling 同构造)。"""
    tr = np.trace(C, axis1=1, axis2=2)
    Cn = C * (D / tr[:, None, None])
    w, V = np.linalg.eigh(Cn.astype(np.float64))
    w = np.maximum(w, 1e-12)
    out = np.empty((len(C), len(IU[0])))
    for i in range(len(C)):
        L = (V[i] * np.log(w[i])) @ V[i].T
        v = L[IU].copy()
        v[IU[0] != IU[1]] *= np.sqrt(2)
        out[i] = v
    return out


def id_purity(F_tr, s_tr, F_te, s_te, k=9):
    kk = KNeighborsClassifier(n_neighbors=k).fit(F_tr, s_tr)
    return float((kk.predict(F_te) == s_te).mean())


def grp(per):
    lo = np.mean([per[SUBJ[i]] for i in LOW])
    hi = np.mean([per[SUBJ[i]] for i in HIGH])
    return float(lo), float(hi)


def run_arm(name, F_tr, F_va, F_te, y_tr, y_va, y_te, sub_te, s_tr):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F_tr, y_tr)
    ok = clf.predict(F_te) == y_te
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    rec = {'dim': int(F_tr.shape[1]), 'val': float((clf.predict(F_va) == y_va).mean()),
           'test': float(ok.mean()), 'low4': lo, 'high5': hi, 'per_subject': per,
           'id_purity': id_purity(F_tr, s_tr, F_te, sub_te)}
    print('%-6s dim=%3d  val %.1f  test %.1f  (低四 %.1f  高五 %.1f)  id纯度 %.3f'
          % (name, rec['dim'], rec['val'] * 100, rec['test'] * 100,
             lo * 100, hi * 100, rec['id_purity']))
    return rec


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, X_va, y_va = ret[0], ret[1], ret[2], ret[3]
    X_te, y_te = ret[4], ret[5]
    sub_te, s_tr = ret[8], ret[6]

    # 协方差(原始量纲)与 LE 切向量(内部再做 trace 归一)
    C_t_tr, C_t_va, C_t_te = (cov_raw(X[:, :, TASK[0]:TASK[1]]) for X in (X_tr, X_va, X_te))
    C_r_tr, C_r_va, C_r_te = (cov_raw(X[:, :, REST[0]:REST[1]]) for X in (X_tr, X_va, X_te))
    L_tr, L_va, L_te = le_feats(C_t_tr), le_feats(C_t_va), le_feats(C_t_te)

    def pw(C):                    # 逐通道功率 log(对角)
        return np.log(np.maximum(np.diagonal(C, axis1=1, axis2=2), 1e-12))

    def gt(C):                    # 全局功率 log(trace)
        return np.log(np.maximum(np.trace(C, axis1=1, axis2=2), 1e-12))[:, None]

    def cat(a, b):
        return np.concatenate([a, b], axis=1)

    arms = {
        'A0': (L_tr, L_va, L_te),
        'A1': (cat(L_tr, gt(C_t_tr)), cat(L_va, gt(C_t_va)), cat(L_te, gt(C_t_te))),
        'A2': (cat(L_tr, pw(C_t_tr)), cat(L_va, pw(C_t_va)), cat(L_te, pw(C_t_te))),
        'A3': (cat(L_tr, pw(C_t_tr) - pw(C_r_tr)), cat(L_va, pw(C_t_va) - pw(C_r_va)),
               cat(L_te, pw(C_t_te) - pw(C_r_te))),
        'A4': (cat(L_tr, pw(C_r_tr)), cat(L_va, pw(C_r_va)), cat(L_te, pw(C_r_te))),
    }

    rec = {'note': 'trace 归一化删除的绝对功率拼回 LE 切空间;共享 shrinkage LDA;零训练',
           'task': list(TASK), 'rest': list(REST), 'arms': {}}
    for k, (a, b, c) in arms.items():
        rec['arms'][k] = run_arm(k, a, b, c, y_tr, y_va, y_te, sub_te, s_tr)

    base = rec['arms']['A0']
    print('\n--- 相对 A0 的增量 ---')
    for k in ('A1', 'A2', 'A3', 'A4'):
        r = rec['arms'][k]
        print('%s  pooled %+.2f  (低四 %+.2f / 高五 %+.2f)'
              % (k, (r['test'] - base['test']) * 100,
                 (r['low4'] - base['low4']) * 100, (r['high5'] - base['high5']) * 100))

    (OUT / 'trace_power.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/trace_power.json' % OUT)


if __name__ == '__main__':
    main()
