# -*- coding: utf-8 -*-
"""SVM 读出细查:超参格点稳定性 + rcs 433 维(后端180⊕单切图253)上核读出是否叠加后端。

承接 nonlinear_readout:单切图 RBF-SVM 73.5(val 74.7)大胜线性族。本脚本补两件事:
① C/γ 格点的 val 面(过拟合检查,读数稳定性);
② 同超参迁移到 rcs_s42 的 433 维特征(深度后端在核读出之上有无增益)。
口径:主协议 rs=42 同折,val 选超参,test 只报一次。零训练(SVM 凸解)。
只写 results/svm_readout/。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import cov_tn, grp                  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                         # noqa: E402

OUT = BASE / 'results' / 'svm_readout'
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
    L_tr = le_feats(cov_tn(X_tr[:, :, TASK[0]:TASK[1]]))
    L_va = le_feats(cov_tn(X_va[:, :, TASK[0]:TASK[1]]))
    L_te = le_feats(cov_tn(X_te[:, :, TASK[0]:TASK[1]]))

    sub_ = np.random.default_rng(1).choice(len(y_tr), 1000, replace=False)
    d2 = ((L_tr[sub_][:, None, :] - L_tr[sub_][None, :]) ** 2).sum(-1)
    med2 = float(np.median(d2[np.triu_indices(len(sub_), 1)]))

    # ---- ① 格点 val 面 ----
    rec = {'grid_val': {}, 'gamma_scale': 'gm/(2*median_sq_dist)'}
    best = None
    for C_ in (10.0, 100.0, 1000.0):
        for gm in (0.5, 1.0, 2.0):
            svm = SVC(kernel='rbf', C=C_, gamma=gm / (2 * med2))
            svm.fit(L_tr, y_tr)
            ovv = float((svm.predict(L_va) == y_va).mean())
            rec['grid_val']['C%.0f_g%.1f' % (C_, gm)] = ovv
            print('C=%5.0f γ=%.1f×(2med²)⁻¹  val %.1f' % (C_, gm, ovv * 100))
            if best is None or ovv > best[0]:
                best = (ovv, C_, gm, svm)
    ovv, C_, gm, svm = best
    ok = svm.predict(L_te) == y_te
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    nsv = int(svm.n_support_.sum())
    rec['chosen'] = {'C': C_, 'gm': gm, 'val': ovv, 'test': float(ok.mean()),
                     'low4': lo, 'high5': hi, 'per_subject': per, 'n_support': nsv,
                     'n_train': len(y_tr)}
    print('选定 C=%.0f γ=%.1f: test %.1f (低%.1f 高%.1f) 支持向量 %d/%d'
          % (C_, gm, ok.mean() * 100, lo * 100, hi * 100, nsv, len(y_tr)))
    print('逐人:', ' '.join('%s %.0f' % (s, per[s] * 100) for s in SUBS))

    # ---- ② rcs 433 维(后端180 ⊕ LE253),同超参迁移 ----
    ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
                local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
                raw_cov_stream=True)
    net = load_matt_model(str(BASE / 'results' / 'rcs_s42' /
                              'rcs42_cross_mix_best_acc.pt'), 375, 188, ARGS)
    net.eval()
    dev = next(net.parameters()).device

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                out.append(net._features(
                    torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy())
        return np.concatenate(out, 0)

    F_tr, F_va, F_te = feats(X_tr), feats(X_va), feats(X_te)
    svm2 = SVC(kernel='rbf', C=C_, gamma=gm / (2 * med2))
    svm2.fit(F_tr, y_tr)
    ok2 = svm2.predict(F_te) == y_te
    per2 = {SUBJ[i]: float(ok2[sub_te == i].mean()) for i in range(9)}
    lo2, hi2 = grp(per2)
    rec['rcs433_svm'] = {'note': 'rcs_s42 后端180⊕LE253, 同 C/γ 迁移(未调)',
                         'val': float((svm2.predict(F_va) == y_va).mean()),
                         'test': float(ok2.mean()), 'low4': lo2, 'high5': hi2,
                         'per_subject': per2, 'n_support': int(svm2.n_support_.sum())}
    print('rcs433 SVM: val %.1f  test %.1f (低%.1f 高%.1f)'
          % (rec['rcs433_svm']['val'] * 100, ok2.mean() * 100, lo2 * 100, hi2 * 100))
    # 配对 McNemar
    b = int(np.sum(ok & ~ok2))
    c = int(np.sum(~ok & ok2))
    from scipy.stats import chi2
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    rec['rcs433_svm']['mcnemar_vs_253'] = [float(chi2.sf(chi, 1)), b, c]
    print('McNemar 433 vs 253: p=%.3f (b=%d c=%d)' % (chi2.sf(chi, 1), b, c))

    (OUT / 'svm_readout.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/svm_readout.json' % OUT)


if __name__ == '__main__':
    main()
