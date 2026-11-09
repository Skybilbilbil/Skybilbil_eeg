# -*- coding: utf-8 -*-
"""出口 1 的量化:少标签逐人校准在最优解码器(rcs 特征)上值多少(零训练)。

## 依据
论文 §10.3 出口 1(少样本校准)写着「量级参考是 §9.3 的 k≈16——任何校准机制必须
显著超过这条线,否则不如直接给目标试次」;但该校准量级从未在**当前最优特征**
(rcs 433 维,共享 LDA 头 = 61.4)上量化过。rcs 特征的逐人−共享差距只剩 +3.7/+3.6
(推进十)——这正是「出口 1 能不能吃掉最后一段」的直接问题。

## 设计
k∈{0,5,10,20}/类校准试次(从**目标被试自己的训练折**抽,测试折不动;无泄漏)。
校准方式 = 收缩插值:全局 LDA 头 + 被试 k 试次(**复制 R=20 倍加权**)重拟合
shrinkage LDA。每个被试、每个 k 重复 20 次随机抽样取均值。
3 个 rcs seed 的特征各做一遍,按 seed 配对报低四/高五/全体。
只写 results/fewshot_calib/。
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
OUT = BASE / 'results' / 'fewshot_calib'
RUNS = {42: 'rcs_s42', 2024: 'rcs_s2024', 7: 'rcs_s7'}
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            raw_cov_stream=True)
KS = (0, 5, 10, 20)
R = 20        # 校准试次复制倍数(给小样本以话语权)
NREP = 20


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
    rec = {'per_seed': {}}
    agg = {k: {s: [] for s in SUBJ} for k in KS}
    for seed, dirname in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'rcs{seed}_cross_mix_best_acc.pt'), 375, 188, ARGS)
        net.eval()
        dev = next(net.parameters()).device

        def feats(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    out.append(net._features(
                        torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy())
            return np.concatenate(out, 0)
        Ftr, Fte = feats(X_tr), feats(X_te)
        rng = np.random.default_rng(seed)
        res = {k: {s: [] for s in SUBJ} for k in KS}
        for i, s in enumerate(SUBJ):
            trm, tem = sub_tr == i, sub_te == i
            Fi, yi = Ftr[trm], y_tr[trm]
            base_clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            base_clf.fit(Ftr, y_tr)
            res[0][s].append(float((base_clf.predict(Fte[tem]) == y_te[tem]).mean()))
            for k in KS:
                if k == 0:
                    continue
                idx_c = [np.where(yi == c)[0] for c in range(4)]
                for rep in range(NREP):
                    pick = np.concatenate([rng.choice(ix, k, replace=False)
                                           for ix in idx_c])
                    Fc = np.concatenate([Ftr, np.tile(Fi[pick], (R, 1))])
                    yc = np.concatenate([y_tr, np.tile(yi[pick], R)])
                    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                    clf.fit(Fc, yc)
                    res[k][s].append(float(
                        (clf.predict(Fte[tem]) == y_te[tem]).mean()))
        rec['per_seed'][str(seed)] = {str(k): {s: float(np.mean(res[k][s]))
                                               for s in SUBJ} for k in KS}
        for k in KS:
            for s in SUBJ:
                agg[k][s].append(np.mean(res[k][s]))
        print('seed%d 完成' % seed, flush=True)
    print('\n少标签逐人校准(rcs 特征,k 试次/类,校准试次复制 %d 倍;3 seed 均值):' % R)
    print('%-4s %7s %7s %7s | %s' % ('k', '低四', '高五', '全体', '低四逐人'))
    rec['summary'] = {}
    for k in KS:
        ps = {s: float(np.mean(agg[k][s])) for s in SUBJ}
        lo = np.mean([ps[SUBJ[i]] for i in LOW])
        hi = np.mean([ps[SUBJ[i]] for i in HIGH])
        ov = np.mean(list(ps.values()))
        rec['summary'][str(k)] = {'per_subject': ps, 'low4': float(lo),
                                  'high5': float(hi), 'overall': float(ov)}
        print('%-4d %7.1f %7.1f %7.1f | %s' % (
            k, lo * 100, hi * 100, ov * 100,
            ' '.join('%s%.0f' % (SUBJ[i], ps[SUBJ[i]] * 100) for i in LOW)))
    (OUT / 'fewshot_calib.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/fewshot_calib.json' % OUT)


if __name__ == '__main__':
    main()
