# -*- coding: utf-8 -*-
"""PhysionetMI 前 50% cohort 瓶颈定位：表示 vs 分类头（零训练，CPU）。

问题：共享深度模型在 cohort 上 train 55.7% / test 52.27%，混淆主轴是「单手 vs 双手」。
BCI-2a 上做过同一诊断（E-1，2026-10-30）：闭式 LDA 头 63.2/63.5 与 SGD 头 62.9 统计无差
⇒ 「训练过程损失」不存在。此处用同一判据检验 Physionet。

做法：用 `get_features`（分类头之前的表示）在训练折上拟合闭式 LDA，
在 cohort 的 val/test 上评估，与 SGD 头对照。
  · 闭式头 ≈ SGD 头 ⇒ 瓶颈在**表示**（特征里没有可线性读出的类信息）；
  · 闭式头 ≫ SGD 头 ⇒ 瓶颈在**头/训练过程**。

产物：results/physionet_head/physionet_head.json
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
CKPT = BASE / 'results' / 'physionet_mix80_s42' / 'pm80_cross_mix_best_acc.pt'
COHORT = BASE / 'results' / 'physionet_gap' / 'cohort_by_trainacc.json'
OUT = BASE / 'results' / 'physionet_head'
ARGS = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=True)
WIN, STEP = 120, 60


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    from scipy.signal import resample_poly
    sd = [(resample_poly(np.asarray(X, np.float32), 1, 2, axis=-1).astype(np.float32), y)
          for X, y in sd]
    X = np.concatenate([s[0] for s in sd], 0); y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste = (
        ret[0], ret[1], ret[6], ret[2], ret[3], ret[7], ret[4], ret[5], ret[8])
    keep = sorted(int(s) - 1 for s in json.load(open(COHORT))['cohort_1based'])
    mva, mte = np.isin(sva, keep), np.isin(ste, keep)

    net = load_matt_model(str(CKPT), WIN, STEP, ARGS).cpu().eval()
    print(f'[特征] 提取中 (M 窗注意力后切空间)…', flush=True)

    def feats(S):
        out = []
        with torch.no_grad():
            for i in range(0, len(S), 256):
                t = torch.Tensor(S[i:i + 256]).unsqueeze(1)
                out.append(net.get_features(t).numpy())
        return np.concatenate(out)

    Ftr, Fva, Fte = feats(Xtr), feats(Xva), feats(Xte)
    print(f'[特征] dim={Ftr.shape[1]} | train {Ftr.shape[0]}', flush=True)

    rec = {'feat_dim': int(Ftr.shape[1]), 'cohort_n': len(keep),
           'arms': {}, 'sgd_head_test': 0.5227}

    def acc(p, y, m):
        # clf.predict 返回标签（1D），不是概率
        return float((p[m] == y[m]).mean())

    for name, clf in (('lda_shrinkage', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
                      ('logreg_C1', LogisticRegression(max_iter=2000, C=1.0))):
        clf.fit(Ftr, ytr)
        ptr, pva, pte = clf.predict(Ftr), clf.predict(Fva), clf.predict(Fte)
        r = {'train': float((ptr == ytr).mean()),
             'val_cohort': acc(pva, yva, mva), 'test_cohort': acc(pte, yte, mte),
             'val_all': float((pva == yva).mean()), 'test_all': float((pte == yte).mean())}
        rec['arms'][name] = r
        print(f'[闭式头 {name:14s}] train {r["train"]*100:5.2f} | cohort val {r["val_cohort"]*100:5.2f} '
              f'| cohort test {r["test_cohort"]*100:5.2f} | 全103 test {r["test_all"]*100:5.2f}',
              flush=True)

    # ---- 预检验：深度特征 ⊕ 原始任务窗协方差切向量（模拟 --raw_cov_stream，零训练） ----
    # 动机：头不是瓶颈 ⇒ 表示缺信息；模型 conv2 把 22 通道压到 8 维 SPD，可能丢掉
    # 「单手 vs 双手」需要的双侧功率对比。直接拼接 253 维全切空间看能不能补上。
    def cov_tn(S):
        s = S.astype(np.float64)
        s = s - s.mean(axis=2, keepdims=True)
        C = np.einsum('nct,ndt->ncd', s, s) / s.shape[-1]
        return C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)

    def le_feats(C):
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
        iu = np.triu_indices(C.shape[1])
        return L[:, iu[0], iu[1]]

    T0, T1 = 80, 361
    Cv = [cov_tn(S[:, :, T0:T1]) for S in (Xtr, Xva, Xte)]
    Lv = [le_feats(c) for c in Cv]
    rec['raw_cov_dim'] = int(Lv[0].shape[1])
    for tag, parts in (('cov_only', (None,)),
                       ('depth+cov', (None,))):
        if tag == 'cov_only':
            A, B, Cc = Lv
        else:
            A = np.concatenate([Ftr, Lv[0]], 1)
            B = np.concatenate([Fva, Lv[1]], 1)
            Cc = np.concatenate([Fte, Lv[2]], 1)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(A, ytr)
        pva, pte = clf.predict(B), clf.predict(Cc)
        r = {'dim': int(A.shape[1]), 'train': float((clf.predict(A) == ytr).mean()),
             'val_cohort': acc(pva, yva, mva), 'test_cohort': acc(pte, yte, mte),
             'test_all': float((pte == yte).mean())}
        rec['arms'][tag] = r
        print(f'[预检验 {tag:10s}] dim {r["dim"]:4d} | train {r["train"]*100:5.2f} | '
              f'cohort val {r["val_cohort"]*100:5.2f} | cohort test {r["test_cohort"]*100:5.2f} | '
              f'全103 test {r["test_all"]*100:5.2f}', flush=True)

    (OUT / 'physionet_head.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/physionet_head.json', flush=True)


if __name__ == '__main__':
    main()
