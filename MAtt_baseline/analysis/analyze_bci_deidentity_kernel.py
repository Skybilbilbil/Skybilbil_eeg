# -*- coding: utf-8 -*-
"""BCI-2a 判决实验：共享核读出 73.5 的增益到底来自身份还是结构。

## 问题（2026-11-08，Physionet 侧反哺）
Physionet 上共享 RBF-SVM（LE 253 维）在全 103 人上 49.32%、拆成逐人 LDA 只有 53.91%
⇒ 真共享。而 BCI-2a 上共享核 73.5%、拆开 73.8%（+0.3）⇒ 判为「假共享」（实为 9 个解码器）。
但「假共享」一直是从「拆开涨不涨」推断的，**没有直接测量核读出的增益是否来自身份**。

## 做法
沿用 analyze_deidentity.py 的 BCI 口径（逐人均值对齐 → 对角白化 → 逐人协方差白化），
把读出从 LDA 扩成 **LDA + RBF-SVM**（C=100、γ=3/(2·median d²)，与 svm_readout 同款）。
如果核的 73.5 主要读的是身份，那么去身份后核应当大幅下降；若几乎不动，则说明核里
确实有跨人共享的结构，73.5 的价值需要重估。

⚠ 去身份按人拟合（用 subject_id）⇒ **诊断模式，不是主协议读数**。

产物：results/bci_deidentity_kernel/bci_deidentity_kernel.json
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'bci_deidentity_kernel'
TASK = (375, 1126)          # cue 后 0.5s 起至 epoch 末尾（analyze_individual_diff 同口径）


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


def med2(F, n=800, seed=1):
    r = np.random.default_rng(seed).choice(len(F), min(n, len(F)), replace=False)
    d2 = ((F[r][:, None, :] - F[r][None, :, :]) ** 2).sum(-1)
    return float(np.median(d2[np.triu_indices(len(r), 1)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X = np.concatenate([s[0] for s in sd], 0)
    y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(sd)])
    n_sub = len(sd)
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste = (
        ret[0], ret[1], ret[6], ret[2], ret[3], ret[7], ret[4], ret[5], ret[8])
    print(f'[数据] {n_sub} 人 | train {len(ytr)} / val {len(yva)} / test {len(yte)}', flush=True)

    L = [le_feats(cov_tn(S[:, :, TASK[0]:TASK[1]])) for S in (Xtr, Xva, Xte)]
    Ltr, Lva, Lte = L
    MU = np.stack([Ltr[str_ == s].mean(0) for s in range(n_sub)])
    SD = np.stack([Ltr[str_ == s].std(0) for s in range(n_sub)]) + 1e-9

    def build(fn, F, ix):
        Z = np.empty_like(F)
        for s in range(n_sub):
            m = ix == s
            if m.any():
                Z[m] = fn(s, F[m])
        return Z

    arms = {'baseline': (Ltr, Lva, Lte)}

    MA = [build(lambda s, x: x - MU[s], F, ix) for F, ix in
          ((Ltr, str_), (Lva, sva), (Lte, ste))]
    arms['mean_align'] = tuple(MA)

    DW = [build(lambda s, x: (x - MU[s]) / SD[s], F, ix) for F, ix in
          ((Ltr, str_), (Lva, sva), (Lte, ste))]
    arms['diag_white'] = tuple(DW)

    lam = 0.99
    As = []
    for s in range(n_sub):
        Xs = Ltr[str_ == s] - MU[s]
        Sg = (Xs.T @ Xs) / max(len(Xs), 1)
        tr = np.trace(Sg) / Sg.shape[0]
        Sl = (1 - lam) * Sg + lam * tr * np.eye(Sg.shape[0])
        w, V = np.linalg.eigh(Sl)
        As.append((V / np.sqrt(np.maximum(w, 1e-12))) @ V.T)
    CW = [build(lambda s, x, As=As: (x - MU[s]) @ As[s], F, ix) for F, ix in
          ((Ltr, str_), (Lva, sva), (Lte, ste))]
    arms['cov_white_l0.99'] = tuple(CW)

    rec = {'note': 'DIAGNOSTIC: 去身份按人拟合(用 subject_id)；任务窗 %s' % (TASK,),
           'task_window': list(TASK), 'n_sub': n_sub, 'n_test': int(len(yte)), 'arms': {}}
    for name, (A, B, Cc) in arms.items():
        m2 = med2(A)
        for rname, clf in (
            ('lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')),
            ('svm', SVC(kernel='rbf', C=100., gamma=3.0 / (2 * m2))),
        ):
            clf.fit(A, ytr)
            r = {'train': float((clf.predict(A) == ytr).mean()),
                 'val': float((clf.predict(B) == yva).mean()),
                 'test': float((clf.predict(Cc) == yte).mean())}
            rec['arms'][f'{name}_{rname}'] = r
            print(f'[{name:16s} {rname}] train {r["train"]*100:5.2f} | val {r["val"]*100:5.2f} '
                  f'| test {r["test"]*100:5.2f}', flush=True)

    (OUT / 'bci_deidentity_kernel.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/bci_deidentity_kernel.json', flush=True)


if __name__ == '__main__':
    main()
