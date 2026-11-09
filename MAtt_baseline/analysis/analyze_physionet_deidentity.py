# -*- coding: utf-8 -*-
"""PhysionetMI 去身份检验（零训练，CPU）:把身份方向压掉后类别可分性会不会转正。

## 动机（2026-11-08）
gap 诊断给出三条一致读数：
  · 身份 103 类 LDA 5 折 = 1.0000（随机 0.0097）、最近邻同人率 k=1/15/30 全 1.000；
  · 同一个 253 维切空间里类别 silhouette = **-0.011**（类别簇互相重叠）；
  · 跨人类差分方向余弦 +0.080 vs 劈半地板 +0.398（比值 0.20）。
⇒ 表示空间被「这是谁」占满。于是检验:压掉身份后，类别信息能不能回到可读状态。

## 口径（严格照 analyze_deidentity.py 的 BCI 版，便于对照）
  · 特征 = trace 归一协方差 → logm → 上三角 253 维（与 gap 脚本 C2 同口径）；
  · 变换按人拟合（用 subject_id）⇒ **诊断模式，不是主协议读数**；
  · 四臂:baseline / mean_align(x-μ_s) / diag_white((x-μ_s)/σ_s) /
    cov_white(逐人散布 Sg，收缩 Sl=(1-λ)Sg+λ·tr(Sg)/n·I，λ∈{0.9,0.99,0.999})；
  · 读出 = LDA(lsqr, shrinkage=auto)；指标 = train/test 精度 + 类别 silhouette + 身份 kNN(train→test)。

## BCI 对照（analyze_deidentity.py，2026-10-25）
  · LE 线性 62.9 → mean_align 62.2 → diag_white 61.1 → cov_white 61.1-62.3 ⇒ **不涨**；
  · 但 BCI 的类别 silhouette 是 **+0.013（正）**，类别本来就分得开；
  · Physionet 是 **-0.011（负）**，类别簇本来就叠着 —— 这是 BCI 上无法区分的情形。

用法: CUDA_VISIBLE_DEVICES= python analysis/analyze_physionet_deidentity.py
产物: results/physionet_deidentity/physionet_deidentity.json
"""
import sys, json
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
OUT = BASE / 'results' / 'physionet_deidentity'
RESAMPLE = 2                       # 160 -> 80 Hz（与 gap 脚本同一个 ckpt 口径）
FS = 160 // RESAMPLE
T0_SEC, T1_SEC = 1.0, 10.0         # analyze_physionet_adapt.TASK 同口径


def cov(seg):
    s = seg - seg.mean(axis=2, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / seg.shape[-1]


def le_feats(C):
    """trace 归一 → logm → 上三角（253 维），与 gap 脚本 C2 一致。"""
    C = C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    if RESAMPLE > 1:
        from scipy.signal import resample_poly
        sd = [(resample_poly(np.asarray(X, np.float32), 1, RESAMPLE, axis=-1).astype(np.float32), y)
              for X, y in sd]
    X = np.concatenate([s[0] for s in sd], 0)
    y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste = (
        ret[0], ret[1], ret[6], ret[2], ret[3], ret[7], ret[4], ret[5], ret[8])
    t0, t1 = int(T0_SEC * FS), min(int(T1_SEC * FS), X.shape[-1])
    print(f'[数据] {len(sd)} 人; train {len(ytr)} / val {len(yva)} / test {len(yte)}; '
          f'任务窗 {t0}:{t1}（{FS} Hz）')

    L = [le_feats(cov(S[:, :, t0:t1].astype(np.float64))) for S in (Xtr, Xva, Xte)]
    Ltr, Lva, Lte = L
    n_sub = len(sd)
    MU = np.stack([Ltr[str_ == s].mean(0) if (str_ == s).any() else Ltr.mean(0)
                   for s in range(n_sub)])
    SD = np.stack([Ltr[str_ == s].std(0) if (str_ == s).any() else Ltr.std(0)
                   for s in range(n_sub)]) + 1e-9

    def build(fn, F, ix):
        Z = np.empty_like(F)
        for s in range(n_sub):
            m = ix == s
            if m.any():
                Z[m] = fn(s, F[m])
        return Z

    def eval_arm(name, Ftr, Fte):
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
        ok_tr = float((clf.predict(Ftr) == ytr).mean())
        ok_te = float((clf.predict(Fte) == yte).mean())
        sil = float(silhouette_score(Fte, yte))
        knn = KNeighborsClassifier(n_neighbors=9).fit(Ftr, str_)
        id_te = float((knn.predict(Fte) == ste).mean())
        print(f'[{name:20s}] train {ok_tr*100:5.2f} | test {ok_te*100:5.2f} | '
              f'silhouette(类别) {sil:+.3f} | 身份kNN {id_te:.3f}', flush=True)
        return {'train': ok_tr, 'test': ok_te, 'silhouette_class': sil,
                'identity_knn': id_te}

    rec = {'note': 'DIAGNOSTIC: 变换按人拟合(用 subject_id)，不是主协议读数',
           'fs': FS, 'task_window': [T0_SEC, T1_SEC],
           'n_subjects': n_sub, 'n_test': int(len(yte)),
           'bci_reference': {'le_lda': 0.629, 'mean_align': 0.622,
                             'diag_white': 0.611, 'cov_white': '0.611-0.623',
                             'note': 'BCI 九人；类别 silhouette 为 +0.013（物理上类别可分）'},
           'arms': {}}

    rec['arms']['baseline'] = eval_arm('baseline', Ltr, Lte)

    MA = [build(lambda s, x: x - MU[s], F, ix) for F, ix in
          ((Ltr, str_), (Lva, sva), (Lte, ste))]
    rec['arms']['mean_align'] = eval_arm('mean_align', MA[0], MA[2])

    DW = [build(lambda s, x: (x - MU[s]) / SD[s], F, ix) for F, ix in
          ((Ltr, str_), (Lva, sva), (Lte, ste))]
    rec['arms']['diag_white'] = eval_arm('diag_white', DW[0], DW[2])

    for lam in (0.9, 0.99, 0.999):
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
        rec['arms'][f'cov_white_l{lam}'] = eval_arm(f'cov_white_l{lam}', CW[0], CW[2])

    (OUT / 'physionet_deidentity.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/physionet_deidentity.json', flush=True)


if __name__ == '__main__':
    main()
