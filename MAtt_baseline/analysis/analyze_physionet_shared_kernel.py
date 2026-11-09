# -*- coding: utf-8 -*-
"""PhysionetMI 前 50% cohort 上的共享核读出（零训练，CPU）。

口径（用户 2026-11-08 定）:
  · 评估 cohort = 训练集深度 acc 排序前 52 人（cohort_by_trainacc.json，不碰 test 排序）；
  · **模型不接收 subject_id**（主协议硬约束）—— 这里是一个共享 RBF-SVM / Nyström 读出，
    训练用**全 103 人训练折**（与共享预训练同可见度），只在 cohort 的 test 上评估。
  · 因此 3 个 '''+'''标注必须有:① pooled 精度;② 身份纯度 kNN9;③ 拆成逐人 LDA 后涨多少
    （BCI 上 73.5 就是被这三条判定为「实为 9 个解码器」的假共享）。

参照（同一 cohort，已落盘）:
  深度 52.27 / 共享 CSP 50.99 / 共享 LE 42.90 / 深度⊕CSP 融合 53.55。

产物: results/physionet_shared_kernel/physionet_shared_kernel.json
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from scipy.spatial.distance import cdist

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
COHORT = BASE / 'results' / 'physionet_gap' / 'cohort_by_trainacc.json'
OUT = BASE / 'results' / 'physionet_shared_kernel'
RESAMPLE = 2
T0_SEC, T1_SEC = 1.0, 10.0


def le_feats(C):
    C = C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]]


def cov(seg):
    s = seg.astype(np.float64)
    s = s - s.mean(axis=2, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / s.shape[-1]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    if RESAMPLE > 1:
        from scipy.signal import resample_poly
        sd = [(resample_poly(np.asarray(X, np.float32), 1, RESAMPLE, axis=-1).astype(np.float32), y)
              for X, y in sd]
    X = np.concatenate([s[0] for s in sd], 0); y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste = (
        ret[0], ret[1], ret[6], ret[2], ret[3], ret[7], ret[4], ret[5], ret[8])
    keep = sorted(int(s) - 1 for s in json.load(open(COHORT))['cohort_1based'])
    mte, mva = np.isin(ste, keep), np.isin(sva, keep)
    fs = 160 // RESAMPLE
    t0, t1 = int(T0_SEC * fs), min(int(T1_SEC * fs), X.shape[-1])
    Ftr, Fva, Fte = (le_feats(cov(S[:, :, t0:t1])) for S in (Xtr, Xva, Xte))
    print(f'[数据] {len(sd)} 人 | cohort {len(keep)} | test {int(mte.sum())} | dim {Ftr.shape[1]}',
          flush=True)

    rng = np.random.default_rng(1)
    ridx = rng.choice(len(Ftr), min(1200, len(Ftr)), replace=False)
    d2 = ((Ftr[ridx][:, None, :] - Ftr[ridx][None, :, :]) ** 2).sum(-1)
    med2 = float(np.median(d2[np.triu_indices(len(ridx), 1)]))
    print(f'[核] median d2 = {med2:.3f}', flush=True)

    rec = {'cohort_n': len(keep), 'test_n': int(mte.sum()), 'feat_dim': int(Ftr.shape[1]),
           'reference': {'deep': 0.5227, 'csp': 0.5099, 'le': 0.4290, 'deep_csp_fusion': 0.5355},
           'arms': {}}

    def acc(p, m):
        return float((p[m] == yte[m]).mean())

    def report(name, clf, Xtr_=None):
        pva, pte = clf.predict(Fva), clf.predict(Fte)
        r = {'cohort_val': float((pva[mva] == yva[mva]).mean()),
             'cohort_test': acc(pte, mte),
             'val_all': float((pva == yva).mean()),
             'test_all': float((pte == yte).mean())}
        # 身份纯度 + 拆开检验（同特征、同 cohort）
        knn = KNeighborsClassifier(n_neighbors=9).fit(Ftr, str_)
        r['identity_knn9'] = float((knn.predict(Fte) == ste).mean())
        per = []
        for s in keep:
            m_tr, m_te = str_ == s, ste == s
            if m_tr.sum() < 10 or m_te.sum() < 3:
                continue
            clf_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr[m_tr], ytr[m_tr])
            per.append(float((clf_s.predict(Fte[m_te]) == yte[m_te]).mean()))
        r['per_subject_lda'] = float(np.mean(per)) if per else None
        rec['arms'][name] = r
        print(f'[{name:18s}] cohort val {r["cohort_val"]*100:5.2f} | test {r["cohort_test"]*100:5.2f}'
              f' | 身份kNN9 {r["identity_knn9"]:.3f}'
              f' | 拆9个逐人 {r["per_subject_lda"]*100:5.2f}', flush=True)
        return r

    # 线性基线（同特征、同 cohort，用于确认口径）
    report('le_lda', LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr))

    for gm in (8.0,):
        for C in (100.0,):
            clf = SVC(kernel='rbf', C=C, gamma=gm / (2 * med2)).fit(Ftr, ytr)
            report(f'rbf_C{int(C)}_gm{gm}', clf)

    (OUT / 'physionet_shared_kernel.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/physionet_shared_kernel.json', flush=True)


if __name__ == '__main__':
    main()
