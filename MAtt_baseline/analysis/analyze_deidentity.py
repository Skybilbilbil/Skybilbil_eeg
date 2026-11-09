# -*- coding: utf-8 -*-
"""共享特征来源盘点(一):去身份 + 相位连通性 —— 两种「其它方面」的信息源能不能当共享特征

## 动机(用户 2026-10-25:「找能用作共享参数模型的特征,不要只从算法角度」)
推进十八已证:LE 空间前 30 近邻 100% 同人 ⇒ 共享核 73.5 实为 9 个解码器。
本轮从文献找两条「让特征跨被试一致」的路:
  ① 去身份:Riemannian mean alignment + tangent-space scatter whitening
     (arXiv 2610.15626;身份识别 90.8%→15.6%,MI 代价 7.0pp)
  ② 相位连通性 PLV:协方差只含零滞后相关,相位同步是另一类信息源(本项目从未测过)

## 判定
「真共享」= 去掉身份结构后共享读出仍保留的精度。
参照:LE 线性 62.9 / 核 73.5(固定 C=100, gm=2)。

## 口径
主协议 rs=42 同折;去身份按人拟合(用 subject_id)⇒ **诊断模式,不是主协议读数**。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import hilbert
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import cov_tn, grp, fit_lda         # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'deidentity'
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


def plv_feats(Xe, chunk=256):
    """PLV_ab = |(1/T) Σ_t exp(i(φ_a−φ_b))|,Fisher z 后取 vech。"""
    Xt = Xe[:, :, TASK[0]:TASK[1]]
    out = []
    for i in range(0, len(Xt), chunk):
        xb = Xt[i:i + chunk]
        E = np.exp(1j * np.angle(hilbert(xb, axis=-1)))
        T = E.shape[-1]
        P = np.abs(np.einsum('nat,nbt->nab', E, E.conj())) / T
        out.append(np.stack([spdvec(P[k]) for k in range(len(P))]))
    return np.arctanh(np.clip(np.concatenate(out, 0), 0, 0.9999))


def med2(F, n=800):
    r = np.random.default_rng(1).choice(len(F), min(n, len(F)), replace=False)
    d2 = ((F[r][:, None, :] - F[r][None, :, :]) ** 2).sum(-1)
    return float(np.median(d2[np.triu_indices(len(r), 1)]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X = np.concatenate([s[0] for s in sd], 0).astype(np.float64)
    y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                          for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_ = ret[0], ret[1], ret[6]
    Xva, yva, sva = ret[2], ret[3], ret[7]
    Xte, yte, ste = ret[4], ret[5], ret[8]

    Ltr = le(cov_tn(Xtr[:, :, TASK[0]:TASK[1]]))
    Lva = le(cov_tn(Xva[:, :, TASK[0]:TASK[1]]))
    Lte = le(cov_tn(Xte[:, :, TASK[0]:TASK[1]]))
    Ptr, Pva, Pte = (plv_feats(X) for X in (Xtr, Xva, Xte))

    rec = {'note': 'DIAGNOSTIC: 去身份按人拟合(用 subject_id),不是主协议读数', 'arms': {}}
    m2 = med2(Ltr)

    def rep(name, ok, base=None):
        per = {SUBS[i]: float(ok[ste == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        d = '' if base is None else '  d=%+5.1f' % ((ok.mean() - base) * 100)
        print('%-30s TEST %5.1f (low %5.1f high %5.1f)%s'
              % (name, ok.mean() * 100, lo * 100, hi * 100, d), flush=True)
        rec['arms'][name] = {'test': float(ok.mean()), 'low4': lo, 'high5': hi,
                             'per_subject': per}
        return np.asarray(ok)

    ok_lda = fit_lda(Ltr, ytr).predict(Lte) == yte
    ok_svm = SVC(kernel='rbf', C=100., gamma=2.0 / (2 * m2)).fit(Ltr, ytr).predict(Lte) == yte
    rec['ref_lda'], rec['ref_svm'] = float(ok_lda.mean()), float(ok_svm.mean())
    rep('ref_LE_lda', ok_lda)
    rep('ref_LE_svm', ok_svm)

    MU = np.stack([Ltr[str_ == s].mean(0) for s in range(9)])
    SD = np.stack([Ltr[str_ == s].std(0) + 1e-9 for s in range(9)])

    def build(fn):
        outs = []
        for x, ix in ((Ltr, str_), (Lva, sva), (Lte, ste)):
            Z = np.empty_like(x)
            for s in range(9):
                m = ix == s
                if m.sum():
                    Z[m] = fn(s, x[m])
            outs.append(Z)
        return outs

    MA = build(lambda s, x: x - MU[s])
    rep('mean_align_lda', fit_lda(MA[0], ytr).predict(MA[2]) == yte, ok_lda.mean())
    rep('mean_align_svm', SVC(kernel='rbf', C=100., gamma=2.0 / (2 * med2(MA[0])))
        .fit(MA[0], ytr).predict(MA[2]) == yte, ok_svm.mean())

    DW = build(lambda s, x: (x - MU[s]) / SD[s])
    rep('diag_white_lda', fit_lda(DW[0], ytr).predict(DW[2]) == yte, ok_lda.mean())
    rep('diag_white_svm', SVC(kernel='rbf', C=100., gamma=2.0 / (2 * med2(DW[0])))
        .fit(DW[0], ytr).predict(DW[2]) == yte, ok_svm.mean())

    for lam in (0.9, 0.99, 0.999):
        As = []
        for s in range(9):
            Xs = Ltr[str_ == s] - MU[s]
            Sg = (Xs.T @ Xs) / len(Xs)
            tr = np.trace(Sg) / Sg.shape[0]
            Sl = (1 - lam) * Sg + lam * tr * np.eye(Sg.shape[0])
            w, V = np.linalg.eigh(Sl)
            As.append((V / np.sqrt(np.maximum(w, 1e-12))) @ V.T)
        CW = build(lambda s, x, As=As: (x - MU[s]) @ As[s])
        rep('covwhite_l%.3f_lda' % lam, fit_lda(CW[0], ytr).predict(CW[2]) == yte, ok_lda.mean())
        rep('covwhite_l%.3f_svm' % lam, SVC(kernel='rbf', C=100., gamma=2.0 / (2 * med2(CW[0])))
            .fit(CW[0], ytr).predict(CW[2]) == yte, ok_svm.mean())

    for nm, F in (('raw', Ltr), ('mean_align', MA[0]), ('diag_white', DW[0])):
        acc = float((KNeighborsClassifier(n_neighbors=9).fit(F, str_).predict(F) == str_).mean())
        print('  identity kNN(9) %-11s = %.3f (chance 0.111)' % (nm, acc), flush=True)
        rec['arms']['idknn_' + nm] = acc

    rep('PLV_lda', fit_lda(Ptr, ytr).predict(Pte) == yte)
    rep('PLV_svm', SVC(kernel='rbf', C=100., gamma=2.0 / (2 * med2(Ptr)))
        .fit(Ptr, ytr).predict(Pte) == yte)
    Fuse = [np.concatenate([a, b], 1) for a, b in ((Ltr, Ptr), (Lva, Pva), (Lte, Pte))]
    rep('PLV_plus_LE_lda', fit_lda(Fuse[0], ytr).predict(Fuse[2]) == yte, ok_lda.mean())
    PMU = np.stack([Ptr[str_ == s].mean(0) for s in range(9)])
    PA = [x - PMU[ix] for x, ix in ((Ptr, str_), (Pva, sva), (Pte, ste))]
    rep('PLV_mean_align_lda', fit_lda(PA[0], ytr).predict(PA[2]) == yte)

    (OUT / 'deidentity.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                         encoding='utf-8')
    print('[OK] %s/deidentity.json' % OUT, flush=True)


if __name__ == '__main__':
    main()
