# -*- coding: utf-8 -*-
"""文献做法补测:多频带协方差的「黎曼均值融合」与「OAS 收缩」

## 背景(用户 2026-10-25:「参考别的在这方面擅长的文献的模型做法,他们从算法和原本特征中做什么处理」)
跨被试 MI 文献的标准 pipeline 分四层:信号(带通/filter bank/CAR) → 对齐(EA/RA/RPA/ITSA/DCT)
→ 特征(协方差估计 OAS、切空间 LE、vech) → 模型(LDA/SVM/CNN+Transformer)。
本项目已内建 EA(pkl 为 22×22 全通道 EA)、并已判死 RA/RPA/filter bank 拼接/多参考点。
**未测过的两处**:①多频带协方差用「黎曼均值融合」(而非拼接);②协方差 OAS 收缩。

## 数学
- 黎曼(此实现用 log-欧氏)均值融合:C̄ = exp(mean_k log C_k),在 SPD 流形上取几何平均,
  避免拼接导致的维度稀释(拼接会把 253 维扩到 759 维,shrinkage LDA 被稀释)。
- OAS 收缩:Σ̂ = (1−ρ)Σ̂_ML + ρ·tr(Σ̂)/p·I,ρ 由 Oracle Approximating Shrinkage 闭式给出;
  本项目默认是 trace 归一 + 无收缩(样本量 n≈750 ≫ p=22,收缩本应几乎无效)。

## 口径
主协议 rs=42 同折;SVM 的 γ 网格含 {1,2,3},故参照行为 gm3 的 72.1(固定 gm2 为 73.5)。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.covariance import OAS
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import grp, fit_lda, logm_pd, expm_sym  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'literature_pipeline'
TASK = ED.TASK
SUBS = ['S%02d' % i for i in range(1, 10)]
SFREQ = 250.0
BANDS = {'bb': (8, 30), 'mu': (8, 13), 'beta': (15, 25)}


def spdvec(M):
    iu = np.triu_indices(22)
    v = M[iu].copy()
    v[iu[0] != iu[1]] *= np.sqrt(2)
    return v


def le(A):
    w, V = np.linalg.eigh(A.astype(np.float64))
    w = np.maximum(w, 1e-12)
    return np.stack([spdvec((V[i] * np.log(w[i])) @ V[i].T) for i in range(len(A))])


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (22 / np.trace(C, axis1=1, axis2=2)[:, None, None])


def bandpass(x, lo, hi, order=4):
    b, a = butter(order, [lo / (SFREQ / 2), hi / (SFREQ / 2)], btype='band')
    return filtfilt(b, a, x, axis=-1)


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
    Xtr, ytr = ret[0], ret[1]
    Xva, yva = ret[2], ret[3]
    Xte, yte, ste = ret[4], ret[5], ret[8]

    rec = {'note': '文献做法补测;参照口径:LE253 LDA=62.9, SVM gm3=72.1(固定 gm2 为 73.5)', 'arms': {}}

    def rep(name, ok, val=None):
        per = {SUBS[i]: float(ok[ste == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        vs = '' if val is None else '  val %5.1f' % (val * 100)
        print('%-30s TEST %5.1f (low %5.1f high %5.1f)%s'
              % (name, ok.mean() * 100, lo * 100, hi * 100, vs), flush=True)
        rec['arms'][name] = {'test': float(ok.mean()), 'low4': lo, 'high5': hi, 'per_subject': per}
        return np.asarray(ok)

    def svm_grid(F):
        m2 = med2(F[0])
        best = None
        for C_ in (10., 100.):
            for gm in (1., 2., 3.):
                s = SVC(kernel='rbf', C=C_, gamma=gm / (2 * m2)).fit(F[0], ytr)
                v = float((s.predict(F[1]) == yva).mean())
                if best is None or v > best[0]:
                    best = (v, gm, s)
        return best

    Cbb = [cov_tn(x[:, :, TASK[0]:TASK[1]]) for x in (Xtr, Xva, Xte)]
    Lbb = [le(c) for c in Cbb]
    rep('ref_LE_lda', fit_lda(Lbb[0], ytr).predict(Lbb[2]) == yte)
    b = svm_grid(Lbb)
    rep('ref_LE_svm_gm%.0f' % b[1], b[2].predict(Lbb[2]) == yte, b[0])

    Cs = {k: [cov_tn(bandpass(x, lo, hi)[:, :, TASK[0]:TASK[1]]) for x in (Xtr, Xva, Xte)]
          for k, (lo, hi) in BANDS.items()}
    for combo in (('bb', 'mu'), ('bb', 'beta'), ('bb', 'mu', 'beta')):
        def merge(i, combo=combo):
            return expm_sym(np.mean([logm_pd(Cs[k][i]) for k in combo], axis=0))
        Fm = [le(merge(i)) for i in range(3)]
        rep('rmean_%s_lda' % '+'.join(combo), fit_lda(Fm[0], ytr).predict(Fm[2]) == yte)
        b = svm_grid(Fm)
        rep('rmean_%s_svm_gm%.0f' % ('+'.join(combo), b[1]), b[2].predict(Fm[2]) == yte, b[0])
    Fc = [np.concatenate([le(Cs[k][i]) for k in ('bb', 'mu', 'beta')], 1) for i in range(3)]
    rep('concat_bb+mu+beta_lda', fit_lda(Fc[0], ytr).predict(Fc[2]) == yte)

    def cov_oas(x):
        out = []
        for i in range(len(x)):
            o = OAS(assume_centered=True).fit(x[i, :, TASK[0]:TASK[1]].T)
            out.append(o.covariance_)
        C = np.stack(out)
        return C * (22 / np.trace(C, axis1=1, axis2=2)[:, None, None])
    Lo = [le(cov_oas(x)) for x in (Xtr, Xva, Xte)]
    rep('oas_LE_lda', fit_lda(Lo[0], ytr).predict(Lo[2]) == yte)
    b = svm_grid(Lo)
    rep('oas_LE_svm_gm%.0f' % b[1], b[2].predict(Lo[2]) == yte, b[0])

    (OUT / 'literature_pipeline.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/literature_pipeline.json' % OUT, flush=True)


if __name__ == '__main__':
    main()
