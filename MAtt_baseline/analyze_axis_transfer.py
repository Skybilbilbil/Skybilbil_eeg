# -*- coding: utf-8 -*-
"""路线 A:可迁移的判别结构在"空间轴"还是"时间/频率轴"?

动机(结论见 AGENTS.md):
  - 固定 22 通道协方差 -> 切空间:被试内 5 折 72.5%,跨被试 9x9 迁移 30.5%
    => 表示本身有内容,瓶颈是"坐标系/域偏移"。
  - 跨被试梯度是独立随机方向,对齐比按 1/sqrt(N) 衰减 => 共享梯度本身没有方向。
  - 解剖先验低通基(空间平滑方向)不多于随机基 => "共享空间方向"族已判死。

剩下的问法:**有没有哪条轴上的判别结构可以跨被试共享?** 本脚本把表示按轴拆开,
分别测"被试内可分性"与"跨被试可迁移性",看哪条轴衰减最小:

  1 轴向消融:full / diag(通道功率) / off-diag(纯空间耦合) / 频带细化 spec
  2 逐频带迁移曲线:哪个频带跨被试还有信号
  3 逐频带 CSP 空间子空间重叠 vs 随机基线 vs 被试内上限
  4 频率轴一致性 + 按 IAF 平移对齐能否恢复迁移(唯一可操作的一条)

纯 CPU、不训练。一次带通滤波同时产出全部量。运行:
  python analyze_axis_transfer.py
"""
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, welch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from utils.GetPhysionetMI import load_preprocessed_dataset
from analyze_spd_separability import MI_W0, MI_W1, trial_cov, tangent

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
FS = 250

FC0, FSTEP, FBW = 5.0, 0.5, 2.0            # 细栅格:2Hz 宽、0.5Hz 步(IAF 对齐的分辨率)
FINE_CENTERS = np.arange(FC0, 32.0 + 1e-9, FSTEP)
COARSE_CENTERS = np.arange(7.0, 29.0 + 1e-9, 1.0)   # 粗栅格:7..29Hz
COARSE_IDX = [int(round((c - FC0) / FSTEP)) for c in COARSE_CENTERS]
BROAD = (8.0, 30.0)
N_FOLDS = 5


def bp(X, lo, hi):
    """零相位带通。(N,C,T) -> (N,C,T)"""
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, X, axis=-1)


def _clf(scale):
    if scale:
        return make_pipeline(StandardScaler(),
                             LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')


def lda_cv(F, y, seed=42, scale=True):
    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    sc = []
    for tr, te in skf.split(F, y):
        clf = _clf(scale)
        clf.fit(F[tr], y[tr])
        sc.append(clf.score(F[te], y[te]))
    return float(np.mean(sc))


def lda_transfer(Fs, Ys, scale=True):
    """9x9 迁移:行=训,列=测。返回 (矩阵, 非对角均值, 对角均值)"""
    n = len(Fs)
    M = np.zeros((n, n))
    for i in range(n):
        clf = _clf(scale)
        clf.fit(Fs[i], Ys[i])
        for j in range(n):
            M[i, j] = clf.score(Fs[j], Ys[j]) * 100
    return M, (M.sum() - np.trace(M)) / (n * n - n), float(np.mean(np.diag(M)))


def per_target_transfer(Fs, Ys):
    """返回每个目标被试的"接收性"(别人训它来测)"""
    n = len(Fs)
    S = np.zeros((n, n))
    for i in range(n):
        clf = _clf(True)
        clf.fit(Fs[i], Ys[i])
        for j in range(n):
            if i != j:
                S[i, j] = clf.score(Fs[j], Ys[j]) * 100
    return S.sum(0) / (n - 1)


def csp_subspace(C, y, k=8):
    """one-vs-rest CSP 的前 k 个滤波器列空间 P(22 x k)"""
    Ws = []
    for c in np.unique(y):
        m = y == c
        Ca, Cb = C[m].mean(0), C[~m].mean(0)
        w, V = np.linalg.eigh(np.linalg.solve(Cb + 1e-6 * np.eye(22), Ca))
        Ws.append(V[:, np.argsort(w)[::-1][:3]])
    Q, _ = np.linalg.qr(np.concatenate(Ws, axis=1))
    return Q[:, :k]


def overlap(P, Q):
    """tr(P P^T Q Q^T)/k = 平均 cos^2(主角)"""
    return float(np.trace((P.T @ Q) @ (Q.T @ P)) / P.shape[1])


def random_overlap(d=22, k=8, n=500, seed=0):
    rng = np.random.default_rng(seed)
    v = [overlap(np.linalg.qr(rng.standard_normal((d, k)))[0],
                 np.linalg.qr(rng.standard_normal((d, k)))[0]) for _ in range(n)]
    return float(np.mean(v)), float(np.std(v))


def est_iaf(X):
    """用静息基线(epoch 索引 0:250 = cue 前 1s)的 6-13Hz 峰做 IAF,抛物线插值细化"""
    f, P = welch(X[:, :, 0:FS], fs=FS, nperseg=FS, nfft=1024, axis=-1)
    P = P.mean(axis=(0, 1))
    mk = (f >= 6) & (f <= 13)
    idx = np.flatnonzero(mk)[np.argmax(P[mk])]
    if 0 < idx < len(f) - 1 and (f[idx + 1] - f[idx]) > 0:
        y0, y1, y2 = P[idx - 1], P[idx], P[idx + 1]
        den = y0 - 2 * y1 + y2
        off = 0.5 * (y0 - y2) / den if den != 0 else 0.0
        return float(f[idx] + off * (f[1] - f[0]))
    return float(f[idx])


def main():
    sd, _ = load_preprocessed_dataset(str(DATA))
    n = len(sd)
    Xs = [np.asarray(X, dtype=np.float64) for X, _ in sd]
    Ys = [np.asarray(y) for _, y in sd]
    print('被试数=%d  每被试试次=%d  MI 窗=%.1f-%.1f s\n'
          % (n, len(Ys[0]), MI_W0 / FS, MI_W1 / FS))

    # ---- 一次滤波,产出:细栅格功率谱 / 粗栅格 CSP / 宽带切空间 / IAF ----
    iaf = np.zeros(n)
    spec = []                                        # (N,22,nfine) 对数功率
    Ps = np.zeros((n, len(COARSE_CENTERS), 22, 8))   # 粗栅格 CSP 子空间
    Ceil = np.zeros((n, len(COARSE_CENTERS)))        # 被试内 test-retest 上限
    broad = []                                       # 宽带 253 维切空间
    cmap = {ci: k for k, ci in enumerate(COARSE_IDX)}
    for s in range(n):
        X = Xs[s]
        iaf[s] = est_iaf(X)
        sp = np.zeros((len(X), 22, len(FINE_CENTERS)), dtype=np.float32)
        for bi, fc in enumerate(FINE_CENTERS):
            S = bp(X, fc - FBW / 2, fc + FBW / 2)[:, :, MI_W0:MI_W1]
            sp[:, :, bi] = np.log(np.var(S, axis=-1) + 1e-12)
            if bi in cmap:
                C = np.einsum('nct,ndt->ncd', S, S) / (MI_W1 - MI_W0)
                ci = cmap[bi]
                Ps[s, ci] = csp_subspace(C, Ys[s])
                h = len(Xs[s]) // 2
                Ceil[s, ci] = overlap(csp_subspace(C[:h], Ys[s][:h]),
                                      csp_subspace(C[h:], Ys[s][h:]))
        spec.append(sp)
        broad.append(tangent(trial_cov(bp(X, *BROAD), MI_W0, MI_W1)).astype(np.float32))
        print('  S%02d 完成  IAF=%.2fHz' % (s + 1, iaf[s]))

    print('\nIAF: 均值 %.2fHz  范围 %.2f~%.2fHz  std %.2fHz\n'
          % (iaf.mean(), iaf.min(), iaf.max(), iaf.std()))

    iu = np.triu_indices(22)
    diag_i = np.array([k for k in range(len(iu[0])) if iu[0][k] == iu[1][k]])
    off_i = np.array([k for k in range(len(iu[0])) if iu[0][k] != iu[1][k]])
    spec_c = [sp[:, :, COARSE_IDX].reshape(len(sp), -1) for sp in spec]

    # ---------- 1 轴向消融 ----------
    print('=' * 78)
    print('1 轴向消融:被试内 5 折 CV vs 跨被试 9x9 迁移(逐列 z-score)')
    print('   族                             被试内CV  跨被试迁移  迁移/被试内')
    for name, F in [('full   (253,空间耦合+功率)', broad),
                    ('diag   (22,只有通道功率)  ', [t[:, diag_i] for t in broad]),
                    ('offdiag(231,纯空间耦合)   ', [t[:, off_i] for t in broad]),
                    ('spec_c (506,频率轴细化)   ', spec_c),
                    ('spec_f (22x55,频率轴极细) ', [sp.reshape(len(sp), -1) for sp in spec])]:
        w = np.mean([lda_cv(f, y) for f, y in zip(F, Ys)])
        _, off, _ = lda_transfer(F, Ys)
        print('   %s  %6.1f%%   %6.1f%%      %5.2f' % (name, w * 100, off, off / (w * 100)))
    _, off0, dg0 = lda_transfer(broad, Ys, scale=False)
    print('   [回连既有口径] full 不标准化: 迁移 %.1f%%(非对角)  对角(乐观) %.1f%%'
          % (off0, dg0))
    print('   对照:随机 25%')

    # ---------- 2 逐频带迁移曲线 ----------
    print('\n' + '=' * 78)
    print('2 逐频带迁移曲线(单频带 22 通道对数功率)')
    print('   频带   被试内CV  接收性  迁移/被试内')
    bw, br = [], []
    for ci, fc in enumerate(COARSE_CENTERS):
        F = [sp[:, :, COARSE_IDX[ci]] for sp in spec]
        w = np.mean([lda_cv(f, y) for f, y in zip(F, Ys)])
        recv = per_target_transfer(F, Ys).mean()
        bw.append(w * 100); br.append(recv)
        print('   %4.1fHz    %5.1f%%   %5.1f%%     %5.2f' % (fc, w * 100, recv, recv / (w * 100)))
    bw, br = np.array(bw), np.array(br)
    print('   -> 被试内最优 %.0fHz(%.1f%%),其接收性 %.1f%%'
          % (COARSE_CENTERS[bw.argmax()], bw.max(), br[bw.argmax()]))
    print('   -> 接收性最优 %.0fHz(%.1f%%),其被试内 %.1f%%'
          % (COARSE_CENTERS[br.argmax()], br.max(), bw[br.argmax()]))
    print('   -> 接收性 %.1f~%.1f%%  跨度 %.1fpp   被试内 %.1f~%.1f%%'
          % (br.min(), br.max(), br.max() - br.min(), bw.min(), bw.max()))

    # ---------- 3 逐频带 CSP 空间子空间重叠 ----------
    print('\n' + '=' * 78)
    rnd_m, rnd_s = random_overlap()
    print('3 逐频带 CSP 空间子空间重叠(k=8),随机基线 %.3f+-%.3f(平均主角度 %.1f 度)'
          % (rnd_m, rnd_s, np.degrees(np.arccos(np.sqrt(rnd_m)))))
    print('   频带   跨被试重叠   被试内上限   可共享成分')
    for ci, fc in enumerate(COARSE_CENTERS):
        ov = [overlap(Ps[i, ci], Ps[j, ci]) for i in range(n) for j in range(i + 1, n)]
        ceil = float(Ceil[:, ci].mean())
        print('   %4.1fHz    %.3f        %.3f        %6.1f%%'
              % (fc, np.mean(ov), ceil, (np.mean(ov) - rnd_m) / (ceil - rnd_m) * 100))

    # ---------- 4 频率轴一致性与 IAF 对齐 ----------
    print('\n' + '=' * 78)
    print('4 频率轴一致性 + 按 IAF 平移对齐')
    prof = np.array([[lda_cv(sp[:, :, COARSE_IDX[ci]], Ys[s]) for ci in range(len(COARSE_CENTERS))]
                     for s, sp in enumerate(spec)])
    obs = [np.corrcoef(prof[i], prof[j])[0, 1] for i in range(n) for j in range(i + 1, n)]
    rng = np.random.default_rng(0)
    nul = [np.corrcoef(prof[i], rng.permutation(prof[j]))[0, 1]
           for i, j in rng.integers(0, n, (2000, 2)) if i != j]
    print('   被试间"频带重要性 profile"相关 r = %+.3f  (随机置换基线 %+.3f+-%.3f)'
          % (np.mean(obs), np.mean(nul), np.std(nul)))
    best = np.array([COARSE_CENTERS[np.argmax(prof[s])] for s in range(n)])
    print('   各被试最有判别力的频带: %s'
          % ', '.join('S%02d=%.0fHz' % (s + 1, b) for s, b in enumerate(best)))
    print('   与 IAF 的相关 r = %+.3f' % np.corrcoef(best, iaf)[0, 1])

    print('\n   IAF 对齐(每被试频带索引平移 -(IAF_s - mean),0.5Hz 分辨率):')
    print('     频带   原始接收性   IAF对齐接收性   Δ')
    t0, t1 = [], []
    for ci, fc in enumerate(COARSE_CENTERS):
        F0 = [sp[:, :, COARSE_IDX[ci]] for sp in spec]
        F1 = [sp[:, :, int(np.clip(COARSE_IDX[ci] - round((iaf[s] - iaf.mean()) / FSTEP),
                                   0, len(FINE_CENTERS) - 1))] for s, sp in enumerate(spec)]
        _, r0, _ = lda_transfer(F0, Ys)
        _, r1, _ = lda_transfer(F1, Ys)
        t0.append(r0); t1.append(r1)
        print('     %4.1fHz    %5.1f%%       %5.1f%%       %+5.1fpp' % (fc, r0, r1, r1 - r0))
    print('     均值      %5.1f%%       %5.1f%%       %+5.1fpp'
          % (np.mean(t0), np.mean(t1), np.mean(t1) - np.mean(t0)))


if __name__ == '__main__':
    main()
