# -*- coding: utf-8 -*-
"""路线 A 决定性检验(第 2 版):频带内共享空间滤波,以及多频带拼接能否超过宽带?

第 1 版(analyze_axis_transfer.py 第 3 节)显示逐频带 CSP 子空间重叠在
10-11Hz / 22-27Hz 有 50-69% 可共享成分;但用"共享 CSP 真迁移"检验
(本脚本第 1 节)后发现:宽带 (8-30) 迁移 41.5% **高于任何单频带**,
而 13-18Hz 的共享 CSP 只有 25.6%(=随机) -> 高重叠只是"共同主源",不是判别方向。

本节回答最后一个可能:**多频带拼接(FBCSP)能否超过宽带?**
若不能,则"频带结构化共享空间前端"这条架构路线判死。

纯 CPU、不训练。
"""
import sys
import time
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
FS = 250
W0, W1 = 375, 1126
K_CSP = 8
RIDGE = 1e-4

# 通带定义(单频带或"两段带通相加"构成的挖空带)
PB = {
    '8-30':      [(8, 30)],
    '8-12':      [(8, 12)],
    '12-16':     [(12, 16)],
    '13-18':     [(13, 18)],
    '16-20':     [(16, 20)],
    '20-24':     [(20, 24)],
    '24-28':     [(24, 28)],
    '8-13+18-30': [(8, 13), (18, 30)],
}
# 每个组合 = 若干通带的共享 CSP 对数方差特征拼接
COMBOS = [
    ('8-30 宽带(现模型等效)', ['8-30']),
    ('8-12 mu',               ['8-12']),
    ('13-18 死区',            ['13-18']),
    ('20-28 hi-beta',         ['20-24', '24-28']),
    ('8-12 + 20-28 拼接',     ['8-12', '20-24', '24-28']),
    ('五子带全拼接(FBCSP)',   ['8-12', '12-16', '16-20', '20-24', '24-28']),
    ('挖掉 13-18 的宽带',     ['8-13+18-30']),
]


def bp(X, lo, hi):
    b, a = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band')
    return filtfilt(b, a, X, axis=-1)


def filt(X, key):
    """按通带列表相加得到带限信号"""
    out = None
    for lo, hi in PB[key]:
        s = bp(X, lo, hi)
        out = s if out is None else out + s
    return out


def trial_cov(S):
    C = np.einsum('nct,ndt->ncd', S, S) / S.shape[2]
    return C + RIDGE * np.trace(C, axis1=1, axis2=2)[:, None, None] / 22 * np.eye(22)


def csp(C, y, k=K_CSP, n_cls=4, eps=1e-6):
    # reg 必须与 C 自身同量级:trial_cov 里的 RIDGE 是 1e-4×平均特征值(~5e-16),
    # 而原先这里加的绝对 1e-6 比 Cb 的最大特征值(~1e-10)还大 4 个数量级,
    # 会把 Cb 整个抹掉 ⇒ CSP 退化成类平均协方差的 PCA。2026-09-24 改为相对 ridge。
    reg = eps * np.trace(C.mean(0)) / C.shape[-1] * np.eye(C.shape[-1])
    Ws = []
    for c in range(n_cls):
        m = y == c
        w, V = np.linalg.eigh(np.linalg.solve(C[~m].mean(0) + reg, C[m].mean(0)))
        Ws.append(V[:, np.argsort(w)[::-1][:k // n_cls + 1]])
    return np.concatenate(Ws, axis=1)[:, :k]


def logvar(W, S):
    # 地板同样改为相对量:绝对 1e-12 会把低功率滤波器的方差压掉最多 ~0.3 nat(实测)
    v = np.var(np.einsum('ck,nct->nkt', W, S), axis=-1)
    return np.log(v + 1e-9 * v.mean())


def lda():
    return make_pipeline(StandardScaler(),
                         LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))


def main():
    t0 = time.time()
    sd, _ = load_preprocessed_dataset(str(DATA))
    n = len(sd)
    Ys = [np.asarray(y) for _, y in sd]

    used = sorted({k for _, ks in COMBOS for k in ks})
    sig, cov = {}, {}
    for key in used:
        sig[key], cov[key] = [], []
        for X, _ in sd:
            S = filt(np.asarray(X, dtype=np.float64), key)[:, :, W0:W1]
            cov[key].append(trial_cov(S))
            sig[key].append(S.astype(np.float32))
        print('  通带 %-11s 完成 (%.0fs)' % (key, time.time() - t0))

    print('\n共享 CSP 跨被试迁移(leave-one-subject-out,9 折);"被试内"用测试被试自己的 CSP 5 折 CV')
    print('   组合                       维数  共享迁移   被试内   衰减    最差折  最好折')
    for name, keys in COMBOS:
        accs, within = [], []
        for H in range(n):
            tr = [i for i in range(n) if i != H]
            yp = np.concatenate([Ys[i] for i in tr])
            Ftr, Fte = [], []
            for key in keys:
                W = csp(np.concatenate([cov[key][i] for i in tr], 0), yp)
                Ftr.append(np.concatenate([logvar(W, sig[key][i]) for i in tr], 0))
                Fte.append(logvar(W, sig[key][H]))
            clf = lda().fit(np.concatenate(Ftr, 1), yp)
            accs.append(clf.score(np.concatenate(Fte, 1), Ys[H]))
            # 被试内:自己的 CSP + 自己的协方差,5 折 CV
            Fs, Ws = [], []
            for key in keys:
                Wh = csp(cov[key][H], Ys[H])
                Fs.append(logvar(Wh, sig[key][H]))
            Fh = np.concatenate(Fs, 1)
            skf = StratifiedKFold(5, shuffle=True, random_state=42)
            within.append(np.mean([lda().fit(Fh[a], Ys[H][a]).score(Fh[b], Ys[H][b])
                                   for a, b in skf.split(Fh, Ys[H])]))
        a = np.array(accs) * 100
        w = np.mean(within) * 100
        print('   %-26s %3d   %5.1f%%   %5.1f%%  %5.2f   %5.1f   %5.1f'
              % (name, sum(K_CSP for _ in keys), a.mean(), w, a.mean() / w, a.min(), a.max()))
    print('\n总耗时 %.0fs' % (time.time() - t0))


if __name__ == '__main__':
    main()
