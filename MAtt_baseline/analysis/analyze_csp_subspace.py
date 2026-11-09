# -*- coding: utf-8 -*-
"""跨被试"最优空间滤波器"子空间重叠诊断。

回答一个前提性问题:**各被试的最优空间滤波器方向是否互相正交?**
- CSP(one-vs-rest)给出被试自己的最优空间滤波器组(经典意义)。
- 用 tr(P_i P_j)/k = 平均 cos²(主角) 度量两个 k 维子空间的重叠:
    1.0 = 完全相同;  k/d = 随机子空间的期望重叠。
- 关键是要有两条基线,否则绝对数字无法解释:
    ① 随机 k 维子空间(理论上 = k/d);
    ② **被试内 test-retest**(同一被试劈半各算一次 CSP)——这是本测量的可信上限。
另附:被试内模型 conv1 权重的"滤波器级最佳匹配 |cos|" + 随机基线
(conv1 是 22×22 满秩,子空间重叠恒为 1,只能按滤波器逐个比)。
"""
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.linalg import eigh

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
K = 8          # 4 类 × (最大1 + 最小1)
N_CLASSES = 4


def load_all():
    sd, _ = load_preprocessed_dataset(str(DATA))
    return [(np.asarray(X, dtype=np.float64), np.asarray(y)) for X, y in sd]


def trial_covs(X):
    return np.einsum('nct,ndt->ncd', X, X) / X.shape[-1]


def csp(X, y, per_class=1, eps=1e-6):
    """one-vs-rest CSP:每类取广义特征问题两端各 per_class 个向量"""
    covs = [trial_covs(X[y == c]).mean(0) for c in range(N_CLASSES)]
    d = covs[0].shape[0]
    reg = eps * np.trace(sum(covs) / N_CLASSES) / d * np.eye(d)
    W = []
    for c in range(N_CLASSES):
        Crest = sum(covs[i] for i in range(N_CLASSES) if i != c) / (N_CLASSES - 1)
        _, V = eigh(covs[c], Crest + reg)
        W.append(V[:, -per_class:])
        W.append(V[:, :per_class])
    return np.linalg.qr(np.concatenate(W, axis=1))[0]


def overlap(Q1, Q2):
    """tr(P1 P2)/k = 平均 cos²(主角);返回 (重叠, 主角角度数组)"""
    s = np.linalg.svd(Q1.T @ Q2, compute_uv=False)
    return float(np.mean(s ** 2)), np.degrees(np.arccos(np.clip(s, -1.0, 1.0)))


def rand_overlap(k, d, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    v = [np.mean(np.linalg.svd(np.linalg.qr(rng.normal(size=(d, k)))[0].T @
                               np.linalg.qr(rng.normal(size=(d, k)))[0],
                               compute_uv=False) ** 2) for _ in range(n)]
    return float(np.mean(v)), float(np.std(v))


def main():
    data = load_all()
    d = data[0][0].shape[1]
    print(f'被试数={len(data)}  通道={d}  每被试试次={[len(y) for _, y in data]}')
    print(f'CSP 每被试滤波器数 k={K}\n')

    Q = [csp(X, y) for X, y in data]

    vals, angs = [], []
    for i in range(len(Q)):
        for j in range(i + 1, len(Q)):
            o, a = overlap(Q[i], Q[j])
            vals.append(o)
            angs.append(a.mean())
    rk, rs = rand_overlap(K, d)
    ws = []
    for X, y in data:
        p = np.random.default_rng(0).permutation(len(y))
        h = len(y) // 2
        ws.append(overlap(csp(X[p[:h]], y[p[:h]]), csp(X[p[h:]], y[p[h:]]))[0])

    print('子空间重叠 tr(PiPj)/k   (1.0=同一子空间,  随机期望 = k/d = %.3f)' % (K / d))
    print(f'  跨被试 (36 对)       {np.mean(vals):.3f} ± {np.std(vals):.3f}   平均主角度 {np.mean(angs):.1f}°')
    print(f'  随机子空间基线        {rk:.3f} ± {rs:.3f}')
    print(f'  被试内 test-retest    {np.mean(ws):.3f} ± {np.std(ws):.3f}   <- 本测量的可信上限')
    print()
    print('结论阈值:跨被试 ≈ 随机基线 → 各被试最优空间滤波器与随机子空间不可区分(正交成立);')
    print('          跨被试 ≈ 被试内上限 → 存在共享结构,瓶颈在优化而非几何。')
    print()

    # ---- 被试内模型的 conv1 权重:滤波器级最佳匹配 ----
    print('被试内 conv1 权重滤波器级最佳匹配 |cos|(22 维通道向量,每被试 22 个滤波器)')
    paths = {3: 'results/profile_final/final_within_S3_best_acc.pt',
             9: 'results/profile_final/final_within_S9_best_acc.pt'}
    Wl = {}
    for s, p in paths.items():
        sd = torch.load(BASE / p, map_location='cpu', weights_only=True)
        W = sd['conv1.weight'].reshape(22, -1).numpy().astype(np.float64)
        Wl[s] = W / np.linalg.norm(W, axis=1, keepdims=True)
    G = np.abs(Wl[3] @ Wl[9].T)
    obs = float(G.max(1).mean())
    rng = np.random.default_rng(0)
    null = []
    for _ in range(3000):
        A = rng.normal(size=(22, 22)); A /= np.linalg.norm(A, axis=1, keepdims=True)
        B = rng.normal(size=(22, 22)); B /= np.linalg.norm(B, axis=1, keepdims=True)
        null.append(np.abs(A @ B.T).max(1).mean())
    print(f'  S3 vs S9 实测        {obs:.3f}')
    print(f'  随机滤波器组基线      {np.mean(null):.3f} ± {np.std(null):.3f}')
    print(f'  (满秩 22×22,子空间重叠恒为 1,只能逐滤波器比)')


if __name__ == '__main__':
    main()
