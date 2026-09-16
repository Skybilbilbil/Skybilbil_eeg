# -*- coding: utf-8 -*-
"""解剖共享基测验:跨被试的"类别判别方向"是否共享空间平滑结构?

诊断已确认各被试的 CSP 子空间近乎正交(重叠 0.399 vs 随机 0.364)。本脚本检验
一个**决定后续算法方向**的问题:这种正交是"被试各用不同的平滑空间模式"(→ 一个
固定的解剖基就能把它们对齐,共享基有救),还是"差异根本不在平滑与否"(→ 所有
'共享/对齐方向'的算法都死)?

做法:把各被试的 CSP 子空间投影到 scalp 图 Laplacian 的**最低频 m 个模式**上,
再测跨被试两两重叠;与**同维数的随机基**做对照(控制维度效应)。
  - 解剖基 >> 随机基  -> 共享结构是空间平滑的,固定解剖基可取
  - 解剖基 ≈ 随机基   -> 被试差异与平滑性无关,该方向判死
"""
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
import analyze_csp_subspace as A  # noqa: E402
from mAtt.mAtt import BCI2A_CHANNELS  # noqa: E402


def laplacian_basis(channels=BCI2A_CHANNELS):
    """scalp 2D 坐标 -> 高斯邻接 -> 归一化 Laplacian -> 特征向量(低频在前)"""
    from mne.channels import make_standard_montage
    mont = make_standard_montage('standard_1020')
    pos = np.array([mont.get_positions()['ch_pos'][c] for c in channels], dtype=float)
    pos = pos[:, :2]                       # 俯视 2D 投影,避免外接球畸变
    D = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
    sig = np.median(D[D > 0])
    Adj = np.exp(-(D ** 2) / (2 * sig ** 2))
    np.fill_diagonal(Adj, 0.0)
    deg = Adj.sum(1)
    Ln = np.eye(len(channels)) - Adj / np.sqrt(deg[:, None] * deg[None, :])
    w, B = np.linalg.eigh(Ln)              # 特征值小 = 空间越平滑
    return w, B, sig


def main():
    w, B, sig = laplacian_basis()
    print(f'通道 {len(BCI2A_CHANNELS)} 个, 高斯 sigma={sig:.3f} m')
    print(f'Laplacian 特征值最低 6 个: {w[:6].round(3)}\n')

    sd, _ = A.load_preprocessed_dataset(str(A.DATA))
    Q = [A.csp(np.asarray(X, dtype=np.float64), np.asarray(y)) for X, y in sd]
    k = Q[0].shape[1]
    raw = np.mean([A.overlap(Q[i], Q[j])[0] for i in range(9) for j in range(i + 1, 9)])
    print(f'各被试 CSP 子空间维数 k={k};原始跨被试重叠(不投影) = {raw:.3f}')
    print(f'(随机 k 维子空间基线 = k/d = {k / 22:.3f})\n')

    def proj_overlap(Bm):
        Bm = np.linalg.qr(Bm)[0]
        Qp = [np.linalg.qr(Bm @ (Bm.T @ q))[0] for q in Q]   # 投影到 Bm 张成的子空间
        return np.mean([A.overlap(Qp[i], Qp[j])[0]
                        for i in range(9) for j in range(i + 1, 9)])

    rng = np.random.default_rng(0)
    print('  m    解剖低通基    随机 m 维基(200 次)    解剖-随机')
    for m in [10, 12, 14, 16, 18]:
        ov_a = proj_overlap(B[:, :m])
        ov_r = float(np.mean([proj_overlap(rng.normal(size=(22, m))) for _ in range(200)]))
        print(f' {m:3d}      {ov_a:.3f}          {ov_r:.3f}           {ov_a - ov_r:+.3f}')
    print('\n判据:解剖低通基明显高于随机同维基 -> 各被试判别方向共享空间平滑结构,')
    print('      固定解剖基(即 conv1 参数化 W=P·V)有立足点;两者相当则该方向判死。')


if __name__ == '__main__':
    main()
