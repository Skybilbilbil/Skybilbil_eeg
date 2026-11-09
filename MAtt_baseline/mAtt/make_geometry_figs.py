#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成《黎曼几何讲义》的全部插图（8 张），纯 CPU，秒级。

图目录: MAtt_baseline/mAtt/geometry_figs/
所有图都由本脚本现场计算绘制，不引用外部图片资源，故可随文档一起复现。
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Arc
from scipy.linalg import expm, fractional_matrix_power, logm

HERE = Path(__file__).resolve().parent
OUT = HERE / 'geometry_figs'
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.sans-serif': ['Noto Sans CJK JP', 'Noto Sans CJK SC', 'Droid Sans Fallback', 'DejaVu Sans'],
    'axes.unicode_minus': False,
    'font.size': 10,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': ':',
    'figure.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.dpi': 160,
})
C_EUC = '#1f77b4'      # 欧氏
C_LE = '#d62728'       # Log-Euclidean
C_AIRM = '#2ca02c'     # 仿射不变
C_SPD = '#ff7f0e'      # SPD 流形
C_TAN = '#9467bd'      # 切空间

rng = np.random.default_rng(7)


def le_geo(a, b, t):
    return expm((1 - t) * logm(a) + t * logm(b))


def airm_geo(a, b, t):
    w = fractional_matrix_power(a, 0.5)
    wi = np.linalg.inv(w)
    return w @ fractional_matrix_power(wi @ b @ wi, t) @ w


# =====================================================================
# 图 0: 从一组数到空间里的一个点（零基础预备章的配图）
# =====================================================================
def fig00():
    fig = plt.figure(figsize=(13.4, 4.9))
    C_PT = '#d62728'
    C_AX = '#37474f'

    # ---- (a) 1 个数 = 数轴上的一个点 ----
    ax = fig.add_axes([0.035, 0.64, 0.19, 0.24])
    ax.axis('off')
    ax.plot([0, 3.2], [0.5, 0.5], color=C_AX, lw=2)
    for v in (0, 1, 2, 3):
        ax.plot([v, v], [0.44, 0.56], color=C_AX, lw=1.4)
        ax.text(v, 0.30, str(v), ha='center', fontsize=9, color=C_AX)
    ax.plot([2.4], [0.5], 'o', color=C_PT, ms=9, zorder=5)
    ax.text(2.4, 0.68, '2.4', ha='center', fontsize=10, color=C_PT)
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(0.1, 1.0)
    ax.set_title('(a) 1 个数 = 1 个点', fontsize=10.5)

    # ---- (b) 2 个数 = 平面上的一个点 ----
    ax = fig.add_axes([0.265, 0.60, 0.185, 0.30])
    ax.annotate('', xy=(1.25, 0), xytext=(-0.08, 0),
                arrowprops=dict(arrowstyle='-|>', color=C_AX, lw=1.6))
    ax.annotate('', xy=(0, 1.25), xytext=(0, -0.08),
                arrowprops=dict(arrowstyle='-|>', color=C_AX, lw=1.6))
    ax.plot([0.85], [0.62], 'o', color=C_PT, ms=9, zorder=5)
    ax.plot([0.85, 0.85], [0, 0.62], ls=':', color=C_PT, lw=1.2)
    ax.plot([0, 0.85], [0.62, 0.62], ls=':', color=C_PT, lw=1.2)
    ax.text(0.9, 0.66, '(身高, 体重)', fontsize=8.6, color=C_PT)
    ax.text(1.18, -0.16, '身高', fontsize=8.6, color=C_AX)
    ax.text(-0.16, 1.10, '体重', fontsize=8.6, color=C_AX)
    ax.set_xlim(-0.25, 1.45)
    ax.set_ylim(-0.25, 1.45)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title('(b) 2 个数 = 1 个点', fontsize=10.5)

    # ---- (c) 3 个数 = 三维空间里的一个点 ----
    ax = fig.add_axes([0.475, 0.56, 0.205, 0.38], projection='3d')
    ax.plot([0, 1.15], [0, 0], [0, 0], color=C_AX, lw=1.4)
    ax.plot([0, 0], [0, 1.15], [0, 0], color=C_AX, lw=1.4)
    ax.plot([0, 0], [0, 0], [0, 1.15], color=C_AX, lw=1.4)
    ax.scatter([0.72], [0.55], [0.68], color=C_PT, s=60, depthshade=False)
    ax.plot([0.72, 0.72], [0.55, 0.55], [0, 0.68], ls=':', color=C_PT, lw=1.1)
    ax.text(0.78, 0.58, 0.72, '(身高,体重,年龄)', fontsize=8.2, color=C_PT)
    ax.set_xlim(0, 1.25)
    ax.set_ylim(0, 1.25)
    ax.set_zlim(0, 1.25)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_box_aspect((1, 1, 1))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_visible(False)
        axis.line.set_color('#b0bec5')
    ax.grid(False)
    ax.view_init(elev=18, azim=-55)
    ax.set_title('(c) 3 个数 = 1 个点', fontsize=10.5, pad=-2)

    # ---- (d) 矩阵 = 一串数 = 高维空间里的一个点 ----
    ax = fig.add_axes([0.70, 0.56, 0.285, 0.40])
    ax.axis('off')
    ax.add_patch(FancyBboxPatch((0.02, 0.42), 0.26, 0.34,
                                boxstyle='round,pad=0.02', fc='#eceff1',
                                ec='#37474f', lw=1.2))
    ax.text(0.15, 0.585, '1.0   0.5', ha='center', fontsize=11)
    ax.text(0.15, 0.475, '0.5   3.0', ha='center', fontsize=11)
    ax.text(0.15, 0.82, '一个 2×2 矩阵', ha='center', fontsize=9)
    ax.text(0.15, 0.33, '里面有 4 个数', ha='center', fontsize=9, color='#546e7a')
    ax.annotate('', xy=(0.42, 0.59), xytext=(0.30, 0.59),
                arrowprops=dict(arrowstyle='-|>', color=C_PT, lw=2))
    ax.text(0.36, 0.66, '摊平', ha='center', fontsize=9, color=C_PT)
    ax.text(0.44, 0.585, '(1.0, 0.5, 0.5, 3.0)', fontsize=11, va='center')
    ax.text(0.44, 0.44, '四个数', fontsize=9, color='#546e7a')
    ax.text(0.44, 0.30, '＝ 四维空间里的一个点', fontsize=10.5, color=C_PT)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0.15, 1.0)
    ax.set_title('(d) 矩阵也是一串数，所以也是"点"', fontsize=10.5)

    # ---- (e) 不是每个点都合法 ----
    ax = fig.add_axes([0.035, 0.03, 0.95, 0.45])
    ax.axis('off')
    ax.text(0.5, 0.92, '但不是每个点都是合法的协方差矩阵：它必须"正定"（特征值全为正）',
            ha='center', fontsize=10.5, color='#263238')

    ax.add_patch(FancyBboxPatch((0.06, 0.22), 0.30, 0.52,
                                boxstyle='round,pad=0.03', fc='#e8f5e9',
                                ec='#2e7d32', lw=1.4))
    ax.text(0.21, 0.63, '[[1.0, 0.5], [0.5, 3.0]]', ha='center', fontsize=11)
    ax.text(0.21, 0.45, '特征值 0.42 和 3.58', ha='center', fontsize=9.5)
    ax.text(0.21, 0.29, '两个都是正数 → 合法', ha='center', fontsize=9.5,
            color='#2e7d32')

    ax.add_patch(FancyBboxPatch((0.40, 0.22), 0.30, 0.52,
                                boxstyle='round,pad=0.03', fc='#ffebee',
                                ec='#c62828', lw=1.4))
    ax.text(0.55, 0.63, '[[1.0, 2.0], [2.0, 1.0]]', ha='center', fontsize=11)
    ax.text(0.55, 0.45, '特征值 3.00 和 −1.00', ha='center', fontsize=9.5)
    ax.text(0.55, 0.29, '有负数 → 不是协方差', ha='center', fontsize=9.5,
            color='#c62828')

    ax.annotate('', xy=(0.755, 0.48), xytext=(0.72, 0.48),
                arrowprops=dict(arrowstyle='-|>', color='#546e7a', lw=1.6))
    ax.text(0.76, 0.60, '所有"合法点"', fontsize=10)
    ax.text(0.76, 0.47, '组成的区域', fontsize=10)
    ax.text(0.76, 0.31, '= SPD 流形', fontsize=11.5, color=C_PT, weight='bold')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0.1, 1.05)

    fig.savefig(OUT / 'fig00_matrix_to_point.png', bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 图 1: SPD 锥、欧氏弦 vs 测地线、行列式膨胀
# =====================================================================
def fig01():
    fig = plt.figure(figsize=(12.8, 4.9))

    # --- (a) 2x2 SPD 锥 ---
    ax = fig.add_axes([0.03, 0.07, 0.42, 0.82], projection='3d')
    g = np.linspace(0.06, 2.2, 30)
    x, y = np.meshgrid(g, g)
    z = np.sqrt(x * y)
    ax.plot_surface(x, y, z, alpha=0.12, color=C_SPD, linewidth=0, antialiased=True)
    ax.plot_wireframe(x, y, z, color=C_SPD, lw=0.35, alpha=0.30, rstride=3, cstride=3)

    # 两个"高度相关但方向相反"的 2x2 协方差（共享特征基 => LE 测地线 = 特征值的几何插值）
    a = np.array([[1.0, 0.9], [0.9, 1.0]])          # det = 0.19
    b = np.array([[1.0, -0.9], [-0.9, 1.0]])        # det = 0.19
    ts = np.linspace(0, 1, 80)
    chord = np.array([(1 - t) * a + t * b for t in ts])
    geo = np.array([le_geo(a, b, t) for t in ts])

    ax.plot(chord[:, 0, 0], chord[:, 1, 1], chord[:, 0, 1], color=C_EUC, lw=2.6,
            label='欧氏直线 (1-t)A + tB')
    ax.plot(geo[:, 0, 0], geo[:, 1, 1], geo[:, 0, 1], color=C_LE, lw=2.6, ls='--',
            label='LE 测地线 exp((1-t)logA + t logB)')
    ax.scatter([a[0, 0]], [a[1, 1]], [a[0, 1]], s=80, color='k', depthshade=False, zorder=6)
    ax.scatter([b[0, 0]], [b[1, 1]], [b[0, 1]], s=80, color='k', depthshade=False, zorder=6)
    ax.text(a[0, 0], a[1, 1], a[0, 1] + 0.30, 'A', fontsize=14, weight='bold')
    ax.text(b[0, 0], b[1, 1], b[0, 1] - 0.55, 'B', fontsize=14, weight='bold')
    ax.set_xlim(0, 2.1)
    ax.set_ylim(0, 2.1)
    ax.set_zlim(-1.5, 1.6)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([0.5, 1.0, 1.5, 2.0])
    ax.set_zticks([-1.0, 0.0, 1.0])
    ax.set_box_aspect((1.0, 1.0, 1.15))
    ax.tick_params(labelsize=8, pad=-1)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_zlabel('')
    ax.text2D(0.02, 0.02, r'横轴 $C_{11}$、纵轴 $C_{22}$、竖轴 $C_{12}$'
              r'；锥内 = 正定', transform=ax.transAxes, fontsize=8.4)
    ax.set_title('(a) 2×2 SPD 锥：两条连法都落在锥内，\n但只有虚线是“测地线”', fontsize=11, pad=0)
    ax.view_init(elev=18, azim=-62)
    ax.legend(loc='upper right', fontsize=7.6, framealpha=0.92, bbox_to_anchor=(1.02, 1.0))
    ax.text2D(0.60, 0.42, '中点 = 0.436·I', transform=ax.transAxes, fontsize=8.2, color=C_LE)
    ax.text2D(0.62, 0.66, '中点 = I', transform=ax.transAxes, fontsize=8.2, color=C_EUC)

    # --- (b) 行列式沿两条路径 ---
    ax2 = fig.add_axes([0.57, 0.15, 0.40, 0.70])
    det_chord = np.array([np.linalg.det(m) for m in chord])
    det_geo = np.array([np.linalg.det(m) for m in geo])
    mid = len(ts) // 2
    ratio = det_chord[mid] / det_geo[mid]
    ax2.plot(ts, det_chord, color=C_EUC, lw=2.8, label='欧氏弦上的 det（凭空被抬高）')
    ax2.plot(ts, det_geo, color=C_LE, lw=2.8, ls='--',
             label=r'LE 测地线上 $\det = \det A^{1-t}\det B^{t}$（几何插值）')
    ax2.set_xlabel('插值参数 t')
    ax2.set_ylabel('det C')
    ax2.set_ylim(0.10, 1.18)
    ax2.set_title('(b) 行列式“膨胀”(swelling)：两个 det 都等于 0.19 的协方差，\n欧氏中点却让 det 变成 1.0', fontsize=11)
    ax2.legend(fontsize=8.2, loc='upper center')
    ax2.annotate('', xy=(0.5, det_chord[mid]), xytext=(0.5, det_geo[mid]),
                 arrowprops=dict(arrowstyle='<->', color='#37474f', lw=1.5))
    ax2.text(0.535, 0.55, f'×{ratio:.1f}', fontsize=13, color='#37474f')
    ax2.text(0.5, 0.215, '虚线：det 全程不变（相对变化被正确表达）', ha='center', fontsize=8.8, color=C_LE)
    fig.savefig(OUT / 'fig01_spd_cone_swelling.png', bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 图 2: 矩阵对数 = 把乘法变成加法的坐标系；距离的压缩；绝对地板的坑
# =====================================================================
def fig02():
    fig = plt.figure(figsize=(13.2, 3.9))

    ax = fig.add_subplot(1, 3, 1)
    ax.set_xlim(-0.2, 8.6)
    ax.set_ylim(-0.25, 4.6)
    ax.axis('off')
    ax.plot([0, 8.4], [3.2, 3.2], color='#333', lw=2)
    ax.text(8.55, 3.2, '×2', fontsize=10, va='center', color='#333')
    for v in (1, 2, 4, 8):
        ax.plot([np.log(v), np.log(v)], [3.0, 3.42], color='#333', lw=1.6)
        ax.text(np.log(v), 3.55, f'{v}', fontsize=11, ha='center')
    ax.text(0, 4.15, 'SPD 上的乘法（SPD 本身，尺度是乘性的）', fontsize=9.6, color='#333')

    ax.plot([0, 8.4], [1.0, 1.0], color=C_LE, lw=2)
    ax.text(8.55, 1.0, '+log2', fontsize=10, va='center', color=C_LE)
    for v in (1, 2, 4, 8):
        ax.plot([np.log(v), np.log(v)], [0.78, 1.22], color=C_LE, lw=1.6)
        ax.text(np.log(v), 0.5, f'{np.log(v):.2f}', fontsize=10, ha='center', color=C_LE)
    ax.text(0, 1.75, r'$\log$ 之后：对称矩阵的加法（只要会线性代数就行）', fontsize=9.6, color=C_LE)
    for v in (2, 4, 8):
        ax.annotate('', xy=(np.log(v), 1.25), xytext=(np.log(v), 2.95),
                    arrowprops=dict(arrowstyle='-|>', color='#999', lw=1.2, ls=':'))
    ax.set_title('(a) 矩阵对数 = 坐标系：等比的乘法 → 等距的加法', fontsize=11)

    ax = fig.add_subplot(1, 3, 2)
    c = np.linspace(0.05, 6, 400)
    ax.plot(c, np.abs(c - 1), color=C_EUC, lw=2.4, label=r'欧氏 $|c-1|$（线性、无上界）')
    ax.plot(c, np.abs(np.log(c)), color=C_LE, lw=2.4, ls='--', label=r'LE $|\log c|$（对数、压缩）')
    ax.set_yscale('symlog', linthresh=0.5)
    ax.set_xlabel('尺度因子 c（把 C 换成 cC）')
    ax.set_ylabel('到单位阵的距离')
    ax.annotate('c=1→2 与 c=100→200\n"相对变化"相同，LE 给出同一个距离',
                xy=(2, np.log(2)), xytext=(2.05, 0.045), fontsize=9,
                arrowprops=dict(arrowstyle='->', color='#444'))
    ax.set_title('(b) 距离只该看"相对变化"：LE 的尺度不变性', fontsize=11)
    ax.legend(fontsize=8.5, loc='upper left')

    ax = fig.add_subplot(1, 3, 3)
    n = 22
    q, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam = np.exp(np.linspace(np.log(1e-3), np.log(1.0), n))
    c_raw = (q @ np.diag(lam) @ q.T) * 1e-10
    bad = np.linalg.eigvalsh(0.5 * (logm(c_raw + 1e-6 * np.eye(n)) + logm(c_raw + 1e-6 * np.eye(n)).T))
    good = np.linalg.eigvalsh(
        0.5 * (logm(c_raw + 1e-6 * np.trace(c_raw) / n * np.eye(n))
               + logm(c_raw + 1e-6 * np.trace(c_raw) / n * np.eye(n)).T))
    ax.plot(np.sort(bad) - np.mean(bad), 'o-', ms=3.4, color=C_EUC, lw=1.4,
            label='绝对地板 +1e-6·I（量级大 4 个数量级）')
    ax.plot(np.sort(good) - np.mean(good), 's-', ms=3.4, color=C_LE, lw=1.4,
            label='相对地板 +1e-6·tr(C)/n·I')
    ax.set_xlabel('排序后的特征值序号')
    ax.set_ylabel('log 特征值（已减均值）')
    ax.set_title('(c) 地板必须相对：否则 log 退化成线性映射，几何缺席', fontsize=11)
    ax.legend(fontsize=8, loc='upper left')
    fig.tight_layout()
    fig.savefig(OUT / 'fig02_log_map.png', bbox_inches='tight')
    plt.close(fig)


# =====================================================================
# 图 3: 三种度量 —— 对角切片上 LE=AIRM；分叉只由"不交换"决定
# =====================================================================
def fig03():
    fig = plt.figure(figsize=(12.8, 4.7))

    # ---- (a) 对角切片：log 坐标把弯曲拉平 ----
    ax = fig.add_axes([0.07, 0.15, 0.40, 0.70])
    a = np.diag([1.0, 9.0])
    b = np.diag([9.0, 1.0])
    ts = np.linspace(0, 1, 80)
    # a、b 都是对角阵，直接取对角元（不能用 eigvalsh：它会把特征值升序重排，路径会断开）
    chord = np.array([[((1 - t) * a + t * b)[0, 0], ((1 - t) * a + t * b)[1, 1]] for t in ts])
    geo = np.array([[le_geo(a, b, t)[0, 0], le_geo(a, b, t)[1, 1]] for t in ts])
    ax.plot(chord[:, 0], chord[:, 1], color=C_EUC, lw=2.8, label='欧氏直线（在 log 坐标里是弯的）')
    ax.plot(geo[:, 0], geo[:, 1], color=C_LE, lw=2.8, ls='--',
            label='LE 测地线（在 log 坐标里是直线）= 本切片上的 AIRM 测地线')
    ax.scatter([1, 9], [9, 1], s=90, color='k', zorder=6)
    ax.text(0.95, 11.6, 'A = diag(1,9)', fontsize=9.6)
    ax.text(6.4, 0.72, 'B = diag(9,1)', fontsize=9.6)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.7, 13.0)
    ax.set_ylim(0.55, 16.0)
    ax.set_xlabel(r'特征值 $\lambda_1$（对数轴）')
    ax.set_ylabel(r'特征值 $\lambda_2$（对数轴）')
    ax.set_title('(a) 只有 2 个自由度时可以直接画：log 坐标把弯曲空间拉平', fontsize=11)
    ax.legend(fontsize=8.0, loc='lower left')

    # ---- (b) LE 与 AIRM 的分歧只由"特征向量不交换"决定 ----
    ax = fig.add_axes([0.57, 0.15, 0.40, 0.70])
    n = 5
    q1, _ = np.linalg.qr(rng.normal(size=(n, n)))
    lam1 = np.array([1.0, 2.0, 3.0, 5.0, 8.0])
    lam2 = np.array([5.0, 0.6, 2.5, 1.4, 3.6])
    thetas = np.linspace(0, np.pi / 2, 40)
    gaps = []
    for th in thetas:
        rot = np.eye(n)
        c, s = np.cos(th), np.sin(th)
        rot[0, 0] = rot[1, 1] = c
        rot[0, 1] = -s
        rot[1, 0] = s
        aa = q1 @ np.diag(lam1) @ q1.T
        bb = q1 @ rot @ np.diag(lam2) @ rot.T @ q1.T
        gaps.append(max(np.linalg.norm(le_geo(aa, bb, t) - airm_geo(aa, bb, t)) for t in ts))
    ax.plot(np.degrees(thetas), gaps, color=C_AIRM, lw=2.8)
    ax.set_xlabel('A 与 B 的特征基之间的夹角（度）')
    ax.set_ylabel(r'两条测地线的最大逐点差 $\|\cdot\|_F$')
    ax.set_ylim(-0.004, 0.095)
    ax.set_title('(b) LE 与 AIRM 的分歧只来自“特征向量不交换”', fontsize=11)
    ax.annotate('夹角 0°：两者完全重合\n（共享特征基 = 可交换）', xy=(1.0, 0.004), xytext=(9, 0.045),
                fontsize=8.8, arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))
    ax.annotate('夹角越大，两种“平均”越不同', xy=(52, gaps[22]), xytext=(30, 0.078),
                fontsize=8.8, arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))
    fig.savefig(OUT / 'fig03_three_metrics.png', bbox_inches='tight')
    plt.close(fig)

# =====================================================================
# 图 4: 切空间 / 指数映射 / 对数映射
# =====================================================================
def fig04():
    fig = plt.figure(figsize=(12.6, 4.6))

    # ---- (a) 球面：测地线 = 大圆；切平面 = 切空间 ----
    ax = fig.add_axes([0.02, 0.04, 0.40, 0.84], projection='3d')
    R = 2.0
    th = np.linspace(0, 2 * np.pi, 72)
    ph = np.linspace(0, 1.10, 32)
    PH, TH = np.meshgrid(ph, th)
    X = R * np.sin(PH) * np.cos(TH)
    Y = R * np.sin(PH) * np.sin(TH)
    Zc = R * np.cos(PH)
    ax.plot_surface(X, Y, Zc, alpha=0.22, color=C_SPD, linewidth=0, antialiased=True)
    hp = 1.24
    ax.plot_surface(np.array([[-hp, -hp], [hp, hp]]), np.array([[-hp, hp], [-hp, hp]]),
                    np.full((2, 2), R), alpha=0.20, color=C_TAN, linewidth=0)

    p_hat = np.array([0.0, 0.0, 1.0])
    v_hat = np.array([0.85, 0.55, 0.0])
    v_hat = v_hat / np.linalg.norm(v_hat)
    tt = np.linspace(0, 0.62, 40)
    curve = R * (np.cos(tt)[:, None] * p_hat + np.sin(tt)[:, None] * v_hat)
    ax.plot(curve[:, 0], curve[:, 1], curve[:, 2], color=C_LE, lw=3.0)
    ax.plot([0, R * v_hat[0]], [0, R * v_hat[1]], [R, R], color=C_TAN, lw=2.4, ls=':')
    ax.plot([0], [0], [R], 'ko', ms=9)
    ax.plot([curve[-1, 0]], [curve[-1, 1]], [curve[-1, 2]], 'r^', ms=11)

    ax.text2D(0.06, 0.80, r'$P$（流形上的点 = 协方差 $C$）', transform=ax.transAxes, fontsize=9.0)
    ax.text2D(0.46, 0.90, r'切平面 $T_P\mathcal{M}$（= 切空间）', transform=ax.transAxes,
              fontsize=9.0, color=C_TAN)
    ax.text2D(0.42, 0.55, r'$v$：切向量（对称矩阵）', transform=ax.transAxes, fontsize=9.0, color=C_TAN)
    ax.text2D(0.10, 0.09, r'$\mathrm{Exp}_P(v)$：沿测地线走一段（大圆）',
              transform=ax.transAxes, fontsize=9.0, color=C_LE)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.set_box_aspect((1, 1, 0.75))
    ax.view_init(elev=26, azim=-58)
    ax.set_title('(a) 切空间 = 在 P 处"贴住"流形的平面；Exp/Log 在两片空间往返', fontsize=10.6, pad=0)

    # ---- (b) 1 维示意 ----
    ax = fig.add_axes([0.50, 0.13, 0.47, 0.74])
    ax.axhline(0, color='#37474f', lw=1.6)
    ax.axhline(1, color=C_SPD, lw=2.6)
    ax.text(-3.1, 1.10, r'SPD：$(0,\infty)$ 这条射线（1 维时就是"正数"）', fontsize=9, color=C_SPD)
    ax.text(-3.1, -0.55, r'切空间 $T_I$：整条实数轴（对称矩阵）', fontsize=9, color=C_TAN)
    xs = np.linspace(-2.6, 1.55, 200)
    ax.plot(xs, np.exp(xs), color=C_LE, lw=2.2, ls='--',
            label=r'$\exp: x\mapsto e^{x}$（保序、处处可逆）')
    for x0 in (-1.9, -1.2, -0.5, 0.2, 0.9):
        ax.annotate('', xy=(x0, np.exp(x0)), xytext=(x0, 0),
                    arrowprops=dict(arrowstyle='-|>', color='#9e9e9e', lw=1.0))
        ax.plot([x0], [np.exp(x0)], 'o', color=C_SPD, ms=5.5)
        ax.plot([x0], [0], '|', color=C_TAN, ms=9)
    ax.set_xlim(-3.2, 1.95)
    ax.set_ylim(-0.9, 6.2)
    ax.set_xlabel(r'切空间坐标（1 维时就是 $\log\lambda$）')
    ax.set_ylabel(r'流形上的点（1 维时就是 $\lambda$）')
    ax.set_title('(b) 1 维示意：log 把"乘法射线"拉成"加法直线"，且是整体同构', fontsize=10.6)
    ax.legend(fontsize=8.6, loc='upper left')
    fig.savefig(OUT / 'fig04_tangent_exp_log.png', bbox_inches='tight')
    plt.close(fig)



# =====================================================================
# 图 5: 共轭 X→WᵀXW 与 EA 白化；AIRM 等距 vs LE 改变
# =====================================================================
def fig05():
    fig = plt.figure(figsize=(12.6, 4.3))

    ax = fig.add_subplot(1, 3, 1)
    n = 8
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(y, y)
    base = np.exp(-((X - 0.28) ** 2 + (Y - 0.62) ** 2) / 0.05) + \
        0.8 * np.exp(-((X - 0.72) ** 2 + (Y - 0.30) ** 2) / 0.07)
    c0 = base + 0.06 * np.eye(n)
    ax.imshow(c0, cmap='magma')
    ax.set_title('协方差 C（通道×通道）', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    ax = fig.add_subplot(1, 3, 2)
    w = np.linalg.qr(rng.normal(size=(n, 4)))[0]
    c1 = w.T @ c0 @ w
    ax.imshow(c1, cmap='magma')
    ax.set_title(r'$W^{\top}CW$，$W\in\mathrm{St}(8,4)$' + '\n= 换成 4 个空间滤波器的坐标', fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)

    # ---- (c) 距离的变化量：AIRM 数学上恒等，LE 会变 ----
    ax = fig.add_axes([0.71, 0.17, 0.27, 0.66])
    d_airm_d, d_le_d = [], []
    for _ in range(12):
        q1, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        q2, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        aa = q1 @ np.diag(np.exp(rng.normal(scale=0.7, size=6))) @ q1.T
        bb = q2 @ np.diag(np.exp(rng.normal(scale=0.7, size=6))) @ q2.T

        def ad(x, y_):
            w_ = np.linalg.inv(fractional_matrix_power(x, 0.5))
            m_ = w_ @ y_ @ w_
            m_ = 0.5 * (m_ + m_.T)
            return float(np.linalg.norm(np.log(np.maximum(np.linalg.eigvalsh(m_), 1e-300))))

        def ld(x, y_):
            return float(np.linalg.norm(logm(x) - logm(y_)))

        r_ = 0.5 * (aa + bb)
        p_ = np.linalg.inv(fractional_matrix_power(r_, 0.5))
        a2, b2 = p_ @ aa @ p_.T, p_ @ bb @ p_.T
        d_airm_d.append(abs(ad(aa, bb) - ad(a2, b2)))
        d_le_d.append(abs(ld(aa, bb) - ld(a2, b2)))
    idx = np.arange(12)
    ax.bar(idx - 0.2, np.maximum(d_airm_d, 1e-18), 0.4, color=C_AIRM,
           label=r'AIRM：$|d_{\rm before}-d_{\rm after}|$')
    ax.bar(idx + 0.2, d_le_d, 0.4, color=C_LE, alpha=0.85,
           label=r'LE：$|d_{\rm before}-d_{\rm after}|$')
    ax.set_yscale('log')
    ax.set_ylim(1e-18, 1)
    ax.set_xlabel('随机矩阵对')
    ax.set_ylabel('白化前后的距离变化（对数轴）')
    ax.set_title('(c) 同一个共轭：AIRM 不动，LE 动', fontsize=10.4)
    ax.legend(fontsize=6.8, loc='upper right')
    fig.savefig(OUT / 'fig05_congruence_ea.png', bbox_inches='tight')
    plt.close(fig)

def fig06():
    fig = plt.figure(figsize=(12.6, 4.8))

    # ---- (a) p=1：Stiefel = 单位圆 ----
    ax = fig.add_axes([0.02, 0.04, 0.44, 0.86])
    th = np.linspace(0, 2 * np.pi, 500)
    ax.plot(np.cos(th), np.sin(th), color=C_SPD, lw=2.6)
    w = np.array([0.50, 0.8660])
    g = np.array([0.60, 0.10])
    radial = w * (w @ g)
    z = g - radial
    w_lin = w + 0.5 * z
    w_new = w_lin / np.linalg.norm(w_lin)

    ax.plot([w[0] - 0.9 * (-w[1]), w[0] + 0.9 * (-w[1])],
            [w[1] - 0.9 * w[0], w[1] + 0.9 * w[0]], color='#9e9e9e', lw=1.1, ls=':')
    ax.annotate('', xy=w + g, xytext=w,
                arrowprops=dict(arrowstyle='-|>', color=C_EUC, lw=1.9, ls='--'))
    ax.annotate('', xy=w + z, xytext=w,
                arrowprops=dict(arrowstyle='-|>', color=C_AIRM, lw=2.8))
    ax.annotate('', xy=w_new, xytext=w_lin,
                arrowprops=dict(arrowstyle='-|>', color=C_LE, lw=1.8))
    ax.plot([w[0]], [w[1]], 'ko', ms=9, zorder=6)
    ax.plot([w_new[0]], [w_new[1]], 'r^', ms=11, zorder=6)

    ax.text(w[0] - 0.34, w[1] + 0.09, r'$W$', fontsize=12)
    ax.text(w[0] - 0.30, w[1] - 0.10, r'$W^{\top}W=I$', fontsize=9.6)
    ax.text(w[0] + g[0] + 0.04, w[1] + g[1] + 0.06,
            r'$\nabla$：欧氏梯度（指向流形外）', fontsize=8.8, color=C_EUC)
    ax.text(w[0] + z[0] + 0.06, w[1] + z[1] - 0.34,
            r'$Z=\nabla-W\,\mathrm{sym}(W^{\top}\nabla)$', fontsize=8.8, color=C_AIRM)
    ax.text(w[0] + z[0] + 0.06, w[1] + z[1] - 0.47, '投影到切空间', fontsize=8.8, color=C_AIRM)
    ax.text(w_new[0] + 0.06, w_new[1] - 0.26, 'QR 回缩后重新落回圆上', fontsize=8.8, color=C_LE)
    ax.set_xlim(-1.55, 2.05)
    ax.set_ylim(-1.30, 1.55)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('(a) $p=1$ 时 Stiefel 流形就是单位圆：先投影掉法向分量，再拉回流形',
                 fontsize=10.6)

    # ---- (b) 两种更新方式 40 步的对比 ----
    ax = fig.add_axes([0.55, 0.14, 0.42, 0.74])
    w = np.array([[1.0], [0.0]])
    nrm, orth = [], []
    for _ in range(40):
        gg = rng.normal(size=(2, 1))
        scl = 0.35 * (w.T @ gg)
        zz = gg - w @ (0.5 * (scl + scl.T))
        w = w - 0.35 * zz
        q, r = np.linalg.qr(w)
        w = q @ np.diag(np.sign(np.diag(r)))
        nrm.append(float(np.linalg.norm(w)))
        orth.append(float(np.linalg.norm(w.T @ w - np.eye(1))))
    w2 = np.array([[1.0], [0.0]])
    orth_bad = []
    for _ in range(40):
        gg = rng.normal(size=(2, 1))
        w2 = w2 - 0.35 * gg
        orth_bad.append(float(np.linalg.norm(w2.T @ w2 - np.eye(1))))
    ax.semilogy(np.maximum(orth, 1e-17), color=C_AIRM, lw=2.4,
                label=r'投影 + 回缩：$\|W^{\top}W-I\|\approx10^{-16}$（机器精度）')
    ax.semilogy(orth_bad, color=C_EUC, lw=2.4, ls='--',
                label=r'欧氏梯度直接减：$\|W^{\top}W-I\|$ 单调发散')
    ax.semilogy(nrm, color=C_LE, lw=2.0, ls=':',
                label=r'投影 + 回缩：$\|W\|_F$ 恒为 $\sqrt{p}=1$（构造性守恒）')
    ax.set_xlabel('优化步数')
    ax.set_ylabel('偏差（对数轴）')
    ax.set_ylim(1e-17, 1e3)
    ax.set_title('(b) 回缩保证“永远在流形上”；范数守恒是构造性的，不代表没学到', fontsize=10.6)
    ax.legend(fontsize=7.8, loc='center right')
    fig.savefig(OUT / 'fig06_stiefel.png', bbox_inches='tight')
    plt.close(fig)

# =====================================================================
# 图 7: 整套模型的几何层次图
# =====================================================================
def fig07():
    fig, ax = plt.subplots(figsize=(13.8, 5.8))
    ax.set_xlim(0, 13.8)
    ax.set_ylim(0, 5.8)
    ax.axis('off')

    top = [
        ('EEG 试次', '$(B,22,T)$', '#cfd8dc'),
        ('通道先验加权', '逐通道缩放', '#cfd8dc'),
        ('conv1', '空间滤波 $22\\to 22$', '#cfd8dc'),
        ('conv2', '混通道+时序 $22\\to 8$', '#cfd8dc'),
        ('Hann 滑窗', '$M$ 个窗', '#cfd8dc'),
    ]
    bot = [
        ('向量化 + 线性', r'$\mathbb{R}^{d(d+1)/2}$', C_TAN),
        ('切空间 log', r'$\mathrm{SPD}\to\mathrm{Sym}$', C_TAN),
        ('平滑整流', r'$\lambda\mapsto\frac{\lambda+\sqrt{\lambda^2+4\epsilon}}{2}$', C_SPD),
        ('流形注意力', r'$Q,K,V=W^{\top}XW$', C_SPD),
        ('SPDNorm', 'log 域平移 + 缩放', C_SPD),
        ('协方差 + 迹归一化', r'$C=XX^{\top}/\mathrm{tr}$', C_SPD),
    ]

    def draw(x, y, w, h, t, sub, color, fs=9.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.07,rounding_size=0.10',
                                    fc=color, ec='#37474f', lw=1.1, alpha=0.90))
        ax.text(x + w / 2, y + h * 0.64, t, ha='center', va='center', fontsize=fs, weight='bold')
        ax.text(x + w / 2, y + h * 0.26, sub, ha='center', va='center', fontsize=fs - 1.3)

    xs_t = [0.35 + 2.62 * i for i in range(5)]
    xs_b = [0.90 + 2.10 * i for i in range(6)]
    for i, (t, s, c) in enumerate(top):
        draw(xs_t[i], 4.00, 2.30, 0.95, t, s, c)
        if i:
            ax.annotate('', xy=(xs_t[i], 4.47), xytext=(xs_t[i - 1] + 2.30, 4.47),
                        arrowprops=dict(arrowstyle='-|>', color='#455a64', lw=1.4))
    for i, (t, s, c) in enumerate(bot):
        draw(xs_b[i], 1.95, 1.92, 0.95, t, s, c)
        if i:
            ax.annotate('', xy=(xs_b[i], 2.42), xytext=(xs_b[i - 1] + 1.92, 2.42),
                        arrowprops=dict(arrowstyle='-|>', color='#455a64', lw=1.4))
    # U 形拐弯：滑窗 -> 协方差
    ax.annotate('', xy=(xs_b[5] + 0.97, 2.95), xytext=(xs_t[4] + 1.15, 4.00),
                arrowprops=dict(arrowstyle='-|>', color='#455a64', lw=1.6))

    ax.text(0.35, 5.15, '欧氏空间  $\\mathbb{R}^{C\\times T}$', fontsize=11, color='#37474f')
    ax.text(0.90, 1.62, '切空间  $\\mathrm{Sym}(n)$（平坦）', fontsize=10.5, color=C_TAN)
    ax.text(5.10, 1.62, 'SPD 流形  $\\mathcal{P}(n)$（弯曲）', fontsize=10.5, color='#c0620a')
    ax.text(0.35, 0.90,
            '几何要点：① 迹归一化把“绝对能量”换成“相对结构”，数据落在 tr≈1 的曲面上；'
            '② 所有可学习变换都是共轭 $W^{\\top}(\\cdot)W$，天然保正定；'
            '③ 打分 = Log-Euclidean 测地距离 $-\\|\\log Q-\\log K\\|^2$，聚合 = $\\exp(\\sum_j w\\log V_j)$；',
            fontsize=9.2, color='#263238')
    ax.text(0.35, 0.52,
            '④ 只在最后一步“下坡”到切空间做线性读出；'
            '⑤ 正交参数 $W$ 走的是 Stiefel 流形上的投影梯度 + QR 回缩，整条链路有四个空间在轮换。',
            fontsize=9.2, color='#263238')
    ax.set_title('mAtt 的几何层次：一个协方差矩阵从“数字”变成“流形上的点”再变回“数字”',
                 fontsize=12.5, pad=4)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02)
    fig.savefig(OUT / 'fig07_pipeline_geometry.png', bbox_inches='tight')
    plt.close(fig)

# =====================================================================
# 图 8: 三种"平均"的差别
# =====================================================================
def fig08():
    fig = plt.figure(figsize=(12.4, 4.4))
    ax = fig.add_subplot(1, 2, 1)
    n = 3
    pts = []
    for _ in range(8):
        q, _ = np.linalg.qr(rng.normal(size=(n, n)))
        pts.append(q @ np.diag(np.exp(rng.normal(scale=0.55, size=n))) @ q.T)
    eu = sum(pts) / len(pts)
    le = expm(np.mean([logm(p) for p in pts], axis=0))
    m = np.eye(n)
    for _ in range(300):
        wi = np.linalg.inv(fractional_matrix_power(m, 0.5))
        tang = np.mean([logm(wi @ p @ wi) for p in pts], axis=0)
        ms = fractional_matrix_power(m, 0.5)
        m = ms @ expm(tang) @ ms
        if np.linalg.norm(tang) < 1e-13:
            break
    eigs = lambda mm: np.sort(np.linalg.eigvalsh(mm))
    ax.plot(eigs(eu), 'o-', color=C_EUC, lw=2, label=r'欧氏平均 $(1/N)\sum C_i$')
    ax.plot(eigs(le), 's--', color=C_LE, lw=2, label=r'LE 平均 $\exp(\frac{1}{N}\sum\log C_i)$')
    ax.plot(eigs(m), '^:', color=C_AIRM, lw=2, label='AIRM 平均（Karcher 迭代）')
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['1', '2', '3'])
    ax.set_xlabel('特征值序号（从小到大）')
    ax.set_ylabel('特征值')
    ax.set_title('(a) 同一批点、三种"平均"给出三个不同的矩阵', fontsize=11)
    ax.legend(fontsize=8.5)
    ax.annotate('欧氏平均把谱尾抬高了（det 膨胀的另一种看法）', xy=(2, eigs(eu)[2]),
                xytext=(0.25, 1.24), fontsize=8.6,
                arrowprops=dict(arrowstyle='->', color='#444'))

    ax = fig.add_subplot(1, 2, 2)
    dets = [np.linalg.det(eu), np.linalg.det(le), np.linalg.det(m),
            np.exp(np.mean([np.log(np.linalg.det(p)) for p in pts]))]
    cols = [C_EUC, C_LE, C_AIRM, '#7f7f7f']
    bars = ax.bar(['欧氏平均', 'LE 平均', 'AIRM 平均', '各点 det 的几何平均'], dets, color=cols, alpha=0.85)
    for bb, v in zip(bars, dets):
        ax.text(bb.get_x() + bb.get_width() / 2, v * 1.02 if v > 0 else v, f'{v:.3f}',
                ha='center', fontsize=9)
    ax.set_ylabel('平均后矩阵的 det')
    ax.set_ylim(0, 1.55)
    ax.set_title('(b) 欧氏平均把"体积"抬高；两种几何平均都精确守住 det', fontsize=10.6)
    ax.text(1.5, 1.33, '后三者精确相等不是巧合：det 是乘性的，几何均值天然把它守住',
            ha='center', fontsize=8.6, color='#37474f')
    fig.subplots_adjust(wspace=0.28)
    fig.savefig(OUT / 'fig08_means.png', bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    for f in (fig00, fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        f()
        print('ok', f.__name__)
