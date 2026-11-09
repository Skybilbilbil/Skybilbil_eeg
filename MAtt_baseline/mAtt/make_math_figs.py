#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""生成《黎曼几何数学基础》的插图，纯 CPU。输出: MAtt_baseline/mAtt/math_figs/"""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch

HERE = Path(__file__).resolve().parent
OUT = HERE / 'math_figs'
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
    'savefig.bbox': 'tight',
})

C1, C2, C3, C4 = '#1F4E79', '#C55A11', '#548235', '#7F6000'
CMU = '#606A72'


def save(fig, name):
    p = OUT / name
    fig.savefig(p)
    plt.close(fig)
    print('saved', p.name)


# ---------- 图 1：三种距离 ----------
def fig_m01():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    A, B = np.array([0.08, 0.50]), np.array([0.92, 0.50])

    ax.annotate('', xy=B, xytext=A,
                arrowprops=dict(arrowstyle='-', color=C1, lw=2.2))
    ax.text(0.50, 0.545, '直线  1080 km', ha='center', color=C1, fontsize=9.5)

    pts = np.array([[0.08, 0.50], [0.30, 0.28], [0.64, 0.30], [0.92, 0.50]])
    ax.plot(pts[:, 0], pts[:, 1], color=C2, lw=2.2, ls='--')
    ax.text(0.50, 0.215, '公路  1400 km', ha='center', color=C2, fontsize=9.5)

    t = np.linspace(0, 1, 200)
    ax.plot(0.08 + 0.84 * t, 0.50 + 0.15 * np.sin(np.pi * t),
            color=C3, lw=2.2, ls='-.')
    ax.text(0.50, 0.705, '铁路  1300 km', ha='center', color=C3, fontsize=9.5)

    ax.scatter([A[0], B[0]], [A[1], B[1]], s=70, color='#222', zorder=5)
    ax.text(A[0], 0.435, '北京', ha='center', fontsize=10)
    ax.text(B[0], 0.435, '上海', ha='center', fontsize=10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0.15, 0.78)
    ax.axis('off')
    ax.set_title('同一对城市，三种都对的距离', fontsize=11, color=C1)
    save(fig, 'fig_m01_three_distances.png')


# ---------- 图 2：SPD 锥 ----------
def fig_m02():
    fig = plt.figure(figsize=(5.6, 4.6))
    ax = fig.add_subplot(111, projection='3d')

    a = np.linspace(0.45, 3.0, 70)
    b = np.linspace(-2.0, 2.0, 70)
    A, B = np.meshgrid(a, b)
    D = B ** 2 / A
    D[D > 5.0] = np.nan
    ax.plot_surface(A, B, D, alpha=0.30, color='#9DC3E6',
                    edgecolor='none', rstride=2, cstride=2)

    rng = np.random.default_rng(3)
    aa = rng.uniform(0.7, 2.7, 60)
    bb = rng.uniform(-1.6, 1.6, 60)
    dd = bb ** 2 / aa + rng.uniform(0.25, 1.9, 60)
    m = dd < 5.0
    ax.scatter(aa[m], bb[m], dd[m], s=12, color=C2, alpha=0.75)

    ax.set_xlabel('a', fontsize=9)
    ax.set_ylabel('b', fontsize=9)
    ax.set_zlabel('d', fontsize=9)
    ax.set_zlim(0, 5.0)
    ax.view_init(elev=20, azim=-56)
    ax.set_title('SPD = 曲面 ad = b\u00b2 之上（2\u00d72 情形）', fontsize=10.5, color=C1, pad=2)
    save(fig, 'fig_m02_spd_cone.png')


# ---------- 图 3：弦长 vs 弧长 ----------
def fig_m03():
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    th = np.linspace(0, np.pi / 2, 200)
    ax.plot(np.cos(th), np.sin(th), color=C1, lw=2.4, label='弧长 = \u03c0/2 \u2248 1.571')

    ax.plot([1, 0], [0, 1], '--', color=C2, lw=2.2, label='弦长 = \u221a2 \u2248 1.414')

    ax.scatter([1, 0], [0, 1], s=70, color='#222', zorder=5)
    ax.text(1.03, -0.06, 'A (1,0)', fontsize=9.5)
    ax.text(-0.28, 1.0, 'B (0,1)', fontsize=9.5)

    ax.set_xlim(-0.35, 1.30)
    ax.set_ylim(-0.18, 1.22)
    ax.set_aspect('equal')
    ax.set_title('圆的距离是 1.571，不是 1.414', fontsize=10.5, color=C1)
    ax.legend(fontsize=9, loc='upper right', framealpha=0.9)
    save(fig, 'fig_m03_chord_arc.png')


# ---------- 图 4/5：切空间与指数映射 ----------
def fig_m04():
    fig, ax = plt.subplots(figsize=(5.8, 4.6))
    th = np.linspace(-0.62 * np.pi, 0.62 * np.pi, 300)
    ax.plot(np.cos(th), np.sin(th), color=C1, lw=2.4, label='单位圆')

    ax.scatter([1], [0], s=78, color='#222', zorder=6)
    ax.text(1.06, -0.12, 'p = (1,0)', fontsize=9.5)

    c = 0.9
    ax.annotate('', xy=(1, c), xytext=(1, 0),
                arrowprops=dict(arrowstyle='-|>', color=C3, lw=2.4,
                                mutation_scale=15))
    ax.text(1.07, c * 0.50, 'v = (0, c)', color=C3, fontsize=10)

    ax.scatter([np.cos(c)], [np.sin(c)], s=72, color=C2, zorder=6)
    ax.annotate('exp(v)：落在圆上', xy=(np.cos(c), np.sin(c)),
                xytext=(0.22, 1.18), fontsize=9.5, color=C2,
                arrowprops=dict(arrowstyle='->', color=C2, lw=1.2))

    ax.scatter([1], [c], s=75, marker='x', color='#999', zorder=5)
    ax.annotate('直接加 (1,c)：出界', xy=(1, c), xytext=(1.14, c + 0.20),
                fontsize=9, color='#888',
                arrowprops=dict(arrowstyle='->', color='#aaa', lw=1.0))

    ax.text(-0.88, 1.24, 'c = 0.9 即弧长', fontsize=8.8, color=CMU)

    ax.set_xlim(0.02, 1.80)
    ax.set_ylim(-1.18, 1.45)
    ax.set_aspect('equal')
    ax.set_title('切向量走一步：直接加会出界，exp 落在圆上',
                 fontsize=10.5, color=C1)
    save(fig, 'fig_m04_tangent_exp.png')


# ---------- 图 6：三种均值 ----------
def fig_m06():
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))

    names = ['算术均值', 'LE 均值', 'AIRM 中点']
    dets = [1.0000, 0.1900, 0.1900]
    cols = ['#C00000', C3, C1]

    ax = axes[0]
    bars = ax.bar(names, dets, color=cols, width=0.55, alpha=0.88)
    ax.axhline(0.19, color='#444', ls=':', lw=1.6)
    ax.text(2.42, 0.235, '端点 det = 0.19', fontsize=8.6, color='#444', ha='right')
    for b, v in zip(bars, dets):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, '%.4f' % v,
                ha='center', fontsize=9)
    ax.set_ylim(0, 1.22)
    ax.set_ylabel('行列式')
    ax.set_title('算术均值把 det 放大 5.3 倍', fontsize=10, color=C1)

    sums = [4.7808, 4.1640, 4.1641]
    ax = axes[1]
    bars = ax.bar(names, sums, color=cols, width=0.55, alpha=0.88)
    ax.axhline(4.1641, color='#444', ls=':', lw=1.6)
    ax.text(2.42, 4.19, '真实距离 4.1641', fontsize=8.6, color='#444', ha='right')
    for b, v in zip(bars, sums):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, '%.4f' % v,
                ha='center', fontsize=9)
    ax.set_ylim(4.0, 4.92)
    ax.set_ylabel('到两端距离之和')
    ax.set_title('只有 AIRM 中点在测地线上', fontsize=10, color=C1)

    fig.tight_layout()
    save(fig, 'fig_m06_three_means.png')


# ---------- 图 7：QR 回缩 ----------
def fig_m07():
    rng = np.random.default_rng(0)
    p, n = 8, 22

    def qr_retract(Y):
        Q, R = np.linalg.qr(Y)
        d = np.sign(np.diag(R))
        d[d == 0] = 1.0
        return Q * d

    W = qr_retract(rng.normal(size=(n, p)))
    W0 = W.copy()
    norms, moves = [], []
    for _ in range(12):
        G = rng.normal(size=(n, p)) * 0.1
        WtG = W.T @ G
        grad = G - W @ ((WtG + WtG.T) / 2)
        W = qr_retract(W - 0.1 * grad)
        norms.append(np.linalg.norm(W))
        moves.append(np.linalg.norm(W - W0))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ks = np.arange(1, len(norms) + 1)
    ax.plot(ks, norms, 'o-', color=C1, lw=2.2, ms=5, label='||W||_F  （恒为 \u221a8）')
    ax.axhline(np.sqrt(p), color=C1, ls=':', lw=1.2, alpha=0.6)

    ax2 = ax.twinx()
    ax2.plot(ks, moves, 's--', color=C2, lw=2.0, ms=5, label='||W - W0||_F')
    ax2.set_ylabel('到起点的距离', color=C2)
    ax2.tick_params(axis='y', colors=C2)
    ax2.grid(False)

    ax.set_xlabel('更新步数')
    ax.set_ylabel('Frobenius 范数', color=C1)
    ax.tick_params(axis='y', colors=C1)
    ax.set_ylim(2.4, 3.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=9, loc='center right', framealpha=0.92)
    ax.set_title('范数不变，但参数确实在移动', fontsize=10.5, color=C1)
    save(fig, 'fig_m07_qr_norm.png')


# ---------- 图 8：softmax 饱和 ----------
def fig_m08():
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ds = np.linspace(-8, 8, 400)
    for scale, col, lab in [(1.0, C3, '已缩放 \u00f7\u221ad'),
                            (np.sqrt(16), C2, 'd = 16，不缩放'),
                            (np.sqrt(64), '#C00000', 'd = 64，不缩放')]:
        ax.plot(ds, 1 / (1 + np.exp(-ds / scale)), color=col, lw=2.2, label=lab)
    ax.axhline(0.99, color='#999', ls=':', lw=1.1)
    ax.axhline(0.01, color='#999', ls=':', lw=1.1)
    ax.set_xlabel('两个 logit 之差')
    ax.set_ylabel('softmax 输出的第一类概率')
    ax.set_ylim(-0.03, 1.03)
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92)
    ax.set_title('不缩放时 softmax 迅速饱和（梯度趋零）', fontsize=10.2, color=C1)
    save(fig, 'fig_m08_softmax_sat.png')


if __name__ == '__main__':
    fig_m01()
    fig_m02()
    fig_m03()
    fig_m04()
    fig_m06()
    fig_m07()
    fig_m08()
    print('done')
