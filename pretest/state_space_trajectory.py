# -*- coding: utf-8 -*-
"""
状态空间轨迹可视化 (State-Space Trajectory Plot)
将 140 维切空间特征通过 PCA 降到 2D，绘制 4 类平均轨迹。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from pathlib import Path
from sklearn.decomposition import PCA
from tqdm import tqdm

from dtw_utils import setup_chinese_font, load_trajectory_data, CLASS_NAMES_CN

WIN_SEC = 1.0  # 滑动窗口长度，用于计算 Epoch 起止


def log_euclidean_mapping(covs, eps=1e-7):
    """对数欧氏映射: [N, C, C] -> [N, D]"""
    N, C, _ = covs.shape
    evals, evecs = np.linalg.eigh(covs)
    log_evals = np.log(np.clip(evals, a_min=eps, a_max=None))
    log_D = np.zeros((N, C, C), dtype=log_evals.dtype)
    log_D[np.arange(N)[:, None], np.arange(C), np.arange(C)] = log_evals
    log_C = evecs @ log_D @ evecs.transpose(0, 2, 1)
    idx = np.triu_indices(C)
    tangent = log_C[:, idx[0], idx[1]]
    off_diag = (idx[0] != idx[1])
    tangent[:, off_diag] *= np.sqrt(2.0)
    return tangent.astype(np.float32)


def covs_to_tangent_features(trajs, eps=1e-7):
    """[trials, windows, bands, C, C] -> [trials, windows, bands*D]"""
    n_trials, n_windows, n_bands, C, _ = trajs.shape
    flat = trajs.reshape(-1, C, C)
    tangent = log_euclidean_mapping(flat, eps=eps)
    D = C * (C + 1) // 2
    tangent = tangent.reshape(n_trials, n_windows, n_bands, D)
    return tangent.reshape(n_trials, n_windows, -1)


def extract_all_features(subjects_data):
    all_features, all_labels = [], []
    for subj_info in tqdm(subjects_data, desc='提取被试切空间特征'):
        feats = covs_to_tangent_features(subj_info['trajectories'])
        all_features.append(feats)
        all_labels.append(subj_info['labels'])
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels


def pca_reduce(features, n_components=2):
    """[trials, windows, D] -> [trials, windows, n_components]"""
    n_trials, n_windows, D = features.shape
    flat = features.reshape(-1, D)
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(flat)
    print(f'PCA 解释方差比: {pca.explained_variance_ratio_}')
    print(f'累计解释方差: {np.sum(pca.explained_variance_ratio_):.3f}')
    return reduced.reshape(n_trials, n_windows, n_components), pca


def average_by_class(trajectories, labels, n_classes=4):
    means = np.zeros((n_classes, trajectories.shape[1], 2))
    counts = np.zeros(n_classes)
    for c in range(n_classes):
        mask = labels == c
        means[c] = trajectories[mask].mean(axis=0)
        counts[c] = mask.sum()
    return means, counts


def plot_state_space_trajectories(class_means, window_centers, save_path, win_sec=1.0):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(11, 9))

    cmap_per_class = ['Blues', 'Reds', 'Greens', 'Oranges']
    norm = Normalize(vmin=window_centers.min(), vmax=window_centers.max())

    all_x, all_y = [], []
    for cls_idx in range(4):
        traj = class_means[cls_idx]
        all_x.extend(traj[:, 0])
        all_y.extend(traj[:, 1])

        points = traj.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap=cmap_per_class[cls_idx], norm=norm, linewidths=3.5)
        lc.set_array(window_centers)
        ax.add_collection(lc)

        # 起点/终点
        ax.scatter(traj[0, 0], traj[0, 1], c='black', s=90, marker='o', zorder=5)
        ax.scatter(traj[-1, 0], traj[-1, 1], c='black', s=90, marker='s', zorder=5)

        # 标注关键时间点：基线 -0.5s、任务期 0/1/2/3s、最后一个窗口中心 3.0s
        for t_anno in [-0.5, 0, 1, 2, 3.0]:
            idx = np.argmin(np.abs(window_centers - t_anno))
            ax.scatter(traj[idx, 0], traj[idx, 1], c='black', s=50, zorder=6)
            label = f'{t_anno:.1f}s' if isinstance(t_anno, float) else f'{t_anno}s'
            ax.annotate(label, (traj[idx, 0], traj[idx, 1]),
                        textcoords='offset points', xytext=(6, 6), fontsize=10,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

    # 图例
    legend_lines = []
    for cls_idx, name in enumerate(CLASS_NAMES_CN):
        color = plt.get_cmap(cmap_per_class[cls_idx])(0.7)
        legend_lines.append(Line2D([0], [0], color=color, linewidth=3.5, label=name))
    legend_lines.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                                markersize=8, label='起点'))
    legend_lines.append(Line2D([0], [0], marker='s', color='w', markerfacecolor='black',
                                markersize=8, label='终点'))
    ax.legend(handles=legend_lines, loc='best')

    # 坐标轴
    padding = 0.12
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_range = x_max - x_min
    y_range = y_max - y_min
    ax.set_xlim(x_min - padding * x_range, x_max + padding * x_range)
    ax.set_ylim(y_min - padding * y_range, y_max + padding * y_range)

    t_min, t_max = window_centers.min(), window_centers.max()
    epoch_start = t_min - win_sec / 2
    epoch_end = t_max + win_sec / 2

    ax.set_xlabel('PC1', fontsize=12)
    ax.set_ylabel('PC2', fontsize=12)
    ax.set_title(
        f'状态空间轨迹（PCA 降维）\nEpoch [{epoch_start:.1f}s, {epoch_end:.1f}s]，'
        f'窗口中心 {t_min:.1f}s ~ {t_max:.1f}s',
        fontsize=14
    )
    ax.grid(True, alpha=0.3)

    # 颜色条表示时间
    sm = ScalarMappable(cmap='viridis', norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('时间 (s)', fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    OUTPUT_DIR = Path('pretest') / 'output'
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    subjects_data, traj_name = load_trajectory_data(n_subjects=None)
    print(f'📂 加载轨迹: {traj_name}')

    print('🔧 切空间特征提取...')
    features, labels = extract_all_features(subjects_data)

    print('📉 PCA 降维到 2D...')
    trajectories_2d, pca = pca_reduce(features, n_components=2)

    print('📊 按类别平均...')
    class_means, counts = average_by_class(trajectories_2d, labels)

    print('各类别试次数:')
    for i, name in enumerate(CLASS_NAMES_CN):
        print(f'  {name}: {int(counts[i])}')

    print('🎨 绘制状态空间轨迹...')
    n_windows = features.shape[1]
    window_centers = -0.5 + np.arange(n_windows) * 0.1
    save_path = OUTPUT_DIR / 'state_space_trajectory.png'
    plot_state_space_trajectories(class_means, window_centers, save_path, win_sec=WIN_SEC)

    print(f'✅ 已保存: {save_path}')


if __name__ == '__main__':
    main()