# -*- coding: utf-8 -*-
"""
DTW 预实验公共工具：
  - 加载轨迹
  - 频带黎曼均值
  - 标量/矩阵 DTW
  - 中文字体 & 常用绘图
"""

import pickle
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from scipy.stats import mannwhitneyu
from sklearn.manifold import MDS
from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.distance import distance_riemann
from tqdm import tqdm

CLASS_NAMES = ['left', 'right', 'feet', 'hands']
CLASS_NAMES_CN = ['左手', '右手', '双脚', '双手']
COLORS = ['C0', 'C1', 'C2', 'C3']

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_chinese_font():
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    matplotlib.rcParams['axes.unicode_minus'] = False


def load_trajectory_data(traj_dir='trajectory_data', n_subjects=None):
    traj_path = sorted(
        Path(traj_dir).glob('trajectories_*.pkl'),
        key=lambda p: p.stat().st_mtime,
        reverse=True
    )[0]
    with open(traj_path, 'rb') as f:
        data = pickle.load(f)
    subjects = data['subject_data']
    if n_subjects is not None:
        subjects = subjects[:n_subjects]
    return subjects, traj_path.name


def average_bands(trajs, tol=1e-5, maxiter=20):
    """对 [trials, windows, bands, C, C] 沿 bands 做黎曼均值。"""
    n_trials, n_windows, n_bands, C, _ = trajs.shape
    avg = np.zeros((n_trials, n_windows, C, C))
    for t in tqdm(range(n_trials), desc='频带均值', leave=False):
        for w in range(n_windows):
            avg[t, w] = mean_riemann(trajs[t, w], tol=tol, maxiter=maxiter)
    return avg


def compute_class_mean_trajectories(trial_trajs, trial_labels, n_classes=4, tol=1e-5, maxiter=20):
    """每个类每个窗口做一个黎曼均值矩阵。"""
    class_mean_trajs = {}
    for cls_idx in range(n_classes):
        cls_stack = np.stack([trial_trajs[i] for i in np.where(trial_labels == cls_idx)[0]])
        n_windows = cls_stack.shape[1]
        mean_traj = np.zeros((n_windows, cls_stack.shape[-2], cls_stack.shape[-1]))
        for w in range(n_windows):
            mean_traj[w] = mean_riemann(cls_stack[:, w], tol=tol, maxiter=maxiter)
        class_mean_trajs[cls_idx] = mean_traj
    return class_mean_trajs


def build_scalar_curves(trial_trajs, class_mean_trajs, n_classes=4):
    """每个 trial 在每个窗口计算到 4 个类均值的黎曼距离。"""
    scalar_curves = []
    for traj in tqdm(trial_trajs, desc='标量曲线'):
        n_windows = traj.shape[0]
        curve = np.zeros((n_windows, n_classes))
        for w in range(n_windows):
            for cls_idx in range(n_classes):
                curve[w, cls_idx] = distance_riemann(traj[w], class_mean_trajs[cls_idx][w])
        scalar_curves.append(curve)
    return np.array(scalar_curves)


def dtw_multivariate(seq1, seq2, radius=2):
    distance, _ = fastdtw(seq1, seq2, dist=euclidean, radius=radius)
    return distance


def pairwise_riemannian_distance_gpu(seq1, seq2, device=DEVICE):
    """AIRM 距离，[T1, C, C] vs [T2, C, C] -> [T1, T2]。"""
    A = torch.tensor(seq1, dtype=torch.float64, device=device)
    B = torch.tensor(seq2, dtype=torch.float64, device=device)

    eigvals_A, eigvecs_A = torch.linalg.eigh(A)
    eigvals_A = torch.clamp(eigvals_A, min=1e-10)
    A_inv_sqrt = eigvecs_A @ torch.diag_embed(eigvals_A ** (-0.5)) @ eigvecs_A.transpose(-2, -1)

    M = A_inv_sqrt.unsqueeze(1) @ B.unsqueeze(0) @ A_inv_sqrt.unsqueeze(1)
    M = (M + M.transpose(-2, -1)) / 2

    eigvals = torch.linalg.eigvalsh(M)
    eigvals = torch.clamp(eigvals, min=1e-10)
    dists = torch.sqrt(torch.sum(torch.log(eigvals) ** 2, dim=-1))
    return dists.cpu().numpy()


def dtw_matrix(seq1, seq2, device=DEVICE):
    """矩阵 DTW：局部代价 GPU 批量算 AIRM，DP 在 CPU 完成。"""
    cost_mat = pairwise_riemannian_distance_gpu(seq1, seq2, device=device)
    T1, T2 = cost_mat.shape
    D = np.full((T1 + 1, T2 + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            D[i, j] = cost_mat[i - 1, j - 1] + min(D[i - 1, j], D[i, j - 1], D[i - 1, j - 1])
    return D[T1, T2]


def compute_distance_matrix(seqs, metric_fn, desc='DTW'):
    n = len(seqs)
    dist_mat = np.zeros((n, n))
    for i in tqdm(range(n), desc=desc):
        for j in range(i + 1, n):
            d = metric_fn(seqs[i], seqs[j])
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat


def collect_pairwise_stats(dist_mat, labels, subjects):
    """返回同被试同类、同被试异类、跨被试同类三组距离。"""
    intra_same, inter_same, intra_cross = [], [], []
    n = len(labels)
    for i in range(n):
        for j in range(i + 1, n):
            si, ci = subjects[i], labels[i]
            sj, cj = subjects[j], labels[j]
            if si == sj:
                if ci == cj:
                    intra_same.append(dist_mat[i, j])
                else:
                    inter_same.append(dist_mat[i, j])
            elif ci == cj:
                intra_cross.append(dist_mat[i, j])
    return intra_same, inter_same, intra_cross


def plot_boxplot(groups, labels, title, ylabel, save_path):
    setup_chinese_font()
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=groups)
    plt.xticks(range(len(labels)), labels)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_mds(dist_mat, labels, save_path, title='MDS 投影'):
    setup_chinese_font()
    emb = MDS(n_components=2, dissimilarity='precomputed', random_state=42).fit_transform(dist_mat)
    plt.figure(figsize=(10, 8))
    for cls_idx in range(4):
        mask = labels == cls_idx
        plt.scatter(emb[mask, 0], emb[mask, 1], c=COLORS[cls_idx],
                    label=CLASS_NAMES[cls_idx], alpha=0.7, s=80)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def mannwhitney_summary(group_a, group_b, label_a='A', label_b='B'):
    stat, p = mannwhitneyu(group_a, group_b, alternative='less')
    print(f"  {label_a} vs {label_b}: p={p:.2e}")
    return p