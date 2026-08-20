# -*- coding: utf-8 -*-
"""
黎曼轨迹核心可视化
对应用户需求的图5-10
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
import torch
warnings.filterwarnings('ignore')

from utils import log_euclidean_mapping, normalize_trace, riemann_distance_logeuc

# 样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
CLASS_NAMES = ['左手', '右手', '双脚', '双手']
BAND_NAMES = ['Mu(8-12Hz)', '低β(12-14Hz)', '中β(14-20Hz)', '高β(20-25Hz)', 'γ(25-30Hz)']
CHANNEL_NAMES = ['FCz', 'FC3', 'FC4', 'C4', 'C3', 'CPz', 'Cz']


class RiemannianVisualizer:
    def __init__(self):
        self.traj_file = Path("trajectory_data") / "trajectories_500ms_win.pkl"
        self.output_dir = Path("figures")
        self.output_dir.mkdir(exist_ok=True)

        with open(self.traj_file, 'rb') as f:
            data = pickle.load(f)
        self.subjects_data = data['subject_data']
        print(f"✅ 加载 {len(self.subjects_data)} 被试")

    def save(self, name):
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 已保存: {path.name}")
        plt.close()

    # ==================== 图5: 多频带PCA轨迹 ====================
    def plot_riemann_trajectory_pca(self):
        """单个trial的5频带协方差矩阵在PCA降维后的2D轨迹"""
        subj = self.subjects_data[0]
        covs = subj['trajectories']  # [trials, windows, bands, 7, 7]
        labels = subj['labels']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.flatten()

        for cls in range(4):
            mask = labels == cls
            if mask.sum() == 0:
                continue

            trial_covs = covs[np.where(mask)[0][0]]  # [windows, bands, 7, 7]
            n_windows = trial_covs.shape[0]

            # 将所有窗口所有频带的协方差展平
            all_cov_flat = trial_covs.reshape(-1, 7*7)  # [windows*bands, 49]
            pca = PCA(n_components=2)
            traj_2d = pca.fit_transform(all_cov_flat)

            ax = axes[cls]
            for band in range(5):
                start = band * n_windows
                end = (band + 1) * n_windows
                band_traj = traj_2d[start:end]
                ax.plot(band_traj[:, 0], band_traj[:, 1], 'o-',
                       color=COLORS[band], linewidth=1.5, markersize=4,
                       label=BAND_NAMES[band], alpha=0.8)
                ax.scatter(band_traj[0, 0], band_traj[0, 1],
                          color=COLORS[band], s=100, marker='*', zorder=5)

            ax.set_xlabel('主成分 1', fontsize=10)
            ax.set_ylabel('主成分 2', fontsize=10)
            ax.set_title(f'{CLASS_NAMES[cls]}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle('PCA 空间中的多频带协方差轨迹\n（被试1，每类首个试次）',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save('riem_01_PCA多频带轨迹')

    # ==================== 图6: 不同类别平均轨迹对比 ====================
    def plot_class_average_trajectory(self):
        """每类所有trial在mu频带的平均黎曼轨迹"""
        band_idx = 0  # mu节律
        all_trajectories = {cls: [] for cls in range(4)}

        # 收集前10个被试的数据
        for subj in self.subjects_data[:10]:
            covs = subj['trajectories'][:, :, band_idx, :, :]  # [trials, windows, 7, 7]
            labels = subj['labels']
            for cls in range(4):
                mask = labels == cls
                if mask.sum() > 0:
                    all_trajectories[cls].append(covs[mask])  # list of [n_trials, windows, 7, 7]

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # 左图: 基于黎曼距离的1D轨迹
        ax = axes[0]
        for cls in range(4):
            if len(all_trajectories[cls]) == 0:
                continue
            cls_covs = np.concatenate(all_trajectories[cls], axis=0)  # [N, windows, 7, 7]
            # 计算每个窗口到第一个窗口的黎曼距离
            ref_cov = np.mean(cls_covs[:, 0, :, :], axis=0)  # 第一个窗口的平均
            n_windows = cls_covs.shape[1]
            mean_dists = []
            std_dists = []
            for w in range(n_windows):
                dists = [riemann_distance_logeuc(ref_cov, cls_covs[t, w, :, :])
                         for t in range(min(50, len(cls_covs)))]
                mean_dists.append(np.mean(dists))
                std_dists.append(np.std(dists))
            mean_dists = np.array(mean_dists)
            std_dists = np.array(std_dists)
            ax.plot(range(n_windows), mean_dists, 'o-', color=COLORS[cls],
                   linewidth=2, label=CLASS_NAMES[cls])
            ax.fill_between(range(n_windows), mean_dists - std_dists, mean_dists + std_dists,
                           color=COLORS[cls], alpha=0.2)

        ax.set_xlabel('时间窗序号', fontsize=11)
        ax.set_ylabel('相对于基线的黎曼距离', fontsize=11)
        ax.set_title('(a) 各类别平均轨迹\n（Mu 频带，10名被试）', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 右图: 协方差矩阵Frobenius范数演化
        ax = axes[1]
        for cls in range(4):
            if len(all_trajectories[cls]) == 0:
                continue
            cls_covs = np.concatenate(all_trajectories[cls], axis=0)
            n_windows = cls_covs.shape[1]
            norms = np.array([[np.linalg.norm(cls_covs[t, w], 'fro')
                               for w in range(n_windows)]
                              for t in range(min(30, len(cls_covs)))])
            mean_norms = norms.mean(axis=0)
            std_norms = norms.std(axis=0)
            ax.plot(range(n_windows), mean_norms, 'o-', color=COLORS[cls],
                   linewidth=2, label=CLASS_NAMES[cls])
            ax.fill_between(range(n_windows), mean_norms - std_norms, mean_norms + std_norms,
                           color=COLORS[cls], alpha=0.2)

        ax.set_xlabel('时间窗序号', fontsize=11)
        ax.set_ylabel('协方差 Frobenius 范数', fontsize=11)
        ax.set_title('(b) 协方差范数演化\n（Mu 频带，10名被试）', fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        self.save('riem_02_类别平均轨迹')

    # ==================== 图7: 协方差矩阵演化帧 ====================
    def plot_covariance_evolution_frames(self):
        """单个trial的协方差矩阵随时间演化的关键帧"""
        subj = self.subjects_data[0]
        covs = subj['trajectories']
        labels = subj['labels']

        # 选左手类别
        cls = 0
        mask = labels == cls
        if mask.sum() == 0:
            return
        trial_covs = covs[np.where(mask)[0][0]]  # [windows, bands, 7, 7]
        n_windows = trial_covs.shape[0]

        # 选4个关键窗口
        frame_indices = [0, n_windows//3, 2*n_windows//3, n_windows-1]
        fig, axes = plt.subplots(len(frame_indices), 5, figsize=(15, 10))

        for row, w_idx in enumerate(frame_indices):
            for band in range(5):
                ax = axes[row, band]
                cov = trial_covs[w_idx, band, :, :]
                vmax = np.abs(cov).max()
                im = ax.imshow(cov, cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
                ax.set_xticks(range(7))
                ax.set_yticks(range(7))
                ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha='right', fontsize=6)
                ax.set_yticklabels(CHANNEL_NAMES, fontsize=6)

                if row == 0:
                    ax.set_title(BAND_NAMES[band], fontsize=9, fontweight='bold')
                if band == 0:
                    ax.set_ylabel(f'窗{w_idx}\n({w_idx*0.1:.1f}秒)', fontsize=9, fontweight='bold')

                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle(f'协方差矩阵演化帧 — {CLASS_NAMES[cls]}\n（被试1，首个试次）',
                    fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save('riem_03_协方差演化帧')

    # ==================== 图8: 单频带切空间t-SNE ====================
    def plot_single_band_tsne(self):
        """单个频带(mu)的切空间特征t-SNE"""
        print("  🔄 计算单频带切空间t-SNE...")

        band_idx = 0  # mu节律
        all_covs = []
        all_labels = []

        for s in self.subjects_data[:15]:
            covs = s['trajectories'][:, :, band_idx, :, :]  # [trials, windows, 7, 7]
            labels = s['labels']
            # 每个trial取前5个window
            n = min(len(labels), 30)
            indices = np.random.choice(len(labels), n, replace=False)
            all_covs.append(covs[indices, :5, :, :].reshape(-1, 7, 7))
            all_labels.extend(np.repeat(labels[indices], 5))

        all_covs = np.concatenate(all_covs, axis=0)  # [N, 7, 7]
        all_labels = np.array(all_labels)

        # 黎曼映射
        covs_tensor = torch.tensor(all_covs, dtype=torch.float64)
        covs_norm = normalize_trace(covs_tensor)
        tangent = log_euclidean_mapping(covs_norm).cpu().numpy()  # [N, 28]

        # t-SNE
        n_samples = min(1500, len(tangent))
        idx = np.random.choice(len(tangent), n_samples, replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(tangent[idx])
        y_plot = all_labels[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in range(4):
            mask = y_plot == cls
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=COLORS[cls], label=CLASS_NAMES[cls],
                      alpha=0.5, s=25, edgecolors='none')
        ax.set_xlabel('t-SNE 维度 1', fontsize=11)
        ax.set_ylabel('t-SNE 维度 2', fontsize=11)
        ax.set_title(f'切空间特征的 t-SNE\n（{BAND_NAMES[band_idx]}，15名被试，窗口1-5）',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        plt.tight_layout()
        self.save('riem_04_单频带tSNE')

    # ==================== 图9: 140维特征序列热力图 ====================
    def plot_feature_sequence_heatmap(self):
        """单个trial的140维特征序列"""
        subj = self.subjects_data[0]
        covs = subj['trajectories']  # [trials, windows, bands, 7, 7]
        labels = subj['labels']

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()

        for cls in range(4):
            mask = labels == cls
            if mask.sum() == 0:
                continue

            trial_covs = covs[np.where(mask)[0][0]]  # [windows, bands, 7, 7]
            n_windows = trial_covs.shape[0]

            # 计算每个窗口的切空间特征并拼接
            features = []
            for w in range(n_windows):
                window_covs = torch.tensor(trial_covs[w], dtype=torch.float64)  # [bands, 7, 7]
                window_covs = normalize_trace(window_covs)
                tangent = log_euclidean_mapping(window_covs).cpu().numpy()  # [bands, 28]
                features.append(tangent.flatten())  # [140]

            features = np.array(features)  # [windows, 140]

            ax = axes[cls]
            im = ax.imshow(features.T, cmap='RdBu_r', aspect='auto', interpolation='nearest')
            ax.set_xlabel('时间窗序号', fontsize=10)
            ax.set_ylabel('特征维度', fontsize=10)
            ax.set_title(f'{CLASS_NAMES[cls]}', fontsize=11, fontweight='bold')

            # 标注频带分界
            for b in range(1, 5):
                ax.axhline(y=b*28 - 0.5, color='white', linewidth=1)
            ax.set_yticks([14, 42, 70, 98, 126])
            ax.set_yticklabels(BAND_NAMES, fontsize=7)

            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle('每试次的140维特征序列热力图\n（被试1，每类首个试次）',
                    fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save('riem_05_特征序列热力图')

    def run_all(self):
        print("\n" + "=" * 60)
        print("🎨 黎曼轨迹核心可视化")
        print("=" * 60)
        self.plot_riemann_trajectory_pca()
        self.plot_class_average_trajectory()
        self.plot_covariance_evolution_frames()
        self.plot_single_band_tsne()
        self.plot_feature_sequence_heatmap()
        print(f"\n✅ 已保存到 {self.output_dir}/")


if __name__ == "__main__":
    viz = RiemannianVisualizer()
    viz.run_all()