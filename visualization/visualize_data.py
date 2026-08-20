# -*- coding: utf-8 -*-
"""
EEG 算法数据层面可视化脚本
不依赖训练结果，基于 trajectory_data 和 saved_data 生成
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
import mne
warnings.filterwarnings('ignore')

# ==================== 全局样式设置 ====================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['figure.dpi'] = 150

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
CLASS_NAMES = ['左手', '右手', '双脚', '双手']
BAND_NAMES = ['Mu\n(8-12Hz)', '低β\n(12-14Hz)', '中β\n(14-20Hz)', '高β\n(20-25Hz)', 'γ\n(25-30Hz)']
CHANNEL_NAMES = ['FCz', 'FC3', 'FC4', 'C4', 'C3', 'CPz', 'Cz']

# ==================== 通道拓扑坐标（用于画头皮图） ====================
CHANNEL_POSITIONS = {
    'FCz': (0.0, 0.6), 'FC3': (-0.4, 0.5), 'FC4': (0.4, 0.5),
    'C3': (-0.5, 0.0), 'C4': (0.5, 0.0), 'CPz': (0.0, -0.4), 'Cz': (0.0, 0.0),
}


class EEGDataVisualizer:
    def __init__(self):
        self.traj_file = Path("trajectory_data") / "trajectories_500ms_win.pkl"
        self.prep_file = self._find_preprocessed_file()
        self.output_dir = Path("figures")
        self.output_dir.mkdir(exist_ok=True)

        # 加载轨迹数据
        with open(self.traj_file, 'rb') as f:
            traj_data = pickle.load(f)
        self.subjects_data = traj_data['subject_data']
        self.num_subjects = len(self.subjects_data)
        print(f"✅ 加载轨迹数据: {self.num_subjects} 被试")

        # 尝试加载预处理数据
        self.preprocessed_data = None
        if self.prep_file:
            try:
                with open(self.prep_file, 'rb') as f:
                    self.preprocessed_data = pickle.load(f)
                print(f"✅ 加载预处理数据: {self.prep_file.name}")
            except Exception as e:
                print(f"⚠️ 预处理数据加载失败: {e}")

    def _find_preprocessed_file(self):
        prep_dir = Path("saved_data")
        if not prep_dir.exists():
            return None
        pkl_files = list(prep_dir.glob("preprocessed_data_*.pkl"))
        if not pkl_files:
            return None
        pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return pkl_files[0]

    def save(self, name):
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 已保存: {path.name}")
        plt.close()

    # ==================== 图1: 类别分布 ====================
    def plot_class_distribution(self):
        """图1-2: 总体和每被试类别分布"""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # 总体分布
        all_labels = np.concatenate([s['labels'] for s in self.subjects_data])
        counts = np.bincount(all_labels, minlength=4)
        bars = axes[0].bar(range(4), counts, color=COLORS, edgecolor='black', alpha=0.85, linewidth=1.2)
        axes[0].set_xticks(range(4))
        axes[0].set_xticklabels(CLASS_NAMES, fontsize=10)
        axes[0].set_ylabel('试次数量', fontsize=11)
        axes[0].set_title('(a) 总体类别分布', fontsize=12, fontweight='bold')
        for bar, v in zip(bars, counts):
            axes[0].text(bar.get_x() + bar.get_width()/2, v + max(counts)*0.02,
                        f'{v}', ha='center', va='bottom', fontweight='bold', fontsize=10)

        # 每被试分布（前15个）
        n_show = min(15, self.num_subjects)
        subj_labels = [s['subject_key'] for s in self.subjects_data[:n_show]]
        class_counts = np.array([np.bincount(s['labels'], minlength=4) for s in self.subjects_data[:n_show]])

        x = np.arange(n_show)
        width = 0.2
        for i in range(4):
            axes[1].bar(x + i*width, class_counts[:, i], width,
                       label=CLASS_NAMES[i], color=COLORS[i], alpha=0.85, edgecolor='black', linewidth=0.5)
        axes[1].set_xticks(x + width * 1.5)
        axes[1].set_xticklabels([s.split('_S')[-1] for s in subj_labels], rotation=45, ha='right', fontsize=8)
        axes[1].set_ylabel('试次数量', fontsize=11)
        axes[1].set_title('(b) 各被试类别分布', fontsize=12, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=8, framealpha=0.9)

        plt.tight_layout()
        self.save('data_01_类别分布')

    # ==================== 图2: 通道拓扑位置 ====================
    def plot_channel_topology(self):
        """图3: 7个选择通道的头皮位置"""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal')
        ax.axis('off')

        # 画头皮轮廓（简化为圆）
        theta = np.linspace(0, 2*np.pi, 100)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', linewidth=2)
        ax.plot(0.5*np.cos(theta), 0.5*np.sin(theta), 'k--', linewidth=1, alpha=0.3)  # 鼻尖参考

        # 鼻根
        ax.plot([0], [1.05], 'kv', markersize=8)

        for ch_name, (x, y) in CHANNEL_POSITIONS.items():
            idx = CHANNEL_NAMES.index(ch_name)
            circle = plt.Circle((x, y), 0.12, color=COLORS[idx % len(COLORS)], alpha=0.7, ec='black', linewidth=1.5)
            ax.add_patch(circle)
            ax.text(x, y, ch_name, ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        ax.set_title('选取的7个中央区通道', fontsize=13, fontweight='bold', pad=10)
        self.save('data_02_通道拓扑')

    # ==================== 图3: 协方差矩阵热力图 ====================
    def plot_covariance_heatmaps(self):
        """图4: 不同类别和频带的平均协方差矩阵"""
        # 取前3个被试做平均
        n_subj_avg = min(3, self.num_subjects)
        all_covs = []
        all_labels = []

        for s in self.subjects_data[:n_subj_avg]:
            all_covs.append(s['trajectories'][:, 0, :, :, :])  # 取第1个window
            all_labels.extend(s['labels'])

        all_covs = np.concatenate(all_covs, axis=0)  # [N, bands, 7, 7]
        all_labels = np.array(all_labels)

        fig, axes = plt.subplots(4, 5, figsize=(16, 13))
        for cls in range(4):
            mask = all_labels == cls
            if mask.sum() == 0:
                continue
            avg_cov = all_covs[mask].mean(axis=0)  # [bands, 7, 7]

            for band in range(5):
                ax = axes[cls, band]
                vmax = np.abs(avg_cov[band]).max()
                im = ax.imshow(avg_cov[band], cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
                ax.set_xticks(range(7))
                ax.set_yticks(range(7))
                ax.set_xticklabels(CHANNEL_NAMES, rotation=45, ha='right', fontsize=7)
                ax.set_yticklabels(CHANNEL_NAMES, fontsize=7)

                if cls == 0:
                    ax.set_title(BAND_NAMES[band], fontsize=10, fontweight='bold')
                if band == 0:
                    ax.set_ylabel(CLASS_NAMES[cls], fontsize=10, fontweight='bold', rotation=90, labelpad=10)

                # 添加colorbar
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.suptitle('各类别与频带的平均协方差矩阵\n（前3名被试平均，第1个时间窗）',
                     fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save('data_03_协方差热力图')

    # ==================== 图4: 协方差特征值分布 ====================
    def plot_eigenvalue_distribution(self):
        """图5: 协方差矩阵特征值分布"""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

        # 收集特征值
        all_eigenvalues = {cls: [] for cls in range(4)}
        for s in self.subjects_data[:5]:  # 前5个被试
            covs = s['trajectories'][:, 0, 0, :, :]  # [trials, 7, 7]
            labels = s['labels']
            for trial_idx, label in enumerate(labels):
                eigs = np.linalg.eigvalsh(covs[trial_idx])
                all_eigenvalues[label].append(eigs)

        # 左图: 各类别特征值分布箱线图
        eigenvalue_data = []
        tick_labels = []
        positions = []
        pos = 1
        for cls in range(4):
            if len(all_eigenvalues[cls]) == 0:
                continue
            eigs_arr = np.array(all_eigenvalues[cls])  # [n_trials, 7]
            for dim in range(7):
                eigenvalue_data.append(eigs_arr[:, dim])
                tick_labels.append(f'λ{dim+1}')
                positions.append(pos)
                pos += 1
            pos += 1  # 类别间隔

        # 简化为画各类别的平均特征值
        axes[0].set_prop_cycle(color=COLORS)
        for cls in range(4):
            if len(all_eigenvalues[cls]) > 0:
                mean_eigs = np.array(all_eigenvalues[cls]).mean(axis=0)
                axes[0].plot(range(1, 8), mean_eigs, 'o-', linewidth=2, markersize=6,
                            label=CLASS_NAMES[cls])
        axes[0].set_xlabel('特征值序号', fontsize=11)
        axes[0].set_ylabel('特征值', fontsize=11)
        axes[0].set_title('(a) 各类别平均特征值谱', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=9)
        axes[0].set_yscale('log')
        axes[0].grid(True, alpha=0.3)

        # 右图: 所有特征值的直方图
        all_eigs_flat = []
        for cls in range(4):
            if len(all_eigenvalues[cls]) > 0:
                all_eigs_flat.extend(np.array(all_eigenvalues[cls]).flatten())
        axes[1].hist(all_eigs_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        axes[1].axvline(x=1e-4, color='red', linestyle='--', linewidth=2, label='正则化阈值 (1e-4)')
        axes[1].set_xlabel('特征值', fontsize=11)
        axes[1].set_ylabel('频数', fontsize=11)
        axes[1].set_title('(b) 特征值分布', fontsize=12, fontweight='bold')
        axes[1].set_xscale('log')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        self.save('data_04_特征值分布')

    # ==================== 图5: t-SNE可视化黎曼特征 ====================
    def plot_tsne_tangent_space(self):
        """图6: t-SNE降维可视化"""
        from utils import log_euclidean_mapping, normalize_trace
        import torch

        print("  🔄 正在计算黎曼切空间特征用于t-SNE...")

        # 采样数据（避免太多）
        max_samples_per_subj = 20
        all_covs = []
        all_labels = []

        for s in self.subjects_data[:20]:  # 前20个被试
            covs = s['trajectories'][:, 0, :, :, :]  # [trials, bands, 7, 7]
            labels = s['labels']
            n = min(len(labels), max_samples_per_subj)
            indices = np.random.choice(len(labels), n, replace=False)
            all_covs.append(covs[indices])
            all_labels.extend(labels[indices])

        all_covs = np.concatenate(all_covs, axis=0)  # [N, bands, 7, 7]
        all_labels = np.array(all_labels)

        # 计算切空间特征
        N = all_covs.shape[0]
        covs_tensor = torch.tensor(all_covs.reshape(-1, 7, 7), dtype=torch.float64)
        covs_norm = normalize_trace(covs_tensor)
        tangent = log_euclidean_mapping(covs_norm).cpu().numpy()  # [N*5, 28]

        # 合并5个频带: 平均
        tangent = tangent.reshape(N, 5, 28).mean(axis=1)  # [N, 28]

        # t-SNE
        print("  🔄 运行t-SNE...")
        n_samples = min(1000, len(tangent))
        idx = np.random.choice(len(tangent), n_samples, replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//5))
        X_tsne = tsne.fit_transform(tangent[idx])
        y_plot = all_labels[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in range(4):
            mask = y_plot == cls
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=COLORS[cls], label=CLASS_NAMES[cls],
                      alpha=0.6, s=35, edgecolors='none')
        ax.set_xlabel('t-SNE 维度 1', fontsize=11)
        ax.set_ylabel('t-SNE 维度 2', fontsize=11)
        ax.set_title('黎曼切空间特征的 t-SNE 可视化\n（20名被试，5频带平均）',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        plt.tight_layout()
        self.save('data_05_切空间tSNE')

    # ==================== 图6: 动态轨迹演化 ====================
    def plot_trajectory_evolution(self):
        """图7: 单个trial的协方差矩阵动态演化"""
        subj = self.subjects_data[0]
        covs = subj['trajectories']  # [trials, windows, bands, 7, 7]
        labels = subj['labels']

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        axes = axes.flatten()

        for cls in range(4):
            mask = labels == cls
            if mask.sum() == 0:
                continue

            trial_covs = covs[np.where(mask)[0][0]]  # [windows, bands, 7, 7]
            n_windows = trial_covs.shape[0]

            ax = axes[cls]

            # 计算每window的Frobenius范数作为轨迹幅值
            for band in range(5):
                band_covs = trial_covs[:, band, :, :]  # [windows, 7, 7]
                traj_norm = np.array([np.linalg.norm(c, 'fro') for c in band_covs])
                ax.plot(range(n_windows), traj_norm, 'o-', color=COLORS[band % len(COLORS)],
                       linewidth=1.5, markersize=4, label=BAND_NAMES[band], alpha=0.8)

            ax.set_xlabel('时间窗序号（步长0.1秒）', fontsize=10)
            ax.set_ylabel('协方差 Frobenius 范数', fontsize=10)
            ax.set_title(f'{CLASS_NAMES[cls]}', fontsize=11, fontweight='bold')
            ax.legend(fontsize=7, loc='best')
            ax.grid(True, alpha=0.3)

        plt.suptitle('动态黎曼轨迹演化\n（被试1，每类首个试次）',
                     fontsize=13, fontweight='bold', y=1.02)
        plt.tight_layout()
        self.save('data_06_轨迹演化')

    # ==================== 图7: EA对齐效果 ====================
    def plot_ea_alignment_effect(self):
        """基于trajectory_data展示EA对齐效果:
        1) 不同被试同类协方差对比; 2) 多被试t-SNE聚类; 3) 类内 vs 被试间同类距离
        """
        from utils import riemann_distance_logeuc as riemann_dist
        import torch
        from utils import log_euclidean_mapping, normalize_trace

        n_show = min(6, self.num_subjects)
        band_idx = 0  # Mu频带

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)

        # ========== 子图1: 4个被试左手协方差热力图对比 ==========
        ax1 = fig.add_subplot(gs[0, 0])
        cls = 0  # 左手
        subj_covs = []
        for s in range(min(4, n_show)):
            subj = self.subjects_data[s]
            covs = subj['trajectories'][:, 0, band_idx, :, :]  # [trials, 7, 7]
            labels = subj['labels']
            mask = labels == cls
            if mask.sum() > 0:
                avg_cov = covs[mask].mean(axis=0)
                subj_covs.append(avg_cov)

        if len(subj_covs) >= 2:
            # 计算被试间平均协方差的差异（ Frobenius范数差异）
            diff_matrix = np.zeros((len(subj_covs), len(subj_covs)))
            for i in range(len(subj_covs)):
                for j in range(len(subj_covs)):
                    diff_matrix[i, j] = np.linalg.norm(subj_covs[i] - subj_covs[j], 'fro')
            im1 = ax1.imshow(diff_matrix, cmap='YlOrRd', vmin=0, aspect='equal')
            ax1.set_xticks(range(len(subj_covs)))
            ax1.set_yticks(range(len(subj_covs)))
            ax1.set_xticklabels([f'被试{i+1}' for i in range(len(subj_covs))])
            ax1.set_yticklabels([f'被试{i+1}' for i in range(len(subj_covs))])
            ax1.set_title('(a) 不同被试左手类协方差差异\n（EA对齐后，Frobenius范数）', fontsize=11, fontweight='bold')
            plt.colorbar(im1, ax=ax1, fraction=0.046)

        # ========== 子图2: 多被试各类别协方差平均热力图 ==========
        ax2 = fig.add_subplot(gs[0, 1:])
        # 收集前6个被试的4类平均协方差
        all_avg_covs = []
        all_labels_for_covs = []
        all_subj_ids = []
        for s in range(n_show):
            subj = self.subjects_data[s]
            covs = subj['trajectories'][:, 0, band_idx, :, :]
            labels = subj['labels']
            for cls_idx in range(4):
                mask = labels == cls_idx
                if mask.sum() > 0:
                    avg_cov = covs[mask].mean(axis=0)
                    all_avg_covs.append(avg_cov)
                    all_labels_for_covs.append(cls_idx)
                    all_subj_ids.append(s)

        # 将所有平均协方差展平后计算两两距离
        n_covs = len(all_avg_covs)
        dist_mat = np.zeros((n_covs, n_covs))
        for i in range(n_covs):
            for j in range(n_covs):
                try:
                    dist_mat[i, j] = riemann_dist(all_avg_covs[i], all_avg_covs[j])
                except:
                    dist_mat[i, j] = 0

        im2 = ax2.imshow(dist_mat, cmap='viridis', aspect='auto')
        # 构造标签：每4个一组（4类），每组间加间隔线
        tick_labels = []
        for s in range(n_show):
            for c in range(4):
                tick_labels.append(f'S{s+1}-{CLASS_NAMES[c][:2]}')
        ax2.set_xticks(range(n_covs))
        ax2.set_yticks(range(n_covs))
        ax2.set_xticklabels(tick_labels, rotation=90, ha='right', fontsize=5)
        ax2.set_yticklabels(tick_labels, fontsize=5)
        # 加被试间隔线
        for s in range(1, n_show):
            ax2.axhline(y=s*4 - 0.5, color='white', linewidth=1.5)
            ax2.axvline(x=s*4 - 0.5, color='white', linewidth=1.5)
        ax2.set_title('(b) 多被试多类别黎曼距离矩阵\n（EA对齐后，块对角=同被试四类，块间=不同被试）',
                     fontsize=11, fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046)

        # ========== 子图3: t-SNE聚类（颜色=类别）==========
        ax3 = fig.add_subplot(gs[1, 0])
        all_covs_tsne = []
        all_labels_tsne = []
        all_subj_tsne = []
        for s in range(n_show):
            subj = self.subjects_data[s]
            covs = subj['trajectories'][:, 0, band_idx, :, :]  # [trials, 7, 7]
            labels = subj['labels']
            # 每类最多采样5个trial
            for cls_idx in range(4):
                mask = labels == cls_idx
                if mask.sum() == 0:
                    continue
                idx = np.where(mask)[0]
                n_sample = min(5, len(idx))
                sampled = np.random.choice(idx, n_sample, replace=False)
                all_covs_tsne.append(covs[sampled])
                all_labels_tsne.extend([cls_idx] * n_sample)
                all_subj_tsne.extend([s] * n_sample)

        all_covs_tsne = np.concatenate(all_covs_tsne, axis=0)  # [N, 7, 7]
        all_labels_tsne = np.array(all_labels_tsne)
        all_subj_tsne = np.array(all_subj_tsne)

        # 切空间映射
        covs_tensor = torch.tensor(all_covs_tsne, dtype=torch.float64)
        covs_norm = normalize_trace(covs_tensor)
        tangent = log_euclidean_mapping(covs_norm).cpu().numpy()

        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tangent)//5))
        X_tsne = tsne.fit_transform(tangent)

        for cls_idx in range(4):
            mask = all_labels_tsne == cls_idx
            ax3.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       c=COLORS[cls_idx], label=CLASS_NAMES[cls_idx],
                       alpha=0.7, s=50, edgecolors='none')
        ax3.set_xlabel('t-SNE 维度 1', fontsize=10)
        ax3.set_ylabel('t-SNE 维度 2', fontsize=10)
        ax3.set_title('(c) 切空间特征 t-SNE\n（颜色=类别，EA对齐后）', fontsize=11, fontweight='bold')
        ax3.legend(fontsize=8, loc='best')
        ax3.grid(True, alpha=0.3)

        # ========== 子图4: 被试分散度检查（颜色=被试编号）==========
        ax4 = fig.add_subplot(gs[1, 1])
        subj_colors = plt.cm.tab10(np.linspace(0, 1, n_show))
        for s in range(n_show):
            mask = all_subj_tsne == s
            ax4.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                       color=subj_colors[s], label=f'被试{s+1}',
                       alpha=0.7, s=50, edgecolors='none')
        ax4.set_xlabel('t-SNE 维度 1', fontsize=10)
        ax4.set_ylabel('t-SNE 维度 2', fontsize=10)
        ax4.set_title('(d) 被试分散度检查\n（颜色=被试编号，EA对齐后）', fontsize=11, fontweight='bold')
        ax4.legend(fontsize=6, loc='best', ncol=2)
        ax4.grid(True, alpha=0.3)

        # ========== 子图5: 距离统计（类内 vs 被试间同类 vs 类间）==========
        ax5 = fig.add_subplot(gs[1, 2])
        # 基于trial级协方差计算三种距离
        intra_cls_dists = []      # 同被试同类
        inter_subj_same_cls = []  # 不同被试同类
        inter_cls_dists = []      # 不同类（无论是否同被试）

        for s in range(n_show):
            subj = self.subjects_data[s]
            covs = subj['trajectories'][:, 0, band_idx, :, :]
            labels = subj['labels']
            for c in range(4):
                mask_c = labels == c
                if mask_c.sum() < 2:
                    continue
                idx_c = np.where(mask_c)[0]
                # 同被试同类距离
                for i in range(len(idx_c)):
                    for j in range(i+1, len(idx_c)):
                        try:
                            d = riemann_dist(covs[idx_c[i]], covs[idx_c[j]])
                            intra_cls_dists.append(d)
                        except:
                            pass

        # 不同被试同类 & 不同类
        for s1 in range(n_show):
            for s2 in range(s1+1, n_show):
                subj1 = self.subjects_data[s1]
                subj2 = self.subjects_data[s2]
                covs1 = subj1['trajectories'][:, 0, band_idx, :, :]
                labels1 = subj1['labels']
                covs2 = subj2['trajectories'][:, 0, band_idx, :, :]
                labels2 = subj2['labels']
                for i in range(len(labels1)):
                    for j in range(len(labels2)):
                        try:
                            d = riemann_dist(covs1[i], covs2[j])
                            if labels1[i] == labels2[j]:
                                inter_subj_same_cls.append(d)
                            else:
                                inter_cls_dists.append(d)
                        except:
                            pass

        bp = ax5.boxplot([intra_cls_dists, inter_subj_same_cls, inter_cls_dists],
                         labels=['类内\n（同被试）', '被试间同类\n（EA对齐）', '类间\n（不同类）'],
                         patch_artist=True,
                         medianprops=dict(color='red', linewidth=2))
        colors_box = ['#A8DADC', '#457B9D', '#E63946']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax5.set_ylabel('黎曼距离', fontsize=11)
        ax5.set_title('(e) 距离统计对比\n（EA对齐后，Mu频带）', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')

        # 添加说明文字
        if inter_subj_same_cls and intra_cls_dists:
            ratio = np.mean(inter_subj_same_cls) / np.mean(intra_cls_dists)
            ax5.text(0.98, 0.98, f'被试间同类/类内 = {ratio:.2f}\n（越接近1说明EA效果越好）',
                    transform=ax5.transAxes, ha='right', va='top',
                    fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.suptitle('EA（欧几里得对齐）效果可视化\n基于 trajectory_data 中已EA对齐的协方差数据',
                    fontsize=14, fontweight='bold', y=1.01)
        self.save('data_07_EA对齐效果')

    # ==================== 主入口 ====================
    def run_all(self):
        print("\n" + "=" * 60)
        print("🎨 EEG 数据层面可视化")
        print("=" * 60)

        self.plot_class_distribution()
        self.plot_channel_topology()
        self.plot_covariance_heatmaps()
        self.plot_eigenvalue_distribution()
        self.plot_tsne_tangent_space()
        self.plot_trajectory_evolution()
        self.plot_ea_alignment_effect()

        print(f"\n✅ 所有图形已保存到 {self.output_dir}/")


if __name__ == "__main__":
    viz = EEGDataVisualizer()
    viz.run_all()