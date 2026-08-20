# -*- coding: utf-8 -*-
"""
ICA 与 EA 处理效果评估可视化
生成多维度对比图，量化评估每一步预处理的效果

运行方式:
    python visualization/evaluate_ica_ea.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mne
from pathlib import Path
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform

import preprocessing
import alignment
import moabb_loader
from visualization.utils import riemann_distance_logeuc, normalize_trace

# ==================== 全局样式 ====================
# 完整的中文字体配置
plt.rcParams['font.sans-serif'] = [
    'Microsoft YaHei',  # 微软雅黑
    'SimHei',          # 黑体
    'Arial Unicode MS',
    'DejaVu Sans',
    'WenQuanYi Micro Hei',
    'PingFang SC',
    'STHeiti',
    'SimSun',
    'FangSong',
    'KaiTi',
    'sans-serif'
]
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

# 设置 mne 的字体
mne.set_log_level('WARNING')

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
CLASS_NAMES = ['左手', '右手', '双脚', '双手']


class ICAEAEvaluator:
    """
    评估 ICA 去噪与 EA 对齐效果的综合可视化器
    """

    def __init__(self, subjects_to_load=[1, 2, 3, 4, 5], output_dir="figures"):
        self.subjects_to_load = subjects_to_load
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # 中间状态存储
        self.raw_data = {}      # 选通道+重参考后（ICA前）
        self.ica_data = {}      # ICA去噪后
        self.epochs_no_ea = {}  # 滤波+切分后（EA前）
        self.epochs_ea = {}     # EA对齐后
        self.ica_metrics = {}   # ICA统计指标

    def save(self, name):
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 已保存: {path.name}")
        plt.close()

    # ==================== 数据加载与处理 ====================

    def load_and_process(self):
        """加载被试并按步骤保存中间状态"""
        print(f"🔧 加载并处理 {len(self.subjects_to_load)} 个被试...")
        for _, _, raw, meta in moabb_loader.iter_all_subjects(
            subjects_to_load=self.subjects_to_load, verbose=False
        ):
            subj_key = meta['subject_key']
            print(f"  ⏳ {subj_key} ...", end=" ", flush=True)

            # Step 1: 64 通道 1-45Hz 滤波（ICA 前）
            raw_1_45 = raw.copy().filter(l_freq=1.0, h_freq=45.0, picks='eeg', verbose=False)
            self.raw_data[subj_key] = raw_1_45.copy()

            # Step 2: ICA 去噪（64 通道）
            raw_ica, metrics = preprocessing.ica_denoise(raw_1_45)
            self.ica_data[subj_key] = raw_ica.copy()
            self.ica_metrics[subj_key] = metrics

            # Step 3: 选 22 通道 + 8-30Hz 滤波 + Epochs（EA 前）
            raw_sel = preprocessing.select_channels_physionetmi(raw_ica)
            raw_filt = preprocessing.filter(raw_sel)
            left, right, feet, hands = preprocessing.epochs(raw_filt, tmin=-1.0, tmax=3.5)
            self.epochs_no_ea[subj_key] = {
                'left': left, 'right': right, 'feet': feet, 'hands': hands
            }

            # Step 4: EA 对齐
            P = alignment.compute_ea_matrix_from_epochs([left, right, feet, hands])
            left_a, right_a, feet_a, hands_a = alignment.apply_ea_to_epochs_list(
                [left, right, feet, hands], P
            )
            self.epochs_ea[subj_key] = {
                'left': left_a, 'right': right_a, 'feet': feet_a, 'hands': hands_a
            }
            print("完成")

    # ==================== ICA 评估图表 ====================

    def plot_ica_timeseries(self, subj_key=None, duration=5.0):
        """ICA 前后时域波形对比（关键运动想象通道）"""
        if subj_key is None:
            subj_key = list(self.raw_data.keys())[0]
        raw_before = self.raw_data[subj_key]
        raw_after = self.ica_data[subj_key]

        # 确保使用两个数据都存在的 EEG 通道
        picks = ['Cz', 'C3', 'C4', 'FCz', 'Pz']
        picks = [ch for ch in picks if ch in raw_before.ch_names and ch in raw_after.ch_names]

        fig, axes = plt.subplots(len(picks), 2, figsize=(14, 2.5 * len(picks)), sharex=True)
        if len(picks) == 1:
            axes = axes.reshape(1, -1)

        n_samples = int(duration * raw_before.info['sfreq'])
        t = np.arange(n_samples) / raw_before.info['sfreq']

        for i, ch in enumerate(picks):
            idx = raw_before.ch_names.index(ch)
            data_b = raw_before.get_data()[idx, :n_samples]
            data_a = raw_after.get_data()[idx, :n_samples]

            axes[i, 0].plot(t, data_b * 1e6, color=COLORS[0], linewidth=0.8)
            axes[i, 0].set_ylabel(f'{ch}\n(μV)', fontsize=9)
            if i == 0:
                axes[i, 0].set_title('ICA 前', fontsize=11, fontweight='bold')
            axes[i, 0].grid(True, alpha=0.3)

            axes[i, 1].plot(t, data_a * 1e6, color=COLORS[1], linewidth=0.8)
            if i == 0:
                axes[i, 1].set_title('ICA 后', fontsize=11, fontweight='bold')
            axes[i, 1].grid(True, alpha=0.3)

        axes[-1, 0].set_xlabel('时间 (s)', fontsize=10)
        axes[-1, 1].set_xlabel('时间 (s)', fontsize=10)
        fig.suptitle(f'{subj_key} 时域波形对比 (ICA 去噪前后)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('ica_01_时域波形对比')

    def plot_ica_psd(self, subj_key=None):
        """ICA 前后功率谱密度对比"""
        if subj_key is None:
            subj_key = list(self.raw_data.keys())[0]
        raw_before = self.raw_data[subj_key]
        raw_after = self.ica_data[subj_key]

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        picks = mne.pick_types(raw_before.info, eeg=True)

        for ax, raw, title, color in zip(
            axes, [raw_before, raw_after], ['ICA 前', 'ICA 后'], [COLORS[0], COLORS[1]]
        ):
            psds, freqs = mne.time_frequency.psd_array_welch(
                raw.get_data(picks=picks), sfreq=raw.info['sfreq'],
                fmin=1, fmax=40, n_fft=512, n_overlap=256, verbose=False
            )
            psd_mean = psds.mean(axis=0)
            ax.semilogy(freqs, psd_mean * 1e12, color=color, linewidth=1.5)
            ax.axvspan(8, 12, alpha=0.15, color='blue', label='Mu节律 (8-12Hz)')
            ax.axvspan(13, 30, alpha=0.15, color='red', label='Beta节律 (13-30Hz)')
            ax.set_xlabel('频率 (Hz)', fontsize=11)
            ax.set_ylabel('功率谱密度 (μV²/Hz)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(loc='upper right', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([1, 40])

        fig.suptitle(f'{subj_key} 功率谱密度对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('ica_02_PSD对比')

    def plot_ica_topography(self, subj_key=None):
        """ICA 前后频段功率拓扑图对比"""
        if subj_key is None:
            subj_key = list(self.raw_data.keys())[0]
        raw_before = self.raw_data[subj_key]
        raw_after = self.ica_data[subj_key]

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        configs = [
            (raw_before, (8, 12), 'ICA去噪前 - Mu节律 (8-12Hz)', COLORS[0]),
            (raw_after, (8, 12), 'ICA去噪后 - Mu节律 (8-12Hz)', COLORS[1]),
            (raw_before, (13, 30), 'ICA去噪前 - Beta节律 (13-30Hz)', COLORS[0]),
            (raw_after, (13, 30), 'ICA去噪后 - Beta节律 (13-30Hz)', COLORS[1]),
        ]

        for ax, (raw, band, title, _) in zip(axes.flat, configs):
            picks = mne.pick_types(raw.info, eeg=True)
            psds, freqs = mne.time_frequency.psd_array_welch(
                raw.get_data(picks=picks), sfreq=raw.info['sfreq'],
                fmin=band[0], fmax=band[1], n_fft=512, n_overlap=256, verbose=False
            )
            freq_mask = (freqs >= band[0]) & (freqs <= band[1])
            power = psds[:, freq_mask].mean(axis=1)

            # 确保有电极位置信息
            info = raw.copy().pick(picks).info
            if info.get_montage() is None:
                try:
                    info.set_montage('standard_1020', match_case=False, on_missing='warn')
                except:
                    pass
            
            im, _ = mne.viz.plot_topomap(power, info, axes=ax, show=False,
                                         cmap='RdBu_r', contours=6)
            ax.set_title(title, fontsize=11, fontweight='bold')
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label('功率 (μV²)', fontsize=9)

        fig.suptitle(f'{subj_key} 频段功率拓扑图对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('ica_03_拓扑图对比')

    def plot_ica_metrics_summary(self):
        """多被试 ICA 指标统计汇总"""
        if not self.ica_metrics:
            return

        subjects = list(self.ica_metrics.keys())
        var_red = [self.ica_metrics[s].get('variance_reduction_percent', 0) for s in subjects]
        amp_red = [self.ica_metrics[s].get('amplitude_reduction_percent', 0) for s in subjects]
        kurt_red = [self.ica_metrics[s].get('kurtosis_reduction', 0) for s in subjects]
        n_comp = [self.ica_metrics[s].get('components_excluded', 0) for s in subjects]

        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        x = np.arange(len(subjects))
        width = 0.6

        plots = [
            (var_red, '方差降低率', '降低率 (%)', COLORS[0]),
            (amp_red, '振幅降低率', '降低率 (%)', COLORS[1]),
            (kurt_red, '峰度降低量', '峰度降低量', COLORS[2]),
            (n_comp, '排除 ICA 成分数', '成分数', COLORS[3]),
        ]

        for ax, data, title, ylabel, color in zip(axes.flat, *zip(*plots)):
            bars = ax.bar(x, data, width, color=color, edgecolor='black', alpha=0.85)
            ax.set_xticks(x)
            ax.set_xticklabels([s.split('_')[-1] for s in subjects], rotation=45, ha='right')
            ax.set_ylabel(ylabel, fontsize=10)
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
            ax.grid(True, alpha=0.3, axis='y')
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8)

        fig.suptitle('ICA 去噪效果统计汇总', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('ica_04_指标统计汇总')

    # ==================== EA 评估图表 ====================

    def _extract_epochs_numpy(self, epochs_dict):
        """将各类别 Epochs 合并为 numpy 数组并打标签"""
        data_list, labels_list = [], []
        for label_idx, cls_name in enumerate(['left', 'right', 'feet', 'hands']):
            arr = epochs_dict[cls_name].get_data()  # (trials, ch, time)
            data_list.append(arr)
            labels_list.append(np.full(len(arr), label_idx))
        return np.concatenate(data_list, axis=0), np.concatenate(labels_list)

    def plot_ea_tsne(self):
        """EA 前后 t-SNE 分布对比"""
        print("  ⏳ 计算 t-SNE...", end=" ", flush=True)
        data_no_ea, labels_no_ea, subj_no_ea = [], [], []
        data_ea, labels_ea, subj_ea = [], [], []

        for subj_idx, (subj_key, epochs_dict) in enumerate(self.epochs_no_ea.items()):
            arr, labs = self._extract_epochs_numpy(epochs_dict)
            data_no_ea.append(arr)
            labels_no_ea.append(labs)
            subj_no_ea.append(np.full(len(arr), subj_idx))

            arr_ea, labs_ea = self._extract_epochs_numpy(self.epochs_ea[subj_key])
            data_ea.append(arr_ea)
            labels_ea.append(labs_ea)
            subj_ea.append(np.full(len(arr_ea), subj_idx))

        X_no_ea = np.concatenate(data_no_ea, axis=0)
        X_ea = np.concatenate(data_ea, axis=0)
        y_lab = np.concatenate(labels_no_ea)
        y_subj = np.concatenate(subj_no_ea)

        # 特征：时间维度平均
        feat_no_ea = X_no_ea.mean(axis=2)
        feat_ea = X_ea.mean(axis=2)

        # 随机降采样加速
        n_samples = min(2000, len(feat_no_ea))
        n_samples = max(100, n_samples)  # 确保至少有100个样本
        rng = np.random.RandomState(42)
        idx = rng.choice(len(feat_no_ea), n_samples, replace=False)

        feat_no_ea_s = feat_no_ea[idx]
        feat_ea_s = feat_ea[idx]
        y_lab_s = y_lab[idx]
        y_subj_s = y_subj[idx]

        # 确保 perplexity 不超过样本数
        perplexity = min(30, len(feat_no_ea_s) // 3)
        perplexity = max(5, perplexity)

        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
        emb_no_ea = tsne.fit_transform(feat_no_ea_s)
        emb_ea = tsne.fit_transform(feat_ea_s)
        print("完成")

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # 左图：EA 前按被试着色（预期被试聚类明显）
        scatter1 = axes[0].scatter(emb_no_ea[:, 0], emb_no_ea[:, 1], c=y_subj_s,
                                   cmap='tab10', s=15, alpha=0.7)
        axes[0].set_title('EA对齐前 (按被试着色)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('t-SNE 维度1')
        axes[0].set_ylabel('t-SNE 维度2')
        plt.colorbar(scatter1, ax=axes[0], label='被试编号')

        # 右图：EA 后按类别着色（预期类别聚类更明显）
        for cls in range(4):
            mask = y_lab_s == cls
            axes[1].scatter(emb_ea[mask, 0], emb_ea[mask, 1],
                           color=COLORS[cls], s=15, alpha=0.7, label=CLASS_NAMES[cls])
        axes[1].set_title('EA对齐后 (按类别着色)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('t-SNE 维度1')
        axes[1].set_ylabel('t-SNE 维度2')
        axes[1].legend(loc='best', fontsize=9)

        fig.suptitle('EA 对齐前后 t-SNE 分布对比', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('ea_01_tSNE对比')

    def _compute_subject_mean_covs(self, use_ea=False):
        """计算每个被试的跨类别平均协方差矩阵"""
        epochs_source = self.epochs_ea if use_ea else self.epochs_no_ea
        covs = []
        for subj_key, epochs_dict in epochs_source.items():
            all_data = np.concatenate([
                epochs_dict[cls].get_data() for cls in ['left', 'right', 'feet', 'hands']
            ], axis=0)
            all_data_2d = all_data.transpose(1, 0, 2).reshape(all_data.shape[1], -1)
            cov = all_data_2d @ all_data_2d.T
            cov = cov / (np.trace(cov) + 1e-10)
            covs.append(cov)
        return np.stack(covs)

    def plot_ea_distance_matrix(self):
        """被试间黎曼距离热力图"""
        print("  ⏳ 计算黎曼距离矩阵...", end=" ", flush=True)
        covs_no_ea = self._compute_subject_mean_covs(use_ea=False)
        covs_ea = self._compute_subject_mean_covs(use_ea=True)
        n = len(covs_no_ea)

        dist_no = np.zeros((n, n))
        dist_ea = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d1 = riemann_distance_logeuc(covs_no_ea[i], covs_no_ea[j])
                d2 = riemann_distance_logeuc(covs_ea[i], covs_ea[j])
                dist_no[i, j] = dist_no[j, i] = d1
                dist_ea[i, j] = dist_ea[j, i] = d2
        print("完成")

        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        subj_labels = [k.split('_')[-1] for k in self.epochs_no_ea.keys()]

        for ax, dist, title in zip(axes, [dist_no, dist_ea], ['EA 前', 'EA 后']):
            sns.heatmap(dist, annot=True, fmt='.2f', cmap='YlOrRd',
                       xticklabels=subj_labels, yticklabels=subj_labels,
                       square=True, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
            ax.set_title(f'{title} 被试间黎曼距离', fontsize=12, fontweight='bold')

        fig.suptitle('EA 对齐前后 被试间协方差距离热力图', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('ea_02_距离热力图')

    def plot_ea_inter_subject_variance(self):
        """跨被试通道方差对比（EA 应显著降低被试间差异）"""
        print("  ⏳ 计算跨被试方差...", end=" ", flush=True)
        means_no_ea = []
        means_ea = []
        for subj_key in self.epochs_no_ea.keys():
            arr_no = np.concatenate([
                self.epochs_no_ea[subj_key][cls].get_data()
                for cls in ['left', 'right', 'feet', 'hands']
            ], axis=0).mean(axis=0)
            arr_ea = np.concatenate([
                self.epochs_ea[subj_key][cls].get_data()
                for cls in ['left', 'right', 'feet', 'hands']
            ], axis=0).mean(axis=0)
            means_no_ea.append(arr_no)
            means_ea.append(arr_ea)

        means_no_ea = np.stack(means_no_ea)
        means_ea = np.stack(means_ea)

        std_no = means_no_ea.std(axis=0).mean(axis=1)  # axis=0跨被试求标准差，axis=1跨时间求平均
        std_ea = means_ea.std(axis=0).mean(axis=1)

        ch_names = self.epochs_no_ea[list(self.epochs_no_ea.keys())[0]]['left'].ch_names
        print("完成")

        fig, ax = plt.subplots(figsize=(12, 5))
        x = np.arange(len(ch_names))
        width = 0.35
        ax.bar(x - width / 2, std_no, width, label='EA 前', color=COLORS[0], alpha=0.85, edgecolor='black')
        ax.bar(x + width / 2, std_ea, width, label='EA 后', color=COLORS[1], alpha=0.85, edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(ch_names, rotation=45, ha='right')
        ax.set_ylabel('跨被试标准差 (μV)', fontsize=11)
        ax.set_title('EA 对齐前后 跨被试通道波形标准差', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')

        overall_red = (std_no.mean() - std_ea.mean()) / std_no.mean() * 100
        ax.text(0.98, 0.95, f'总体降低幅度: {overall_red:.1f}%',
                transform=ax.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        plt.tight_layout()
        self.save('ea_03_跨被试方差')

    # ==================== 综合流程汇总 ====================

    def plot_processing_pipeline_summary(self):
        """全流程信号质量演变（从原始到 EA 后）"""
        print("  ⏳ 生成全流程演变图...", end=" ", flush=True)
        subj_key = list(self.raw_data.keys())[0]
        raw_orig = self.raw_data[subj_key]
        raw_ica = self.ica_data[subj_key]

        epochs_dict = self.epochs_no_ea[subj_key]
        epochs_no_ea_arr = np.concatenate([
            epochs_dict[cls].get_data() for cls in ['left', 'right', 'feet', 'hands']
        ], axis=0)
        epochs_ea_arr = np.concatenate([
            self.epochs_ea[subj_key][cls].get_data() for cls in ['left', 'right', 'feet', 'hands']
        ], axis=0)

        # 定位 Cz 通道（确保在两个数据中都存在）
        cz_idx = None
        for i, ch in enumerate(epochs_dict['left'].ch_names):
            if ch == 'Cz':
                cz_idx = i
                break
        if cz_idx is None:
            cz_idx = 0

        # 检查 cz_idx 是否在 raw_orig 的通道范围内
        eeg_picks_raw = mne.pick_types(raw_orig.info, eeg=True)
        if cz_idx >= len(eeg_picks_raw):
            cz_idx = 0
        actual_cz_idx_in_raw = eeg_picks_raw[cz_idx] if len(eeg_picks_raw) > cz_idx else 0

        fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=False)

        # 步骤 1: 选通道+重参考
        data_1 = raw_orig.get_data()[actual_cz_idx_in_raw, :int(5 * raw_orig.info['sfreq'])]
        t1 = np.arange(len(data_1)) / raw_orig.info['sfreq']
        axes[0].plot(t1, data_1 * 1e6, color=COLORS[0], linewidth=0.7)
        axes[0].set_ylabel('Cz通道 (μV)', fontsize=10)
        axes[0].set_title('步骤1：选择通道 + 重参考后', fontsize=11, fontweight='bold', loc='left')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('时间 (s)', fontsize=10)

        # 步骤 2: ICA 去噪后
        eeg_picks_ica = mne.pick_types(raw_ica.info, eeg=True)
        actual_cz_idx_ica = eeg_picks_ica[min(cz_idx, len(eeg_picks_ica)-1)] if len(eeg_picks_ica) > 0 else 0
        data_2 = raw_ica.get_data()[actual_cz_idx_ica, :int(5 * raw_ica.info['sfreq'])]
        t2 = np.arange(len(data_2)) / raw_ica.info['sfreq']
        axes[1].plot(t2, data_2 * 1e6, color=COLORS[1], linewidth=0.7)
        axes[1].set_ylabel('Cz通道 (μV)', fontsize=10)
        axes[1].set_title('步骤2：ICA去噪后', fontsize=11, fontweight='bold', loc='left')
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlabel('时间 (s)', fontsize=10)

        # 步骤 3: EA 对齐前 Epochs 平均
        data_3 = epochs_no_ea_arr[:, cz_idx, :].mean(axis=0)
        t3 = np.arange(len(data_3)) / epochs_dict['left'].info['sfreq'] + epochs_dict['left'].tmin
        axes[2].plot(t3, data_3 * 1e6, color=COLORS[2], linewidth=1.2)
        axes[2].set_ylabel('Cz通道 (μV)', fontsize=10)
        axes[2].set_title('步骤3：滤波 + 切分Epochs (EA对齐前，跨试次平均)', fontsize=11, fontweight='bold', loc='left')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlabel('时间 (s)', fontsize=10)

        # 步骤 4: EA 对齐后
        data_4 = epochs_ea_arr[:, cz_idx, :].mean(axis=0)
        axes[3].plot(t3, data_4 * 1e6, color=COLORS[3], linewidth=1.2)
        axes[3].set_ylabel('Cz通道 (μV)', fontsize=10)
        axes[3].set_title('步骤4：EA欧氏对齐后 (跨试次平均)', fontsize=11, fontweight='bold', loc='left')
        axes[3].set_xlabel('时间 (s)', fontsize=11)
        axes[3].grid(True, alpha=0.3)

        fig.suptitle(f'{subj_key} 预处理全流程信号演变 (Cz 通道)', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('pipeline_01_全流程演变')
        print("完成")

    # ==================== 主控 ====================

    def run_all(self):
        print("=" * 60)
        print("ICA & EA 处理效果评估可视化")
        print("=" * 60)
        self.load_and_process()

        print("\n📊 生成 ICA 评估图表...")
        self.plot_ica_timeseries()
        self.plot_ica_psd()
        self.plot_ica_topography()
        self.plot_ica_metrics_summary()

        print("\n📊 生成 EA 评估图表...")
        self.plot_ea_tsne()
        self.plot_ea_distance_matrix()
        self.plot_ea_inter_subject_variance()

        print("\n📊 生成全流程汇总...")
        self.plot_processing_pipeline_summary()

        print(f"\n✅ 全部完成！图表保存在: {self.output_dir.absolute()}")


if __name__ == '__main__':
    evaluator = ICAEAEvaluator(subjects_to_load=[1, 2, 3, 4, 5])
    evaluator.run_all()