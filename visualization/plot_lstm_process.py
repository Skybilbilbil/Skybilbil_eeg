# -*- coding: utf-8 -*-
"""
LSTM 分类过程可视化
展示模型如何随时间窗口逐步做出分类决策
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# ==================== 全局样式 ====================
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
CLASS_NAMES = ['左手', '右手', '双脚', '双手']
NUM_CLASSES = 4


# ========== 模型定义（与 Classification.py 一致）==========
class FBCSP_RiemannianLSTM(nn.Module):
    def __init__(self, input_size=140, hidden_size=32, num_classes=4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        self.lstm = nn.LSTM(64, hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        # 时间注意力
        self.attention_vector = nn.Parameter(torch.randn(hidden_size * 2))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes)
        )

    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.size()
        x_flat = x.contiguous().view(-1, x.size(-1))
        x_proj = self.proj(x_flat)
        x_proj_seq = x_proj.view(batch_size, seq_len, -1)
        out, _ = self.lstm(x_proj_seq)
        attn_scores = torch.matmul(out, self.attention_vector)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.fc(pooled)
        if return_attention:
            return logits, attn_weights
        return logits

    def forward_intermediate(self, x):
        """返回中间层输出，用于可视化"""
        batch_size, seq_len, _ = x.size()
        x_flat = x.contiguous().view(-1, x.size(-1))
        x_proj = self.proj(x_flat)
        x_proj_seq = x_proj.view(batch_size, seq_len, -1)
        out, (h_n, c_n) = self.lstm(x_proj_seq)
        attn_scores = torch.matmul(out, self.attention_vector)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.fc(pooled)
        return {
            'proj_out': x_proj_seq,
            'lstm_out': out,
            'hidden_final': h_n,
            'pooled': pooled,
            'logits': logits,
            'attention_weights': attn_weights
        }


# ========== 数据预处理工具函数 ==========
def normalize_trace(covs):
    traces = torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    return covs / traces


def log_euclidean_mapping(cov_matrices):
    C = cov_matrices.shape[-1]
    evals, evecs = torch.linalg.eigh(cov_matrices)
    log_evals = torch.log(torch.clamp(evals, min=1e-7))
    log_D = torch.diag_embed(log_evals)
    log_C = torch.matmul(torch.matmul(evecs, log_D), evecs.mT)
    idx = torch.triu_indices(C, C)
    tangent_vectors = log_C[:, idx[0], idx[1]]
    off_diag_mask = (idx[0] != idx[1])
    tangent_vectors[:, off_diag_mask] *= np.sqrt(2.0)
    return tangent_vectors.to(torch.float32)


# ========== 加载测试数据 ==========
def load_test_data(min_windows=None):
    input_file = Path("trajectory_data") / "trajectories_500ms_win.pkl"
    with open(input_file, 'rb') as f:
        dataset = pickle.load(f)

    subjects_data = dataset['subject_data']
    num_bands = 5
    num_channels = 7
    TANGENT_DIM_PER_BAND = num_channels * (num_channels + 1) // 2
    TANGENT_DIM = num_bands * TANGENT_DIM_PER_BAND

    test_indices = []
    test_keys = []
    for i, subj in enumerate(subjects_data):
        key = subj.get('subject_key', '')
        if "PhysionetMI_S" in key:
            num = int(key.split("S")[-1])
            if 1 <= num <= 10:
                test_indices.append(i)
                test_keys.append(key)
    if not test_indices:
        test_indices = list(range(min(10, len(subjects_data))))
        test_keys = [subjects_data[i].get('subject_key', f"S{i+1}") for i in test_indices]

    test_covs_list, test_labels_list = [], []
    for i in test_indices:
        test_covs_list.append(subjects_data[i]['trajectories'])
        test_labels_list.append(subjects_data[i]['labels'])

    if min_windows is None:
        min_windows = min(c.shape[1] for c in test_covs_list)
    test_covs_list = [c[:, :min_windows, :, :, :] for c in test_covs_list]
    test_trials_per_subject = [c.shape[0] for c in test_covs_list]

    test_covs = np.concatenate(test_covs_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    band_tangents_test = []
    for b in range(num_bands):
        band_test = test_covs[:, :, b, :, :]
        t_test = torch.tensor(band_test.reshape(-1, num_channels, num_channels),
                              dtype=torch.float64, device=device)
        t_test = normalize_trace(t_test)
        test_band_tangent = log_euclidean_mapping(t_test)
        band_tangents_test.append(test_band_tangent)

    test_tangent = torch.cat(band_tangents_test, dim=1)
    X_test = test_tangent.view(test_covs.shape[0], min_windows, TANGENT_DIM)

    model_weights_file = Path("results") / "lstm_model_weights.pth"
    if model_weights_file.exists():
        ckpt = torch.load(model_weights_file, map_location=device)
        feat_mean = ckpt['feat_mean'].to(device)
        feat_std = ckpt['feat_std'].to(device)
        X_test = (X_test - feat_mean.unsqueeze(0)) / feat_std.unsqueeze(0)
    else:
        print("⚠️ 未找到模型权重文件，无法标准化数据")
        return None, None, None, None

    return X_test, test_labels, test_trials_per_subject, test_keys


# ========== 逐步截断预测 ==========
def stepwise_predict(model, X_test, max_steps=None):
    model.eval()
    n_trials, total_steps, feat_dim = X_test.shape
    if max_steps is None:
        max_steps = total_steps
    probs_history = []
    with torch.no_grad():
        for t in range(1, max_steps + 1):
            X_trunc = X_test[:, :t, :]
            outputs = model(X_trunc)
            probs = torch.softmax(outputs, dim=1)
            probs_history.append(probs.cpu().numpy())
    return np.stack(probs_history, axis=1)


# ========== 可视化类 ==========
class LSTMProcessVisualizer:
    def __init__(self):
        self.output_dir = Path("figures")
        self.output_dir.mkdir(exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.X_test = None
        self.y_test = None
        self.probs_history = None
        self._load()

    def _load(self):
        model_weights_file = Path("results") / "lstm_model_weights.pth"
        if not model_weights_file.exists():
            print("⚠️ 未找到模型权重文件，请先运行 Classification.py")
            return
        ckpt = torch.load(model_weights_file, map_location=self.device)
        self.model = FBCSP_RiemannianLSTM(
            input_size=ckpt['input_size'],
            hidden_size=ckpt['hidden_size'],
            num_classes=ckpt['num_classes']
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        print(f"✅ 加载模型: hidden={ckpt['hidden_size']}, classes={ckpt['num_classes']}")

        X_test, y_test, trials_per_subj, keys = load_test_data(min_windows=ckpt.get('min_windows'))
        if X_test is None:
            return
        self.X_test = X_test.to(self.device)
        self.y_test = y_test

        print("  🔄 正在进行逐步截断预测...")
        self.probs_history = stepwise_predict(self.model, self.X_test)
        print(f"  ✅ 概率历史 shape: {self.probs_history.shape}")

    def save(self, name):
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 已保存: {path.name}")
        plt.close()

    def _check_ready(self):
        if self.probs_history is None:
            print("  ⚠️ 数据未准备好，跳过")
            return False
        return True

    # ==================== 图1: 单个试次分类概率演化 ====================
    def plot_stepwise_prob_evolution(self):
        if not self._check_ready():
            return
        n_steps = self.probs_history.shape[1]
        time_axis = np.arange(1, n_steps + 1) * 0.5
        fig, axes = plt.subplots(2, 4, figsize=(18, 8))
        axes = axes.flatten()
        for cls in range(4):
            cls_mask = self.y_test == cls
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) == 0:
                continue
            final_preds = np.argmax(self.probs_history[:, -1, :], axis=1)
            correct_idx = cls_indices[final_preds[cls_indices] == cls]
            wrong_idx = cls_indices[final_preds[cls_indices] != cls]
            selected, labels_sel = [], []
            if len(correct_idx) > 0:
                selected.append(correct_idx[0])
                labels_sel.append('正确')
            if len(wrong_idx) > 0:
                selected.append(wrong_idx[0])
                labels_sel.append('错误')
            for row, (idx, label) in enumerate(zip(selected, labels_sel)):
                ax = axes[row * 4 + cls]
                probs = self.probs_history[idx]
                for c in range(4):
                    style = '-' if c == cls else '--'
                    alpha = 1.0 if c == cls else 0.5
                    lw = 2.5 if c == cls else 1.5
                    ax.plot(time_axis, probs[:, c], style, color=COLORS[c],
                           linewidth=lw, alpha=alpha, label=CLASS_NAMES[c])
                final_pred = np.argmax(probs[-1])
                final_prob = probs[-1, final_pred]
                ax.axvline(x=time_axis[-1], color='gray', linestyle=':', alpha=0.5)
                ax.set_xlabel('时间 (秒)', fontsize=9)
                ax.set_ylabel('预测概率', fontsize=9)
                ax.set_ylim([-0.05, 1.05])
                ax.set_title(f'{CLASS_NAMES[cls]} | {label}分类\n最终: {CLASS_NAMES[final_pred]} ({final_prob:.2f})',
                            fontsize=9, fontweight='bold')
                ax.legend(fontsize=6, loc='best')
                ax.grid(True, alpha=0.3)
        plt.suptitle('LSTM 分类概率随时间演化\n（每类展示一个正确和一个错误分类试次）',
                    fontsize=13, fontweight='bold', y=1.01)
        plt.tight_layout()
        self.save('lstm_01_逐步分类概率演化')

    # ==================== 图2: 正确vs错误演化对比 ====================
    def plot_correct_vs_wrong_evolution(self):
        if not self._check_ready():
            return
        n_steps = self.probs_history.shape[1]
        time_axis = np.arange(1, n_steps + 1) * 0.5
        final_preds = np.argmax(self.probs_history[:, -1, :], axis=1)
        correct_mask = final_preds == self.y_test
        wrong_mask = ~correct_mask
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax_idx, (mask, title) in enumerate([(correct_mask, '(a) 正确分类样本'),
                                                  (wrong_mask, '(b) 错误分类样本')]):
            ax = axes[ax_idx]
            for cls in range(4):
                cls_mask = (self.y_test == cls) & mask
                if cls_mask.sum() == 0:
                    continue
                mean_probs = self.probs_history[cls_mask].mean(axis=0)
                std_probs = self.probs_history[cls_mask].std(axis=0)
                ax.plot(time_axis, mean_probs[:, cls], '-', color=COLORS[cls],
                       linewidth=2.5, label=CLASS_NAMES[cls])
                ax.fill_between(time_axis,
                               mean_probs[:, cls] - std_probs[:, cls],
                               mean_probs[:, cls] + std_probs[:, cls],
                               color=COLORS[cls], alpha=0.2)
            ax.set_xlabel('时间 (秒)', fontsize=11)
            ax.set_ylabel('预测概率', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.set_ylim([-0.05, 1.05])
            ax.grid(True, alpha=0.3)
        plt.suptitle('LSTM 分类置信度演化：正确 vs 错误样本', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('lstm_02_正确错误演化对比')

    # ==================== 图3: 决策热力图 ====================
    def plot_decision_heatmap(self):
        if not self._check_ready():
            return
        n_steps = self.probs_history.shape[1]
        fig, axes = plt.subplots(1, 4, figsize=(16, 5))
        for cls in range(4):
            cls_mask = self.y_test == cls
            cls_indices = np.where(cls_mask)[0]
            if len(cls_indices) == 0:
                continue
            final_probs_cls = self.probs_history[cls_indices, -1, cls]
            sort_idx = np.argsort(final_probs_cls)[::-1]
            sorted_probs = self.probs_history[cls_indices[sort_idx], :, cls]
            ax = axes[cls]
            im = ax.imshow(sorted_probs, aspect='auto', cmap='RdYlGn',
                          vmin=0, vmax=1, interpolation='nearest')
            ax.set_xlabel('时间窗序号', fontsize=10)
            ax.set_ylabel('试次（按最终置信度排序）', fontsize=10)
            ax.set_title(f'{CLASS_NAMES[cls]}', fontsize=11, fontweight='bold')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='概率')
        plt.suptitle('LSTM 决策置信度热力图（真实标签各类别）', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('lstm_03_决策热力图')

    # ==================== 图4: 最终概率分布 ====================
    def plot_final_probability_distribution(self):
        if not self._check_ready():
            return
        final_probs = np.max(self.probs_history[:, -1, :], axis=1)
        final_preds = np.argmax(self.probs_history[:, -1, :], axis=1)
        correct_mask = final_preds == self.y_test
        wrong_mask = ~correct_mask
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax_idx, (mask, title) in enumerate([(correct_mask, '(a) 正确分类样本'),
                                                  (wrong_mask, '(b) 错误分类样本')]):
            ax = axes[ax_idx]
            data_to_plot = [final_probs[(self.y_test == cls) & mask] for cls in range(4)]
            bp = ax.boxplot(data_to_plot, labels=CLASS_NAMES, patch_artist=True,
                           medianprops=dict(color='red', linewidth=2))
            for patch, color in zip(bp['boxes'], COLORS):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)
            ax.set_ylabel('最终预测概率', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_ylim([0, 1.05])
            ax.grid(True, alpha=0.3, axis='y')
        plt.suptitle('最终预测概率分布', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('lstm_04_最终概率分布')

    # ==================== 图5: LSTM隐藏状态t-SNE ====================
    def plot_lstm_hidden_tsne(self):
        if self.X_test is None or self.model is None:
            print("  ⚠️ 模型或数据未加载，跳过")
            return
        print("  🔄 提取 LSTM 隐藏状态...")
        with torch.no_grad():
            intermediates = self.model.forward_intermediate(self.X_test)
            lstm_out = intermediates['lstm_out']
        hidden_mean = lstm_out.mean(dim=1).cpu().numpy()
        n_samples = min(800, len(hidden_mean))
        idx = np.random.choice(len(hidden_mean), n_samples, replace=False)
        X_plot = hidden_mean[idx]
        y_plot = self.y_test[idx]
        print("  🔄 运行 t-SNE...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, n_samples//5))
        X_tsne = tsne.fit_transform(X_plot)
        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in range(4):
            mask = y_plot == cls
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=COLORS[cls], label=CLASS_NAMES[cls],
                      alpha=0.6, s=35, edgecolors='none')
        ax.set_xlabel('t-SNE 维度 1', fontsize=11)
        ax.set_ylabel('t-SNE 维度 2', fontsize=11)
        ax.set_title('LSTM 隐藏状态 t-SNE 可视化\n（双向LSTM输出平均）', fontsize=12, fontweight='bold')
        ax.legend(fontsize=10, framealpha=0.9)
        plt.tight_layout()
        self.save('lstm_05_隐藏状态tSNE')

    # ==================== 图6: LSTM时间注意力权重 ====================
    def plot_temporal_attention(self):
        if self.X_test is None or self.model is None:
            print("  ⚠️ 模型或数据未加载，跳过")
            return
        print("  🔄 提取 LSTM 时间注意力权重...")
        with torch.no_grad():
            intermediates = self.model.forward_intermediate(self.X_test)
            attn_weights = intermediates['attention_weights'].cpu().numpy()  # [n_samples, n_steps]
        n_steps = attn_weights.shape[1]
        time_axis = np.arange(1, n_steps + 1) * 0.5

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        axes = axes.flatten()

        # (a) 各类别平均注意力权重
        ax = axes[0]
        for cls in range(4):
            cls_mask = self.y_test == cls
            if cls_mask.sum() == 0:
                continue
            mean_attn = attn_weights[cls_mask].mean(axis=0)
            std_attn = attn_weights[cls_mask].std(axis=0)
            ax.plot(time_axis, mean_attn, '-', color=COLORS[cls], linewidth=2.5, label=CLASS_NAMES[cls])
            ax.fill_between(time_axis, mean_attn - std_attn, mean_attn + std_attn,
                           color=COLORS[cls], alpha=0.2)
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_ylabel('注意力权重', fontsize=10)
        ax.set_title('(a) 各类别平均注意力权重', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.set_ylim([0, max(0.5, attn_weights.max() * 1.1)])
        ax.grid(True, alpha=0.3)

        # (b) 正确vs错误分类的注意力
        ax = axes[1]
        final_preds = np.argmax(self.probs_history[:, -1, :], axis=1)
        correct_mask = final_preds == self.y_test
        wrong_mask = ~correct_mask
        for mask, label, style in [(correct_mask, '正确分类', '-'), (wrong_mask, '错误分类', '--')]:
            if mask.sum() == 0:
                continue
            mean_attn = attn_weights[mask].mean(axis=0)
            ax.plot(time_axis, mean_attn, style, color='steelblue' if '正确' in label else 'coral',
                   linewidth=2.5, label=label)
        ax.set_xlabel('时间 (秒)', fontsize=10)
        ax.set_ylabel('注意力权重', fontsize=10)
        ax.set_title('(b) 正确 vs 错误分类的注意力', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # (c) 注意力热力图（按类别和试次排序）
        ax = axes[2]
        sorted_attn = []
        yticks = []
        for cls in range(4):
            cls_mask = self.y_test == cls
            cls_attn = attn_weights[cls_mask]
            if len(cls_attn) == 0:
                continue
            # 按最大注意力位置排序
            max_pos = np.argmax(cls_attn, axis=1)
            sort_idx = np.argsort(max_pos)
            sorted_attn.append(cls_attn[sort_idx])
            yticks.append(len(cls_attn))
        if sorted_attn:
            all_sorted = np.concatenate(sorted_attn, axis=0)
            im = ax.imshow(all_sorted, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xlabel('时间窗序号', fontsize=10)
            ax.set_ylabel('试次', fontsize=10)
            ax.set_title('(c) 注意力权重热力图', fontsize=11, fontweight='bold')
            # 添加类别分隔线
            cumsum = np.cumsum([0] + [len(a) for a in sorted_attn[:-1]])
            for y in cumsum[1:]:
                ax.axhline(y - 0.5, color='white', linewidth=1.5)
            plt.colorbar(im, ax=ax, fraction=0.046, label='权重')

        # (d) 注意力峰值位置分布
        ax = axes[3]
        peak_positions = np.argmax(attn_weights, axis=1) * 0.5
        for cls in range(4):
            cls_mask = self.y_test == cls
            if cls_mask.sum() == 0:
                continue
            data = peak_positions[cls_mask]
            ax.hist(data, bins=n_steps, alpha=0.6, color=COLORS[cls], label=CLASS_NAMES[cls], edgecolor='none')
        ax.set_xlabel('注意力峰值时间 (秒)', fontsize=10)
        ax.set_ylabel('试次数', fontsize=10)
        ax.set_title('(d) 注意力峰值位置分布', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        plt.suptitle('LSTM 时间注意力权重分析', fontsize=13, fontweight='bold')
        plt.tight_layout()
        self.save('lstm_06_时间注意力权重')

    def run_all(self):
        print("\n" + "=" * 60)
        print("🎨 LSTM 分类过程可视化")
        print("=" * 60)
        self.plot_stepwise_prob_evolution()
        self.plot_correct_vs_wrong_evolution()
        self.plot_decision_heatmap()
        self.plot_final_probability_distribution()
        self.plot_lstm_hidden_tsne()
        self.plot_temporal_attention()
        print(f"\n✅ 所有图形已保存到 {self.output_dir}/")


if __name__ == "__main__":
    viz = LSTMProcessVisualizer()
    viz.run_all()