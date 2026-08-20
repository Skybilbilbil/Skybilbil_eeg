# -*- coding: utf-8 -*-
"""
分类结果可视化
对应用户需求的图11-16
需要 Classification.py 保存的结果文件
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.manifold import TSNE
import warnings
import torch
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
plt.rcParams['figure.dpi'] = 150

COLORS = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
CLASS_NAMES = ['左手', '右手', '双脚', '双手']


class ResultsVisualizer:
    def __init__(self):
        self.results_file = Path("results") / "classification_results_full.pkl"
        self.traj_file = Path("trajectory_data") / "trajectories_500ms_win.pkl"
        self.output_dir = Path("figures")
        self.output_dir.mkdir(exist_ok=True)

        self.results = None
        if self.results_file.exists():
            with open(self.results_file, 'rb') as f:
                self.results = pickle.load(f)
            # 兼容新旧键名
            if 'subject_results' in self.results:
                self.results['fold_results'] = self.results['subject_results']
            print(f"✅ 加载结果: {len(self.results['fold_results'])} 个测试被试, "
                  f"平均准确率: {self.results['mean_accuracy']*100:.2f}%")
        else:
            print("⚠️ 未找到分类结果文件，部分图表将无法生成")
            print("   请先运行 Classification.py 并保存结果")

    def save(self, name):
        path = self.output_dir / f"{name}.png"
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  📊 已保存: {path.name}")
        plt.close()

    def _check_results(self):
        if self.results is None:
            print("  ⚠️ 跳过（无结果文件）")
            return False
        return True

    # ==================== 图10: 混淆矩阵 ====================
    def plot_confusion_matrix(self):
        if not self._check_results():
            return
        y_true = np.concatenate([f['y_true'] for f in self.results['fold_results']])
        y_pred = np.concatenate([f['y_pred'] for f in self.results['fold_results']])
        cm = confusion_matrix(y_true, y_pred, normalize='true')

        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
                   square=True, cbar_kws={'shrink': 0.8},
                   annot_kws={'size': 12, 'weight': 'bold'},
                   linewidths=1, linecolor='white', ax=ax)
        ax.set_xlabel('预测标签', fontsize=12)
        ax.set_ylabel('真实标签', fontsize=12)
        ax.set_title(f'归一化混淆矩阵（固定训练-测试划分）\n'
                    f'整体准确率: {self.results["mean_accuracy"]*100:.2f}% ± {self.results["std_accuracy"]*100:.2f}%',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        self.save('cls_01_混淆矩阵')

    # ==================== 图11: 每被试准确率 ====================
    def plot_per_subject_accuracy(self):
        if not self._check_results():
            return
        subjects = [f['subject'] for f in self.results['fold_results']]
        accs = [f['accuracy'] * 100 for f in self.results['fold_results']]
        mean_acc = self.results['mean_accuracy'] * 100
        std_acc = self.results['std_accuracy'] * 100

        fig, ax = plt.subplots(figsize=(10, 5))
        colors_bar = [COLORS[i % len(COLORS)] for i in range(len(accs))]
        bars = ax.bar(range(len(accs)), accs, color=colors_bar, alpha=0.85, edgecolor='black', linewidth=1)
        ax.axhline(y=mean_acc, color='red', linestyle='--', linewidth=2, label=f'均值: {mean_acc:.1f}%')
        ax.axhspan(mean_acc - std_acc, mean_acc + std_acc, alpha=0.15, color='red', label=f'±1 标准差: {std_acc:.1f}%')
        ax.axhline(y=25, color='gray', linestyle=':', linewidth=1.5, label='随机水平 (25%)')

        ax.set_xticks(range(len(subjects)))
        ax.set_xticklabels([s.split('_S')[-1] for s in subjects], rotation=45, ha='right')
        ax.set_ylabel('准确率 (%)', fontsize=12)
        ax.set_xlabel('被试', fontsize=12)
        ax.set_title('各测试被试准确率（固定划分）', fontsize=13, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim([0, 105])

        for bar, acc in zip(bars, accs):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1.5,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.tight_layout()
        self.save('cls_02_被试准确率')

    # ==================== 图12: ROC曲线 ====================
    def plot_roc_curves(self):
        if not self._check_results():
            return
        # 需要softmax概率，但当前结果只有硬标签
        # 改为展示每类别的Precision/Recall/F1
        from sklearn.metrics import precision_recall_fscore_support

        y_true = np.concatenate([f['y_true'] for f in self.results['fold_results']])
        y_pred = np.concatenate([f['y_pred'] for f in self.results['fold_results']])

        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(4)
        width = 0.25
        ax.bar(x - width, precision, width, label='Precision', color=COLORS[0], alpha=0.85, edgecolor='black')
        ax.bar(x, recall, width, label='Recall', color=COLORS[1], alpha=0.85, edgecolor='black')
        ax.bar(x + width, f1, width, label='F1-Score', color=COLORS[2], alpha=0.85, edgecolor='black')

        ax.set_xticks(x)
        ax.set_xticklabels(CLASS_NAMES, fontsize=10)
        ax.set_ylabel('分数', fontsize=12)
        ax.set_title('各类别精确率 / 召回率 / F1（固定划分）', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.set_ylim([0, 1.15])
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        self.save('cls_03_各类指标')

    # ==================== 图13: 训练Loss曲线 ====================
    def plot_training_curves(self):
        if not self._check_results() or 'training_histories' not in self.results:
            print("  ⚠️ 跳过训练曲线（无训练历史）")
            return

        histories = self.results['training_histories']
        fig, axes = plt.subplots(2, 5, figsize=(16, 7))
        axes = axes.flatten()

        for idx, (hist, fold_res) in enumerate(zip(histories, self.results['fold_results'])):
            if idx >= len(axes):
                break
            ax = axes[idx]
            # 兼容新旧键名
            loss_key = 'loss' if 'loss' in hist else 'train_loss'
            ax.plot(hist['epoch'], hist[loss_key], 'b-', linewidth=1.5, alpha=0.8)
            ax.set_xlabel('轮次', fontsize=8)
            ax.set_ylabel('损失', fontsize=8)
            ax.set_title(f'被试 {idx+1}: {fold_res["subject"].split("_S")[-1]}\n'
                        f'准确率: {fold_res["accuracy"]*100:.1f}%', fontsize=9, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # 隐藏多余子图
        for idx in range(len(histories), len(axes)):
            axes[idx].axis('off')

        # 平均曲线
        fig2, ax2 = plt.subplots(figsize=(8, 5))
        loss_key = 'loss' if 'loss' in histories[0] else 'train_loss'
        max_epochs = max(len(h[loss_key]) for h in histories)
        loss_matrix = np.array([h[loss_key] + [np.nan]*(max_epochs - len(h[loss_key]))
                                for h in histories])
        mean_loss = np.nanmean(loss_matrix, axis=0)
        std_loss = np.nanstd(loss_matrix, axis=0)
        epochs = range(1, max_epochs + 1)
        ax2.plot(epochs, mean_loss, 'b-', linewidth=2, label='Mean Loss')
        ax2.fill_between(epochs, mean_loss - std_loss, mean_loss + std_loss,
                        alpha=0.3, color='blue', label='±1 Std')
        ax2.set_xlabel('轮次', fontsize=12)
        ax2.set_ylabel('训练损失', fontsize=12)
        n_test = len(self.results['fold_results'])
        ax2.set_title(f'平均训练损失曲线（{n_test}个测试被试）', fontsize=13, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        self.save('cls_04_训练损失曲线')

    # ==================== 图14: LSTM输出t-SNE（需要重新推理） ====================
    def plot_lstm_output_tsne(self):
        """需要重新运行模型获取LSTM输出 - 简化版：用原始特征代替"""
        if not self.traj_file.exists():
            return

        with open(self.traj_file, 'rb') as f:
            data = pickle.load(f)

        from utils import log_euclidean_mapping, normalize_trace

        # 收集测试被试(1-10)的切空间特征
        test_subjects = []
        for s in data['subject_data']:
            key = s.get('subject_key', '')
            if 'PhysionetMI_S' in key:
                num = int(key.split('S')[-1])
                if 1 <= num <= 10:
                    test_subjects.append(s)

        if not test_subjects:
            test_subjects = data['subject_data'][:10]

        all_features = []
        all_labels = []

        for subj in test_subjects:
            covs = subj['trajectories'][:, 0, :, :, :]  # [trials, bands, 7, 7]
            labels = subj['labels']
            covs_tensor = torch.tensor(covs.reshape(-1, 7, 7), dtype=torch.float64)
            covs_norm = normalize_trace(covs_tensor)
            tangent = log_euclidean_mapping(covs_norm).cpu().numpy()
            tangent = tangent.reshape(len(labels), 5, 28).mean(axis=1)  # [trials, 28]
            all_features.append(tangent)
            all_labels.extend(labels)

        X = np.concatenate(all_features, axis=0)
        y = np.array(all_labels)

        n_samples = min(800, len(X))
        idx = np.random.choice(len(X), n_samples, replace=False)
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X[idx])
        y_plot = y[idx]

        fig, ax = plt.subplots(figsize=(8, 6))
        for cls in range(4):
            mask = y_plot == cls
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                      c=COLORS[cls], label=CLASS_NAMES[cls],
                      alpha=0.6, s=30, edgecolors='none')
        ax.set_xlabel('t-SNE 维度 1', fontsize=11)
        ax.set_ylabel('t-SNE 维度 2', fontsize=11)
        ax.set_title('LSTM 输入特征的 t-SNE\n（测试被试1-10，频带平均）',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        plt.tight_layout()
        self.save('cls_05_LSTM输入tSNE')

    def run_all(self):
        print("\n" + "=" * 60)
        print("🎨 分类结果可视化")
        print("=" * 60)
        self.plot_confusion_matrix()
        self.plot_per_subject_accuracy()
        self.plot_roc_curves()
        self.plot_training_curves()
        self.plot_lstm_output_tsne()
        print(f"\n✅ 已保存到 {self.output_dir}/")


if __name__ == "__main__":
    viz = ResultsVisualizer()
    viz.run_all()