# -*- coding: utf-8 -*-
"""
试次质量评分数据箱线图可视化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from dtw_utils import setup_chinese_font, CLASS_NAMES_CN


def load_quality_scores(xlsx_path='pretest/output/trial_quality_results.xlsx'):
    df = pd.read_excel(xlsx_path, sheet_name='scores')
    return df


def _plot_one_page(df, metrics, main_title, save_path):
    """绘制一页箱线图（内部辅助函数）"""
    setup_chinese_font()
    
    n_metrics = len(metrics)
    n_cols = 2
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    class_labels_sorted = sorted(df['label'].unique())
    colors = ['C0', 'C1', 'C2', 'C3']
    
    for idx, (col, title) in enumerate(metrics):
        ax = axes[idx]
        data_groups = [df[df['label'] == c][col].values for c in class_labels_sorted]
        labels = [CLASS_NAMES_CN[int(c)] for c in class_labels_sorted]
        
        bp = ax.boxplot(data_groups, labels=labels, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('任务类别', fontsize=11)
        ax.set_ylabel(col, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(main_title, fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 箱线图已保存: {save_path}")


def plot_quality_boxplots(df, output_dir='pretest/output'):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 把数字标签映射为中文类别名（新增一列方便查看）
    df['类别'] = df['label'].map(lambda x: CLASS_NAMES_CN[int(x)])
    
    # 原始值指标
    raw_metrics = [
        ('F1', 'F1 类间区分度（原始值，越大越好）'),
        ('F2', 'F2 类内偏差（原始值，越小越好）'),
        ('F3', 'F3 试次内波动（原始值，越小越好）'),
        ('score', '综合评分 score（越大越好）'),
    ]
    
    # 归一化后指标
    norm_metrics = [
        ('F1_norm', 'F1_norm 类间区分度（归一化，越大越好）'),
        ('F2_norm', 'F2_norm 类内稳定性（归一化，越大越好）'),
        ('F3_norm', 'F3_norm 试次内平滑度（归一化，越大越好）'),
        ('score', '综合评分 score（越大越好）'),
    ]
    
    _plot_one_page(
        df, raw_metrics,
        '试次质量评分指标分布（原始值）',
        output_dir / 'quality_boxplots_raw.png'
    )
    
    _plot_one_page(
        df, norm_metrics,
        '试次质量评分指标分布（归一化后）',
        output_dir / 'quality_boxplots_normalized.png'
    )


def main():
    output_dir = Path('pretest') / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    xlsx_path = output_dir / 'trial_quality_results.xlsx'
    if not xlsx_path.exists():
        print(f"❌ 找不到文件: {xlsx_path}")
        print("请先运行 python pretest/trial_quality_scoring.py")
        return
    
    df = load_quality_scores(xlsx_path)
    plot_quality_boxplots(df, output_dir)


if __name__ == '__main__':
    main()