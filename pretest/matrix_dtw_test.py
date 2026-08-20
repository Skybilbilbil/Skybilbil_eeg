# -*- coding: utf-8 -*-
"""
矩阵 DTW 预实验（矩阵轨迹直接做 GPU 加速的 AIRM-DTW）
"""

import numpy as np
import pandas as pd
from pathlib import Path

from dtw_utils import (
    setup_chinese_font,
    load_trajectory_data,
    average_bands,
    compute_class_mean_trajectories,
    dtw_matrix,
    compute_distance_matrix,
    plot_boxplot,
    plot_mds,
    mannwhitney_summary,
)

N_SUBJECTS_ALL = 10
N_SUBJECTS_WITHIN = 3
MAX_TRIALS_PER_CLASS = 8
TOP_PERCENTILE = 25  # 取前 75%，即 score >= 25% 分位数
OUTPUT_DIR = Path('pretest') / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_quality_filter(xlsx_path, percentile=TOP_PERCENTILE):
    """加载质量评分，返回前 (100-percentile)% 试次的布尔掩码。"""
    df = pd.read_excel(xlsx_path, sheet_name='scores')
    scores = df['score'].values
    threshold = np.percentile(scores, percentile)
    selected = scores >= threshold
    print(f'📊 质量评分阈值（{percentile}% 分位数）: {threshold:.4f}')
    print(f'   高质量试次占比: {selected.mean()*100:.1f}%')
    return selected


def run_matrix_within(subject_trajs, subject_labels, n_subjects, title, output_prefix):
    """同被试内矩阵 DTW 分析。"""
    print(f'\n🧪 {title}：前 {n_subjects} 名被试内矩阵 DTW...')
    trial_list, trial_meta = [], []
    for subj_idx in range(min(n_subjects, len(subject_trajs))):
        trajs, labels = subject_trajs[subj_idx], subject_labels[subj_idx]
        for cls_idx in range(4):
            for t in np.where(labels == cls_idx)[0][:MAX_TRIALS_PER_CLASS]:
                trial_list.append(trajs[t])
                trial_meta.append((subj_idx, labels[t]))

    dist_within = compute_distance_matrix(trial_list, dtw_matrix, desc=f'{title} 同被试 DTW')

    intra, inter = [], []
    for i in range(len(trial_meta)):
        for j in range(i + 1, len(trial_meta)):
            si, ci = trial_meta[i]
            sj, cj = trial_meta[j]
            if si == sj:
                (intra if ci == cj else inter).append(dist_within[i, j])

    plot_boxplot(
        [intra, inter],
        ['同被试同类', '同被试异类'],
        f'{title}：前 {n_subjects} 名被试矩阵曲线 DTW',
        '矩阵 DTW 距离 (AIRM)',
        OUTPUT_DIR / f'{output_prefix}_matrix_dtw_within_subject.png',
    )
    p = mannwhitney_summary(intra, inter, '同被试同类', '同被试异类')

    return {
        'intra': np.array(intra),
        'inter': np.array(inter),
        'p': p,
    }


def run_matrix_cross(subject_trajs, subject_labels, title, output_prefix):
    """跨被试同类矩阵 DTW 分析（基于被试级类均值轨迹）。"""
    print(f'\n🧪 {title}：跨被试同类矩阵 DTW...')
    mean_trajs, mean_meta = [], []
    for subj_idx, (trajs, labels) in enumerate(zip(subject_trajs, subject_labels)):
        for cls_idx in range(4):
            mask = labels == cls_idx
            if mask.sum() == 0:
                continue
            # 已按 cls_idx 筛选为单一类别，标签统一设为 0 以适配 n_classes=1
            single_class_labels = np.zeros(mask.sum(), dtype=int)
            mean_trajs.append(compute_class_mean_trajectories(trajs[mask], single_class_labels, n_classes=1)[0])
            mean_meta.append((subj_idx, cls_idx))

    dist_cross = compute_distance_matrix(mean_trajs, dtw_matrix, desc=f'{title} 跨被试 DTW')

    same_class, diff_class = [], []
    for i in range(len(mean_meta)):
        for j in range(i + 1, len(mean_meta)):
            _, ci = mean_meta[i]
            _, cj = mean_meta[j]
            (same_class if ci == cj else diff_class).append(dist_cross[i, j])

    plot_boxplot(
        [same_class, diff_class],
        ['跨被试同类', '跨被试异类'],
        f'{title}：被试级类均值轨迹矩阵曲线 DTW',
        '矩阵 DTW 距离 (AIRM)',
        OUTPUT_DIR / f'{output_prefix}_matrix_dtw_cross_subject.png',
    )
    p = mannwhitney_summary(same_class, diff_class, '跨被试同类', '跨被试异类')

    plot_mds(
        dist_cross,
        np.array([m[1] for m in mean_meta]),
        OUTPUT_DIR / f'{output_prefix}_matrix_dtw_mds.png',
        title=f'{title}：被试级类均值轨迹的 MDS 投影',
    )

    return {
        'same_class': np.array(same_class),
        'diff_class': np.array(diff_class),
        'p': p,
    }


def print_comparison(full_within, filtered_within, full_cross, filtered_cross,
                     output_dir, plot_prefix='matrix_dtw_key_comparison'):
    """打印并绘制核心对比：跨被试同类 vs 同被试异类。"""
    print('\n' + '=' * 60)
    print('矩阵 DTW 核心对比：跨被试同类 vs 同被试异类')
    print('=' * 60)

    for stage, within, cross in [
        ('全部试次', full_within, full_cross),
        ('前75%高质量试次', filtered_within, filtered_cross),
    ]:
        cross_same = cross['same_class']
        within_inter = within['inter']
        print(f'\n[{stage}]')
        print(f'  跨被试同类均值: {cross_same.mean():.4f}')
        print(f'  同被试异类均值: {within_inter.mean():.4f}')
        ratio = within_inter.mean() / (cross_same.mean() + 1e-10)
        print(f'  同被试异类 / 跨被试同类 比值: {ratio:.3f}')
        print(f'  （比值 > 1 表示类别信息强于被试个体差异）')

    # 绘制前后对比箱线图
    fig_data = [
        full_cross['same_class'],
        full_within['inter'],
        filtered_cross['same_class'],
        filtered_within['inter'],
    ]
    labels = [
        '跨被试同类\n（全部）',
        '同被试异类\n（全部）',
        '跨被试同类\n（前75%）',
        '同被试异类\n（前75%）',
    ]
    plot_boxplot(
        fig_data,
        labels,
        '矩阵 DTW 核心对比：跨被试同类 vs 同被试异类',
        '矩阵 DTW 距离 (AIRM)',
        output_dir / f'{plot_prefix}.png',
    )

    # 统计检验
    print('\n统计检验（Mann-Whitney U）:')
    p_full = mannwhitney_summary(
        full_cross['same_class'], full_within['inter'],
        '跨被试同类（全部）', '同被试异类（全部）'
    )
    p_top75 = mannwhitney_summary(
        filtered_cross['same_class'], filtered_within['inter'],
        '跨被试同类（前75%）', '同被试异类（前75%）'
    )
    return p_full, p_top75


def main():
    setup_chinese_font()

    # 1. 加载质量评分并筛选前75%
    xlsx_path = OUTPUT_DIR / 'trial_quality_results.xlsx'
    selected_mask = load_quality_filter(xlsx_path)

    # 2. 加载轨迹数据
    subjects_data, traj_name = load_trajectory_data(n_subjects=N_SUBJECTS_ALL)
    print(f'📂 加载轨迹: {traj_name}')

    # 3. 沿频带维度做黎曼均值，同时记录全局索引
    print('🔧 沿频带维度做黎曼均值...')
    subject_trajs_full, subject_labels_full = [], []
    subject_trajs_top75, subject_labels_top75 = [], []
    global_idx = 0

    for subj_idx, subj_info in enumerate(subjects_data):
        avg = average_bands(subj_info['trajectories'])
        labels = subj_info['labels']
        n_trials_subj = len(labels)
        print(f"  {subj_info['subject_key']}: {avg.shape}")

        # 该被试的全局索引对应的筛选掩码
        subj_global_idx = np.arange(global_idx, global_idx + n_trials_subj)
        keep = selected_mask[subj_global_idx]

        subject_trajs_full.append(avg)
        subject_labels_full.append(labels)

        subject_trajs_top75.append(avg[keep])
        subject_labels_top75.append(labels[keep])

        global_idx += n_trials_subj

    # 4. 同被试矩阵 DTW 对比
    full_within = run_matrix_within(
        subject_trajs_full, subject_labels_full, N_SUBJECTS_WITHIN,
        '全部试次', 'full'
    )
    filtered_within = run_matrix_within(
        subject_trajs_top75, subject_labels_top75, N_SUBJECTS_WITHIN,
        '前75%高质量试次', 'top75'
    )

    # 5. 跨被试矩阵 DTW 对比
    full_cross = run_matrix_cross(
        subject_trajs_full, subject_labels_full,
        '全部试次', 'full'
    )
    filtered_cross = run_matrix_cross(
        subject_trajs_top75, subject_labels_top75,
        '前75%高质量试次', 'top75'
    )

    # 6. 打印并绘制核心对比结果
    p_full, p_top75 = print_comparison(
        full_within, filtered_within, full_cross, filtered_cross,
        OUTPUT_DIR
    )

    # 7. 保存结果
    np.savez(
        OUTPUT_DIR / 'matrix_dtw_results.npz',
        full_within=full_within,
        top75_within=filtered_within,
        full_cross=full_cross,
        top75_cross=filtered_cross,
        p_full=p_full,
        p_top75=p_top75,
    )
    print('\n💾 结果已保存: ' + str(OUTPUT_DIR / 'matrix_dtw_results.npz'))
    print('✅ 全部完成')


if __name__ == '__main__':
    main()