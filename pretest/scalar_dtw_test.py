# -*- coding: utf-8 -*-
"""
标量曲线 DTW 预实验（从矩阵轨迹 -> 到 4 个类均值的黎曼距离曲线 -> DTW）
"""

import numpy as np
import pandas as pd
from pathlib import Path

from dtw_utils import (
    setup_chinese_font,
    load_trajectory_data,
    average_bands,
    compute_class_mean_trajectories,
    build_scalar_curves,
    dtw_multivariate,
    compute_distance_matrix,
    collect_pairwise_stats,
    plot_boxplot,
    plot_mds,
    mannwhitney_summary,
)

N_SUBJECTS = 10
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


def run_scalar_analysis(trial_trajs, trial_labels, trial_subjects, title, output_prefix):
    """对给定试次集合运行完整的标量 DTW 分析。"""
    print(f'\n📊 {title}: 计算全局类均值轨迹...')
    class_mean_trajs = compute_class_mean_trajectories(trial_trajs, trial_labels)

    print(f'📈 {title}: 构建 4D 标量曲线...')
    scalar_curves = build_scalar_curves(trial_trajs, class_mean_trajs)

    print(f'🔄 {title}: 计算两两 DTW 距离...')
    dist_mat = compute_distance_matrix(scalar_curves, dtw_multivariate, desc=f'{title} 标量 DTW')

    intra_same, inter_same, intra_cross = collect_pairwise_stats(
        dist_mat, trial_labels, trial_subjects
    )

    plot_boxplot(
        [intra_same, inter_same, intra_cross],
        ['同被试同类', '同被试异类', '跨被试同类'],
        f'{title}：标量曲线 DTW 距离分布',
        'DTW 距离',
        OUTPUT_DIR / f'{output_prefix}_scalar_dtw_boxplot.png',
    )

    p1 = mannwhitney_summary(intra_same, inter_same, '同被试同类', '同被试异类')
    p2 = mannwhitney_summary(intra_same, intra_cross, '同被试同类', '跨被试同类')

    plot_mds(dist_mat, trial_labels, OUTPUT_DIR / f'{output_prefix}_scalar_dtw_mds.png',
             title=f'{title}：标量曲线 DTW 距离的 MDS 投影')

    return {
        'dist_mat': dist_mat,
        'trial_labels': trial_labels,
        'trial_subjects': trial_subjects,
        'intra_same': np.array(intra_same),
        'inter_same': np.array(inter_same),
        'intra_cross': np.array(intra_cross),
        'p_intra_inter': p1,
        'p_intra_cross': p2,
    }


def print_comparison(full, filtered):
    """打印全部试次 vs 前75%试次的对比。"""
    print('\n' + '=' * 60)
    print('标量 DTW：全部试次 vs 前75%高质量试次')
    print('=' * 60)
    for key in ['intra_same', 'inter_same', 'intra_cross']:
        f_mean = full[key].mean()
        fl_mean = filtered[key].mean()
        change = (fl_mean - f_mean) / (f_mean + 1e-10) * 100
        print(f'  {key:15s}: 全部={f_mean:.4f}, 前75%={fl_mean:.4f}, 变化={change:+.1f}%')

    f_ratio = full['intra_same'].mean() / (full['inter_same'].mean() + 1e-10)
    fl_ratio = filtered['intra_same'].mean() / (filtered['inter_same'].mean() + 1e-10)
    print(f'  同被试类内/类间比值: 全部={f_ratio:.3f}, 前75%={fl_ratio:.3f}')

    f_ratio_cross = full['intra_cross'].mean() / (full['inter_same'].mean() + 1e-10)
    fl_ratio_cross = filtered['intra_cross'].mean() / (filtered['inter_same'].mean() + 1e-10)
    print(f'  跨被试同类/同被试异类比值: 全部={f_ratio_cross:.3f}, 前75%={fl_ratio_cross:.3f}')


def main():
    setup_chinese_font()

    # 1. 加载质量评分并筛选前75%
    xlsx_path = OUTPUT_DIR / 'trial_quality_results.xlsx'
    selected_mask = load_quality_filter(xlsx_path)

    # 2. 加载轨迹数据
    subjects_data, traj_name = load_trajectory_data(n_subjects=N_SUBJECTS)
    print(f'📂 加载轨迹: {traj_name}')

    # 3. 构建试次池（带全局索引，与 quality scoring 文件中的 trial_idx 对应）
    trial_trajs, trial_labels, trial_subjects, trial_global_idx = [], [], [], []
    global_idx = 0
    print('🔧 沿频带维度做黎曼均值...')
    for subj_idx, subj_info in enumerate(subjects_data):
        avg = average_bands(subj_info['trajectories'])
        labels = subj_info['labels']
        n_trials_subj = len(labels)
        for cls_idx in range(4):
            for t in np.where(labels == cls_idx)[0][:MAX_TRIALS_PER_CLASS]:
                trial_trajs.append(avg[t])
                trial_labels.append(labels[t])
                trial_subjects.append(subj_idx)
                trial_global_idx.append(global_idx + t)
        global_idx += n_trials_subj

    trial_trajs = np.array(trial_trajs)
    trial_labels = np.array(trial_labels)
    trial_subjects = np.array(trial_subjects)
    trial_global_idx = np.array(trial_global_idx)
    n_trials = len(trial_trajs)
    print(f'✅ 总试次数: {n_trials}')

    # 4. 筛选前75%试次
    keep = selected_mask[trial_global_idx]
    filtered_trajs = trial_trajs[keep]
    filtered_labels = trial_labels[keep]
    filtered_subjects = trial_subjects[keep]
    print(f'✅ 筛选后试次数: {len(filtered_trajs)}')

    # 5. 分别对全部试次和前75%试次运行标量 DTW
    full_results = run_scalar_analysis(
        trial_trajs, trial_labels, trial_subjects,
        '全部试次', 'full'
    )
    filtered_results = run_scalar_analysis(
        filtered_trajs, filtered_labels, filtered_subjects,
        '前75%高质量试次', 'top75'
    )

    # 6. 打印对比结果
    print_comparison(full_results, filtered_results)

    # 7. 保存结果
    np.savez(
        OUTPUT_DIR / 'scalar_dtw_results.npz',
        full=full_results,
        top75=filtered_results,
    )
    print('\n💾 结果已保存: ' + str(OUTPUT_DIR / 'scalar_dtw_results.npz'))
    print('✅ 全部完成')


if __name__ == '__main__':
    main()