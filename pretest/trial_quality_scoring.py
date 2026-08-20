# -*- coding: utf-8 -*-
"""
Trial Quality Scoring (基于 MITT 思想的试次可分类性评分)

核心思想：
1. 类间区分度 (F1)：试次应靠近同类平均轨迹、远离最近异类平均轨迹。
2. 试次间稳定性 (F2)：试次不应偏离同类中心太远。
3. 试次内波动度 (F3)：试次内部时序轨迹应平滑，避免剧烈抖动。

使用说明：
- 输入 X 形状: (n_trials, n_epochs, n_features)
- 输入 y 形状: (n_trials,)
- 可选 subject_ids: (n_trials,)，用于被试内中心化
"""

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm


# ========== 工具函数 ==========

def _log_euclidean_mapping(covs, eps=1e-7):
    """
    对数欧氏映射: [N, C, C] -> [N, D]
    自包含实现，避免依赖 visualization/utils.py。
    """
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


def subject_wise_centering(X, subject_ids):
    """
    被试内中心化：消除个体间绝对差异，保留任务相关动态。

    参数:
        X: np.ndarray, (n_trials, n_epochs, n_features)
        subject_ids: np.ndarray, (n_trials,)

    返回:
        X_centered: 去基线后的数据
    """
    X_centered = np.zeros_like(X)
    for sid in np.unique(subject_ids):
        mask = subject_ids == sid
        subject_mean = X[mask].mean(axis=0)  # (n_epochs, n_features)
        X_centered[mask] = X[mask] - subject_mean
    return X_centered


def compute_class_mean_trajectories(X, y, n_classes=None):
    """
    计算每类任务的平均轨迹。

    参数:
        X: (n_trials, n_epochs, n_features)
        y: (n_trials,)
        n_classes: 类别数，默认从 y 推断

    返回:
        mean_trajs: (n_classes, n_epochs, n_features)
    """
    if n_classes is None:
        n_classes = len(np.unique(y))
    mean_trajs = np.zeros((n_classes, X.shape[1], X.shape[2]), dtype=X.dtype)
    for c in range(n_classes):
        mean_trajs[c] = X[y == c].mean(axis=0)
    return mean_trajs


def _percentile_clip_normalize(values, lower=5, upper=95, flip=False, eps=1e-8):
    """
    百分位裁剪归一化 (Winsorizing + Min-Max)。

    步骤:
        1. 计算 lower% 和 upper% 分位数
        2. 将数组裁剪到 [P_lower, P_upper]
        3. Min-Max 归一化到 [0, 1]
        4. 若 flip=True，返回 1 - 归一化值（用于“越小越好”的指标）
    """
    p_low = np.percentile(values, lower)
    p_high = np.percentile(values, upper)
    clipped = np.clip(values, p_low, p_high)

    vmin, vmax = clipped.min(), clipped.max()
    if vmax - vmin < eps:
        normalized = np.ones_like(values)
    else:
        normalized = (clipped - vmin) / (vmax - vmin + eps)

    return 1.0 - normalized if flip else normalized


# ========== 核心评分函数 ==========

def compute_trial_quality_scores(
    X,
    y,
    subject_ids=None,
    weights=(0.5, 0.3, 0.2),
    eps=1e-8,
    normalize=True,
    verbose=True
):
    """
    计算每个试次的可分类性综合评分。

    参数:
        X: np.ndarray, (n_trials, n_epochs, n_features)
        y: np.ndarray, (n_trials,), 类别标签
        subject_ids: np.ndarray, (n_trials,), 可选，用于被试内中心化
        weights: tuple, F1/F2/F3 的权重，默认 (0.5, 0.3, 0.2)
        eps: 防止除零
        normalize: 是否对子指标做 Min-Max 归一化
        verbose: 是否打印过程信息

    返回:
        dict: {
            'scores': (n_trials,) 综合评分,
            'F1': (n_trials,) 类间区分度,
            'F2': (n_trials,) 试次间稳定性,
            'F3': (n_trials,) 试次内波动度,
            'F1_norm': ...,
            'F2_norm': ...,
            'F3_norm': ...,
            'mean_trajs': (n_classes, n_epochs, n_features) 类平均轨迹,
            'threshold_60': 60% 分位数阈值,
        }
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y)

    # --- 步骤 0: 被试内中心化 (可选) ---
    if subject_ids is not None:
        if verbose:
            print("🧠 执行被试内中心化...")
        X = subject_wise_centering(X, subject_ids)

    n_trials, n_epochs, n_features = X.shape
    n_classes = len(np.unique(y))
    y_true = y.astype(int)

    # --- 步骤 1: 计算各类平均轨迹 (MITT: class mean trajectory) ---
    if verbose:
        print(f"📊 计算 {n_classes} 类平均轨迹...")
    mean_trajs = compute_class_mean_trajectories(X, y, n_classes)

    # --- 步骤 2: 计算每个试次到各类平均轨迹的累积欧氏距离 ---
    # dist_to_class[i, c] = sum over epochs/features of (X[i] - mean_trajs[c])^2
    dist_to_class = np.zeros((n_trials, n_classes), dtype=np.float64)
    for c in range(n_classes):
        diff = X - mean_trajs[c][None, ...]          # (n_trials, n_epochs, n_features)
        dist_to_class[:, c] = np.sum(diff ** 2, axis=(1, 2))

    # --- 同时计算类内散布程度（标准差），按要求输出原始数值 ---
    # class_std[c] = 类别 c 内所有试次轨迹相对类中心的标准差
    class_std = np.zeros(n_classes, dtype=np.float64)
    for c in range(n_classes):
        class_trials = X[y == c]                     # (n_c, n_epochs, n_features)
        diff = class_trials - mean_trajs[c][None, ...]
        class_std[c] = np.std(diff)
    class_std_per_trial = class_std[y_true]

    # --- 子指标 1: 类间区分度 (F1) ---
    # F1 = Dist_between / (Dist_within + epsilon)
    # Dist_within: 到同类中心的距离
    # Dist_between: 到最近异类中心的距离
    dist_within = dist_to_class[np.arange(n_trials), y_true]

    # 对每个试次，找最近异类中心的距离
    mask = np.ones((n_trials, n_classes), dtype=bool)
    mask[np.arange(n_trials), y_true] = False
    dist_between = np.min(dist_to_class[mask].reshape(n_trials, n_classes - 1), axis=1)

    F1 = dist_between / (dist_within + eps)

    # --- 子指标 2: 试次间稳定性 (F2, InterTrialVar 思想) ---
    # F2 = Deviation_i = Dist_within（正值，越小代表越稳定）
    F2 = dist_within

    # --- 子指标 3: 试次内波动度 (F3, IntraTrialVar 思想) ---
    # F3 = 试次内方差均值（正值，越小代表越平滑）
    F3 = np.var(X, axis=1).mean(axis=1)

    # --- 步骤 3: 百分位裁剪归一化 ---
    # F1 越大越好，不反转；F2/F3 越小越好，归一化后 1 - norm 反转
    if normalize:
        F1_norm = _percentile_clip_normalize(F1, lower=5, upper=95, flip=False)
        F2_norm = _percentile_clip_normalize(F2, lower=5, upper=95, flip=True)
        F3_norm = _percentile_clip_normalize(F3, lower=5, upper=95, flip=True)
    else:
        F1_norm, F2_norm, F3_norm = F1, F2, F3

    # --- 步骤 4: 加权综合评分 ---
    # Score_i = 0.5 * norm(F1) + 0.3 * norm(F2) + 0.2 * norm(F3)
    w1, w2, w3 = weights
    scores = w1 * F1_norm + w2 * F2_norm + w3 * F3_norm

    if verbose:
        print(f"✅ 评分完成: mean={scores.mean():.4f}, std={scores.std():.4f}")
        print(f"   F1 (区分度) raw range=[{F1.min():.3f}, {F1.max():.3f}]")
        print(f"   F2 (偏差)   raw range=[{F2.min():.3f}, {F2.max():.3f}]")
        print(f"   F3 (波动)   raw range=[{F3.min():.3f}, {F3.max():.3f}]")
        print(f"   F1_norm range=[{F1_norm.min():.3f}, {F1_norm.max():.3f}]")
        print(f"   F2_norm range=[{F2_norm.min():.3f}, {F2_norm.max():.3f}]")
        print(f"   F3_norm range=[{F3_norm.min():.3f}, {F3_norm.max():.3f}]")

    return {
        'scores': scores,
        'F1': F1,
        'F2': F2,
        'F3': F3,
        'F1_norm': F1_norm,
        'F2_norm': F2_norm,
        'F3_norm': F3_norm,
        'class_std': class_std_per_trial,
        'mean_trajs': mean_trajs,
    }


# ========== 本项目数据适配器 ==========

def prepare_features_from_cov_trajectories(subjects_data, eps=1e-7, verbose=True):
    """
    将本项目的协方差轨迹数据转换为 (n_trials, n_epochs, n_features) 格式。

    输入:
        subjects_data: list of dict, 来自 trajectory_data/trajectories_*.pkl 的 subject_data
        每个 dict 包含:
            - 'trajectories': (n_trials, n_windows, n_bands, C, C)
            - 'labels': (n_trials,)
            - 'subject_key': str

    返回:
        X: (n_total_trials, n_windows, n_features)
        y: (n_total_trials,)
        subject_ids: (n_total_trials,)
    """
    all_features, all_labels, all_subject_ids = [], [], []
    for subj_idx, subj_info in enumerate(tqdm(subjects_data, desc='转换协方差轨迹')):
        trajs = subj_info['trajectories']  # (trials, windows, bands, C, C)
        labels = subj_info['labels']
        n_trials, n_windows, n_bands, C, _ = trajs.shape

        # 展平所有窗口和频带: (trials*windows*bands, C, C)
        flat = trajs.reshape(-1, C, C)
        tangent = _log_euclidean_mapping(flat, eps=eps)
        D = C * (C + 1) // 2
        # 恢复形状并沿频带维度拼接: (trials, windows, n_bands*D)
        tangent = tangent.reshape(n_trials, n_windows, n_bands, D)
        features = tangent.reshape(n_trials, n_windows, -1)

        all_features.append(features)
        all_labels.append(labels)
        all_subject_ids.append(np.full(n_trials, subj_idx, dtype=int))

    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)
    subject_ids = np.concatenate(all_subject_ids, axis=0)

    if verbose:
        print(f"✅ 转换完成: X.shape={X.shape}, y.shape={y.shape}, subjects={len(subjects_data)}")
        print(f"   类别分布: {dict(zip(*np.unique(y, return_counts=True)))}")

    return X, y, subject_ids


# ========== 结果保存 ==========

def save_quality_results(result, y, subject_ids, output_path):
    """
    保存试次质量评分的原始计算数值到 Excel (.xlsx)。
    只包含一个工作表 scores，每行一个试次，列包含原始指标与归一化后的指标。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_trials = len(y)

    # 每试次原始计算数值表
    df_scores = pd.DataFrame({
        'trial_idx': np.arange(n_trials),
        'subject_id': subject_ids,
        'label': y,
        'score': result['scores'],
        'F1': result['F1'],
        'F2': result['F2'],
        'F3': result['F3'],
        'F1_norm': result['F1_norm'],
        'F2_norm': result['F2_norm'],
        'F3_norm': result['F3_norm'],
        'class_std': result['class_std'],
    })

    xlsx_path = output_path.with_suffix('.xlsx')
    df_scores.to_excel(xlsx_path, sheet_name='scores', index=False)
    print(f"✅ Excel 已保存: {xlsx_path}")


# ========== 示例入口 ==========

def main():
    """
    示例：加载本项目的轨迹数据，计算试次质量评分并筛选高质量试次。
    """
    import pickle
    from dtw_utils import CLASS_NAMES_CN

    # 1. 加载轨迹数据
    traj_dir = Path('trajectory_data')
    traj_path = sorted(traj_dir.glob('trajectories_*.pkl'),
                       key=lambda p: p.stat().st_mtime, reverse=True)[0]
    print(f"📂 加载轨迹: {traj_path.name}")
    with open(traj_path, 'rb') as f:
        data = pickle.load(f)
    subjects_data = data['subject_data']

    # 2. 转换为特征矩阵 (可选只取前 N 被试做快速测试)
    X, y, subject_ids = prepare_features_from_cov_trajectories(subjects_data)

    # 3. 计算质量评分（启用被试内中心化）
    result = compute_trial_quality_scores(
        X, y,
        subject_ids=subject_ids,
        weights=(0.5, 0.3, 0.2),
        normalize=True,
        verbose=True
    )

    # 4. 保存原始计算数值到 Excel
    output_dir = Path('pretest') / 'output'
    save_quality_results(
        result, y, subject_ids,
        output_dir / 'trial_quality_results'
    )

    # 5. 各类别得分分布（仅查看，不做筛选）
    print("\n📊 各类别 score 分布:")
    for c in np.unique(y):
        cls_scores = result['scores'][y == c]
        print(f"   {CLASS_NAMES_CN[int(c)]}: mean={cls_scores.mean():.4f}, "
              f"std={cls_scores.std():.4f}, n={len(cls_scores)}")


if __name__ == '__main__':
    main()