# -*- coding: utf-8 -*-
"""
MITT (Multi-class Intra-Trial Trajectory) 分析 — 矩阵运算优化版
包含：切空间投影、K-means 预测强度聚类、网格图、连续轨迹子空间、三个核心指标
"""

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import linalg
from sklearn.cluster import KMeans
from sklearn.model_selection import ShuffleSplit
from tqdm import tqdm

from pyriemann.utils.mean import mean_riemann
from pyriemann.utils.tangentspace import tangent_space

from dtw_utils import setup_chinese_font, load_trajectory_data, average_bands, CLASS_NAMES_CN


OUTPUT_DIR = Path('pretest') / 'output' / 'mitt'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_K_RANGE = range(2, 11)      # K 取值 2~10
N_SPLITS = 20                 # 预测强度随机分割次数
REG = 1e-5                    # 协方差矩阵正则化
FLOAT = np.float32            # 统一用 float32 节省内存


# ========== 第一部分：切空间投影 ==========

def regularize_covs(covs, reg=REG):
    """对协方差矩阵加正则化：C += reg * I"""
    n = covs.shape[-1]
    return covs + reg * np.eye(n)[(None,) * (covs.ndim - 2)]


def tangent_space_projection(covs):
    """
    输入：covs (n_trials, n_epochs, C, C)
    输出：v (n_trials, n_epochs, n_features)，已标准化

    注意：pyriemann 的 tangent_space() 已经直接返回向量化的切空间特征，
    形状为 (n_samples, n_features)，不需要再手动对上三角做展平。
    """
    n_trials, n_epochs, C, _ = covs.shape
    covs_flat = covs.reshape(-1, C, C)
    ref = mean_riemann(covs_flat)
    v = tangent_space(covs_flat, ref).astype(FLOAT)  # (n_trials*n_epochs, n_features)
    n_features = v.shape[1]
    v = v.reshape(n_trials, n_epochs, n_features)

    # 按特征维度标准化
    mean = v.reshape(-1, n_features).mean(axis=0)
    std = v.reshape(-1, n_features).std(axis=0) + 1e-8
    v = (v - mean) / std
    return v, ref, mean, std


# ========== 第二部分：K-means 聚类与预测强度（向量化） ==========

def prediction_strength_one_split(V_tr, V_te, K, rng):
    """
    单次分割的预测强度。向量化计算簇内样本对的共现比例。
    """
    km_tr = KMeans(n_clusters=K, random_state=rng.randint(1e9), n_init=10)
    labels_pred = km_tr.fit(V_tr).predict(V_te)

    km_te = KMeans(n_clusters=K, random_state=rng.randint(1e9), n_init=10)
    labels_te = km_te.fit_predict(V_te)

    cluster_scores = []
    for k in range(K):
        mask = labels_pred == k
        n_k = mask.sum()
        if n_k < 2:
            cluster_scores.append(1.0)
            continue

        labels_k = labels_te[mask]
        # 向量化：构造所有样本对的相同标签指示矩阵
        same_te = labels_k[:, None] == labels_k[None, :]
        # 扣掉对角线，上三角求和即为相同对的数量
        same_pairs = (same_te.sum() - n_k) / 2
        total_pairs = n_k * (n_k - 1) / 2
        cluster_scores.append(same_pairs / total_pairs)

    return min(cluster_scores)


def compute_prediction_strength(V_all, k_range=N_K_RANGE, n_splits=N_SPLITS, random_state=42):
    """计算各 K 值的预测强度。"""
    rng = np.random.RandomState(random_state)
    ps_values = []

    for K in tqdm(list(k_range), desc='预测强度'):
        split_ps = []
        ss = ShuffleSplit(n_splits=n_splits, test_size=0.5, random_state=rng.randint(1e9))
        for train_idx, test_idx in ss.split(V_all):
            V_tr, V_te = V_all[train_idx], V_all[test_idx]
            split_ps.append(prediction_strength_one_split(V_tr, V_te, K, rng))
        ps_values.append(np.mean(split_ps))

    return np.array(ps_values)


def select_best_k(ps_values, k_range=N_K_RANGE, threshold=0.3):
    """选择 ps > threshold 的最大 K"""
    valid = [k for k, ps in zip(k_range, ps_values) if ps > threshold]
    return max(valid) if valid else min(k_range)


def plot_prediction_strength(k_range, ps_values, best_k, save_path):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(list(k_range), ps_values, 'o-', linewidth=2, markersize=8)
    ax.axhline(0.3, color='r', linestyle='--', label='阈值 0.3')
    ax.axvline(best_k, color='g', linestyle='--', label=f'最优 K={best_k}')
    ax.set_xlabel('K (聚类数)', fontsize=12)
    ax.set_ylabel('预测强度', fontsize=12)
    ax.set_title('K-means 预测强度曲线', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 预测强度曲线已保存: {save_path}")


# ========== 第三部分：网格图（向量化） ==========

def compute_trellis_data(state_labels, n_epochs, K):
    """
    state_labels: (n_trials, n_epochs)
    返回 state_freq (n_epochs, K) 和 trans_prob (n_epochs-1, K, K)
    """
    n_trials = len(state_labels)

    # state_freq: 用 one-hot + sum 一次完成
    eye = np.eye(K, dtype=FLOAT)
    freq_3d = eye[state_labels]                 # (n_trials, n_epochs, K)
    state_freq = freq_3d.sum(axis=0) / n_trials  # (n_epochs, K)

    # trans_prob: 用 bincount 向量化
    trans_prob = np.zeros((n_epochs - 1, K, K), dtype=FLOAT)
    for e in range(n_epochs - 1):
        pairs = state_labels[:, e] * K + state_labels[:, e + 1]
        counts = np.bincount(pairs, minlength=K * K).reshape(K, K).astype(FLOAT)
        from_counts = counts.sum(axis=1, keepdims=True)
        trans_prob[e] = counts / (from_counts + 1e-10)

    return state_freq, trans_prob


def plot_trellis(state_freq, trans_prob, class_name, window_centers, save_path):
    setup_chinese_font()
    n_epochs, K = state_freq.shape

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(-0.5, n_epochs - 0.5)
    ax.set_ylim(-0.5, K - 0.5)
    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels([f'{x:.1f}' for x in window_centers], rotation=45)
    ax.set_yticks(range(K))
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel('状态', fontsize=12)
    ax.set_title(f'{class_name}：状态网格图', fontsize=14)
    ax.grid(True, alpha=0.2)

    max_freq = state_freq.max()
    yy, xx = np.meshgrid(np.arange(K), np.arange(n_epochs))
    radii = 0.25 * np.sqrt(state_freq / max_freq) + 0.03
    for x, y, r in zip(xx.ravel(), yy.ravel(), radii.ravel()):
        if r > 0.03:
            circle = plt.Circle((x, y), r, color='C0', alpha=0.6)
            ax.add_patch(circle)

    max_trans = trans_prob.max()
    threshold = 0.05
    for e in range(n_epochs - 1):
        active = trans_prob[e] > threshold
        ys, xs = np.where(active)
        for k1, k2 in zip(xs, ys):
            p = trans_prob[e, k1, k2]
            ax.annotate(
                '',
                xy=(e + 1, k2),
                xytext=(e, k1),
                arrowprops=dict(
                    arrowstyle='->',
                    color='gray',
                    lw=0.5 + 2.5 * (p / max_trans),
                    alpha=0.5 + 0.4 * (p / max_trans),
                ),
            )

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 网格图已保存: {save_path}")


# ========== 第四部分：连续轨迹分析与子空间 ==========

def build_svd_subspace(centroids):
    """
    centroids: (K, n_features)
    返回 U (K-1, n_features) 和 centroid_mean (n_features,)
    """
    centroid_mean = centroids.mean(axis=0)
    W = centroids - centroid_mean
    _, _, Vt = linalg.svd(W, full_matrices=False)
    U = Vt[:len(centroids) - 1].astype(FLOAT)
    return U, centroid_mean.astype(FLOAT)


def project_trajectories(v, U, centroid_mean):
    """
    v: (n_trials, n_epochs, n_features)
    返回 proj: (n_trials, n_epochs, K-1)
    """
    return np.einsum('tef,kf->tek', v - centroid_mean, U).astype(FLOAT)


def plot_continuous_trajectories(proj, y, class_name, save_path, max_trials=200):
    setup_chinese_font()
    cls_idx = list(CLASS_NAMES_CN).index(class_name)
    cls_proj = proj[y == cls_idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    for traj in cls_proj[:max_trials]:
        ax.plot(traj[:, 0], traj[:, 1], alpha=0.15, color='gray', linewidth=0.8)

    mean_traj = cls_proj.mean(axis=0)
    ax.plot(mean_traj[:, 0], mean_traj[:, 1], 'o-', color='red', linewidth=2.5,
            markersize=6, label='平均轨迹')

    ax.set_xlabel('子空间维度 1', fontsize=12)
    ax.set_ylabel('子空间维度 2', fontsize=12)
    ax.set_title(f'{class_name}：连续轨迹投影（前2维）', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ 连续轨迹图已保存: {save_path}")


# ========== 第五部分：三个核心指标（向量化） ==========

def per_class_mean_cov(X, Y, n_classes, eps=1e-6):
    """
    向量化计算每个类别的均值和协方差。
    X: (n_trials, n_dims)
    Y: (n_trials, n_classes) one-hot
    返回:
        class_means: (n_classes, n_dims)
        class_covs:  (n_classes, n_dims, n_dims)
        class_counts: (n_classes,)
    """
    counts = Y.sum(axis=0)                                    # (n_classes,)
    class_means = (Y.T @ X) / (counts[:, None] + eps)         # (n_classes, n_dims)

    n_dims = X.shape[1]
    class_covs = np.zeros((n_classes, n_dims, n_dims), dtype=FLOAT)
    for l in range(n_classes):
        if counts[l] < 2:
            class_covs[l] = np.eye(n_dims, dtype=FLOAT) * eps
            continue
        diff = X[y == l] - class_means[l]                     # 仅对该类别取差分
        class_covs[l] = (diff.T @ diff / (counts[l] - 1)).astype(FLOAT) + eps * np.eye(n_dims, dtype=FLOAT)

    return class_means, class_covs, counts


def compute_inter_task_diff(proj, y, n_classes=4):
    """
    对每个 epoch 和每个任务，计算 InterTaskDiff。
    返回 (n_epochs, n_classes)
    """
    n_epochs = proj.shape[1]
    Y = np.eye(n_classes, dtype=FLOAT)[y]
    result = np.zeros((n_epochs, n_classes), dtype=FLOAT)

    for e in range(n_epochs):
        X = proj[:, e, :].astype(FLOAT)
        mu_e = X.mean(axis=0)
        Sigma_e = np.cov(X, rowvar=False).astype(FLOAT) + 1e-4 * np.eye(X.shape[1], dtype=FLOAT)

        class_means, class_covs, counts = per_class_mean_cov(X, Y, n_classes)
        diffs = class_means - mu_e  # (n_classes, n_dims)

        # 分子：||mu_l - mu_e||^2
        numer = np.einsum('ld,ld->l', diffs, diffs)
        # 分母：trace(Sigma_l + Sigma_e)
        denom = np.trace(Sigma_e) + np.trace(class_covs, axis1=1, axis2=2) + 1e-10
        result[e] = numer / denom

    return result


def compute_inter_trial_var(proj, y, n_classes=4):
    """
    对每个 epoch 和每个任务，计算 InterTrialVar。
    返回 (n_epochs, n_classes)
    """
    n_epochs = proj.shape[1]
    Y = np.eye(n_classes, dtype=FLOAT)[y]
    result = np.zeros((n_epochs, n_classes), dtype=FLOAT)

    for e in range(n_epochs):
        X = proj[:, e, :].astype(FLOAT)
        Sigma_e = np.cov(X, rowvar=False).astype(FLOAT) + 1e-4 * np.eye(X.shape[1], dtype=FLOAT)
        trace_e = np.trace(Sigma_e)

        _, class_covs, counts = per_class_mean_cov(X, Y, n_classes)
        trace_l = np.trace(class_covs, axis1=1, axis2=2)
        result[e] = trace_l / (trace_e + 1e-10)

    return result


def compute_intra_trial_var(proj, y, n_classes=4):
    """
    对每个任务，计算 IntraTrialVar 标量。
    返回 (n_classes,)
    """
    n_dims = proj.shape[2]
    overall_cov = np.cov(proj.reshape(-1, n_dims), rowvar=False).astype(FLOAT)
    denom = np.trace(overall_cov) + 1e-10

    result = np.zeros(n_classes, dtype=FLOAT)
    for l in range(n_classes):
        cls_proj = proj[y == l]              # (n_trials_l, n_epochs, n_dims)
        mean_traj = cls_proj.mean(axis=0)    # (n_epochs, n_dims)
        numerator = np.var(mean_traj, axis=0).sum()
        result[l] = numerator / denom

    return result


# ========== 可视化三个指标 ==========

def plot_metric_over_time(values, y, metric_name, ylabel, save_path, window_centers):
    setup_chinese_font()
    n_classes = values.shape[1]
    fig, ax = plt.subplots(figsize=(10, 6))
    for l in range(n_classes):
        ax.plot(window_centers, values[:, l], 'o-', label=CLASS_NAMES_CN[l], linewidth=2)
    ax.set_xlabel('时间 (s)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{metric_name} 随时间变化', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ {metric_name} 图已保存: {save_path}")


def plot_intra_trial_var_bar(values, save_path):
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(values))
    ax.bar(x, values, color=['C0', 'C1', 'C2', 'C3'][:len(values)], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES_CN[:len(values)])
    ax.set_ylabel('IntraTrialVar', fontsize=12)
    ax.set_title('各任务 IntraTrialVar', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"✅ IntraTrialVar 柱状图已保存: {save_path}")


# ========== 报告生成 ==========

def generate_report(best_k, ps_values, inter_task_diff, inter_trial_var, intra_trial_var,
                    report_path):
    n_classes = inter_task_diff.shape[1]

    mean_inter_task = inter_task_diff.mean(axis=0)
    mean_inter_trial = inter_trial_var.mean(axis=0)

    best_disc = np.argmax(mean_inter_task)
    best_stable = np.argmin(mean_inter_trial)
    best_smooth = np.argmin(intra_trial_var)

    lines = [
        "# MITT 分析报告\n",
        f"## 1. K-means 聚类\n",
        f"- 最优 K 值：**{best_k}**\n",
        f"- 阈值：0.3\n",
        f"- 各 K 预测强度：{dict(zip(N_K_RANGE, ps_values.round(3).tolist()))}\n",
        "\n## 2. 各任务核心指标解读\n",
    ]

    for l in range(n_classes):
        lines.append(
            f"### {CLASS_NAMES_CN[l]}\n"
            f"- 平均 InterTaskDiff（区分度）：{mean_inter_task[l]:.4f}\n"
            f"- 平均 InterTrialVar（稳定性）：{mean_inter_trial[l]:.4f}\n"
            f"- IntraTrialVar（波动度）：{intra_trial_var[l]:.4f}\n"
        )

    lines.extend([
        "\n## 3. 综合判断\n",
        f"- **类间区分度最好**：{CLASS_NAMES_CN[best_disc]} "
        f"(InterTaskDiff={mean_inter_task[best_disc]:.4f})\n",
        f"- **试次间最稳定**：{CLASS_NAMES_CN[best_stable]} "
        f"(InterTrialVar={mean_inter_trial[best_stable]:.4f})\n",
        f"- **试次内波动最小**：{CLASS_NAMES_CN[best_smooth]} "
        f"(IntraTrialVar={intra_trial_var[best_smooth]:.4f})\n",
        "\n## 4. 试次筛选建议\n",
        "- 优先选择 InterTaskDiff 高、InterTrialVar 低、IntraTrialVar 低的任务/试次用于训练分类器。\n",
        "- 对于 InterTrialVar 持续较高的 epoch，可能存在较大的被试内变异，建议进一步检查数据质量。\n",
    ])

    with open(report_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print(f"✅ 报告已保存: {report_path}")


# ========== 主流程 ==========

def main():
    setup_chinese_font()
    matplotlib.use('Agg')  # 无头后端，避免弹窗，节省显存

    # 1. 加载数据
    subjects_data, traj_name = load_trajectory_data(n_subjects=None)
    print(f"📂 加载轨迹: {traj_name}")

    all_covs, all_labels = [], []
    for subj_idx, subj_info in enumerate(tqdm(subjects_data, desc='加载被试')):
        avg = average_bands(subj_info['trajectories'])
        all_covs.append(avg.astype(FLOAT))
        all_labels.append(subj_info['labels'])

    covs = np.concatenate(all_covs, axis=0)
    y = np.concatenate(all_labels, axis=0)

    n_trials, n_epochs, C, _ = covs.shape
    print(f"✅ 数据形状: covs={covs.shape}, n_trials={n_trials}, n_epochs={n_epochs}")

    # 释放中间变量
    del all_covs, all_labels

    # 2. 切空间投影
    print("🔧 切空间投影...")
    covs = regularize_covs(covs)
    v, ref, v_mean, v_std = tangent_space_projection(covs)
    n_features = v.shape[2]
    print(f"✅ 切空间特征形状: {v.shape}")
    del covs

    V_all = v.reshape(-1, n_features)
    print(f"✅ 聚类样本数: {V_all.shape[0]}, 特征维度: {V_all.shape[1]}")

    # 3. K-means + 预测强度
    print("🔍 K-means 预测强度...")
    ps_values = compute_prediction_strength(V_all, k_range=N_K_RANGE, n_splits=N_SPLITS)
    best_k = select_best_k(ps_values, k_range=N_K_RANGE, threshold=0.3)
    print(f"✅ 最优 K = {best_k}")

    plot_prediction_strength(N_K_RANGE, ps_values, best_k,
                             OUTPUT_DIR / 'prediction_strength.png')

    # 用最优 K 重新聚类
    print(f"🔍 用 K={best_k} 重新聚类...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    state_labels_all = kmeans.fit_predict(V_all)
    state_labels = state_labels_all.reshape(n_trials, n_epochs)
    centroids = kmeans.cluster_centers_.astype(FLOAT)

    # 4. 网格图
    print("🕸️ 绘制网格图...")
    window_centers = -0.5 + np.arange(n_epochs) * 0.1
    for cls_idx, cls_name in enumerate(CLASS_NAMES_CN):
        mask = y == cls_idx
        sf, tp = compute_trellis_data(state_labels[mask], n_epochs, best_k)
        plot_trellis(sf, tp, cls_name, window_centers,
                     OUTPUT_DIR / f'trellis_{cls_name}.png')

    # 5. 连续轨迹子空间
    print("📉 构建 SVD 子空间...")
    U, centroid_mean = build_svd_subspace(centroids)
    proj = project_trajectories(v, U, centroid_mean)
    print(f"✅ 投影后轨迹形状: {proj.shape}")
    del v, V_all, state_labels_all

    for cls_name in CLASS_NAMES_CN:
        plot_continuous_trajectories(proj, y, cls_name,
                                     OUTPUT_DIR / f'trajectory_{cls_name}.png',
                                     max_trials=200)

    # 6. 三个核心指标
    print("📊 计算三个核心指标...")
    inter_task_diff = compute_inter_task_diff(proj, y, n_classes=4)
    inter_trial_var = compute_inter_trial_var(proj, y, n_classes=4)
    intra_trial_var = compute_intra_trial_var(proj, y, n_classes=4)

    plot_metric_over_time(inter_task_diff, y, 'InterTaskDiff',
                          'InterTaskDiff', OUTPUT_DIR / 'inter_task_diff.png',
                          window_centers)
    plot_metric_over_time(inter_trial_var, y, 'InterTrialVar',
                          'InterTrialVar', OUTPUT_DIR / 'inter_trial_var.png',
                          window_centers)
    plot_intra_trial_var_bar(intra_trial_var, OUTPUT_DIR / 'intra_trial_var.png')

    # 7. 保存数值
    np.savez(
        OUTPUT_DIR / 'mitt_metrics.npz',
        v=v if 'v' in dir() else None,
        state_labels=state_labels,
        centroids=centroids,
        proj=proj,
        U=U,
        centroid_mean=centroid_mean,
        inter_task_diff=inter_task_diff,
        inter_trial_var=inter_trial_var,
        intra_trial_var=intra_trial_var,
        ps_values=ps_values,
        best_k=best_k,
    )

    # 8. 生成报告
    generate_report(best_k, ps_values, inter_task_diff, inter_trial_var,
                    intra_trial_var, OUTPUT_DIR / 'report.md')

    print("\n✅ MITT 分析全部完成")


if __name__ == '__main__':
    main()