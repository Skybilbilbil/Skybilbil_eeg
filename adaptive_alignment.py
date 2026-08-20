# -*- coding: utf-8 -*-
"""
自适应EA (Adaptive Euclidean Alignment)
支持四种模式:
  - 'filter':   剔除离群被试后，用"干净"的被试计算全局EA矩阵（默认，推荐）
  - 'weighted': 加权全局EA，离群被试自动降权
  - 'group':    聚类分组，每组共享EA矩阵
  - 'global':   所有人一起计算一个全局EA矩阵（不做任何筛选）
"""

import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


# ==================== 基础计算 ====================

def compute_riemann_distance(c1, c2):
    """计算两个协方差矩阵之间的黎曼距离（对数欧氏近似）"""
    try:
        inv_c1 = np.linalg.inv(c1 + 1e-8 * np.eye(c1.shape[0]))
        matrix = inv_c1 @ c2
        evals = np.linalg.eigvalsh(matrix)
        return np.sqrt(np.sum(np.log(np.maximum(evals, 1e-10)) ** 2))
    except Exception:
        return np.inf


def compute_reference_covariance(data_list):
    """
    从多个 numpy 数组计算参考协方差矩阵。
    data_list: list of np.ndarray [trials, ch, time]
    """
    all_data = np.concatenate(data_list, axis=0)
    n_channels = all_data.shape[1]
    data_2d = all_data.transpose(1, 0, 2).reshape(n_channels, -1)
    cov = data_2d @ data_2d.T
    cov = cov / (np.trace(cov) + 1e-10)
    return cov


def compute_ea_matrix_from_covs(subject_covs, subject_weights=None, eps=1e-10):
    """
    从一组协方差矩阵计算全局EA矩阵 P = Σ^(-1/2)，支持加权。
    不需要所有被试数据拼接，直接累加协方差矩阵，避免时间维度不一致。
    """
    n = len(subject_covs)
    n_channels = subject_covs[0].shape[0]

    if subject_weights is None:
        subject_weights = np.ones(n)
    else:
        subject_weights = np.array(subject_weights) / np.sum(subject_weights)

    total_cov = np.zeros((n_channels, n_channels))
    for i in range(n):
        total_cov += subject_weights[i] * subject_covs[i]

    total_cov = total_cov / (np.trace(total_cov) + eps)
    eigvals, eigvecs = np.linalg.eigh(total_cov)
    eigvals = np.maximum(eigvals, eps)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    P = eigvecs @ D_inv_sqrt @ eigvecs.T
    return P


def compute_ea_matrix(data, weights=None, eps=1e-10):
    """
    计算EA矩阵 P = Σ^(-1/2)，支持加权。
    data: np.ndarray [trials, ch, time]
    weights: np.ndarray [trials] 或 None
    """
    n_trials, n_channels, n_times = data.shape

    if weights is None:
        data_2d = data.transpose(1, 0, 2).reshape(n_channels, -1)
        cov = data_2d @ data_2d.T
    else:
        weights = weights / weights.sum()
        cov = np.zeros((n_channels, n_channels))
        for i in range(n_trials):
            trial = data[i]
            w = weights[i]
            cov += w * (trial @ trial.T)

    cov = cov / (np.trace(cov) + eps)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    P = eigvecs @ D_inv_sqrt @ eigvecs.T
    return P


def apply_ea_to_data(data, P):
    """对数据应用EA变换: X_aligned = P @ X"""
    return np.einsum('ij, bjt -> bit', P, data).astype(np.float32)


def riemannian_mean(covs, max_iter=10, tol=1e-6):
    """
    计算一组协方差矩阵的Riemannian均值（迭代算法）。
    covs: list of np.ndarray [ch, ch]
    """
    n = len(covs)
    if n == 0:
        return None
    if n == 1:
        return covs[0]

    mean = np.mean(covs, axis=0)

    for _ in range(max_iter):
        try:
            mean_chol = np.linalg.cholesky(mean + 1e-8 * np.eye(mean.shape[0]))
            mean_inv_sqrt = np.linalg.inv(mean_chol)
            mean_sqrt = mean_chol
        except np.linalg.LinAlgError:
            break

        tangent_sum = np.zeros_like(mean)
        for c in covs:
            temp = mean_inv_sqrt @ c @ mean_inv_sqrt.T
            evals, evecs = np.linalg.eigh(temp)
            log_temp = evecs @ np.diag(np.log(np.maximum(evals, 1e-10))) @ evecs.T
            tangent_sum += log_temp

        tangent_mean = tangent_sum / n
        exp_temp = mean_sqrt @ tangent_mean @ mean_sqrt.T
        evals, evecs = np.linalg.eigh(exp_temp)
        new_mean = evecs @ np.diag(np.exp(evals)) @ evecs.T

        if np.linalg.norm(new_mean - mean, 'fro') < tol:
            break
        mean = new_mean

    return mean


def detect_outliers_paired(subject_covs, outlier_threshold=1.5):
    """
    基于两两距离检测离群被试：对于每个被试，计算它与其它所有被试距离的中位数，
    如果这个中位数距离 > Q3 + threshold*IQR，则判定为离群。
    """
    n = len(subject_covs)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = compute_riemann_distance(subject_covs[i], subject_covs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    median_distances = []
    for i in range(n):
        others_dists = dist_matrix[i, np.arange(n) != i]
        median_distances.append(np.median(others_dists))
    median_distances = np.array(median_distances)

    q1 = np.percentile(median_distances, 25)
    q3 = np.percentile(median_distances, 75)
    iqr = q3 - q1
    cutoff = q3 + outlier_threshold * iqr

    mask = median_distances <= cutoff
    return mask, median_distances, dist_matrix


# ==================== 主函数 ====================

def adaptive_group_ea(all_subject_data, mode='filter', outlier_threshold=1.2,
                      max_clusters=5, verbose=True):
    """
    自适应EA对齐。

    参数:
        all_subject_data: list of dicts, keys: 'left','right','feet','hands'
        mode:
            'filter'   — 剔除离群被试后做全局EA（默认，推荐）
            'weighted' — 加权全局EA
            'group'    — 聚类分组EA
            'global'   — 所有人一起EA
        outlier_threshold: 离群检测的严格程度（越小越严格）

    返回:
        aligned_data: list of dicts
        group_info: dict
    """
    n_subjects = len(all_subject_data)
    if verbose:
        print(f"\n🔧 自适应EA: 共 {n_subjects} 个被试, 模式='{mode}'")

    # Step 1: 提取参考协方差
    subject_covs = []
    for subj_data in all_subject_data:
        data_list = [subj_data[k] for k in ['left', 'right', 'feet', 'hands']]
        cov = compute_reference_covariance(data_list)
        subject_covs.append(cov)

    # Step 2: 全局Riemannian均值
    global_mean = riemannian_mean(subject_covs)
    distances_to_mean = np.array([compute_riemann_distance(global_mean, c) for c in subject_covs])

    # Step 3: 基于两两距离的离群检测
    mask, median_dists, dist_matrix = detect_outliers_paired(subject_covs, outlier_threshold=outlier_threshold)

    if verbose:
        print(f"  两两距离中位数范围: [{median_dists.min():.4f}, {median_dists.max():.4f}]")
        removed = np.where(~mask)[0].tolist()
        print(f"  检测到 {len(removed)} 个离群被试: {removed}")

    if mode == 'global':
        selected_indices = list(range(n_subjects))
        subject_weights = None
        if verbose:
            print("  使用全部被试计算全局EA矩阵")

    elif mode == 'filter':
        selected_indices = np.where(mask)[0].tolist()
        removed = np.where(~mask)[0].tolist()
        subject_weights = None
        if verbose:
            print(f"  剔除 {len(removed)} 个离群被试")
            print(f"  保留 {len(selected_indices)} 个被试计算EA矩阵")

    elif mode == 'weighted':
        selected_indices = list(range(n_subjects))
        median_d = np.median(median_dists)
        sigma = median_d * 0.5
        subject_weights = np.exp(-median_dists ** 2 / (2 * sigma ** 2))
        if verbose:
            print(f"  加权EA: 权重范围 [{subject_weights.min():.3f}, {subject_weights.max():.3f}]")
            worst_idx = np.argmin(subject_weights)
            print(f"  最低权重被试 #{worst_idx}: w={subject_weights[worst_idx]:.3f}")

    elif mode == 'group':
        return _group_mode_ea(all_subject_data, subject_covs, max_clusters, verbose)

    else:
        raise ValueError(f"未知模式: {mode}")

    # Step 4: 计算EA矩阵
    selected_covs = [subject_covs[i] for i in selected_indices]
    if mode == 'weighted':
        selected_weights = [subject_weights[i] for i in selected_indices]
        P_global = compute_ea_matrix_from_covs(selected_covs, subject_weights=selected_weights)
    else:
        P_global = compute_ea_matrix_from_covs(selected_covs)

    cond = np.linalg.cond(P_global)
    if verbose:
        print(f"  EA矩阵条件数: {cond:.2f}")

    # Step 5: 应用到所有被试
    aligned_data = []
    for idx in range(n_subjects):
        subj_data = all_subject_data[idx]
        aligned_subj = {}
        for key in ['left', 'right', 'feet', 'hands']:
            aligned_subj[key] = apply_ea_to_data(subj_data[key], P_global)
        aligned_data.append(aligned_subj)

    group_info = {
        'mode': mode,
        'group_labels': np.ones(n_subjects, dtype=int),
        'distance_matrix': dist_matrix,
        'n_groups': 1,
        'distances_to_mean': distances_to_mean,
        'median_distances': median_dists,
        'selected_indices': selected_indices,
        'removed_indices': removed if 'removed' in dir() else [],
    }

    if mode == 'weighted':
        group_info['subject_weights'] = subject_weights

    return aligned_data, group_info


def _group_mode_ea(all_subject_data, subject_covs, max_clusters=5, verbose=True):
    """聚类分组EA子函数"""
    n_subjects = len(all_subject_data)

    dist_matrix = np.zeros((n_subjects, n_subjects))
    for i in range(n_subjects):
        for j in range(i + 1, n_subjects):
            d = compute_riemann_distance(subject_covs[i], subject_covs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    condensed = squareform(dist_matrix)
    Z = linkage(condensed, method='average')

    from sklearn.metrics import silhouette_score
    best_score = -1
    best_labels = None
    for k in range(2, min(max_clusters + 1, n_subjects)):
        labels = fcluster(Z, k, criterion='maxclust')
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < n_subjects:
            try:
                score = silhouette_score(dist_matrix, labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_labels = labels
            except Exception:
                pass

    if best_labels is None:
        best_labels = np.ones(n_subjects, dtype=int)

    unique_groups = np.unique(best_labels)
    n_groups = len(unique_groups)

    if verbose:
        print(f"  📊 自动分组结果: {n_groups} 组")
        for g in unique_groups:
            members = [i for i in range(n_subjects) if best_labels[i] == g]
            print(f"    组 {g}: {len(members)} 个被试 {members}")

    aligned_data = [None] * n_subjects

    for g in unique_groups:
        members = [i for i in range(n_subjects) if best_labels[i] == g]
        group_covs = [subject_covs[idx] for idx in members]
        P_group = compute_ea_matrix_from_covs(group_covs)
        cond = np.linalg.cond(P_group)
        if verbose:
            print(f"  组 {g} EA矩阵条件数: {cond:.2f}")

        for idx in members:
            subj_data = all_subject_data[idx]
            aligned_subj = {}
            for key in ['left', 'right', 'feet', 'hands']:
                aligned_subj[key] = apply_ea_to_data(subj_data[key], P_group)
            aligned_data[idx] = aligned_subj

    group_info = {
        'mode': 'group',
        'group_labels': best_labels,
        'distance_matrix': dist_matrix,
        'n_groups': n_groups,
        'linkage': Z,
    }

    return aligned_data, group_info


# ==================== 诊断图 ====================

def plot_adaptive_ea_results(group_info, output_dir="figures"):
    """绘制自适应EA的诊断图"""
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False
        Path(output_dir).mkdir(exist_ok=True)

        mode = group_info.get('mode', 'unknown')

        if mode in ['filter', 'weighted'] and group_info.get('median_distances') is not None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            median_dists = group_info['median_distances']
            selected = set(group_info.get('selected_indices', []))
            removed = set(group_info.get('removed_indices', []))

            # 左图：中位数距离
            colors = ['#2E86AB' if i in selected else '#E63946' for i in range(len(median_dists))]
            axes[0].bar(range(len(median_dists)), median_dists, color=colors, alpha=0.8, edgecolor='none')
            q1 = np.percentile(median_dists, 25)
            q3 = np.percentile(median_dists, 75)
            iqr = q3 - q1
            cutoff = q3 + 1.2 * iqr
            axes[0].axhline(cutoff, color='red', linestyle='--', linewidth=1.5, label=f'剔除阈值 (Q3+1.2×IQR)')
            axes[0].axhline(np.median(median_dists), color='green', linestyle='--', linewidth=1.5, label='中位数')
            axes[0].set_xlabel('被试编号', fontsize=11)
            axes[0].set_ylabel('与其它被试的中位数距离', fontsize=11)
            axes[0].set_title(f'(a) 离群被试检测（红色=剔除，共{len(removed)}个）', fontsize=11, fontweight='bold')
            axes[0].legend(fontsize=9)
            axes[0].grid(True, alpha=0.3, axis='y')

            # 右图：距离矩阵热力图
            dist_mat = group_info.get('distance_matrix')
            if dist_mat is not None:
                im = axes[1].imshow(dist_mat, cmap='viridis', aspect='auto')
                axes[1].set_xlabel('被试编号', fontsize=11)
                axes[1].set_ylabel('被试编号', fontsize=11)
                axes[1].set_title('(b) 被试间黎曼距离矩阵', fontsize=11, fontweight='bold')
                plt.colorbar(im, ax=axes[1], fraction=0.046)

            plt.suptitle('自适应EA诊断图', fontsize=13, fontweight='bold')
            plt.tight_layout()
            save_path = f"{output_dir}/ea_adaptive_diagnosis.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"📊 诊断图已保存: {save_path}")

        elif mode == 'group':
            from scipy.cluster.hierarchy import dendrogram
            Z = group_info.get('linkage')
            labels = group_info.get('group_labels')
            if Z is not None and labels is not None:
                fig, ax = plt.subplots(figsize=(14, 6))
                dn = dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8)
                unique_groups = np.unique(labels)
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_groups)))
                for i, leaf_id in enumerate(dn['leaves']):
                    g = labels[leaf_id]
                    ax.axvspan(i - 0.4, i + 0.4, ymin=0, ymax=0.05,
                               color=colors[g - 1], alpha=0.7, transform=ax.get_xaxis_transform())
                ax.set_title('自适应EA被试分组树状图', fontsize=13, fontweight='bold')
                ax.set_xlabel('被试编号', fontsize=11)
                ax.set_ylabel('黎曼距离', fontsize=11)
                plt.tight_layout()
                save_path = f"{output_dir}/ea_group_dendrogram.png"
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
                plt.close()
                print(f"📊 诊断图已保存: {save_path}")

    except Exception as e:
        print(f"⚠️ 诊断图绘制失败: {e}")