import numpy as np
import mne
mne.set_log_level('ERROR')

def compute_ea_matrix_from_epochs(epochs_list, eps=1e-10):
    """
    根据一个被试的多个 Epochs（例如 [left, right, feet, hands]）计算 EA 变换矩阵 P。

    参数:
        epochs_list: list[mne.Epochs]，同一被试的若干 Epochs 对象
        eps: 为防止特征值为 0 而加的小正数

    返回:
        P: np.ndarray, 形状 (n_channels, n_channels)，欧氏对齐变换矩阵
    """
    # 先把这个被试的所有 epochs 拼在一起
    concat = mne.concatenate_epochs(epochs_list)
    data = concat.get_data()  # (n_trials, n_channels, n_times)

    n_trials, n_channels, n_times = data.shape
    # 展平成 (n_channels, n_trials * n_times)
    data_2d = data.transpose(1, 0, 2).reshape(n_channels, -1)

    # 计算归一化协方差: Σ = X X^T / trace(X X^T)
    cov = data_2d @ data_2d.T
    cov = cov / np.trace(cov)

    # 特征分解 Σ = U Λ U^T
    eigvals, eigvecs = np.linalg.eigh(cov)

    # 防止数值问题，特征值加一个 eps
    eigvals = np.maximum(eigvals, eps)

    # 计算 Σ^{-1/2} = U Λ^{-1/2} U^T
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    P = eigvecs @ D_inv_sqrt @ eigvecs.T  # (n_channels, n_channels)

    return P


def apply_ea_to_epochs_list(epochs_list, P):
    """
    对同一被试的多个 Epochs 应用 EA 变换矩阵 P。

    参数:
        epochs_list: list[mne.Epochs]
        P: np.ndarray, (n_channels, n_channels)

    返回:
        aligned_list: list[mne.Epochs]，结构与输入一一对应，只是数据被对齐
    """
    aligned_list = []

    for epochs in epochs_list:
        data = epochs.get_data()  # (n_trials, n_channels, n_times)
        # 应用 EA: 对每个 trial，做 P @ X
        # 使用 einsum: i,j 通道, t 时间
        data_aligned = np.einsum('ij, bjt -> bit', P, data)

        # 用对齐后的数据构造新的 Epochs 对象
        epochs_aligned = mne.EpochsArray(
            data_aligned,
            info=epochs.info,
            events=epochs.events,
            tmin=epochs.tmin,
            event_id=epochs.event_id
        )
        aligned_list.append(epochs_aligned)

    return aligned_list


def compute_ea_matrix_from_numpy(data, eps=1e-10):
    """
    从 numpy 数组计算 EA 变换矩阵 P（标准独立 EA）。

    参数:
        data: np.ndarray, 形状 (n_trials, n_channels, n_times)
        eps: 防止特征值为 0

    返回:
        P: np.ndarray, 形状 (n_channels, n_channels)
    """
    n_trials, n_channels, n_times = data.shape
    data_2d = data.transpose(1, 0, 2).reshape(n_channels, -1)
    cov = data_2d @ data_2d.T
    cov = cov / np.trace(cov)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, eps)
    D_inv_sqrt = np.diag(1.0 / np.sqrt(eigvals))
    P = eigvecs @ D_inv_sqrt @ eigvecs.T
    return P


def apply_ea_to_numpy(data, P):
    """
    对 numpy 数组应用 EA 变换。

    参数:
        data: np.ndarray, 形状 (n_trials, n_channels, n_times)
        P: np.ndarray, 形状 (n_channels, n_channels)

    返回:
        aligned: np.ndarray, 形状与 data 相同
    """
    return np.einsum('ij, bjt -> bit', P, data)


def riemannian_distance(A, B):
    """numpy array 黎曼距离"""
    try:
        vals_a, vecs_a = np.linalg.eigh(A)
        vals_a = np.maximum(vals_a, 1e-10)
        A_invsqrt = vecs_a @ np.diag(1.0/np.sqrt(vals_a)) @ vecs_a.T
        M = A_invsqrt @ B @ A_invsqrt
        vals_m, _ = np.linalg.eigh(M)
        vals_m = np.maximum(vals_m, 1e-10)
        return np.sqrt(np.sum(np.log(vals_m)**2))
    except Exception:
        return np.nan


def normalize_trace_np(covs):
    """迹归一化，支持 [..., C, C] 任意 batch 维度"""
    traces = np.trace(covs, axis1=-2, axis2=-1)
    traces = np.where(np.abs(traces) < 1e-12, 1.0, traces)
    return covs / traces[..., None, None]


def log_euclidean_mapping_np(covs):
    """numpy batch log映射，支持 [..., C, C] 任意 batch 维度"""
    original_shape = covs.shape
    C = original_shape[-1]
    covs_2d = covs.reshape(-1, C, C)
    N = covs_2d.shape[0]
    out = np.zeros((N, C * (C + 1) // 2))
    for i in range(N):
        vals, vecs = np.linalg.eigh(covs_2d[i])
        vals = np.maximum(vals, 1e-10)
        logD = np.diag(np.log(vals))
        logC_mat = vecs @ logD @ vecs.T
        idx = 0
        for r in range(C):
            for c in range(r, C):
                scale = np.sqrt(2.0) if r != c else 1.0
                out[i, idx] = logC_mat[r, c] * scale
                idx += 1
    new_shape = original_shape[:-2] + (C * (C + 1) // 2,)
    return out.reshape(new_shape)