import numpy as np
import mne
mne.set_log_level('ERROR')

def compute_ea_matrix_from_numpy(data, eps=1e-10):
    """
    从 numpy 数组计算 EA 变换矩阵 P（标准独立 EA）。

    参数:
        data: np.ndarray, 形状 (n_trials, n_channels, n_times)
        eps: 防止特征值为 0

    返回:
        P: np.ndarray, 形状 (n_channels, n_channels)
    """
    n_channels = data.shape[1]
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
