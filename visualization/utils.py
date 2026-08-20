# -*- coding: utf-8 -*-
"""
可视化专用工具函数
完全独立，不依赖项目其他文件
"""

import numpy as np
import torch


def normalize_trace(cov_matrices):
    """迹归一化: C / trace(C)"""
    if isinstance(cov_matrices, np.ndarray):
        if cov_matrices.ndim == 2:
            traces = np.trace(cov_matrices)
            return cov_matrices / (traces + 1e-10)
        elif cov_matrices.ndim == 3:
            traces = np.trace(cov_matrices, axis1=1, axis2=2)
            traces = traces[:, np.newaxis, np.newaxis]
            return cov_matrices / (traces + 1e-10)
        else:
            raise ValueError(f"不支持的维度: {cov_matrices.ndim}")
    elif isinstance(cov_matrices, torch.Tensor):
        if cov_matrices.dim() == 2:
            traces = torch.trace(cov_matrices)
            return cov_matrices / (traces + 1e-10)
        elif cov_matrices.dim() == 3:
            traces = torch.diagonal(cov_matrices, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
            return cov_matrices / (traces + 1e-10)
        else:
            raise ValueError(f"不支持的维度: {cov_matrices.dim()}")
    else:
        raise TypeError(f"不支持的类型: {type(cov_matrices)}")


def log_euclidean_mapping(cov_matrices, eps=1e-7):
    """
    对数欧氏映射: C -> log(C)
    Args: cov_matrices [N, C, C] (numpy array 或 torch tensor)
    Returns: tangent_vectors [N, D] where D = C*(C+1)//2
    """
    if isinstance(cov_matrices, np.ndarray):
        # NumPy版本
        N, C, _ = cov_matrices.shape
        evals, evecs = np.linalg.eigh(cov_matrices)
        log_evals = np.log(np.clip(evals, a_min=eps, a_max=None))
        log_D = np.zeros((N, C, C), dtype=log_evals.dtype)
        log_D[np.arange(N)[:, None], np.arange(C), np.arange(C)] = log_evals
        log_C = evecs @ log_D @ evecs.transpose(0, 2, 1)
        
        idx = np.triu_indices(C)
        tangent = log_C[:, idx[0], idx[1]]
        
        off_diag = (idx[0] != idx[1])
        tangent[:, off_diag] *= np.sqrt(2.0)
        
        return tangent.astype(np.float32)
    
    elif isinstance(cov_matrices, torch.Tensor):
        # PyTorch版本
        C = cov_matrices.shape[-1]
        N = cov_matrices.shape[0]
        evals, evecs = torch.linalg.eigh(cov_matrices)
        log_evals = torch.log(torch.clamp(evals, min=eps))
        log_D = torch.diag_embed(log_evals)
        log_C = torch.matmul(torch.matmul(evecs, log_D), evecs.mT)
        
        idx = torch.triu_indices(C, C)
        tangent = log_C[:, idx[0], idx[1]]
        
        off_diag = (idx[0] != idx[1])
        tangent[:, off_diag] *= np.sqrt(2.0)
        
        return tangent.to(torch.float32)
    
    else:
        raise TypeError(f"不支持的类型: {type(cov_matrices)}")


def riemann_distance_logeuc(c1, c2):
    """对数欧氏黎曼距离"""
    try:
        inv_c1 = np.linalg.inv(c1 + 1e-8 * np.eye(c1.shape[0]))
        matrix = inv_c1 @ c2
        evals = np.linalg.eigvalsh(matrix)
        return np.sqrt(np.sum(np.log(np.maximum(evals, 1e-10)) ** 2))
    except Exception:
        return 0.0


def tangent_to_full(tangent_vectors, n_channels):
    """切空间向量 -> 对称矩阵 (用于可视化)"""
    D = n_channels * (n_channels + 1) // 2
    idx = np.triu_indices(n_channels)
    matrices = np.zeros((len(tangent_vectors), n_channels, n_channels))
    matrices[:, idx[0], idx[1]] = tangent_vectors
    matrices[:, idx[1], idx[0]] = tangent_vectors
    return matrices