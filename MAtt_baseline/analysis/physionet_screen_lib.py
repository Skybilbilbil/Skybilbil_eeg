# -*- coding: utf-8 -*-
"""筛选协议公共库:个体筛选(训练侧) + 试次级标签盲质量门(2026-10-25,用户指定部署协议)。

设计约束:
- 个体筛选分数 = 该被试**训练折**试次上的 CSP 5 折 CV(全局切分 random_state=42 的
  train 成员)——筛选统计不接触测试折,堵「筛选看过测试」的口子。
- 试次质量门(标签盲): q = logdet(C_task) − logdet(C_rest),C 为试次内任务窗/静息窗
  协方差(8-30Hz 已滤、EA 后数据)。q<0 = 任务期去同步(ERD 存在);**不使用类别标签,
  也不使用符号化侧化信息** ⇒ 测试试次可同规则过滤。另删切空间范数离群试次(MAD>3)。
- 复用:run_physionet_screened.py(分类读数)。
"""
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
REST = (0, 160)      # [-1,0) s @160Hz
TASK = (160, 10**9)  # [0,3.5) s, 由 T 截断


def _cov(seg):
    s = seg - seg.mean(axis=2, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / seg.shape[-1]


def trial_erd_scores(X):
    """逐试次标签盲 ERD 分数 q = slogdet(C_task) − slogdet(C_rest)。"""
    def logdet(seg):
        C = _cov(seg.astype(np.float64))
        return np.array([np.linalg.slogdet(c)[1] for c in C])
    return logdet(X[:, :, TASK[0]:TASK[1]]) - logdet(X[:, :, REST[0]:REST[1]])


def trial_tangent_norm(X):
    """逐试次任务窗协方量的 log-特征值谱范数(对被试中位数谱的偏离)——离群检测。"""
    C = _cov(X[:, :, TASK[0]:TASK[1]].astype(np.float64))
    V = np.array([np.log(np.maximum(np.linalg.eigvalsh(c), 1e-30)) for c in C])
    ref = np.median(V, axis=0)
    return np.linalg.norm(V - ref, axis=1)


def gate_mask(X, keep_frac=0.5):
    """标签盲质量门:保留去同步更强(更小 q)的 keep_frac 比例试次 + 删 MAD>3 离群。"""
    q = trial_erd_scores(X)
    thr = np.quantile(q, keep_frac)   # 保留最小的 keep_frac 比例
    keep = q <= thr
    tn = trial_tangent_norm(X)
    med = np.median(tn)
    mad = np.median(np.abs(tn - med)) * 1.4826
    keep &= tn <= med + 3 * mad
    return keep, q


def subject_train_csp_cv(subject_data, sub_tr, y_tr, idx_tr, k=5):
    """个体筛选分数:只用训练折试次的 CSP 5 折 CV(与全量版同实现)。

    idx_tr = 训练折每行在全池拼接中的原始下标(切分时以 others=[..., idx_all] 带出);
    被试 i 的本地试次下标 = idx_tr[sub_tr==i] − 该被试拼接偏移。
    """
    from analyze_individual_diff import fit_csp
    from sklearn.model_selection import StratifiedKFold
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    scores = []
    off = 0
    for i, (X, y) in enumerate(subject_data):
        m = sub_tr == i
        loc = idx_tr[m] - off
        off += len(y)
        Xs, ys = X[loc], y_tr[m]
        C = _cov(Xs[:, :, TASK[0]:TASK[1]].astype(np.float64))
        accs = []
        for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(C, ys):
            W = fit_csp(C[tr], ys[tr])
            F = lambda Cs: np.log(np.maximum(np.einsum('ia,nij,ja->na', W, Cs, W), 1e-12))
            accs.append(LDA(solver='lsqr', shrinkage='auto')
                        .fit(F(C[tr]), ys[tr]).score(F(C[te]), ys[te]))
        scores.append(float(np.mean(accs)))
    return scores
