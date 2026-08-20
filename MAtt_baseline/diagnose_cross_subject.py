# -*- coding: utf-8 -*-
"""
跨被试差异诊断脚本
测量 PhysionetMI 数据在四个维度上的跨被试差异，输出诊断报告表格。
"""

import os
import sys
import pickle
import json
import numpy as np
import time
from pathlib import Path

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# ---------- 路径 ----------
SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / '../saved_data/preprocessed_data_moabb_raw_ica.pkl'
OUTPUT_DIR = SCRIPT_DIR / 'results' / 'diagnosis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------- 配置 ----------
CONFIG = {
    'n_subjects': 50,
    'max_trials_per_subject': 60,
    'n_classes': 4,
    'window_length': 0.5,
    'window_step': 0.125,
    'sfreq': 160,
    'n_csp_components': 6,
    'random_state': 42,
}

np.random.seed(CONFIG['random_state'])


# ================================================================
# 工具函数
# ================================================================

def load_data():
    """加载 PhysionetMI 预处理数据"""
    with open(DATA_PATH, 'rb') as f:
        data = pickle.load(f)

    left_list = data['aligned_left_epochs_list']
    right_list = data['aligned_right_epochs_list']
    feet_list = data['aligned_feet_epochs_list']
    hands_list = data['aligned_hands_epochs_list']
    metadata = data['subject_metadata']

    n_total = len(metadata)
    n_use = min(CONFIG['n_subjects'], n_total)
    subjects = np.random.choice(n_total, n_use, replace=False)
    subjects.sort()

    print(f'使用 {n_use}/{n_total} 个被试: {[int(s) for s in subjects]}')

    subject_data = []
    for idx in subjects:
        X_list = [left_list[idx], right_list[idx], feet_list[idx], hands_list[idx]]
        X_subj, y_subj = [], []
        for cls_idx, arr in enumerate(X_list):
            if arr is None or len(arr) == 0:
                continue
            X_subj.append(arr)
            y_subj.append(np.full(len(arr), cls_idx, dtype=int))
        X_subj = np.concatenate(X_subj, axis=0).astype(np.float64)
        y_subj = np.concatenate(y_subj, axis=0)

        if X_subj.shape[0] > CONFIG['max_trials_per_subject']:
            idxs = np.random.choice(X_subj.shape[0], CONFIG['max_trials_per_subject'], replace=False)
            X_subj = X_subj[idxs]; y_subj = y_subj[idxs]

        subject_data.append((X_subj, y_subj))
        print(f"  S{idx+1:03d}: {X_subj.shape}, 类别 {np.bincount(y_subj, minlength=4)}")

    return subject_data, [int(s) for s in subjects]


def compute_covariance(X):
    """计算每个试次的协方差矩阵。X: (N, C, T) -> (N, C, C)"""
    N, C, T = X.shape
    X_centered = X - X.mean(axis=-1, keepdims=True)
    cov = X_centered @ X_centered.transpose(0, 2, 1) / (T - 1)
    reg = 1e-6 * np.eye(C)
    return cov + reg


def log_euclidean_distance(A, B):
    """Log-Euclidean distance between two SPD matrices."""
    s_a, u_a = np.linalg.eigh(A)
    s_b, u_b = np.linalg.eigh(B)
    s_a = np.clip(s_a, 1e-10, None)
    s_b = np.clip(s_b, 1e-10, None)
    log_A = u_a @ np.diag(np.log(s_a)) @ u_a.T
    log_B = u_b @ np.diag(np.log(s_b)) @ u_b.T
    return np.sqrt(np.sum((log_A - log_B) ** 2))


# ================================================================
# CSP 实现
# ================================================================

def csp_fit(X, y, n_components=6):
    """
    CSP 空间滤波器。X: (N, C, T), y: (N,) 二分类。
    返回 (C, n_components)，前一半 class 0 方向，后一半 class 1 方向。
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError('CSP 仅支持二分类')

    C = X.shape[1]
    covs = compute_covariance(X)
    mask_c1 = (y == classes[0])
    mask_c2 = (y == classes[1])

    if mask_c1.sum() < 2 or mask_c2.sum() < 2:
        raise ValueError('每个类别至少需要 2 个试次')

    cov1 = covs[mask_c1].mean(axis=0)
    cov2 = covs[mask_c2].mean(axis=0)

    # 自适应正则化：确保 cov1 + cov2 正定
    trace_val = np.trace(cov1 + cov2)
    reg = max(1e-1 * trace_val / C, 1e-8) * np.eye(C)
    cov1_reg = cov1 + reg
    cov2_reg = cov2 + reg
    cov_sum = cov1_reg + cov2_reg

    # 广义特征值分解 (A, B)，使用 scipy
    from scipy.linalg import eigh
    eigvals, eigvecs = eigh(cov1_reg, cov_sum)
    n_pairs = n_components // 2
    idx_min = np.argsort(eigvals)[:n_pairs]
    idx_max = np.argsort(eigvals)[-n_pairs:]
    idx = np.concatenate([idx_min, idx_max])
    return eigvecs[:, idx]


def subspace_principal_angles(W_A, W_B):
    """计算两个子空间之间的主角度（度）"""
    Q_A, _ = np.linalg.qr(W_A)
    Q_B, _ = np.linalg.qr(W_B)
    s = np.linalg.svd(Q_A.T @ Q_B, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    return np.degrees(np.arccos(s))


# ================================================================
# 维度 1：边缘分布偏移
# ================================================================

def diagnose_dim1_marginal_shift(subject_data):
    print('\n' + '='*60)
    print('维度 1：边缘分布偏移（Marginal Distribution Shift）')
    print('='*60)

    covs_by_subject = []
    for X, y in subject_data:
        covs_by_subject.append(compute_covariance(X))

    print('  计算每个被试的 Riemannian 均值...')
    mean_subjects = []
    for i, covs in enumerate(covs_by_subject):
        mean_subjects.append(np.mean(covs, axis=0))  # Log-Euclidean 均值近似
        if (i+1) % 10 == 0:
            print(f'    {i+1}/{len(covs_by_subject)}')

    print('  计算被试间距离...')
    inter_dists = []
    for i in range(len(mean_subjects)):
        for j in range(i+1, len(mean_subjects)):
            inter_dists.append(log_euclidean_distance(mean_subjects[i], mean_subjects[j]))
    inter_mean = np.mean(inter_dists)

    print('  计算被试内距离...')
    intra_dists = []
    for covs in covs_by_subject:
        n = covs.shape[0]
        idxs = np.random.choice(n, min(n, 30), replace=False)
        for i in range(len(idxs)):
            for j in range(i+1, len(idxs)):
                intra_dists.append(log_euclidean_distance(covs[idxs[i]], covs[idxs[j]]))
    intra_mean = np.mean(intra_dists)

    ratio = inter_mean / intra_mean if intra_mean > 0 else float('inf')

    print(f'\n  被试间平均距离: {inter_mean:.4f}')
    print(f'  被试内平均距离: {intra_mean:.4f}')
    print(f'  距离比 (Ratio): {ratio:.2f}')

    if ratio > 2.0:
        verdict = '[!] 显著 — 边缘偏移严重，必须做 Re-bias 对齐'
    elif ratio > 1.0:
        verdict = '[!] 中等 — 建议做 Re-bias 对齐'
    else:
        verdict = '[OK] 轻微 — 边缘偏移不显著'

    print(f'  判定: {verdict}')
    return {
        'dim': 1, 'name': '边缘偏移',
        'metric': f'{ratio:.2f}', 'threshold': '> 2.0',
        'significant': ratio > 2.0, 'verdict': verdict,
        'inter_mean': float(inter_mean), 'intra_mean': float(intra_mean),
        'strategy': 'Re-bias 对齐（Riemannian 均值居中）' if ratio > 1.0 else '无需特殊处理',
    }


# ================================================================
# 维度 2：绝对功率（Trace）个体差异
# ================================================================

def diagnose_dim2_trace_power(subject_data):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import cross_val_score

    print('\n' + '='*60)
    print('维度 2：绝对功率（Trace）个体差异')
    print('='*60)

    mean_traces, within_cvs = [], []
    for X, y in subject_data:
        covs = compute_covariance(X)
        traces = np.trace(covs, axis1=1, axis2=2)
        mean_traces.append(np.mean(traces))
        within_cvs.append(np.std(traces) / (np.mean(traces) + 1e-10))

    mean_traces = np.array(mean_traces)
    cv_between = np.std(mean_traces) / (np.mean(mean_traces) + 1e-10)
    cv_within_avg = np.mean(within_cvs)

    print(f'\n  被试间 Trace CV: {cv_between:.4f}')
    print(f'  被试内 Trace CV 均值: {cv_within_avg:.4f}')
    print(f'  比值: {cv_between / (cv_within_avg + 1e-10):.2f}')

    X_trace, y_all = [], []
    for X, y in subject_data:
        covs = compute_covariance(X)
        log_traces = np.log(np.trace(covs, axis1=1, axis2=2)).reshape(-1, 1)
        X_trace.append(log_traces); y_all.append(y)
    X_trace = np.concatenate(X_trace, axis=0); y_all = np.concatenate(y_all, axis=0)

    lda = LinearDiscriminantAnalysis(solver='svd')
    scores = cross_val_score(lda, X_trace, y_all, cv=5)
    trace_acc = scores.mean()
    chance = 1.0 / CONFIG['n_classes']

    print(f'  仅用 Log-Trace 分类准确率: {trace_acc*100:.2f}% (随机水平: {chance*100:.2f}%)')

    if cv_between > 2 * cv_within_avg and trace_acc > chance * 1.5:
        verdict = '[!] 显著 — 绝对幅值差异极大，建议保留 Trace 双流辅助特征'
    elif cv_between > 2 * cv_within_avg:
        verdict = '[!] 中等 — 幅值差异存在，但分类能力有限'
    else:
        verdict = '[OK] 轻微 — 绝对幅值差异不显著'

    print(f'  判定: {verdict}')
    return {
        'dim': 2, 'name': '绝对功率',
        'metric': f'CV_between={cv_between:.3f}, CV_within={cv_within_avg:.3f}',
        'threshold': 'CV_between > 2×CV_within & Trace Acc > 随机',
        'significant': cv_between > 2 * cv_within_avg and trace_acc > chance * 1.5,
        'verdict': verdict,
        'cv_between': float(cv_between), 'cv_within': float(cv_within_avg),
        'trace_acc': float(trace_acc),
        'strategy': '双流架构保留 Log-Trace' if cv_between > 2 * cv_within_avg else '无需特殊处理',
    }


# ================================================================
# 维度 3：时序动态差异
# ================================================================

def diagnose_dim3_temporal_dynamics(subject_data):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold

    print('\n' + '='*60)
    print('维度 3：时序动态（潜伏期/持续时间）差异')
    print('='*60)

    sfreq = CONFIG['sfreq']
    win_samples = int(CONFIG['window_length'] * sfreq)
    step_samples = int(CONFIG['window_step'] * sfreq)

    n_subj = min(20, len(subject_data))
    subj_data_subset = subject_data[:n_subj]

    print(f'  使用前 {n_subj} 个被试进行时序分析...')
    print(f'  窗口长度: {CONFIG["window_length"]}s ({win_samples}点), 步长: {CONFIG["window_step"]}s ({step_samples}点)')

    all_curves = []
    peak_windows = []

    for subj_idx, (X, y) in enumerate(subj_data_subset):
        C, T = X.shape[1], X.shape[2]
        n_windows = (T - win_samples) // step_samples + 1
        if n_windows < 2:
            continue

        window_accs = []
        for w in range(n_windows):
            start = w * step_samples
            end = start + win_samples
            X_win = X[:, :, start:end]

            try:
                skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=CONFIG['random_state'])
                fold_accs = []
                for train_idx, test_idx in skf.split(X_win, y):
                    X_tr, X_te = X_win[train_idx], X_win[test_idx]
                    y_tr, y_te = y[train_idx], y[test_idx]

                    # 每类 vs 其他类 CSP，拼接 Log-Variance 特征
                    X_tr_feats, X_te_feats = [], []
                    for c in np.unique(y_tr):
                        y_tr_bin = (y_tr == c).astype(int)
                        if y_tr_bin.sum() < 2 or (1 - y_tr_bin).sum() < 2:
                            continue
                        W = csp_fit(X_tr, y_tr_bin, n_components=4)
                        X_tr_feats.append(np.log(np.var(W.T @ X_tr, axis=-1) + 1e-10))
                        X_te_feats.append(np.log(np.var(W.T @ X_te, axis=-1) + 1e-10))

                    if not X_tr_feats:
                        raise ValueError('CSP 特征为空')
                    X_tr_feat = np.concatenate(X_tr_feats, axis=1)
                    X_te_feat = np.concatenate(X_te_feats, axis=1)

                    clf = LinearDiscriminantAnalysis(solver='svd')
                    clf.fit(X_tr_feat, y_tr)
                    fold_accs.append(clf.score(X_te_feat, y_te))
                window_accs.append(np.mean(fold_accs))
            except Exception as e:
                if subj_idx == 0 and w == 0:
                    print(f'    [DEBUG] CSP 失败: {type(e).__name__}: {e}')
                window_accs.append(np.nan)

        window_accs = np.array(window_accs)
        if np.all(np.isnan(window_accs)):
            continue

        window_accs = np.nan_to_num(window_accs, nan=0.25)
        all_curves.append(window_accs)
        peak_windows.append(np.argmax(window_accs))

        if (subj_idx + 1) % 5 == 0:
            print(f'    {subj_idx+1}/{n_subj}')

    # 空保护
    if len(all_curves) == 0:
        print('\n  [!] 没有成功计算任何被试的时序曲线，跳过维度 3')
        return {
            'dim': 3, 'name': '时序动态',
            'metric': 'N/A', 'threshold': '> 1.5 窗口',
            'significant': False,
            'verdict': '无法计算 — 数据不适合滑动窗口 CSP 分析',
            'latency_shift': None,
            'strategy': '检查数据质量或改用其他时序分析方法',
            'curve_path': None,
        }

    peak_windows = np.array(peak_windows)
    latency_shift = np.std(peak_windows)

    print(f'\n  成功计算 {len(all_curves)} 个被试')
    print(f'  峰值窗口偏移标准差: {latency_shift:.2f} 个窗口')
    print(f'  (相当于 {latency_shift * CONFIG["window_step"]:.3f} 秒)')

    if latency_shift > 1.5:
        verdict = '[!] 显著 — 被试间响应速度差异大，固定硬分割 (m=6) 必然导致泛化崩溃'
    elif latency_shift > 1.0:
        verdict = '[!] 中等 — 建议使用多尺度时间窗口'
    else:
        verdict = '[OK] 轻微 — 被试间时序对齐良好'

    print(f'  判定: {verdict}')

    # 保存曲线
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(12, 6))
    x_axis = np.arange(len(all_curves[0])) * CONFIG['window_step']
    for i, curve in enumerate(all_curves):
        ax.plot(x_axis, curve, alpha=0.5, lw=1, label=f'S{i+1}' if i < 10 else None)
    ax.axhline(y=0.25, color='gray', ls='--', alpha=0.5, label='chance')
    ax.set_xlabel('Time (s)'); ax.set_ylabel('Accuracy')
    ax.set_title(f'Sliding-window classification accuracy curves\nLatency shift std = {latency_shift:.2f} windows')
    ax.legend(loc='upper right', ncol=2, fontsize=7)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'temporal_curves.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Curve saved: {OUTPUT_DIR / "temporal_curves.png"}')

    return {
        'dim': 3, 'name': '时序动态',
        'metric': f'Latency Shift = {latency_shift:.2f} 窗口',
        'threshold': '> 1.5 窗口',
        'significant': latency_shift > 1.5, 'verdict': verdict,
        'latency_shift': float(latency_shift),
        'strategy': '多尺度时间窗口 / 动态时序对齐' if latency_shift > 1.0 else '无需特殊处理',
        'curve_path': str(OUTPUT_DIR / 'temporal_curves.png'),
    }


# ================================================================
# 维度 4：判别性子空间方向差异
# ================================================================

def diagnose_dim4_subspace_orientation(subject_data):
    print('\n' + '='*60)
    print('维度 4：判别性子空间方向差异（Subspace Orientation）')
    print('='*60)

    n_subj = len(subject_data)
    subspace_filters = []

    print('  计算每个被试的 CSP 判别子空间...')
    for subj_idx, (X, y) in enumerate(subject_data):
        unique, counts = np.unique(y, return_counts=True)
        target_class = unique[np.argmax(counts)]
        y_bin = (y == target_class).astype(int)
        if y_bin.sum() < 2 or (1 - y_bin).sum() < 2:
            continue
        try:
            W = csp_fit(X, y_bin, n_components=CONFIG['n_csp_components'])
            subspace_filters.append(W)
        except Exception as e:
            if subj_idx == 0:
                print(f'    [DEBUG] 被试 {subj_idx} CSP 失败: {type(e).__name__}: {e}')

        if (subj_idx + 1) % 10 == 0:
            print(f'    {subj_idx+1}/{n_subj}')

    print(f'  成功计算 {len(subspace_filters)} 个被试的子空间')

    if len(subspace_filters) < 2:
        print('  [!] 成功计算子空间的被试不足 2 人，跳过维度 4')
        return {
            'dim': 4, 'name': '子空间方向',
            'metric': 'N/A', 'threshold': '> 30°',
            'significant': False,
            'verdict': '无法计算 — CSP 子空间计算失败',
            'mean_angle': None,
            'strategy': '检查数据质量或改用其他子空间分析方法',
            'angle_path': None,
        }

    all_angles = []
    for i in range(len(subspace_filters)):
        for j in range(i+1, len(subspace_filters)):
            angles = subspace_principal_angles(subspace_filters[i], subspace_filters[j])
            all_angles.extend(angles)

    all_angles = np.array(all_angles)
    mean_angle = np.mean(all_angles)

    print(f'\n  平均主角度: {mean_angle:.2f}°')
    print(f'  最大主角度: {np.max(all_angles):.2f}°')
    print(f'  角度分布: 均值={mean_angle:.2f}°, 中位数={np.median(all_angles):.2f}°')

    if mean_angle > 30.0:
        verdict = '[!] 显著 — 被试间使用完全不同的电极组合，MAtt 投影矩阵 Wv 在跨被试时完全失效'
    elif mean_angle > 20.0:
        verdict = '[!] 中等 — 子空间方向存在差异，建议引入子空间正则化'
    else:
        verdict = '[OK] 轻微 — 被试间判别子空间方向一致'

    print(f'  判定: {verdict}')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(all_angles, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(mean_angle, color='red', ls='--', lw=2, label=f'mean={mean_angle:.1f}°')
    ax.axvline(30.0, color='orange', ls=':', lw=2, label='threshold 30°')
    ax.set_xlabel('Principal angle (°)'); ax.set_ylabel('Count')
    ax.set_title('CSP discriminative subspace principal angle distribution')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / 'subspace_angles.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Angle distribution saved: {OUTPUT_DIR / "subspace_angles.png"}')

    return {
        'dim': 4, 'name': '子空间方向',
        'metric': f'平均主角度 = {mean_angle:.1f}°',
        'threshold': '> 30°',
        'significant': mean_angle > 30.0, 'verdict': verdict,
        'mean_angle': float(mean_angle),
        'strategy': '子空间正则化（Subspace Regularization）' if mean_angle > 20.0 else '无需特殊处理',
        'angle_path': str(OUTPUT_DIR / 'subspace_angles.png'),
    }


# ================================================================
# 综合诊断报告
# ================================================================

def print_final_report(results):
    print('\n\n' + '='*80)
    print('                      跨被试差异综合诊断报告')
    print('='*80)

    print(f'\n{"维度":<8} {"量化指标":<30} {"数值":<25} {"阈值":<30} {"是否显著":<12} {"改造策略"}')
    print('-'*130)

    significant_count = 0
    for r in results:
        sig = '[!] Yes' if r['significant'] else '[OK] No'
        if r['significant']:
            significant_count += 1
        print(f'  {r["dim"]:<6} {r["name"]:<28} {r["metric"]:<25} {r["threshold"]:<30} {sig:<12} {r["strategy"]}')

    print('-'*130)
    print(f'\n  显著差异维度数: {significant_count}/4')

    if significant_count >= 3:
        print('  [!][!][!] 跨被试差异极其严重！建议在 MAtt 中加入以下所有改进：')
        print('    1. 流形对齐（Re-bias）')
        print('    2. 双流架构（保留 Log-Trace）')
        print('    3. 多尺度时间窗口 / 动态时序对齐')
        print('    4. 子空间正则化')
    elif significant_count >= 2:
        print('  [!][!] 跨被试差异较大，需要针对性改进')
    elif significant_count >= 1:
        print('  [!] 存在部分跨被试差异，建议针对性改进')
    else:
        print('  [OK] 跨被试差异不显著，MAtt 可直接用于跨被试')

    print('='*80)

    def convert(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    json_path = OUTPUT_DIR / 'diagnosis_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, default=convert, indent=2, ensure_ascii=False)
    print(f'\n  报告已保存: {json_path}')


# ================================================================
# 主入口
# ================================================================

def main():
    t0 = time.time()

    print('載入 PhysionetMI 数据...')
    subject_data, subject_ids = load_data()

    if len(subject_data) < 5:
        print('❌ 被试数量不足，至少需要 5 人')
        return

    results = []
    results.append(diagnose_dim1_marginal_shift(subject_data))
    results.append(diagnose_dim2_trace_power(subject_data))
    results.append(diagnose_dim3_temporal_dynamics(subject_data))
    results.append(diagnose_dim4_subspace_orientation(subject_data))

    print_final_report(results)
    print(f'\n总耗时: {time.time() - t0:.1f} 秒')
    print(f'输出目录: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()