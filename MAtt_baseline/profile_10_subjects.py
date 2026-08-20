# -*- coding: utf-8 -*-
"""
10 人跨被试对比分析脚本

对 10 个被试分别做被试内训练，再混合做跨被试训练，
在每一层记录中间特征，量化展现被试内 vs 跨被试差距，
同时完成各环节计算量分析。

使用小模型（d_attn_in=8, d_attn_out=8），总计算量降 ~8 倍。
"""

import os, sys, time, json, warnings, pickle
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

# -------------------- 路径设置 --------------------
BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from mAtt.mAtt import mAtt_bci
from mAtt.spd import SPDTransform, SPDNorm, SPDRectified, SPDPower, SPDTangentSpace
from utils.GetPhysionetMI import load_preprocessed_dataset, split_train_valid_test, getAllDataloader
from utils.functions import trainNetwork, testNetwork, _silhouette_with_pca

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

# ==================== 1. 计算量分析 ====================

def profile_computation():
    """分析各环节 FLOPs 和 eigh 调用次数（基于张量形状估算）"""
    print('\n' + '=' * 60)
    print('1. 计算量分析')
    print('=' * 60)

    # 配置
    B, M, C_in, C_out, T, T_win = 64, 8, 8, 8, 721, 160
    n_eigh = 0
    total_flops = 0.0
    items = []

    def add(name, flops, eigh_cnt, note=''):
        nonlocal total_flops, n_eigh
        total_flops += flops
        n_eigh += eigh_cnt
        items.append((name, flops, eigh_cnt, note))

    # [1] 通道加权: 逐元素乘，≈0
    add('[1] ChannelPriorWeighting', 0, 0, 'element-wise, trivial')

    # [2] Conv1: Conv2d(1, 22, (22,1)) → (B,22,1,T)
    flops = 2 * 1 * 22 * 22 * 1 * T / 1e6  # 约 0.7M
    add('[2] Conv1 (spatial)', flops, 0, f'22×22×{T}')

    # [3] InstanceNorm1: trivial
    add('[3] InstanceNorm1', 0, 0, 'trivial')

    # [4] Conv2: Conv2d(22, 8, (1,12), padding=6) → (B,8,1,T)
    flops = 2 * 22 * C_in * 1 * 12 * T / 1e6
    add('[4] Conv2 (temporal)', flops, 0, f'22×{C_in}×12×{T}')

    # [5] InstanceNorm2: trivial
    add('[5] InstanceNorm2', 0, 0, 'trivial')

    # [6] E2R: 协方差 + trace归一化
    # 协方差: B*M*C_in²*T_win
    flops_cov = B * M * C_in * C_in * T_win / 1e6
    # eigh: B*M*C_in³
    flops_eigh = B * M * C_in ** 3 / 1e6
    add('[6] E2R (covariance)', flops_cov + flops_eigh, B * M,
        f'cov: {flops_cov:.1f}M, eigh: {flops_eigh:.1f}M')

    # [7] SPDNorm: 2×eigh
    flops = 2 * B * M * C_in ** 3 / 1e6
    add('[7] SPDNorm', flops, 2 * B * M, f'2×eigh')

    # [8] AttentionManifold
    # SPDTransform: B*M*C_in*C_out²
    flops_spd = B * M * C_in * C_out * C_out / 1e6
    # SPDPower Q/K/V: 3×eigh
    flops_power = 3 * B * M * C_out ** 3 / 1e6
    # 注意力距离: 24对 × eigh
    n_pairs = M * (2 * 1 + 1) + 2 * M  # 约 24 对
    flops_attn = n_pairs * B * C_out ** 3 / 1e6
    # LogEuclideanMean: 1×eigh
    flops_mean = B * C_out ** 3 / 1e6
    add('[8] AttentionManifold',
        flops_spd + flops_power + flops_attn + flops_mean,
        3 * B * M + n_pairs * B + B,
        f'SPD: {flops_spd:.1f}M, Power: {flops_power:.1f}M, '
        f'Attn: {flops_attn:.1f}M, Mean: {flops_mean:.1f}M')

    # [9] SPDRectified: 1×eigh
    flops = B * M * C_out ** 3 / 1e6
    add('[9] SPDRectified', flops, B * M)

    # [10] SPDTangentSpace: 1×eigh
    flops = B * M * C_out ** 3 / 1e6
    add('[10] SPDTangentSpace', flops, B * M)

    # [11] Linear: 微不足道
    add('[11] Linear classifier', 0, 0, 'trivial')

    # 打印表格
    print(f'\n{"环节":<30s} {"FLOPs(M)":<12s} {"eigh次数":<10s} {"备注":<30s}')
    print('-' * 82)
    for name, flops, eigh_cnt, note in items:
        print(f'{name:<30s} {flops:<12.2f} {eigh_cnt:<10d} {note:<30s}')
    print('-' * 82)
    print(f'{"总计":<30s} {total_flops:<12.2f} {n_eigh:<10d}')
    print(f'旧模型(20→18)估计: 总FLOPs ~70M, eigh ~4000 次/batch')
    print(f'新模型(8→8)估计:  总FLOPs ~{total_flops:.1f}M, eigh ~{n_eigh} 次/batch')

    return items


# ==================== 2. 特征记录器 ====================

class FeatureRecorder:
    """注册 forward hook 记录各层输出"""

    def __init__(self, model, name_map=None):
        self.hooks = []
        self.features = {}
        self.timing = {}
        self._register_hooks(model, name_map or {})

    def _register_hooks(self, module, name_map, prefix=''):
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            display_name = name_map.get(full_name, full_name)
            if len(list(child.children())) == 0 or isinstance(child, (nn.Flatten, nn.Dropout)):
                self.hooks.append(child.register_forward_hook(
                    lambda m, i, o, n=display_name: self._record(n, o)
                ))
            else:
                self._register_hooks(child, name_map, full_name)

    def _record(self, name, output):
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    self.features[f'{name}[{i}]'] = o.detach().cpu()
        elif isinstance(output, torch.Tensor):
            self.features[name] = output.detach().cpu()

    def clear(self):
        self.features.clear()
        self.timing.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# ==================== 3. 特征分析 ====================

def analyze_features(feats_dict, label=''):
    """分析记录的特征：统计量、距离等"""
    results = {}
    for name, tensor in feats_dict.items():
        t = tensor.float()
        info = {
            'shape': list(t.shape),
            'mean': t.mean().item(),
            'std': t.std().item(),
            'min': t.min().item(),
            'max': t.max().item(),
            'norm': t.norm().item(),
        }
        # 如果是 SPD 矩阵 (..., n, n)，计算 Log-Euclidean 均值范数
        if t.dim() >= 2 and t.shape[-1] == t.shape[-2] and t.shape[-1] >= 2:
            try:
                s = torch.linalg.eigh(t)[0]
                s = s.abs().clamp(min=1e-10)
                info['log_det_mean'] = s.log().sum(-1).mean().item()
                info['eig_min'] = s.min().item()
                info['eig_max'] = s.max().item()
            except Exception:
                pass
        results[name] = info
    return results


def print_feature_comparison(subject_feats, cross_feats):
    """打印被试内 vs 跨被试各层特征对比"""
    print(f'\n{"=" * 60}')
    print('各层特征对比: 被试内(均值) vs 跨被试')
    print(f'{"=" * 60}')

    # 收集所有层名
    all_layers = set()
    for f in subject_feats.values():
        all_layers.update(f.keys())
    all_layers.update(cross_feats.keys())

    print(f'\n{"层名":<28s} {"指标":<10s} {"被试内均值":<15s} {"跨被试":<15s} {"差异比":<10s}')
    print('-' * 78)

    for layer in sorted(all_layers):
        sub_vals = {}
        for sname, sdict in subject_feats.items():
            if layer in sdict:
                for k, v in sdict[layer].items():
                    if k not in sub_vals:
                        sub_vals[k] = []
                    sub_vals[k].append(v)
        cross_vals = cross_feats.get(layer, {})

        for metric in ['mean', 'std', 'norm']:
            if metric in sub_vals and metric in cross_vals:
                sv = float(np.mean(sub_vals[metric]))
                cv = float(cross_vals[metric])
                ratio = abs(cv - sv) / (abs(sv) + 1e-8)
                metric_short = {'mean': '均值', 'std': '标准差', 'norm': '范数'}.get(metric, metric)
                print(f'{layer:<28s} {metric_short:<10s} {sv:<15.4f} {cv:<15.4f} {ratio:<10.4f}')


# ==================== 4. 10 人对比实验 ====================

def run_10_subject_comparison(dataset='bci2a'):
    """对 10 个被试做被试内 vs 跨被试对比"""
    print('\n' + '=' * 60)
    print('2. 10 人对比实验')
    print('=' * 60)

    # 参数
    if dataset == 'bci2a':
        DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')
        SUBJECTS = list(range(1, 10))  # 被试 1~9
        FS = 250                # 采样率 250Hz
        SAMPLE_RATIO = 0.5      # 每人用 50% 数据（每人576试次，足够）
    else:
        DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl')
        SUBJECTS = list(range(1, 11))  # 被试 1~10
        FS = 160                # 采样率 160Hz
        SAMPLE_RATIO = 0.7      # 每人用 70% 数据

    N_CLASSES = 4
    WIN_SEC = 1.0           # 窗长 1s
    STEP_SEC = 0.5          # 步长 0.5s
    WIN_SAMPLES = int(round(WIN_SEC * FS))
    STEP_SAMPLES = int(round(STEP_SEC * FS))
    ITERATIONS = 50         # 每轮训练迭代数（快速验证）
    LR = 5e-4
    WD = 0.5
    BS = 64
    D_ATTN_IN = 8           # 小模型：8 维
    D_ATTN_OUT = 8

    # 加载数据
    print(f'\n加载数据: {DATA_PATH}')
    subject_data, metadata = load_preprocessed_dataset(DATA_PATH)
    print(f'共 {len(subject_data)} 个被试')

    # 对每个被试提取交叉验证划分
    print(f'\n使用被试: {SUBJECTS}')
    subject_splits = {}
    for sid in SUBJECTS:
        X, y = subject_data[sid - 1]
        if SAMPLE_RATIO < 1.0:
            from sklearn.model_selection import train_test_split
            _, X, _, y = train_test_split(
                X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42
            )
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
        subject_splits[sid] = {
            'train': (torch.Tensor(X_tr).unsqueeze(1), torch.Tensor(y_tr).long()),
            'valid': (torch.Tensor(X_va).unsqueeze(1), torch.Tensor(y_va).long()),
            'test': (torch.Tensor(X_te).unsqueeze(1), torch.Tensor(y_te).long()),
        }
        print(f'  S{sid:02d}: train {X_tr.shape[0]}, valid {X_va.shape[0]}, test {X_te.shape[0]}')

    # 混合数据
    X_all = []
    y_all = []
    for sid in SUBJECTS:
        X, y = subject_data[sid - 1]
        if SAMPLE_RATIO < 1.0:
            from sklearn.model_selection import train_test_split
            _, X, _, y = train_test_split(
                X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42
            )
        X_all.append(X)
        y_all.append(y)
    X_all = np.concatenate(X_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    X_cr_tr, y_cr_tr, X_cr_va, y_cr_va, X_cr_te, y_cr_te = split_train_valid_test(X_all, y_all)
    cross_split = {
        'train': (torch.Tensor(X_cr_tr).unsqueeze(1), torch.Tensor(y_cr_tr).long()),
        'valid': (torch.Tensor(X_cr_va).unsqueeze(1), torch.Tensor(y_cr_va).long()),
        'test': (torch.Tensor(X_cr_te).unsqueeze(1), torch.Tensor(y_cr_te).long()),
    }
    print(f'\n跨被试混合: train {X_cr_tr.shape[0]}, valid {X_cr_va.shape[0]}, test {X_cr_te.shape[0]}')

    # 计算 n_windows（从数据中获取实际时间长度）
    T = subject_data[SUBJECTS[0] - 1][0].shape[-1]
    n_windows = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    print(f'T={T}, n_windows: {n_windows}')

    # 层名映射
    NAME_MAP = {
        'channel_weight': '[1] ChannelWeight',
        'conv1': '[2] Conv1',
        'In1': '[3] InstanceNorm1',
        'conv2': '[4] Conv2',
        'In2': '[5] InstanceNorm2',
        'ract1': '[6] E2R',
        'spdnorm': '[7] SPDNorm',
        'att2': '[8] Attention',
        'ract2': '[9] SPDRectified',
        'tangent': '[10] TangentSpace',
        'linear': '[11] Linear',
    }

    # ========== 被试内训练 ==========
    print('\n' + '-' * 50)
    print('开始被试内训练...')
    print('-' * 50)

    subject_results = {}
    subject_features = {}
    subject_histories = {}
    subject_class_accs = {}

    for sid in SUBJECTS:
        print(f'\n--- 被试 S{sid:02d} ---')
        net = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                       d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT,
                       n_windows=n_windows).to(DEVICE)

        recorder = FeatureRecorder(net, NAME_MAP)

        sp = subject_splits[sid]
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*sp['train']), batch_size=BS, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*sp['valid']), batch_size=BS, shuffle=False)
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*sp['test']), batch_size=BS, shuffle=False)

        # 快速训练
        history, best_loss_path, best_acc_path = trainNetwork(
            net, train_loader, valid_loader, test_loader,
            model_path=str(BASE / 'results' / 'profile_10'),
            iterations=ITERATIONS, lr=LR, wd=WD,
            sub=sid, repeat=1, epochs=4, dataset='profile10',
            patience=10, silhouette_interval=0, grad_clip=1.0,
            use_cosine_schedule=True,
        )

        # 记录特征（最佳模型）
        if best_loss_path and os.path.exists(best_loss_path):
            net.load_state_dict(torch.load(best_loss_path, map_location=DEVICE))
        net.eval()
        recorder.clear()
        with torch.no_grad():
            xb = sp['test'][0][:16].to(DEVICE)
            _ = net(xb)
        subject_features[sid] = {k: v.clone() for k, v in recorder.features.items()}

        # 测试准确率
        acc = testNetwork(net, test_loader)
        # 逐类准确率
        cls_acc = _per_class_accuracy(net, test_loader, N_CLASSES)
        subject_results[sid] = {
            'accuracy': acc,
            'per_class_acc': cls_acc,
            'best_loss_iter': history.get('best_loss_iter', -1),
            'best_acc_iter': history.get('best_acc_iter', -1),
        }
        subject_histories[sid] = history
        subject_class_accs[sid] = cls_acc
        print(f'  S{sid:02d} 测试准确率: {acc * 100:.2f}%  {cls_acc}')
        recorder.remove()

    # ========== 跨被试训练 ==========
    print('\n' + '-' * 50)
    print('开始跨被试训练...')
    print('-' * 50)

    net_cross = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                         d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT,
                         n_windows=n_windows).to(DEVICE)

    cross_recorder = FeatureRecorder(net_cross, NAME_MAP)

    cross_train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*cross_split['train']), batch_size=BS, shuffle=True)
    cross_valid_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*cross_split['valid']), batch_size=BS, shuffle=False)
    cross_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(*cross_split['test']), batch_size=BS, shuffle=False)

    history_cross, best_loss_path_cross, _ = trainNetwork(
        net_cross, cross_train_loader, cross_valid_loader, cross_test_loader,
        model_path=str(BASE / 'results' / 'profile_10'),
        iterations=ITERATIONS, lr=LR, wd=WD,
        sub=0, repeat=1, epochs=4, dataset='profile10_cross',
        patience=10, silhouette_interval=0, grad_clip=1.0,
        use_cosine_schedule=True,
    )

    # 记录跨被试特征
    if best_loss_path_cross and os.path.exists(best_loss_path_cross):
        net_cross.load_state_dict(torch.load(best_loss_path_cross, map_location=DEVICE))
    net_cross.eval()
    cross_recorder.clear()
    with torch.no_grad():
        xb = cross_split['test'][0][:16].to(DEVICE)
        _ = net_cross(xb)
    cross_features = {k: v.clone() for k, v in cross_recorder.features.items()}

    cross_acc = testNetwork(net_cross, cross_test_loader)
    cross_cls_acc = _per_class_accuracy(net_cross, cross_test_loader, N_CLASSES)
    print(f'\n跨被试测试准确率: {cross_acc * 100:.2f}%  {cross_cls_acc}')
    cross_recorder.remove()

    return subject_results, subject_features, cross_features, cross_acc, cross_cls_acc, subject_splits, subject_histories, history_cross


# ==================== 辅助函数 ====================

def _per_class_accuracy(net, loader, n_classes):
    """计算逐类准确率"""
    device = next(net.parameters()).device
    net.eval()
    correct = np.zeros(n_classes, dtype=np.float32)
    total = np.zeros(n_classes, dtype=np.float32)
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)
            pred_c = torch.argmax(pred, dim=1)
            for c in range(n_classes):
                mask = yb == c
                total[c] += mask.sum().item()
                correct[c] += (pred_c[mask] == c).sum().item()
    return (correct / np.maximum(total, 1)).tolist()


# ==================== 5. 可视化 ====================

def generate_visualizations(subject_results, subject_features, cross_features,
                            cross_acc, subject_splits):
    """生成对比可视化"""
    print('\n' + '=' * 60)
    print('3. 生成可视化报告')
    print('=' * 60)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = BASE / 'results' / 'profile_10'
    os.makedirs(out_dir, exist_ok=True)

    # 统计各被试准确率
    sub_accs = [subject_results[s]['accuracy'] * 100 for s in sorted(subject_results)]

    # 创建四合一图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('10-Subject Within-Subject vs Cross-Subject Comparison', fontsize=14, fontweight='bold')

    # (1) 准确率对比
    ax = axes[0, 0]
    subjects = sorted(subject_results.keys())
    x = np.arange(len(subjects))
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(subjects)))
    bars = ax.bar(x, sub_accs, color=colors, edgecolor='navy', linewidth=0.5)
    ax.axhline(y=cross_acc * 100, color='red', linestyle='--', linewidth=2,
               label=f'Cross-Subject: {cross_acc*100:.2f}%')
    ax.axhline(y=25, color='gray', linestyle=':', linewidth=1, label='Chance (25%)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'S{s}' for s in subjects], fontsize=8)
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Test Accuracy: Within vs Cross')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # (2) 各层特征范数对比
    ax = axes[0, 1]
    layers = sorted(cross_features.keys())
    sub_norms = {}
    for layer in layers:
        vals = []
        for s, sdict in subject_features.items():
            if layer in sdict:
                vals.append(sdict[layer].norm().item())
        sub_norms[layer] = np.mean(vals) if vals else 0
    cross_norms = {l: cross_features[l].norm().item() for l in layers if l in cross_features}

    # 只显示非 trivial 且非 0 的层
    display_layers = [l for l in layers if sub_norms.get(l, 0) > 0 and cross_norms.get(l, 0) > 0]
    x = np.arange(len(display_layers))
    w = 0.35
    ax.bar(x - w / 2, [sub_norms[l] for l in display_layers], w, label='Within (avg)', alpha=0.8)
    ax.bar(x + w / 2, [cross_norms[l] for l in display_layers], w, label='Cross', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([l.split('.')[-1][:15] for l in display_layers], fontsize=7, rotation=30)
    ax.set_ylabel('Feature Norm')
    ax.set_title('Feature Norm: Within vs Cross')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # (3) 被试内准确率分布
    ax = axes[1, 0]
    ax.hist(sub_accs, bins=8, color='steelblue', edgecolor='white', alpha=0.7)
    ax.axvline(x=cross_acc * 100, color='red', linestyle='--', linewidth=2,
               label=f'Cross: {cross_acc*100:.2f}%')
    ax.axvline(x=np.mean(sub_accs), color='green', linestyle='-', linewidth=2,
               label=f'Within Mean: {np.mean(sub_accs):.2f}%')
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Count')
    ax.set_title('Within-Subject Accuracy Distribution')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

    # (4) 各层统计量差异比
    ax = axes[1, 1]
    diff_ratios = {}
    for layer in display_layers:
        sub_std = np.std([subject_features[s][layer].norm().item()
                          for s in subject_features if layer in subject_features[s]])
        sub_mean = sub_norms[layer]
        cross_mean = cross_norms[layer]
        ratio = abs(cross_mean - sub_mean) / (sub_mean + 1e-8)
        diff_ratios[layer] = ratio
    if diff_ratios:
        layers_sorted = sorted(diff_ratios, key=diff_ratios.get, reverse=True)
        x = np.arange(len(layers_sorted))
        colors = ['red' if diff_ratios[l] > 0.5 else 'orange' if diff_ratios[l] > 0.2 else 'green'
                  for l in layers_sorted]
        ax.barh(x, [diff_ratios[l] for l in layers_sorted], color=colors, alpha=0.7)
        ax.set_yticks(x)
        ax.set_yticklabels([l.split('.')[-1][:15] for l in layers_sorted], fontsize=7)
        ax.set_xlabel('Difference Ratio |cross - within| / |within|')
        ax.set_title('Cross-vs-Within Gap by Layer')
        ax.axvline(x=0.2, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(x=0.5, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(out_dir / 'comparison_summary.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f'[OK] 保存 {out_dir}/comparison_summary.png')

    # ========== 各层 SPD 分布对比 (PCA + Log-Euclidean) ==========
    # 对每层 SPD 特征做 PCA 投影
    n_rows = 2
    n_cols = 4
    spd_layers = [l for l in display_layers if '[6]' in l or '[7]' in l or '[8]' in l or '[9]' in l]
    if spd_layers:
        fig, axes = plt.subplots(n_rows, min(n_cols, len(spd_layers)),
                                 figsize=(16, 8))
        if len(spd_layers) == 1:
            axes = np.array([[axes]])
        axes = axes.flatten()
        fig.suptitle('SPD Feature Distribution: Within vs Cross (PCA)', fontsize=14)

        for idx, layer in enumerate(spd_layers):
            if idx >= len(axes):
                break
            ax = axes[idx]

            # 收集被试内特征
            sub_feats = []
            sub_labels = []
            for s in sorted(subject_features):
                if layer in subject_features[s]:
                    t = subject_features[s][layer].float()
                    # 展平 SPD 矩阵 (上三角)
                    n = t.shape[-1]
                    triu_idx = torch.triu_indices(n, n)
                    feats = t.reshape(-1, n, n)[:, triu_idx[0], triu_idx[1]]
                    sub_feats.append(feats.numpy())
                    sub_labels.extend([s] * feats.shape[0])

            if len(sub_feats) == 0:
                ax.text(0.5, 0.5, 'No data', transform=ax.transAxes, ha='center')
                continue

            sub_feats = np.concatenate(sub_feats, axis=0)

            # 跨被试特征
            cross_t = cross_features[layer].float()
            n = cross_t.shape[-1]
            triu_idx = torch.triu_indices(n, n)
            cross_feats = cross_t.reshape(-1, n, n)[:, triu_idx[0], triu_idx[1]].numpy()

            # PCA 降维
            from sklearn.decomposition import PCA
            all_feats = np.concatenate([sub_feats, cross_feats], axis=0)
            pca = PCA(n_components=2, random_state=0)
            all_2d = pca.fit_transform(all_feats)
            sub_2d = all_2d[:len(sub_feats)]
            cross_2d = all_2d[len(sub_feats):]

            # 绘制
            ax.scatter(sub_2d[:, 0], sub_2d[:, 1], c=sub_labels, cmap='tab10',
                       s=10, alpha=0.6, label='Within')
            ax.scatter(cross_2d[:, 0], cross_2d[:, 1], c='red', marker='x',
                       s=20, alpha=0.8, label='Cross')
            ax.set_title(f'{layer.split(".")[-1][:20]} (var={pca.explained_variance_ratio_.sum():.2f})')
            ax.set_xlabel('PC1')
            ax.set_ylabel('PC2')
            ax.legend(fontsize=7)

        for idx in range(len(spd_layers), len(axes)):
            axes[idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(str(out_dir / 'spd_pca_comparison.png'), dpi=120, bbox_inches='tight')
        plt.close()
        print(f'[OK] 保存 {out_dir}/spd_pca_comparison.png')

    # ========== 文本报告 ==========
    report = []
    report.append('=' * 60)
    report.append('10-Subject Cross-Subject Comparison Report')
    report.append('=' * 60)
    report.append('')
    report.append(f'Within-Subject Accuracies:')
    for s in sorted(subject_results):
        report.append(f'  S{s:02d}: {subject_results[s]["accuracy"]*100:.2f}%')
    report.append(f'')
    report.append(f'Within-Subject Mean: {np.mean(sub_accs):.2f}%')
    report.append(f'Within-Subject Std:  {np.std(sub_accs):.2f}%')
    report.append(f'Cross-Subject:       {cross_acc*100:.2f}%')
    report.append(f'Within-Cross Gap:    {np.mean(sub_accs) - cross_acc*100:.2f}%')
    report.append(f'Chance Level:        25.00%')
    report.append('')

    # 各层差距
    report.append('Layer-wise Cross-vs-Within Gap:')
    report.append(f'  {"Layer":<25s} {"Norm Ratio":<15s} {"Gap Level":<10s}')
    report.append('  ' + '-' * 50)
    layers = sorted(cross_features.keys())
    for layer in layers:
        sub_norm = sub_norms.get(layer, 0)
        cross_norm = cross_norms.get(layer, 0)
        if sub_norm > 0:
            ratio = abs(cross_norm - sub_norm) / sub_norm
            level = 'HIGH' if ratio > 0.5 else 'MED' if ratio > 0.2 else 'LOW'
            report.append(f'  {layer.split(".")[-1][:25]:<25s} {ratio:<15.4f} {level:<10s}')
    report.append('')

    report_path = out_dir / 'comparison_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
    print(f'[OK] 保存 {report_path}')
    print('\n' + '\n'.join(report))


# ==================== 主流程 ====================

if __name__ == '__main__':
    # 1. 计算量分析
    profile_computation()

    # 2. BCI2A 对比实验
    print('\n' + '=' * 60)
    print('使用 BCI2A 数据集 (9 subjects, 576 trials/subject, 250Hz)')
    print('=' * 60)
    subject_results, subject_features, cross_features, cross_acc, cross_cls_acc, subject_splits, subject_histories, history_cross = \
        run_10_subject_comparison(dataset='bci2a')

    # 3. 特征对比打印
    print_feature_comparison(
        {s: analyze_features(f) for s, f in subject_features.items()},
        analyze_features(cross_features)
    )

    # 4. 可视化
    generate_visualizations(subject_results, subject_features, cross_features,
                            cross_acc, subject_splits)

    # 5. 保存详细数据到 pickle
    out_dir = BASE / 'results' / 'profile_10'
    os.makedirs(out_dir, exist_ok=True)
    detailed_data = {
        'subject_results': subject_results,
        'subject_features': {s: {k: v.numpy() for k, v in f.items()} for s, f in subject_features.items()},
        'cross_features': {k: v.numpy() for k, v in cross_features.items()},
        'cross_acc': cross_acc,
        'cross_cls_acc': cross_cls_acc,
        'subject_histories': subject_histories,
        'history_cross': history_cross,
        'config': {
            'dataset': 'bci2a',
            'subjects': list(range(1, 10)),
            'iterations': 50,
            'd_attn_in': 8,
            'd_attn_out': 8,
            'win_sec': 1.0,
            'step_sec': 0.5,
        }
    }
    pkl_path = out_dir / 'detailed_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(detailed_data, f)
    print(f'[OK] 详细数据已保存到 {pkl_path}')

    print('\n' + '=' * 60)
    print('全部完成!')
    print(f'结果保存在: {BASE / "results" / "profile_10"}')
    print('=' * 60)