# -*- coding: utf-8 -*-
"""
分析已保存的 profile_10 实验结果，提取详细数据到 pickle
加载已有模型 checkpoint，无需重新训练
"""

import os, sys, pickle, json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from mAtt.mAtt import mAtt_bci
from utils.GetPhysionetMI import load_preprocessed_dataset, split_train_valid_test
from utils.functions import testNetwork

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

OUT_DIR = BASE / 'results' / 'profile_10'
os.makedirs(OUT_DIR, exist_ok=True)

# ==================== 配置 ====================
DATASET = 'bci2a'
DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')
SUBJECTS = list(range(1, 10))
FS = 250
SAMPLE_RATIO = 0.5
N_CLASSES = 4
WIN_SEC, STEP_SEC = 1.0, 0.5
WIN_SAMPLES = int(round(WIN_SEC * FS))
STEP_SAMPLES = int(round(STEP_SEC * FS))
D_ATTN_IN, D_ATTN_OUT = 8, 8
BS = 64

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

# ==================== 特征记录器 ====================
class FeatureRecorder:
    def __init__(self, model, name_map=None):
        self.hooks = []
        self.features = {}
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

    def remove(self):
        for h in self.hooks:
            h.remove()


# ==================== 逐类准确率 ====================
def per_class_accuracy(net, loader, n_classes):
    device = next(net.parameters()).device
    net.eval()
    correct = np.zeros(n_classes, dtype=np.float32)
    total = np.zeros(n_classes, dtype=np.float32)
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = net(xb)
            pred_c = torch.argmax(pred, dim=1)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(pred_c.cpu().numpy())
            for c in range(n_classes):
                mask = yb == c
                total[c] += mask.sum().item()
                correct[c] += (pred_c[mask] == c).sum().item()
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = np.concatenate(y_pred_all)
    # 混淆矩阵
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true_all, y_pred_all, labels=range(n_classes))
    return (correct / np.maximum(total, 1)).tolist(), cm


# ==================== 逐层特征详细分析 ====================
def analyze_layer_feature(tensor, name=''):
    """分析单层特征，返回详细统计量"""
    t = tensor.float()
    info = {
        'name': name,
        'shape': list(t.shape),
        'mean': t.mean().item(),
        'std': t.std().item(),
        'min': t.min().item(),
        'max': t.max().item(),
        'norm': t.norm().item(),
        'norm_per_sample': t.reshape(t.shape[0], -1).norm(dim=1).mean().item(),
        'mean_per_sample': t.reshape(t.shape[0], -1).mean(dim=1).mean().item(),
        'std_per_sample': t.reshape(t.shape[0], -1).std(dim=1).mean().item(),
    }
    # 如果是 SPD 矩阵，计算 Log-Euclidean 统计量
    if t.dim() >= 2 and t.shape[-1] == t.shape[-2] and t.shape[-1] >= 2:
        try:
            s = torch.linalg.eigh(t)[0]
            s = s.abs().clamp(min=1e-10)
            info['log_det_mean'] = s.log().sum(-1).mean().item()
            info['eig_min'] = s.min().item()
            info['eig_max'] = s.max().item()
            info['eig_mean'] = s.mean().item()
            info['eig_std'] = s.std().item()
            info['condition_number'] = (s.max(-1).values / s.min(-1).values.clamp(min=1e-10)).mean().item()
            # 对数行列式（协方差估计的熵）
            info['log_det'] = s.log().sum(-1)
            info['log_det_mean'] = info['log_det'].mean().item()
            info['log_det_std'] = info['log_det'].std().item()
            del info['log_det']  # 不保存序列
        except Exception:
            pass
    return info


def analyze_features_batch(feats_dict):
    """分析所有层特征"""
    return {name: analyze_layer_feature(tensor, name) for name, tensor in feats_dict.items()}


# ==================== 主流程 ====================
def main():
    print('\n' + '=' * 60)
    print('加载 BCI2A 数据')
    print('=' * 60)
    subject_data, metadata = load_preprocessed_dataset(DATA_PATH)
    print(f'共 {len(subject_data)} 个被试')

    # 数据划分
    print(f'\n准备数据划分...')
    subject_splits = {}
    for sid in SUBJECTS:
        X, y = subject_data[sid - 1]
        if SAMPLE_RATIO < 1.0:
            from sklearn.model_selection import train_test_split
            _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
        subject_splits[sid] = {
            'train': (torch.Tensor(X_tr).unsqueeze(1), torch.Tensor(y_tr).long()),
            'valid': (torch.Tensor(X_va).unsqueeze(1), torch.Tensor(y_va).long()),
            'test': (torch.Tensor(X_te).unsqueeze(1), torch.Tensor(y_te).long()),
        }

    # 混合数据
    X_all, y_all = [], []
    for sid in SUBJECTS:
        X, y = subject_data[sid - 1]
        if SAMPLE_RATIO < 1.0:
            from sklearn.model_selection import train_test_split
            _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
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

    # n_windows
    T = subject_data[SUBJECTS[0] - 1][0].shape[-1]
    n_windows = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    print(f'T={T}, n_windows: {n_windows}')

    # ========== 加载被试内模型 ==========
    print('\n' + '=' * 60)
    print('加载被试内模型 checkpoint，提取特征')
    print('=' * 60)

    subject_results = {}
    subject_features = {}
    subject_detailed = {}

    for sid in SUBJECTS:
        print(f'\n--- 被试 S{sid:02d} ---')
        # 查找 checkpoint
        ckpt_pattern = f'profile10_repeat1_sub{sid}_epochs4_lr0.0005_wd0.5_best_loss.pt'
        ckpt_path = OUT_DIR / ckpt_pattern
        if not ckpt_path.exists():
            print(f'  [WARN] 找不到 checkpoint: {ckpt_path}, 跳过')
            continue

        net = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                       d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT,
                       n_windows=n_windows).to(DEVICE)
        net.load_state_dict(torch.load(ckpt_path, map_location=DEVICE, weights_only=True))
        net.eval()

        # 提取特征
        recorder = FeatureRecorder(net, NAME_MAP)
        sp = subject_splits[sid]
        test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*sp['test']), batch_size=BS, shuffle=False)

        recorder.clear()
        with torch.no_grad():
            xb = sp['test'][0][:16].to(DEVICE)
            _ = net(xb)
        subject_features[sid] = {k: v.clone() for k, v in recorder.features.items()}

        # 测试准确率 + 逐类准确率
        acc = testNetwork(net, test_loader)
        cls_acc, cm = per_class_accuracy(net, test_loader, N_CLASSES)

        subject_results[sid] = {
            'accuracy': acc,
            'per_class_acc': cls_acc,
            'confusion_matrix': cm.tolist(),
        }
        print(f'  准确率: {acc * 100:.2f}%  逐类: {cls_acc}')
        print(f'  混淆矩阵:\n{cm}')
        recorder.remove()

    # ========== 加载跨被试模型 ==========
    print('\n' + '=' * 60)
    print('加载跨被试模型 checkpoint')
    print('=' * 60)

    cross_ckpt = OUT_DIR / 'profile10_cross_repeat1_sub0_epochs4_lr0.0005_wd0.5_best_loss.pt'
    cross_features = None
    cross_acc = 0
    cross_cls_acc = [0, 0, 0, 0]
    cross_cm = None

    if cross_ckpt.exists():
        net_cross = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                             d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT,
                             n_windows=n_windows).to(DEVICE)
        net_cross.load_state_dict(torch.load(cross_ckpt, map_location=DEVICE, weights_only=True))
        net_cross.eval()

        cross_recorder = FeatureRecorder(net_cross, NAME_MAP)
        cross_test_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(*cross_split['test']), batch_size=BS, shuffle=False)

        cross_recorder.clear()
        with torch.no_grad():
            xb = cross_split['test'][0][:16].to(DEVICE)
            _ = net_cross(xb)
        cross_features = {k: v.clone() for k, v in cross_recorder.features.items()}

        cross_acc = testNetwork(net_cross, cross_test_loader)
        cross_cls_acc, cross_cm = per_class_accuracy(net_cross, cross_test_loader, N_CLASSES)
        print(f'跨被试准确率: {cross_acc * 100:.2f}%  逐类: {cross_cls_acc}')
        print(f'混淆矩阵:\n{cross_cm}')
        cross_recorder.remove()
    else:
        print(f'[WARN] 找不到跨被试 checkpoint: {cross_ckpt}')

    # ========== 计算详细特征分析 ==========
    print('\n' + '=' * 60)
    print('计算逐层特征分析')
    print('=' * 60)

    subject_feat_analysis = {}
    for sid, feats in subject_features.items():
        subject_feat_analysis[sid] = analyze_features_batch(feats)

    cross_feat_analysis = analyze_features_batch(cross_features) if cross_features else {}

    # 计算跨被试差异比
    layer_gaps = {}
    all_layers = set()
    for f in subject_feat_analysis.values():
        all_layers.update(f.keys())
    all_layers.update(cross_feat_analysis.keys())

    for layer in sorted(all_layers):
        sub_norms = []
        for sid, sdict in subject_feat_analysis.items():
            if layer in sdict:
                sub_norms.append(sdict[layer]['norm'])
        if sub_norms and layer in cross_feat_analysis:
            sub_mean_norm = np.mean(sub_norms)
            cross_norm = cross_feat_analysis[layer]['norm']
            ratio = abs(cross_norm - sub_mean_norm) / (sub_mean_norm + 1e-8)
            layer_gaps[layer] = {
                'within_mean_norm': float(sub_mean_norm),
                'within_std_norm': float(np.std(sub_norms)),
                'cross_norm': float(cross_norm),
                'gap_ratio': float(ratio),
                'gap_level': 'HIGH' if ratio > 0.5 else 'MED' if ratio > 0.2 else 'LOW',
            }

    print('\n逐层差距:')
    print(f'  {"Layer":<30s} {"Within Norm":<12s} {"Cross Norm":<12s} {"Gap Ratio":<10s} {"Level":<6s}')
    print('  ' + '-' * 70)
    for layer, info in sorted(layer_gaps.items(), key=lambda x: -x[1]['gap_ratio']):
        print(f'  {layer[:30]:<30s} {info["within_mean_norm"]:<12.4f} {info["cross_norm"]:<12.4f} {info["gap_ratio"]:<10.4f} {info["gap_level"]:<6s}')

    # ========== 保存 pickle ==========
    print('\n' + '=' * 60)
    print('保存详细数据')
    print('=' * 60)

    detailed_data = {
        'subject_results': subject_results,
        'subject_features': {s: {k: v.numpy() for k, v in f.items()} for s, f in subject_features.items()},
        'cross_features': {k: v.numpy() for k, v in cross_features.items()} if cross_features else {},
        'subject_feat_analysis': subject_feat_analysis,
        'cross_feat_analysis': cross_feat_analysis,
        'layer_gaps': layer_gaps,
        'cross_acc': cross_acc,
        'cross_cls_acc': cross_cls_acc,
        'cross_confusion_matrix': cross_cm.tolist() if cross_cm is not None else None,
        'config': {
            'dataset': DATASET,
            'subjects': SUBJECTS,
            'd_attn_in': D_ATTN_IN,
            'd_attn_out': D_ATTN_OUT,
            'win_sec': WIN_SEC,
            'step_sec': STEP_SEC,
            'fs': FS,
            'n_windows': n_windows,
        }
    }

    pkl_path = OUT_DIR / 'detailed_results.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(detailed_data, f)
    print(f'[OK] 保存到 {pkl_path}')
    print(f'     文件大小: {os.path.getsize(pkl_path) / 1024:.1f} KB')

    # ========== 保存 JSON 摘要 ==========
    summary = {
        'within_subject': {
            f'S{s:02d}': {
                'accuracy': float(r['accuracy']),
                'per_class_acc': [float(a) for a in r['per_class_acc']],
            }
            for s, r in subject_results.items()
        },
        'cross_subject': {
            'accuracy': float(cross_acc),
            'per_class_acc': [float(a) for a in cross_cls_acc],
        },
        'layer_gaps': layer_gaps,
        'summary': {
            'within_mean': float(np.mean([r['accuracy'] for r in subject_results.values()])),
            'within_std': float(np.std([r['accuracy'] for r in subject_results.values()])),
            'cross_acc': float(cross_acc),
            'gap': float(np.mean([r['accuracy'] for r in subject_results.values()]) - cross_acc),
        }
    }
    json_path = OUT_DIR / 'detailed_summary.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f'[OK] JSON 摘要保存到 {json_path}')

    print('\n' + '=' * 60)
    print('分析完成!')
    print(f'详细数据: {pkl_path}')
    print(f'JSON摘要: {json_path}')
    print('=' * 60)


if __name__ == '__main__':
    main()