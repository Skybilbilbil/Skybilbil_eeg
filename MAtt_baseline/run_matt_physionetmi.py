# -*- coding: utf-8 -*-
"""
MAtt 在 PhysionetMI 预处理数据上的运行脚本（滑动时间窗 + 稀疏流形注意力版）
"""

import torch
import torch.nn as nn
import json
from pathlib import Path

from mAtt.mAtt import mAtt_bci, E2R
from utils.functions import (
    trainNetwork, testNetwork, save_visualization_data, resolve_path,
)
from utils.GetPhysionetMI import getAllDataloader


# ============================================================
# 实验参数在这里改，不需要命令行
# ============================================================
CONFIG = {
    'dataset': 'physionetmi',  # 可选 'physionetmi' 或 'bci2a'
    'repeat': 1,               # 重复实验编号
    'sub': 1,                  # 被试编号（PhysionetMI: 1~103；BCIC-IV-2a: 1~9）
    'n_classes': 4,            # 类别数（两数据集均为 4 分类）
    'lr': 5e-4,                # 学习率
    'wd': 0.02,                # weight decay（已验证 0.02 避免跨被试塌缩）
    'iterations': 120,         # 训练迭代次数（平衡计算量与效果）
    'grad_clip': 1.0,          # 梯度裁剪 max_norm（注意力 α=-d² 不压缩，靠此保证稳定）
    'bs': 64,                  # batch size
    'patience': 0,             # 早停耐心值：0 表示禁用早停

    # --- 滑动时间窗（E2R）---
    'win_sec': 1.5,            # 窗长 1.5s
    'step_sec': 0.75,          # 步长 0.75s（50% 重叠）

    # --- 稀疏流形注意力 ---
    'local_radius': 6,         # 全连接注意力（窗口数 4~5 时覆盖全部）
    'anchor_stride': None,     # None=无锚点（纯局部注意力，用户要求删除锚点）；int k=每 k 窗一个锚点（含末窗口）
    'n_heads': 1,              # 注意力头数；>1 时 K/V 单组共享（GQA）
    'dropout': 0.3,            # dropout 概率（0.3 平衡正则与效果）

    'model_path': './results/checkpoint/',
    'data_path': None,         # None 时根据 dataset 自动选择默认路径
}

# 不同数据集的默认 pkl 路径与采样率
DATASET_DEFAULT_PATHS = {
    'physionetmi': '../saved_data/preprocessed_data_moabb_raw_ica.pkl',
    'bci2a': '../saved_data/preprocessed_data_bci2a_raw_ica.pkl',
}
DATASET_DEFAULT_SFREQ = {
    'physionetmi': 160,   # MOABB PhysionetMI 原始采样率
    'bci2a': 250,         # BCIC-IV-2a 采样率
}
# ============================================================


def main():
    args = CONFIG.copy()

    # 选择数据集并填充默认路径
    dataset = args.get('dataset', 'physionetmi')
    if dataset not in DATASET_DEFAULT_PATHS:
        raise ValueError(
            f"不支持的数据集 '{dataset}'，可选: {list(DATASET_DEFAULT_PATHS.keys())}"
        )
    if args['data_path'] is None:
        args['data_path'] = DATASET_DEFAULT_PATHS[dataset]

    args['data_path'] = resolve_path(args['data_path'])
    args['model_path'] = resolve_path(args['model_path'])

    # 滑动窗采样点数
    sfreq = DATASET_DEFAULT_SFREQ[dataset]
    win_samples = int(round(args['win_sec'] * sfreq))
    step_samples = int(round(args['step_sec'] * sfreq))

    print(f'==== 数据集: {dataset} | 被试 {args["sub"]:03d} ====')
    print(f'数据路径: {args["data_path"]}')
    print(f'滑动窗: {args["win_sec"]}s ({win_samples} 点) / 步长 {args["step_sec"]}s ({step_samples} 点)')

    trainloader, validloader, testloader = getAllDataloader(
        subject=args['sub'],
        data_path=args['data_path'],
        bs=args['bs'],
        dataset=dataset,
    )

    # 从数据推断窗口数（模型用 LazyLinear 自适应；此值用于打印与 checkpoint 命名）
    T = next(iter(trainloader))[0].shape[-1]
    n_windows = E2R.n_windows(T, win_samples, step_samples)
    args['epochs'] = n_windows  # 兼容 trainNetwork / 可视化命名，含义 = 窗口数
    print(f'时间长度 T={T} → {n_windows} 个窗口；注意力: local_radius={args["local_radius"]}, '
          f'anchor_stride={args["anchor_stride"]}, n_heads={args["n_heads"]}')

    net = mAtt_bci(
        win_samples, step_samples,
        n_classes=args['n_classes'],
        local_radius=args['local_radius'],
        anchor_stride=args['anchor_stride'],
        n_heads=args['n_heads'],
        n_windows=None,          # LazyLinear 自适应实际窗口数（卷积改变时序长度）
    )
    net.dropout = nn.Dropout(args.get('dropout', 0.3))
    if torch.cuda.is_available():
        net = net.cuda()
        print(f'使用 GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('未检测到 GPU，使用 CPU')

    # 训练（新参数不属于 trainNetwork，排除之）
    train_kwargs = {k: v for k, v in args.items()
                    if k not in ['bs', 'data_path', 'win_sec', 'step_sec',
                                 'local_radius', 'anchor_stride', 'n_heads', 'n_classes',
                                 'dropout']}
    history, best_loss_path, best_acc_path = trainNetwork(
        net, trainloader, validloader, testloader,
        silhouette_interval=5,
        **train_kwargs
    )

    # ---------- 用验证 Loss 最低模型测试（A17: state_dict 加载，重建模型） ----------
    def load_model(path):
        m = mAtt_bci(
            win_samples, step_samples,
            n_classes=args['n_classes'],
            local_radius=args['local_radius'],
            anchor_stride=args['anchor_stride'],
            n_heads=args['n_heads'],
            n_windows=None,
        )
        m.dropout = nn.Dropout(args.get('dropout', 0.3))
        m.load_state_dict(torch.load(path, map_location='cpu'))
        return m.cuda() if torch.cuda.is_available() else m

    net_loss = load_model(best_loss_path)
    acc_loss = testNetwork(net_loss, testloader)
    print(f'\n[验证 Loss 最佳模型] 测试准确率: {acc_loss*100:.2f}%')

    # ---------- 用验证准确率最高模型测试 ----------
    net_acc = load_model(best_acc_path)
    acc_acc = testNetwork(net_acc, testloader)
    print(f'[验证 Acc 最佳模型] 测试准确率: {acc_acc*100:.2f}%')

    # 保存两个模型的可视化数据
    save_visualization_data(net_loss, testloader, history, acc_loss, args, criterion='best_loss')
    save_visualization_data(net_acc, testloader, history, acc_acc, args, criterion='best_acc')

    # 保存结果（按数据集+被试命名，同被试运行覆盖）
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / f'{dataset}_sub{args["sub"]:03d}_best_loss_results.json', 'w') as f:
        json.dump({
            'dataset': dataset,
            'subject': args['sub'],
            'win_sec': args['win_sec'],
            'step_sec': args['step_sec'],
            'n_windows': n_windows,
            'criterion': 'val_loss',
            'test_acc': float(acc_loss)
        }, f, indent=2)

    with open(results_dir / f'{dataset}_sub{args["sub"]:03d}_best_acc_results.json', 'w') as f:
        json.dump({
            'dataset': dataset,
            'subject': args['sub'],
            'win_sec': args['win_sec'],
            'step_sec': args['step_sec'],
            'n_windows': n_windows,
            'criterion': 'val_acc',
            'test_acc': float(acc_acc)
        }, f, indent=2)
    print(f'结果已保存: {results_dir}/{dataset}_sub{args["sub"]:03d}_best_{{loss,acc}}_results.json')


if __name__ == '__main__':
    main()
