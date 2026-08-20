# -*- coding: utf-8 -*-
"""
MAtt 跨被试泛化实验入口（滑动时间窗 + 稀疏流形注意力版）

- 90 个 seen subjects：用于训练，每个被试留一部分做被试内测试
- 13 个 unseen subjects：完全跨被试测试
"""

import sys
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from mAtt.mAtt import mAtt_bci, E2R
from utils.functions import (
    trainNetwork, testNetwork, save_visualization_data, resolve_path,
)
from utils.GetPhysionetMI import getCrossSubjectDataloader


# ============================================================
# 实验参数
# ============================================================
CONFIG = {
    'dataset': 'physionetmi',
    'seen_subjects': list(range(1, 91)),      # 90 人用于训练+被试内测试
    'unseen_subjects': list(range(91, 104)),  # 13 人用于完全跨被试测试
    'seen_test_ratio': 0.2,                   # seen subject 留 20% 做被试内测试
    'valid_ratio': 0.2,                       # 验证集比例
    'sample_ratio': 0.5,                      # 每个人用 50% 的数据（对抗过拟合）
    'repeat': 1,
    'n_classes': 4,
    'lr': 5e-4,
    'wd': 0.5,                                # 加大正则化（← 0.01）
    'iterations': 200,   # 数据量大，设大一点；早停会自动提前结束
    'patience': 15,      # 验证指标连续 15 轮不提升就早停
    'grad_clip': 1.0,    # 梯度裁剪（注意力 α=-d² 不压缩，靠此保证稳定）
    'bs': 64,            # 数据量大，batch size 可以放大（看显存，可调到 32/64/128）

    # --- 滑动时间窗（E2R）---
    'win_sec': 1.0,      # 窗长 1s
    'step_sec': 0.5,     # 步长 0.5s（50% 重叠）

    # --- 稀疏流形注意力 ---
    'local_radius': 1,   # 局部注意力半径（窗口数）
    'anchor_stride': None,  # None=仅起始窗口为锚点
    'n_heads': 1,        # >1 时 K/V 单组共享（GQA）

    'model_path': './results/cross_subject_checkpoint/',
    'data_path': '../saved_data/preprocessed_data_moabb_raw_ica.pkl',
}
# ============================================================

SFREQ = 160  # MOABB PhysionetMI 采样率


def main():
    args = CONFIG.copy()
    dataset = args['dataset']
    args['data_path'] = resolve_path(args['data_path'])
    args['model_path'] = resolve_path(args['model_path'])

    win_samples = int(round(args['win_sec'] * SFREQ))
    step_samples = int(round(args['step_sec'] * SFREQ))

    print(f'==== 跨被试实验 | 数据集: {dataset} ====')
    print(f'seen subjects: {len(args["seen_subjects"])} 人')
    print(f'unseen subjects: {len(args["unseen_subjects"])} 人')
    print(f'滑动窗: {args["win_sec"]}s ({win_samples} 点) / 步长 {args["step_sec"]}s ({step_samples} 点)')

    trainloader, validloader, seen_test_loader, unseen_test_loader = getCrossSubjectDataloader(
        seen_subjects=args['seen_subjects'],
        unseen_subjects=args['unseen_subjects'],
        data_path=args['data_path'],
        bs=args['bs'],
        dataset=dataset,
        seen_test_ratio=args['seen_test_ratio'],
        valid_ratio=args['valid_ratio'],
        sample_ratio=args['sample_ratio'],
    )

    # 从数据推断窗口数（LazyLinear 自适应；此值用于打印与 checkpoint 命名）
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
        n_windows=n_windows,
    )
    if torch.cuda.is_available():
        net = net.cuda()
        print(f'使用 GPU: {torch.cuda.get_device_name(0)}')
    else:
        print('未检测到 GPU，使用 CPU')

    # 训练时用 seen_test_loader 作为监控（被试内测试）
    print('\n--- 开始训练 ---')
    train_kwargs = {k: v for k, v in args.items()
                    if k not in ['bs', 'data_path', 'seen_subjects', 'unseen_subjects',
                                 'seen_test_ratio', 'valid_ratio', 'sample_ratio',
                                 'win_sec', 'step_sec', 'local_radius',
                                 'anchor_stride', 'n_heads', 'n_classes']}
    history, best_loss_path, best_acc_path = trainNetwork(
        net, trainloader, validloader, seen_test_loader, **train_kwargs
    )

    # ---------- state_dict 加载（A17: 重建模型） ----------
    def load_model(path):
        m = mAtt_bci(
            win_samples, step_samples,
            n_classes=args['n_classes'],
            local_radius=args['local_radius'],
            anchor_stride=args['anchor_stride'],
            n_heads=args['n_heads'],
            n_windows=n_windows,
        )
        m.load_state_dict(torch.load(path, map_location='cpu'))
        return m.cuda() if torch.cuda.is_available() else m

    # ---------- best loss 模型 ----------
    net_loss = load_model(best_loss_path)
    acc_loss_seen = testNetwork(net_loss, seen_test_loader)
    acc_loss_unseen = testNetwork(net_loss, unseen_test_loader)
    print(f'\n[验证 Loss 最佳模型] seen-subject 测试准确率: {acc_loss_seen*100:.2f}%')
    print(f'[验证 Loss 最佳模型] unseen-subject 跨被试准确率: {acc_loss_unseen*100:.2f}%')

    # ---------- best acc 模型 ----------
    net_acc = load_model(best_acc_path)
    acc_acc_seen = testNetwork(net_acc, seen_test_loader)
    acc_acc_unseen = testNetwork(net_acc, unseen_test_loader)
    print(f'[验证 Acc 最佳模型] seen-subject 测试准确率: {acc_acc_seen*100:.2f}%')
    print(f'[验证 Acc 最佳模型] unseen-subject 跨被试准确率: {acc_acc_unseen*100:.2f}%')

    # 保存可视化数据（best_loss / best_acc × seen / unseen）
    save_visualization_data(net_loss, seen_test_loader, history, acc_loss_seen, args,
                            criterion='best_loss_seen')
    save_visualization_data(net_loss, unseen_test_loader, history, acc_loss_unseen, args,
                            criterion='best_loss_unseen')
    save_visualization_data(net_acc, seen_test_loader, history, acc_acc_seen, args,
                            criterion='best_acc_seen')
    save_visualization_data(net_acc, unseen_test_loader, history, acc_acc_unseen, args,
                            criterion='best_acc_unseen')

    # 保存结果 JSON
    results_dir = Path('./results')
    results_dir.mkdir(parents=True, exist_ok=True)
    base = f'{dataset}_cross_subject'

    for criterion, (acc_seen, acc_unseen) in {
        'val_loss': (acc_loss_seen, acc_loss_unseen),
        'val_acc': (acc_acc_seen, acc_acc_unseen),
    }.items():
        tag = 'best_loss' if criterion == 'val_loss' else 'best_acc'
        with open(results_dir / f'{base}_{tag}_results.json', 'w') as f:
            json.dump({
                'dataset': dataset,
                'seen_subjects': args['seen_subjects'],
                'unseen_subjects': args['unseen_subjects'],
                'win_sec': args['win_sec'],
                'step_sec': args['step_sec'],
                'n_windows': n_windows,
                'criterion': criterion,
                'seen_test_acc': float(acc_seen),
                'unseen_test_acc': float(acc_unseen),
            }, f, indent=2)

    print(f'\n结果已保存: {results_dir}/{base}_best_{{loss,acc}}_results.json')


if __name__ == '__main__':
    main()
