# -*- coding: utf-8 -*-
"""
MAtt 自定义数据加载器：读取 Riemannian_Trajectories 预处理后的 PhysionetMI 数据
"""

import pickle
import numpy as np
import torch
import torch.utils.data as Data
from pathlib import Path
from sklearn.model_selection import train_test_split


def load_preprocessed_dataset(pkl_path):
    """
    通用加载函数，支持 PhysionetMI 与 BCIC-IV-2a 预处理后的 pkl。
    返回：每个被试的 (X, y) 列表
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    left_list = data['aligned_left_epochs_list']
    right_list = data['aligned_right_epochs_list']
    feet_list = data['aligned_feet_epochs_list']
    hands_list = data['aligned_hands_epochs_list']
    metadata = data['subject_metadata']

    n_subjects = len(metadata)
    subject_data = []

    for i in range(n_subjects):
        # 按顺序拼接四类：left=0, right=1, feet=2, hands/tongue=3
        X_list = [left_list[i], right_list[i], feet_list[i], hands_list[i]]
        y_list = []
        X_subj = []
        for cls_idx, arr in enumerate(X_list):
            if arr is None or len(arr) == 0:
                continue
            X_subj.append(arr)
            y_list.append(np.full(len(arr), cls_idx, dtype=int))

        X_subj = np.concatenate(X_subj, axis=0).astype(np.float32)
        y_subj = np.concatenate(y_list, axis=0)

        subject_data.append((X_subj, y_subj))
        print(f"  S{i+1:03d}: {X_subj.shape}, 类别分布 {np.bincount(y_subj, minlength=4)}")

    return subject_data, metadata


def load_preprocessed_physionetmi(pkl_path):
    """（兼容旧接口）加载 PhysionetMI 预处理数据。"""
    return load_preprocessed_dataset(pkl_path)


def split_train_valid_test(X, y, train_ratio=0.6, valid_ratio=0.2, random_state=42):
    """
    分层划分 train / valid / test
    """
    # 先划分 test
    X_trval, X_test, y_trval, y_test = train_test_split(
        X, y, test_size=(1 - train_ratio - valid_ratio),
        stratify=y, random_state=random_state
    )
    # 再划分 valid
    valid_size = valid_ratio / (train_ratio + valid_ratio)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_trval, y_trval, test_size=valid_size,
        stratify=y_trval, random_state=random_state
    )
    return X_train, y_train, X_valid, y_valid, X_test, y_test


def _load_subject(sub, subject_data, sample_ratio, random_state):
    """取出单个被试 (X, y)，校验通道数，并按需做分层采样。"""
    X, y = subject_data[sub - 1]
    if X.shape[1] != 22:
        raise ValueError(f'被试 {sub:03d} 通道数为 {X.shape[1]}，期望 22')

    # 如果指定 sample_ratio < 1.0，先做分层采样
    if sample_ratio < 1.0:
        _, X, _, y = train_test_split(
            X, y, test_size=sample_ratio, stratify=y, random_state=random_state
        )

    return X, y


def getCrossSubjectDataloader(seen_subjects, unseen_subjects, data_path,
                              bs=16, epochs=4, dataset='physionetmi',
                              seen_test_ratio=0.2, valid_ratio=0.2,
                              sample_ratio=1.0, random_state=42):
    """
    跨被试数据加载器。

    seen_subjects:   被试内训练+测试的被试编号列表（如 1~90）
    unseen_subjects: 完全跨被试测试的被试编号列表（如 91~103）
    sample_ratio:    每个人只用多少比例的数据（默认 1.0，例如 0.1 表示只用 10%）

    每个 seen subject 的试次按 train : valid : seen_test 划分：
      train = 1 - valid_ratio - seen_test_ratio
      valid = valid_ratio
      seen_test = seen_test_ratio

    每个 unseen subject 的全部试次都作为跨被试测试集。
    """
    pkl_path = Path(data_path)
    subject_data, metadata = load_preprocessed_dataset(pkl_path)

    n_total = len(subject_data)
    for sub in seen_subjects + unseen_subjects:
        if sub < 1 or sub > n_total:
            raise ValueError(f'被试编号 {sub} 超出范围 1~{n_total}')

    train_ratio = 1.0 - valid_ratio - seen_test_ratio
    if train_ratio <= 0:
        raise ValueError('valid_ratio + seen_test_ratio 必须小于 1')

    # 收集 seen subjects：先全部合并，再统一划分 train/valid/seen_test
    X_seen_all, y_seen_all = [], []
    for sub in seen_subjects:
        X, y = _load_subject(sub, subject_data, sample_ratio, random_state)
        X_seen_all.append(X); y_seen_all.append(y)

    X_seen = np.concatenate(X_seen_all, axis=0)
    y_seen = np.concatenate(y_seen_all, axis=0)
    X_train, y_train, X_valid, y_valid, X_seen_test, y_seen_test = split_train_valid_test(
        X_seen, y_seen, train_ratio=train_ratio, valid_ratio=valid_ratio,
        random_state=random_state
    )

    # 收集 unseen subjects（全部作为跨被试测试）
    X_unseen_all, y_unseen_all = [], []
    for sub in unseen_subjects:
        X, y = _load_subject(sub, subject_data, sample_ratio, random_state)
        X_unseen_all.append(X); y_unseen_all.append(y)

    X_unseen_test = np.concatenate(X_unseen_all, axis=0)
    y_unseen_test = np.concatenate(y_unseen_all, axis=0)

    # 转为 tensor
    def _to_tensor(X, y):
        return torch.Tensor(X.astype(np.float32)).unsqueeze(1), torch.Tensor(y).long()

    X_train, y_train = _to_tensor(X_train, y_train)
    X_valid, y_valid = _to_tensor(X_valid, y_valid)
    X_seen_test, y_seen_test = _to_tensor(X_seen_test, y_seen_test)
    X_unseen_test, y_unseen_test = _to_tensor(X_unseen_test, y_unseen_test)

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}")
    print(f"Seen-subject Test: {X_seen_test.shape}")
    print(f"Unseen-subject Test: {X_unseen_test.shape}")

    trainloader = Data.DataLoader(Data.TensorDataset(X_train, y_train),
                                  batch_size=bs, shuffle=True, num_workers=0)
    validloader = Data.DataLoader(Data.TensorDataset(X_valid, y_valid),
                                  batch_size=1, shuffle=False, num_workers=0)
    seen_test_loader = Data.DataLoader(Data.TensorDataset(X_seen_test, y_seen_test),
                                       batch_size=1, shuffle=False, num_workers=0)
    unseen_test_loader = Data.DataLoader(Data.TensorDataset(X_unseen_test, y_unseen_test),
                                         batch_size=1, shuffle=False, num_workers=0)

    return trainloader, validloader, seen_test_loader, unseen_test_loader


def getAllDataloader(subject, data_path, bs=128, dataset='physionetmi'):
    """
    MAtt 兼容接口。
    subject: 被试编号（1-based）
    data_path: 指向预处理 pkl 的路径
    dataset: 'physionetmi' 或 'bci2a'
    """
    pkl_path = Path(data_path)
    subject_data, metadata = load_preprocessed_dataset(pkl_path)

    if subject < 1 or subject > len(subject_data):
        raise ValueError(
            f'{dataset} 被试编号 {subject} 超出范围 1~{len(subject_data)}'
        )

    X, y = subject_data[subject - 1]

    # 校验通道数：pkl 应为已选好的 22 个任务相关通道
    if X.shape[1] != 22:
        raise ValueError(
            f'被试 {subject:03d} 通道数为 {X.shape[1]}，期望 22。'
            f'请重新运行 Riemannian_Trajectories/main.py 生成 22 通道数据。'
        )

    # 可选：校验通道名是否为 BCI IV-2a 标准 22 通道
    if metadata and len(metadata) >= subject:
        ch_names = metadata[subject - 1].get('channel_names', [])
        expected = [
            'Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4',
            'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
            'CP3', 'CP1', 'CPz', 'CP2', 'CP4',
            'P1', 'Pz', 'P2', 'POz',
        ]
        if ch_names and set(ch_names) != set(expected):
            print(f'⚠️ 被试 {subject:03d} 通道名与标准 BCI IV-2a 不完全一致: {ch_names}')

    # 提示 BCIC-IV-2a 第 4 类语义
    if dataset == 'bci2a' and metadata and len(metadata) >= subject:
        note = metadata[subject - 1].get('note', '')
        if note:
            print(f'ℹ️  被试 {subject:03d}: {note}')

    X_train, y_train, X_valid, y_valid, X_test, y_test = split_train_valid_test(X, y)

    # MAtt 期望输入: (n_trials, 1, channels, timepoints)
    X_train = torch.Tensor(X_train).unsqueeze(1)
    y_train = torch.Tensor(y_train).long()
    X_valid = torch.Tensor(X_valid).unsqueeze(1)
    y_valid = torch.Tensor(y_valid).long()
    X_test = torch.Tensor(X_test).unsqueeze(1)
    y_test = torch.Tensor(y_test).long()

    print(f"Train: {X_train.shape}, Valid: {X_valid.shape}, Test: {X_test.shape}")

    train_dataset = Data.TensorDataset(X_train, y_train)
    valid_dataset = Data.TensorDataset(X_valid, y_valid)
    test_dataset = Data.TensorDataset(X_test, y_test)

    trainloader = Data.DataLoader(train_dataset, batch_size=bs, shuffle=True, num_workers=0)
    validloader = Data.DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)
    testloader = Data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    return trainloader, validloader, testloader