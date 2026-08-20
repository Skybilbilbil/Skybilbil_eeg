# -*- coding: utf-8 -*-
"""
CSV 格式 EEGMMIDB 清洗数据加载器
对应论文: Increasing accessibility to a large brain–computer interface dataset
加载已预处理的 MI 数据，提取四分类试次，返回 numpy 数组。
"""

import pandas as pd
import numpy as np
from pathlib import Path

CSV_DIR = Path("CSV files")
MI_RUNS = [2, 4, 6, 8, 10, 12]
SFREQ = 160.0

# BCI2000 原始 64 通道顺序中 7 个目标通道的列索引
# 顺序: FC5 FC3 FC1 FCz FC2 FC4 FC6 C5 C3 C1 Cz C2 C4 C6 CP5 CP3 CP1 CPz CP2 CP4 CP6 ...
CHANNEL_INDEX_MAP = {
    'FCz': 3,
    'FC3': 1,
    'FC4': 5,
    'C4': 12,
    'C3': 8,
    'CPz': 17,
    'Cz': 10,
}

SELECTED_CHANNEL_NAMES = ['FCz', 'FC3', 'FC4', 'C4', 'C3', 'CPz', 'Cz']
SELECTED_CHANNEL_INDICES = [CHANNEL_INDEX_MAP[ch] for ch in SELECTED_CHANNEL_NAMES]

# 标签映射: 只保留 MI 任务中的动作试次，忽略 Relax
# Run 2,6,10 (Task 2, 单侧想象): Class 5=左拳, Class 6=右拳
# Run 4,8,12 (Task 4, 双侧想象): Class 11=双拳, Class 12=双脚
LABEL_MAP = {
    5: 'left',
    6: 'right',
    11: 'hands',
    12: 'feet',
}


def load_subject_mi_trials(subject_id):
    """
    加载单个被试的所有 MI 试次。

    参数:
        subject_id: int, 被试编号 (1~103)

    返回:
        left, right, feet, hands: 各为 np.ndarray [n_trials, 7, n_times]
        若某类别无试次，返回形状 (0, 7, n_times) 的空数组
    """
    left_trials, right_trials, feet_trials, hands_trials = [], [], [], []

    for run in MI_RUNS:
        sig_file = CSV_DIR / f"SUB_{subject_id:03d}_SIG_{run:02d}.csv"
        ann_file = CSV_DIR / f"SUB_{subject_id:03d}_ANN_{run:02d}.csv"

        if not sig_file.exists() or not ann_file.exists():
            continue

        sig = pd.read_csv(sig_file, header=None).values   # [N, 64]
        ann = pd.read_csv(ann_file, header=None).values   # [30, 5]

        for t in range(ann.shape[0]):
            cls = int(ann[t, 0])
            start = int(ann[t, 3]) - 1   # MATLAB 1-based -> Python 0-based
            end = int(ann[t, 4])          # Python slice 左闭右开，MATLAB end  inclusive 正好对齐

            if cls not in LABEL_MAP:
                continue

            trial = sig[start:end, SELECTED_CHANNEL_INDICES].T  # [7, T]

            cat = LABEL_MAP[cls]
            if cat == 'left':
                left_trials.append(trial)
            elif cat == 'right':
                right_trials.append(trial)
            elif cat == 'feet':
                feet_trials.append(trial)
            elif cat == 'hands':
                hands_trials.append(trial)

    all_trials = left_trials + right_trials + feet_trials + hands_trials
    if len(all_trials) == 0:
        empty = np.empty((0, 7, 0))
        return empty, empty, empty, empty

    min_len = min(tr.shape[1] for tr in all_trials)

    def stack(trials):
        if len(trials) == 0:
            return np.empty((0, 7, min_len))
        trimmed = [tr[:, :min_len] for tr in trials]
        return np.stack(trimmed, axis=0)

    return stack(left_trials), stack(right_trials), stack(feet_trials), stack(hands_trials)