# -*- coding: utf-8 -*-
"""
MOABB BCIC-IV-2a (BNCI2014001) 四类运动想象数据加载模块。
四类标签：left_hand / right_hand / feet / tongue
为兼容 pkl 结构，tongue 占用 'hands' 槽位，使用时应留意语义。
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mne
import numpy as np

try:
    from moabb.datasets import BNCI2014_001 as _BNCICls
except ImportError:
    try:
        from moabb.datasets import BNCI2014001 as _BNCICls
    except ImportError:
        from moabb.datasets.bnci import BNCI2014001 as _BNCICls

from preprocessing import preprocess_bci2a_raw, epochs_to_arrays

SFREQ = 250.0
VALID_SUBJECTS = list(range(1, 10))

SELECTED_CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]

_BCI2A_EVENT_CODE_MAP = {
    "769": "left_hand",
    "770": "right_hand",
    "771": "feet",
    "772": "tongue",
}

_BCI2A_EVENT_NAME_MAP = {
    "left_hand": "left_hand",
    "right_hand": "right_hand",
    "feet": "feet",
    "both feet": "feet",
    "tongue": "tongue",
}


def _map_bci2a_annotations(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    old = raw.annotations
    onsets, durations, descriptions = [], [], []
    for onset, duration, desc in zip(old.onset, old.duration, old.description):
        desc = str(desc).strip().lower()
        mapped = _BCI2A_EVENT_CODE_MAP.get(desc) or _BCI2A_EVENT_NAME_MAP.get(desc)
        if mapped is None:
            continue
        onsets.append(onset)
        durations.append(duration)
        descriptions.append(mapped)
    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    return raw


def _process_raw(raw: mne.io.BaseRaw) -> mne.io.BaseRaw:
    # 保留 EOG 通道（BCI2a 原始 GDF 含 EOG1/EOG2/EOG3），供后续 ICA
    # 用真实眼电参考去伪迹（对齐官方 A01-A05 的 EOG 回归清洗）；
    # 刺激通道（STI）随 stim 类型被 pick_types 排除。
    raw.pick_types(eeg=True, eog=True)
    eog_chs = [c for c in raw.ch_names if 'EOG' in c.upper()]
    if eog_chs:
        raw.set_channel_types({c: 'eog' for c in eog_chs})
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    except Exception as e:
        print(f'warning: 无法设置 montage: {e}')
    raw = _map_bci2a_annotations(raw)
    return raw


def load_subject_mi_raw(subject: int) -> mne.io.BaseRaw:
    dataset = _BNCICls()
    sessions = dataset.get_data(subjects=[subject])
    raws = []
    for sess_name, sess_data in sorted(sessions[subject].items()):
        if hasattr(sess_data, 'pick_types'):
            # 旧版结构: session -> raw
            raws.append(_process_raw(sess_data))
        else:
            # 新版结构: session -> {run: raw}
            for run_name, raw in sorted(sess_data.items()):
                raws.append(_process_raw(raw))
    if len(raws) == 1:
        return raws[0]
    return mne.concatenate_raws(raws, preload=True)


def load_subject_mi_trials(subject_id: int) -> Tuple[np.ndarray, ...]:
    raw = load_subject_mi_raw(subject_id)
    left, right, feet, tongue = preprocess_bci2a_raw(raw)

    epochs_map = {
        "left_hand": left,
        "right_hand": right,
        "feet": feet,
        "tongue": tongue,
    }
    return epochs_to_arrays(
        epochs_map,
        ["left_hand", "right_hand", "feet", "tongue"],
        len(SELECTED_CHANNEL_NAMES),
    )


if __name__ == "__main__":
    sub = 1
    left, right, feet, tongue = load_subject_mi_trials(sub)
    print(f"S{sub:03d} shapes:")
    print(f"  left:  {left.shape}")
    print(f"  right: {right.shape}")
    print(f"  feet:  {feet.shape}")
    print(f"  tongue (slot=hands): {tongue.shape}")