# -*- coding: utf-8 -*-
"""
主流程：MOABB BCIC-IV-2a 原始数据加载 → 22 通道 ICA 去噪 →
       8-30Hz 滤波 → [-1.0, 3.5]s epoch → 标准独立 EA →
       保存供 MAtt 使用
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

import moabb_loader_bci2a
from moabb_loader_bci2a import load_subject_mi_trials, SFREQ, VALID_SUBJECTS, SELECTED_CHANNEL_NAMES
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy

OUTPUT_DIR = Path("saved_data")
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    print("=" * 60)
    print("🚀 MOABB BCIC-IV-2a 原始数据 + ICA 去噪 + 标准独立 EA")
    print("=" * 60)

    aligned_left = []
    aligned_right = []
    aligned_feet = []
    aligned_hands = []
    ea_matrices = []
    metadata = []

    for subj_id in tqdm(VALID_SUBJECTS, desc="处理被试"):
        try:
            left, right, feet, tongue = load_subject_mi_trials(subj_id)
        except Exception as e:
            print(f"\n⚠️ 被试 S{subj_id:03d} 加载失败: {e}，已跳过")
            continue

        if left.shape[0] + right.shape[0] + feet.shape[0] + tongue.shape[0] == 0:
            print(f"⚠️ 被试 S{subj_id:03d} 无有效 MI 试次，已跳过")
            continue

        for name, arr in [("left", left), ("right", right), ("feet", feet), ("tongue", tongue)]:
            if arr.shape[0] > 0 and arr.shape[1] != 22:
                raise ValueError(
                    f"被试 S{subj_id:03d} 类别 {name} 通道数为 {arr.shape[1]}，期望 22。"
                )

        all_data = np.concatenate([left, right, feet, tongue], axis=0)
        P = compute_ea_matrix_from_numpy(all_data)

        aligned_left.append(apply_ea_to_numpy(left, P) if left.shape[0] > 0 else left)
        aligned_right.append(apply_ea_to_numpy(right, P) if right.shape[0] > 0 else right)
        aligned_feet.append(apply_ea_to_numpy(feet, P) if feet.shape[0] > 0 else feet)
        aligned_hands.append(apply_ea_to_numpy(tongue, P) if tongue.shape[0] > 0 else tongue)

        ea_matrices.append(P)
        metadata.append({
            'subject_key': f"BCICIV2a_S{subj_id:03d}",
            'sfreq': SFREQ,
            'channel_names': SELECTED_CHANNEL_NAMES,
            'num_left': left.shape[0],
            'num_right': right.shape[0],
            'num_feet': feet.shape[0],
            'num_hands': tongue.shape[0],
            'trial_length': left.shape[2] if left.shape[0] > 0 else right.shape[2],
            'ea_matrix': P,
            'dataset': 'BCICIV2a',
            'note': '第4类为 tongue，复用 hands 槽位',
        })

    num_subjects = len(metadata)
    print(f"\n✅ 成功处理被试数: {num_subjects}")

    save_data = {
        'num_subjects': num_subjects,
        'subject_metadata': metadata,
        'aligned_left_epochs_list': aligned_left,
        'aligned_right_epochs_list': aligned_right,
        'aligned_feet_epochs_list': aligned_feet,
        'aligned_hands_epochs_list': aligned_hands,
        'ea_matrices': ea_matrices,
        'data_source': 'BCICIV2a_MOABB_Raw_ICA',
    }

    output_path = OUTPUT_DIR / "preprocessed_data_bci2a_raw_ica.pkl"
    with open(output_path, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"💾 数据已保存: {output_path}")
    print(f"   被试数: {num_subjects}")


if __name__ == "__main__":
    main()