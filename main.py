# -*- coding: utf-8 -*-
"""
主流程：从 MOABB PhysionetMI 原始数据加载 → 64 通道 ICA 去噪 →
       选 22 通道 8-30Hz 滤波 → [-1.0, 3.5]s epoch → 标准独立 EA →
       保存供 Dynamic Trajectory 使用

多进程版：按被试并行（ICA/滤波/EA 每被试独立），逐被试断点续跑
（已有 pkl 中被试直接跳过），OMP 线程数压低避免进程间资源竞争。
"""

# 让底层 BLAS/OpenBLAS/MKL 使用多核，提高 ICA/滤波 CPU 利用率
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

import moabb_loader
from moabb_loader import load_subject_mi_trials, SFREQ, VALID_SUBJECTS, SELECTED_CHANNEL_NAMES
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy

OUTPUT_DIR = Path("saved_data")
OUTPUT_DIR.mkdir(exist_ok=True)
OUTPUT_PATH = Path(os.environ.get("PHYSIONET_OUTPUT",
                                  str(OUTPUT_DIR / "preprocessed_data_moabb_raw_ica.pkl")))
N_WORKERS = 8          # 系统 16 核，预处理纯 CPU；OMP=2 × 8 进程 = 16 核打满
REJECT_AMP_UV = None if os.environ.get("REJECT_AMP_UV", "250.0").lower() in ("none", "") \
    else float(os.environ["REJECT_AMP_UV"])  # 峰峰值幅度阈值（µV），None=不剔除；可用环境变量覆盖


def _subject_scope() -> list | None:
    """环境变量 PHYSIONET_SUBJECTS="1,5,9"（逗号分隔原始编号）限定处理范围；
    None=全部 VALID_SUBJECTS。调试/小规模验证用。"""
    v = os.environ.get("PHYSIONET_SUBJECTS")
    if not v:
        return None
    return [int(s) for s in v.split(",") if s]


def _process_one(subject_id: int):
    """处理单个被试（worker 内执行）：加载→ICA→滤波→epoch→EA。

    返回 (subject_id, result_dict | None, error_str | None)。
    result_dict = dict(aligned_left/right/feet/hands, ea_matrix, metadata)
    """
    try:
        left, right, feet, hands = load_subject_mi_trials(subject_id, reject_amp_uv=REJECT_AMP_UV)
    except Exception as e:
        return subject_id, None, f"加载失败: {e}"

    if left.shape[0] + right.shape[0] + feet.shape[0] + hands.shape[0] == 0:
        return subject_id, None, "无有效 MI 试次"

    # 校验每个类别的通道数都应为 22
    for name, arr in [("left", left), ("right", right), ("feet", feet), ("hands", hands)]:
        if arr.shape[0] > 0 and arr.shape[1] != 22:
            return subject_id, None, f"类别 {name} 通道数为 {arr.shape[1]}，期望 22"

    # 拼接该被试全部试次以计算独立 EA 矩阵
    all_data = np.concatenate([left, right, feet, hands], axis=0)  # [N, 22, T]
    P = compute_ea_matrix_from_numpy(all_data)

    res = {
        "left": apply_ea_to_numpy(left, P) if left.shape[0] > 0 else left,
        "right": apply_ea_to_numpy(right, P) if right.shape[0] > 0 else right,
        "feet": apply_ea_to_numpy(feet, P) if feet.shape[0] > 0 else feet,
        "hands": apply_ea_to_numpy(hands, P) if hands.shape[0] > 0 else hands,
        "ea_matrix": P,
        "metadata": {
            'subject_key': f"PhysionetMI_S{subject_id:03d}",
            'sfreq': SFREQ,
            'channel_names': SELECTED_CHANNEL_NAMES,
            'num_left': left.shape[0],
            'num_right': right.shape[0],
            'num_feet': feet.shape[0],
            'num_hands': hands.shape[0],
            'trial_length': left.shape[2] if left.shape[0] > 0 else
                            (right.shape[2] if right.shape[0] > 0 else
                             (feet.shape[2] if feet.shape[0] > 0 else hands.shape[2])),
            'ea_matrix': P,
        },
    }
    return subject_id, res, None


def _load_existing() -> dict:
    """读取已有 pkl 中已处理的被试数据（断点续跑用）。

    返回 {subject_id: {'aligned': {...}, 'ea_matrix', 'metadata'}}，
    供 main() 与新处理结果合并，避免覆盖历史已保存被试。
    """
    if not OUTPUT_PATH.exists():
        return {}
    with open(OUTPUT_PATH, 'rb') as f:
        old = pickle.load(f)
    existing = {}
    sm = old.get('subject_metadata', [])
    for i, m in enumerate(sm):
        sid = int(m['subject_key'].split('S')[-1])
        existing[sid] = {
            'aligned': {'left': old['aligned_left_epochs_list'][i],
                        'right': old['aligned_right_epochs_list'][i],
                        'feet': old['aligned_feet_epochs_list'][i],
                        'hands': old['aligned_hands_epochs_list'][i]},
            'ea_matrix': old['ea_matrices'][i],
            'metadata': m,
        }
    return existing


def main():
    print("=" * 60)
    print("🚀 MOABB PhysionetMI 原始数据 + ICA 去噪 + 标准独立 EA（多进程版）")
    print("=" * 60)

    existing_ids = _load_existing()
    scope = _subject_scope()
    if scope is not None:
        todo = [s for s in scope if s not in existing_ids]
        print(f"调试模式: 限被试 {scope}，已完成 {len(existing_ids)} 人，本次处理 {len(todo)} 人")
    else:
        todo = [s for s in VALID_SUBJECTS if s not in existing_ids]
        print(f"有效被试 {len(VALID_SUBJECTS)} 人，已完成 {len(existing_ids)} 人，本次处理 {len(todo)} 人")

    if not todo:
        print("全部被试已处理，无需重跑。")
        return

    # 按 (类, 被试索引) 收集结果，worker 返回顺序不定，用 subject_id 定位
    # 先并入已有 pkl 中被试，再叠加本次新处理结果（断点续跑不丢历史）
    aligned = {"left": {}, "right": {}, "feet": {}, "hands": {}}
    ea_matrices = {}
    metadata = {}
    for sid, ed in existing_ids.items():
        for k in aligned:
            aligned[k][sid] = ed['aligned'][k]
        ea_matrices[sid] = ed['ea_matrix']
        metadata[sid] = ed['metadata']
    failed = []

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_process_one, sid): sid for sid in todo}
        for fut in tqdm(as_completed(futures), total=len(todo), desc="处理被试"):
            sid, res, err = fut.result()
            if err:
                print(f"\n⚠️ 被试 S{sid:03d}: {err}，已跳过")
                failed.append(sid)
                continue
            for k in aligned:
                aligned[k][sid] = res[k]
            ea_matrices[sid] = res["ea_matrix"]
            metadata[sid] = res["metadata"]

    # 按原始编号排序，保持与历史 pkl 一致的被试顺序
    ordered = [s for s in VALID_SUBJECTS if s in metadata]
    num_subjects = len(ordered)
    print(f"\n✅ 成功处理被试数: {num_subjects}（失败 {len(failed)}: {failed}）")

    save_data = {
        'num_subjects': num_subjects,
        'subject_metadata': [metadata[s] for s in ordered],
        'aligned_left_epochs_list': [aligned["left"][s] for s in ordered],
        'aligned_right_epochs_list': [aligned["right"][s] for s in ordered],
        'aligned_feet_epochs_list': [aligned["feet"][s] for s in ordered],
        'aligned_hands_epochs_list': [aligned["hands"][s] for s in ordered],
        'ea_matrices': [ea_matrices[s] for s in ordered],
        'data_source': 'PhysionetMI_MOABB_Raw_ICA',
    }

    with open(OUTPUT_PATH, 'wb') as f:
        pickle.dump(save_data, f)

    print(f"💾 数据已保存: {OUTPUT_PATH}")
    print(f"   被试数: {num_subjects}")


if __name__ == "__main__":
    main()