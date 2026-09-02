# -*- coding: utf-8 -*-
"""
MOABB PhysionetMI 四类运动想象 (MI) 数据加载模块。

只处理运动想象任务，不处理运动执行 (ME)：
  1) 排除 6 名异常被试 (S088, S089, S092, S100, S104, S106)
  2) 丢弃 Run 1/2 基线，只使用 MI run：4, 6, 8, 10, 12, 14
  3) 将 T0/T1/T2 映射为四类 MI 事件：left_hand / right_hand / hands / feet
  4) 返回 [-1.0, 3.5]s epoch

同时提供与 csv_loader.py 兼容的接口：load_subject_mi_trials。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import mne
import numpy as np
from moabb.datasets import PhysionetMI

from preprocessing import preprocess_physionetmi_raw, _normalize_ch_name

SFREQ = 160.0

# MNE eegbci 数据目录配置键
_EEGBCI_PATH_KEY = "MNE_DATASETS_EEGBCI_PATH"


def _find_existing_eegbci_root() -> Path | None:
    r"""
    自动寻找已下载的 eegbci 数据根目录。
    用户可能通过 download_physionet.py 把数据下到了 MNE 默认目录，
    也可能因为配置损坏下到了项目目录下的异常路径（如 C-\Users\...）。
    """
    project_dir = Path(__file__).resolve().parent
    candidates: list[Path] = []

    # 1) MNE 当前配置的路径
    configured = mne.get_config(_EEGBCI_PATH_KEY)
    if configured:
        candidates.append(Path(configured))

    # 2) 项目目录下常见的异常/损坏路径（用当前用户名替代硬编码）
    _current_user = Path.home().name
    candidates.append(project_dir / "C-\\Users" / _current_user / "mne_data")
    candidates.append(project_dir / "C:\\Users" / _current_user / "mne_data")

    # 3) 系统默认路径
    candidates.append(Path.home() / "mne_data")

    # 去重并保持顺序
    seen = set()
    unique_candidates = []
    for c in candidates:
        norm = str(c.resolve()) if c.exists() else str(c)
        if norm not in seen:
            seen.add(norm)
            unique_candidates.append(c)

    # 用 S001R04.edf 作为存在性探针
    probe = Path("MNE-eegbci-data") / "files" / "eegmmidb" / "1.0.0" / "S001" / "S001R04.edf"
    for root in unique_candidates:
        if (root / probe).exists():
            return root

    return None


# 模块导入时自动定位已下载数据；如果找不到则使用 MNE 默认目录
_EEGBCI_DATA_PATH = _find_existing_eegbci_root()
if _EEGBCI_DATA_PATH is None:
    _EEGBCI_DATA_PATH = Path(mne.get_config(_EEGBCI_PATH_KEY) or Path.home() / "mne_data")
print(f"📁 使用已下载数据目录: {_EEGBCI_DATA_PATH}")

# 排除的异常被试（原始 MOABB ID，对应论文 Table 4）
EXCLUDED_SUBJECTS = {88, 89, 92, 100, 104, 106}

# 有效被试列表（103 人，保留原始编号）
VALID_SUBJECTS = sorted(set(range(1, 110)) - EXCLUDED_SUBJECTS)

# MI run 原始编号：只保留四类运动想象任务 run
MI_RUNS = {4, 6, 8, 10, 12, 14}

# MI run 中 T1/T2 对应的四类事件名
# Run 4,8,12 为单侧想象（左/右拳）；Run 6,10,14 为双侧想象（双拳/双脚）
MI_EVENT_NAMES = {
    ("fist", "T1"): "left_hand",
    ("fist", "T2"): "right_hand",
    ("both", "T1"): "hands",
    ("both", "T2"): "feet",
}

# 与 csv_loader.py / preprocessing.py 一致的 22 通道选择
SELECTED_CHANNEL_NAMES = [
    "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP3", "CP1", "CPz", "CP2", "CP4",
    "P1", "Pz", "P2", "POz",
]


def _extract_run_number(path: str | Path) -> int:
    """从 EDF 文件名（如 S001R03.edf）中提取 run 编号。"""
    stem = Path(path).stem
    match = re.search(r"R(\d{2})$", stem)
    if not match:
        raise ValueError(f"无法从文件名 {stem} 解析 run 编号")
    return int(match.group(1))


def _validate_subject(subject: int) -> int:
    """检查并返回有效被试编号。"""
    if subject in EXCLUDED_SUBJECTS:
        raise ValueError(f"被试 S{subject:03d} 在排除列表 {EXCLUDED_SUBJECTS} 中")
    if not 1 <= subject <= 109:
        raise ValueError(f"被试编号 {subject} 超出范围 1~109")
    return subject


def _build_mi_edf_paths(subject: int, root: Path) -> list[Path]:
    """直接从给定根目录构建该被试的 MI run EDF 路径。"""
    base = root / "MNE-eegbci-data" / "files" / "eegmmidb" / "1.0.0" / f"S{subject:03d}"
    paths = [base / f"S{subject:03d}R{run:02d}.edf" for run in sorted(MI_RUNS)]
    return paths


def get_subject_edf_paths(subject: int, verbose: bool = False) -> list[Path]:
    r"""
    返回单个被试全部 MI run 的原始 EDF 文件路径，按 run 编号排序。
    优先从已发现的数据目录直接构建路径，避免 MNE 对异常路径的解析错误；
    如有缺失，再回退到 MOABB 的 data_path 触发下载（下载目录为默认 C:\Users\<user>\mne_data）。
    """
    _validate_subject(subject)

    # 1) 直接从已发现的数据根目录构建路径
    paths = _build_mi_edf_paths(subject, _EEGBCI_DATA_PATH)
    all_exist = all(p.exists() for p in paths)

    if not all_exist:
        # 2) 有缺失时回退到 MOABB data_path，并强制使用默认目录以下载缺失文件
        default_path = Path.home() / "mne_data"
        print(f"⚠️ 部分 MI run 在 {_EEGBCI_DATA_PATH} 不存在，回退到默认下载目录: {default_path}")
        mne.set_config(_EEGBCI_PATH_KEY, str(default_path))
        dataset = PhysionetMI(imagined=True, executed=False)
        paths = [Path(p) for p in dataset.data_path(subject)]
        paths = [p for p in paths if _extract_run_number(p) in MI_RUNS]
        paths.sort(key=lambda p: _extract_run_number(p))

    if verbose:
        print(f"  数据根目录: {_EEGBCI_DATA_PATH}")
        for p in paths:
            print(f"    {'✅' if p.exists() else '❌'} {p.name} (存在={p.exists()})")

    return paths


def _load_run_raw(path: Path) -> mne.io.BaseRaw:
    """读取单个 run 的 EDF 文件：读入后立即降维到 22 个目标中央区通道。

    直接在单个 run 上 pick 22 通道（而不是 6 个 run 全量 64 通道拼接后再选），
    读盘后的内存占用与后续滤波/ICA 计算量都按 22/64 比例大幅下降。
    """
    raw = mne.io.read_raw_edf(path, preload=True, verbose=False)
    # 仅保留 EEG 通道（MNE 会自动提取 annotation，第 65 列不会被保留为数据通道）
    raw.pick_types(eeg=True)
    try:
        raw.set_montage("standard_1020", match_case=False, on_missing="warn")
    except Exception:
        pass
    # 立即降维：只保留 SELECTED_CHANNEL_NAMES 对应的 22 通道（兼容带后缀的 EDF 命名）
    norm = {_normalize_ch_name(c): c for c in raw.ch_names}
    keep = [norm[n] for n in map(_normalize_ch_name, SELECTED_CHANNEL_NAMES) if n in norm]
    if len(keep) == len(SELECTED_CHANNEL_NAMES):
        raw.pick_channels(keep)
    return raw


def _map_run_annotations(raw: mne.io.BaseRaw, run_number: int) -> mne.io.BaseRaw:
    """
    将单个 MI run 的 T0/T1/T2 标注重命名为：
      - T0 → rest
      - T1/T2 → left_hand / right_hand / hands / feet（根据 run 类型）
    """
    # run 4,8,12 为单侧想象；run 6,10,14 为双侧想象
    task_category = "fist" if run_number in {4, 8, 12} else "both"
    old = raw.annotations
    onsets, durations, descriptions = [], [], []

    for onset, duration, desc in zip(old.onset, old.duration, old.description):
        desc = str(desc).strip()
        if desc == "T0":
            descriptions.append("rest")
        elif desc in ("T1", "T2"):
            descriptions.append(MI_EVENT_NAMES[(task_category, desc)])
        else:
            descriptions.append(desc)

        onsets.append(onset)
        durations.append(duration)

    raw.set_annotations(mne.Annotations(onsets, durations, descriptions))
    return raw


def load_subject_mi_raw(subject: int) -> mne.io.BaseRaw:
    """
    加载单个被试的全部 MI 任务 run（原始 4,6,8,10,12,14），
    将标注统一重命名为 left_hand / right_hand / feet / hands，
    拼接为一条连续 Raw 返回。

    参数:
        subject: 原始被试编号（1~109，排除异常被试）

    返回:
        拼接后的 MNE Raw 对象，仅包含四类 MI 事件标注。
    """
    _validate_subject(subject)
    paths = get_subject_edf_paths(subject)
    if not paths:
        raise ValueError(f"被试 S{subject:03d} 没有有效的 MI run")

    raws = []
    for path in paths:
        run_number = _extract_run_number(path)
        raw_run = _load_run_raw(path)
        raw_run = _map_run_annotations(raw_run, run_number)
        raws.append(raw_run)

    raw_all = raws[0] if len(raws) == 1 else mne.concatenate_raws(raws, preload=True)
    return raw_all


def load_subject_mi_trials(subject_id: int, reject_amp_uv: float = None) -> Tuple[np.ndarray, ...]:
    """
    与 csv_loader.py 接口兼容：加载单个被试的四类 MI 试次。

    参数:
        subject_id: 被试编号（1~109，排除列表除外）
        reject_amp_uv: 峰峰值幅度阈值（µV），剔除超幅坏 trial；None=不剔除

    返回:
        left, right, feet, hands: 各为 np.ndarray [n_trials, 22, n_times]
        若某类别无试次，返回形状 (0, 22, n_times) 的空数组
    """
    raw = load_subject_mi_raw(subject_id)
    left_epochs, right_epochs, feet_epochs, hands_epochs = preprocess_physionetmi_raw(
        raw, reject_amp_uv=reject_amp_uv)

    epochs_map = {
        "left_hand": left_epochs,
        "right_hand": right_epochs,
        "feet": feet_epochs,
        "hands": hands_epochs,
    }

    n_times = 0
    for ep in epochs_map.values():
        if len(ep) > 0:
            n_times = ep.get_data(copy=False).shape[-1]
            break

    result = []
    for name in ["left_hand", "right_hand", "feet", "hands"]:
        ep = epochs_map[name]
        if len(ep) == 0 or n_times == 0:
            arr = np.empty((0, len(SELECTED_CHANNEL_NAMES), n_times))
        else:
            arr = ep.get_data(copy=False).astype(np.float32)
        result.append(arr)

    return tuple(result)


# ---------------------------------------------------------------------------
# 以下保留旧接口，供 Data_saving.py / visualization/evaluate_ica_ea.py 调用
# ---------------------------------------------------------------------------

def discover_four_class_mi_datasets(verbose: bool = True):
    """（旧接口）加载 PhysionetMI 四分类运动想象数据集。"""
    dataset = PhysionetMI(imagined=True, executed=False)
    if verbose:
        print(f"  ✅ PhysionetMI: 109 被试，目标事件 left_hand/right_hand/feet/hands")
    return [dataset]


def get_dataset_summary(datasets=None) -> list[dict]:
    """（旧接口）返回数据集摘要信息。"""
    return [{
        "name": "PhysionetMI",
        "code": "PhysionetMotorImagery",
        "n_subjects": len(VALID_SUBJECTS),
        "events": {
            "left_hand": 1, "right_hand": 2,
            "feet": 3, "hands": 4,
        },
        "sfreq": SFREQ,
        "preprocessor": "physionetmi",
    }]


def iter_all_subjects(datasets=None, verbose: bool = True, subjects_to_load=None):
    """
    （旧接口更新）遍历被试，生成 (dataset, subject_id, raw, meta)。
    现在直接返回 load_subject_mi_raw 得到的 Raw（仅含 MI run）。
    """
    if datasets is None:
        datasets = discover_four_class_mi_datasets(verbose=verbose)

    dataset = datasets[0]

    if subjects_to_load is None:
        subjects_to_load = VALID_SUBJECTS
    else:
        subjects_to_load = [s for s in subjects_to_load if s not in EXCLUDED_SUBJECTS]

    for subject_id in subjects_to_load:
        subject_key = f"PhysionetMI_S{subject_id:03d}"
        if verbose:
            print(f"  ⬇️  被试 {subject_key} ...", end=" ", flush=True)

        raw = load_subject_mi_raw(subject_id)

        meta = {
            "subject_key": subject_key,
            "dataset_name": "PhysionetMI",
            "subject_id": subject_id,
            "sfreq": raw.info["sfreq"],
            "n_channels": len(mne.pick_types(raw.info, eeg=True, exclude=[])),
            "preprocessor": "physionetmi",
        }

        if verbose:
            print(
                f"完成 | {meta['n_channels']}ch, "
                f"{meta['sfreq']:.0f}Hz, {raw.n_times / meta['sfreq']:.1f}s"
            )

        yield dataset, subject_id, raw, meta


if __name__ == "__main__":
    print("=" * 60)
    print("MOABB PhysionetMI 原始数据加载测试")
    print("=" * 60)

    test_subject = 1
    print(f"\n测试加载被试 S{test_subject:03d} 的 MI 数据...")
    raw = load_subject_mi_raw(test_subject)
    print(f"  采样率: {raw.info['sfreq']:.0f} Hz")
    print(f"  通道数: {len(raw.ch_names)}")
    print(f"  时长: {raw.n_times / raw.info['sfreq']:.1f} s")
    print(f"  事件: {set(raw.annotations.description)}")

    left, right, feet, hands = load_subject_mi_trials(test_subject)
    print(f"\n四类 MI epoch 形状:")
    print(f"  左手: {left.shape}")
    print(f"  右手: {right.shape}")
    print(f"  双脚: {feet.shape}")
    print(f"  双手: {hands.shape}")
