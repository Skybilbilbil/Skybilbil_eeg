import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

import preprocessing, alignment, moabb_loader

# ========== 核心优化：数据瘦身函数 ==========
def extract_and_compress(epochs_data):
    """
    提取纯净的 NumPy 数组并转换为 float32，剥离臃肿的元数据
    """
    if hasattr(epochs_data, 'get_data'):
        arr = epochs_data.get_data(copy=False)
    else:
        arr = np.array(epochs_data)

    return arr.astype(np.float32)


def preprocess_subject(raw, preprocessor_name="physionetmi"):
    """
    对被试 Raw 做预处理与 EA 对齐（独立EA模式）。

    预处理流程：
      1) 64 通道 1-45Hz 带通滤波
      2) 64 通道 ICA 去眼动伪迹
      3) 选择 22 个中央区通道
      4) 8-30Hz 精细滤波
      5) [-1.0, 3.5]s 切分四类 MI epoch

    返回四类对齐后的 epochs 数组 (left, right, feet, hands)。
    """
    if preprocessor_name != "physionetmi":
        raise ValueError(f"未支持的数据集预处理管线: {preprocessor_name}")

    left_epochs, right_epochs, feet_epochs, hands_epochs = preprocessing.preprocess_physionetmi_raw(
        raw, tmin=-1.0, tmax=3.5
    )

    P = alignment.compute_ea_matrix_from_epochs(
        [left_epochs, right_epochs, feet_epochs, hands_epochs]
    )
    left_aligned, right_aligned, feet_aligned, hands_aligned = (
        alignment.apply_ea_to_epochs_list(
            [left_epochs, right_epochs, feet_epochs, hands_epochs], P
        )
    )

    return (
        extract_and_compress(left_aligned),
        extract_and_compress(right_aligned),
        extract_and_compress(feet_aligned),
        extract_and_compress(hands_aligned),
        P,
    )


# ========== 数据管理配置 ==========
USE_SAVED_DATA = False
SAVE_DATA = True
SAVE_DIR = Path("saved_data")
SAVE_DIR.mkdir(exist_ok=True)

# PhysionetMI 配置：加载所有有效被试（103 人）用于训练
SUBJECTS_TO_LOAD = None  # None 表示加载所有有效被试

# ========== 主流程 ==========
print("🎯 EEG 四分类运动想象预处理 (左手/右手/脚/双手)")
print("=" * 60)

aligned_left_epochs_list = []
aligned_right_epochs_list = []
aligned_feet_epochs_list = []
aligned_hands_epochs_list = []
ea_matrices = []
subject_metadata = []

if USE_SAVED_DATA:
    print("🚀 使用保存的预处理数据（跳过预处理步骤）")

    pkl_files = list(SAVE_DIR.glob("preprocessed_data_*.pkl"))
    if not pkl_files:
        print("❌ 未找到预处理数据文件，切换到完整预处理模式")
        USE_SAVED_DATA = False
    else:
        pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_file = pkl_files[0]

        try:
            with open(latest_file, 'rb') as f:
                loaded_data = pickle.load(f)

            print(f"✅ 已加载预处理数据: {latest_file.name}")
            print(f"📅 保存时间: {loaded_data['save_time']}")
            print(f"👥 包含被试数: {loaded_data['num_subjects']}")

            aligned_left_epochs_list = loaded_data['aligned_left_epochs_list']
            aligned_right_epochs_list = loaded_data['aligned_right_epochs_list']
            aligned_feet_epochs_list = loaded_data['aligned_feet_epochs_list']
            aligned_hands_epochs_list = loaded_data['aligned_hands_epochs_list']
            subject_metadata = loaded_data.get('subject_metadata', [])

            print("\n⏩ 跳过预处理，直接开始数据整合")

        except Exception as e:
            print(f"❌ 加载失败: {e}")
            print("❌ 切换到完整预处理模式")
            USE_SAVED_DATA = False

if not USE_SAVED_DATA:
    print("🔧 进行完整的数据预处理")

    print("\n📡 数据源: MOABB (PhysionetMI 数据集)")
    summary = moabb_loader.get_dataset_summary()
    for ds_info in summary:
        print(
            f"  - {ds_info['name']}: {ds_info['n_subjects']} 被试, "
            f"事件={list(ds_info['events'].keys())}"
        )

    print(f"\n📋 计划加载被试: {SUBJECTS_TO_LOAD}")

    print("\n📋 使用【独立EA】模式（每个被试各自计算EA矩阵）")

    for _, _, raw, meta in moabb_loader.iter_all_subjects(subjects_to_load=SUBJECTS_TO_LOAD):
        print(f"\n=== 读取并预处理被试 {meta['subject_key']} ===")

        left, right, feet, hands, P = preprocess_subject(
            raw, preprocessor_name=meta["preprocessor"]
        )

        aligned_left_epochs_list.append(left)
        aligned_right_epochs_list.append(right)
        aligned_feet_epochs_list.append(feet)
        aligned_hands_epochs_list.append(hands)
        ea_matrices.append(P)
        meta["channel_names"] = moabb_loader.SELECTED_CHANNEL_NAMES
        meta["ea_matrix"] = P
        subject_metadata.append(meta)

        print(
            f"  - 被试 {meta['subject_key']} 预处理及对齐完毕 "
            f"(L={len(left)}, R={len(right)}, F={len(feet)}, H={len(hands)})"
        )

    if SAVE_DATA:
        print("\n💾 正在保存精简后的预处理数据...")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"preprocessed_data_{timestamp}.pkl"
        save_path = SAVE_DIR / filename

        save_data = {
            'aligned_left_epochs_list': aligned_left_epochs_list,
            'aligned_right_epochs_list': aligned_right_epochs_list,
            'aligned_feet_epochs_list': aligned_feet_epochs_list,
            'aligned_hands_epochs_list': aligned_hands_epochs_list,
            'ea_matrices': ea_matrices,
            'subject_metadata': subject_metadata,
            'data_source': 'PhysionetMI',
            'save_time': datetime.now().isoformat(),
            'num_subjects': len(aligned_left_epochs_list),
            'description': 'PhysionetMI四分类运动想象预处理数据(float32, 已独立EA对齐)',
        }

        try:
            with open(save_path, 'wb') as f:
                pickle.dump(save_data, f)

            file_size = save_path.stat().st_size / 1024 / 1024

            print(f"✅ 预处理数据已保存: {save_path}")
            print(f"📊 文件大小: {file_size:.1f} MB")
            print(f"👥 包含被试数: {len(aligned_left_epochs_list)}")
            if subject_metadata:
                keys = [m['subject_key'] for m in subject_metadata]
                print(f"🧑 被试列表: {keys}")
            print("💡 下次设置 USE_SAVED_DATA = True 可秒速读取数据！")

        except Exception as e:
            print(f"❌ 保存失败: {e}")
