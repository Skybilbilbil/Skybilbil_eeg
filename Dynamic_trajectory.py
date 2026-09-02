import pickle
import numpy as np
import torch
import mne
from pathlib import Path
import time

# ========== 配置参数 ==========
# 输入输出路径
INPUT_DIR = Path("saved_data")
OUTPUT_DIR = Path("trajectory_data")
OUTPUT_DIR.mkdir(exist_ok=True)

# 采样与时频参数 (CSV 清洗数据固定 160Hz)
# 每个被试会使用自己的实际采样率
DEFAULT_SFREQ = 160.0 
BANDS = [(8, 12),(12,14),(14,20),(20,25),(25, 30)]  # Mu 节律和 Beta 节律
WIN_SEC = 1.0   # 滑动窗口 1 秒
STEP_SEC = 0.1 # 步长 0.1 秒

# 🔧 防止过拟合：只选择7个中央顶叶通道
# 新流程保存的是 22 通道 epoch，在此处按名称截取 7 通道
SELECTED_CHANNEL_NAMES = ['FCz', 'FC3', 'FC4', 'C4', 'C3', 'CPz', 'Cz']


def _select_channels_from_metadata(all_trials, channel_names):
    """
    从 22 通道 trial 数组中按名称选取 7 个目标通道，并保持 SELECTED_CHANNEL_NAMES 顺序。

    参数:
        all_trials: np.ndarray, shape [trials, C, T]
        channel_names: list[str], 长度 C，与 all_trials 的通道维度对应

    返回:
        selected: np.ndarray, shape [trials, 7, T]
    """
    if all_trials.shape[1] == len(SELECTED_CHANNEL_NAMES):
        # 已经是 7 通道，直接返回
        return all_trials

    name_to_idx = {name: i for i, name in enumerate(channel_names)}
    indices = []
    for ch in SELECTED_CHANNEL_NAMES:
        if ch not in name_to_idx:
            raise ValueError(f"目标通道 {ch} 不在输入通道列表 {channel_names} 中")
        indices.append(name_to_idx[ch])

    return all_trials[:, indices, :]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def filter_data_mne(data, sfreq, l_freq, h_freq):
    """利用 MNE 进行快速滤波 (data shape:[trials, channels, time])"""
    # mne.filter.filter_data 默认支持多维数组
    return mne.filter.filter_data(data.astype(np.float64), sfreq, l_freq, h_freq, n_jobs=-1, verbose=False)

def build_dynamic_trajectories():
    print("🚀 开始构建动态流形轨迹 (Riemannian Trajectories)")
    print(f"💻 计算设备: {device} | 频段: {BANDS} | 窗口: {WIN_SEC}s, 步长: {STEP_SEC}s")
    
    # 1. 查找最新的预处理文件
    pkl_files = list(INPUT_DIR.glob("preprocessed_data_*.pkl"))
    if not pkl_files:
        raise FileNotFoundError("未找到预处理数据，请先运行预处理脚本！")
    pkl_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    latest_file = pkl_files[0]
    
    print(f"📂 加载数据: {latest_file.name}")
    with open(latest_file, 'rb') as f:
        data = pickle.load(f)
    
    num_subjects = data['num_subjects']
    subject_metadata = data.get('subject_metadata', [])
    subject_trajectories =[]
    
    print(f"✅ 数据集来源: {data.get('data_source', 'unknown')}")
    print(f"📊 包含被试: {num_subjects}")
    
    start_time = time.time()
    
    # 2. 按被试处理
    for subj_idx in range(num_subjects):
        if subj_idx < len(subject_metadata):
            subj_label = subject_metadata[subj_idx]['subject_key']
            subj_sfreq = subject_metadata[subj_idx].get('sfreq', DEFAULT_SFREQ)
        else:
            subj_label = f"A{subj_idx+1:02d}T"
            subj_sfreq = DEFAULT_SFREQ

        print(f"\n🧠 正在处理被试 {subj_label} ...")

        win_samples = int(WIN_SEC * subj_sfreq)
        step_samples = int(STEP_SEC * subj_sfreq)
        
        # 提取四类数据 (shape: [trials, channels, times])
        epochs_list =[
            data['aligned_left_epochs_list'][subj_idx],
            data['aligned_right_epochs_list'][subj_idx],
            data['aligned_feet_epochs_list'][subj_idx],
            data['aligned_hands_epochs_list'][subj_idx]
        ]
        
        # 创建标签 (0: Left, 1: Right, 2: Feet, 3: Hands)
        labels = np.concatenate([[i] * len(ep) for i, ep in enumerate(epochs_list)])
        all_trials = np.concatenate(epochs_list, axis=0) # [total_trials, C, T]
        
        # 从 22 通道 epoch 中选取 7 个目标通道
        # 优先从 metadata 读取通道名，否则使用标准 22 通道顺序兜底
        channel_names = subject_metadata[subj_idx].get('channel_names', None) if subj_idx < len(subject_metadata) else None
        if channel_names is None:
            channel_names = [
                "Fz", "FC3", "FC1", "FCz", "FC2", "FC4",
                "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
                "CP3", "CP1", "CPz", "CP2", "CP4",
                "P1", "Pz", "P2", "POz",
            ]
        all_trials = _select_channels_from_metadata(all_trials, channel_names)

        num_channels = all_trials.shape[1]
        print(f"   通道数: {num_channels} | {SELECTED_CHANNEL_NAMES}")
        
        # --- 步骤 A: 频域扩展 (Filter Bank) - 分别处理每个频带 ---
        filtered_bands = []
        for (lf, hf) in BANDS:
            filt_data = filter_data_mne(all_trials, subj_sfreq, lf, hf)
            filtered_bands.append(filt_data)
        
        num_bands = len(BANDS)
        
        # --- 步骤 B: 分别计算每个频带的协方差矩阵 ---
        all_band_covs = []
        
        for band_idx, filt_data in enumerate(filtered_bands):
            print(f"   处理频带 {band_idx+1}/{num_bands}: {BANDS[band_idx]}...")
            
            # 单个频带: [total_trials, num_channels, times]
            tensor_data = torch.tensor(filt_data, dtype=torch.float64, device=device) * 1e6
            
            # 利用 unfold 生成滑动窗口
            # [trials, C, times -> [trials, C, num_windows, WIN_SAMPLES]
            windows = tensor_data.unfold(dimension=2, size=win_samples, step=step_samples)
            
            # 维度重排: -> [trials, num_windows, C, WIN_SAMPLES]
            windows = windows.permute(0, 2, 1, 3)
            num_windows = windows.shape[1]
            
            # 中心化
            mean = windows.mean(dim=-1, keepdim=True)
            windows_centered = windows - mean
            
            # 批量计算协方差矩阵: (X @ X^T) / (N-1)
            cov_matrices = torch.matmul(windows_centered, windows_centered.mT) / (win_samples - 1)
            
            # 正则化
            reg_matrix = torch.eye(num_channels, dtype=torch.float64, device=device) * 1e-4
            cov_matrices = cov_matrices + reg_matrix
            
            all_band_covs.append(cov_matrices.cpu().numpy())
        
        # --- 步骤 C: 合并所有频带的协方差矩阵 ---
        # 形状: [trials, num_windows, num_bands, C, C]
        cov_matrices_combined = np.stack(all_band_covs, axis=2)
        
        subject_trajectories.append({
            'trajectories': cov_matrices_combined,
            'labels': labels,
            'subject_key': subj_label,
            'sfreq': subj_sfreq,
        })
        print(f"   生成轨迹形状: {cov_matrices_combined.shape} (Trials x Windows x Bands x Chs x Chs)")
    
    # 3. 保存结果
    save_path = OUTPUT_DIR / "trajectories_moabb_7ch.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({
            'subject_data': subject_trajectories,
            'subject_metadata': subject_metadata,
            'metadata': {
                'sfreq': DEFAULT_SFREQ,
                'bands': BANDS,
                'win_sec': WIN_SEC,
                'step_sec': STEP_SEC,
                'data_source': data.get('data_source', 'unknown'),
            }
        }, f)
        
    print(f"\n✅ 所有被试动态轨迹已保存至 {save_path}")
    print(f"⏱️ 总耗时: {time.time() - start_time:.2f} 秒")

if __name__ == "__main__":
    build_dynamic_trajectories()