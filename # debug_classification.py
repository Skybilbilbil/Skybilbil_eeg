# debug_classification.py
import pickle
import numpy as np
import torch
from pathlib import Path

INPUT_FILE = Path("trajectory_data") / "trajectories_500ms_win.pkl"

print("="*60)
print("数据诊断")
print("="*60)

with open(INPUT_FILE, 'rb') as f:
    dataset = pickle.load(f)

subjects_data = dataset['subject_data']
print(f"\n被试数量: {len(subjects_data)}")

for i, subj in enumerate(subjects_data):
    traj = subj['trajectories']
    labels = subj['labels']
    print(f"\n{'='*40}")
    print(f"被试 {i+1}:")
    print(f"  轨迹形状: {traj.shape}")
    print(f"  标签形状: {labels.shape}")
    print(f"  标签唯一值: {np.unique(labels)}")
    print(f"  标签分布: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    # 检查协方差矩阵
    print(f"  协方差矩阵范围: [{traj.min():.6f}, {traj.max():.6f}]")
    print(f"  协方差矩阵均值: {traj.mean():.6f}")
    print(f"  协方差矩阵标准差: {traj.std():.6f}")
    
    # 检查是否对称
    is_sym = np.allclose(traj[0,0], traj[0,0].T)
    print(f"  是否对称: {is_sym}")
    
    # 检查特征值（是否正定）
    eigvals = np.linalg.eigvalsh(traj[0,0])
    print(f"  第一个矩阵特征值范围: [{eigvals.min():.6f}, {eigvals.max():.6f}]")
    print(f"  是否正定: {np.all(eigvals > 0)}")

# 检查切空间映射后的数据
print(f"\n{'='*60}")
print("切空间映射诊断")
print(f"{'='*60}")

from pyriemann.utils.mean import mean_covariance

# 模拟第一个 fold
test_idx = 0
train_covs_list = [subj['trajectories'] for i, subj in enumerate(subjects_data) if i != test_idx]
train_covs = np.concatenate(train_covs_list, axis=0)
train_labels = np.concatenate([subj['labels'] for i, subj in enumerate(subjects_data) if i != test_idx])

C = train_covs.shape[-1]
flat_train_covs = train_covs.reshape(-1, C, C)

# 计算参考点
sample_indices = np.random.choice(len(flat_train_covs), min(2000, len(flat_train_covs)), replace=False)
ref_cov_np = mean_covariance(flat_train_covs[sample_indices], metric='logeuclid')

print(f"\n参考矩阵:")
print(f"  形状: {ref_cov_np.shape}")
print(f"  范围: [{ref_cov_np.min():.6f}, {ref_cov_np.max():.6f}]")
print(f"  是否对称: {np.allclose(ref_cov_np, ref_cov_np.T)}")

# 测试切空间映射
from Classification import batched_tangent_space_mapping

ref_cov_tensor = torch.tensor(ref_cov_np, dtype=torch.float64, device='cpu')
test_covs = torch.tensor(flat_train_covs[:10], dtype=torch.float64, device='cpu')

tangent = batched_tangent_space_mapping(test_covs, ref_cov_tensor, batch_size=5)
print(f"\n切空间映射结果:")
print(f"  形状: {tangent.shape}")
print(f"  范围: [{tangent.min():.6f}, {tangent.max():.6f}]")
print(f"  均值: {tangent.mean():.6f}")
print(f"  标准差: {tangent.std():.6f}")
print(f"  是否包含NaN: {torch.isnan(tangent).any()}")
print(f"  是否包含Inf: {torch.isinf(tangent).any()}")

# 检查LSTM输入
trials_train, wins, _, _ = train_covs.shape
TANGENT_DIM = C * (C + 1) // 2
train_tangent = batched_tangent_space_mapping(
    torch.tensor(flat_train_covs[:100], dtype=torch.float64, device='cpu'),
    ref_cov_tensor, batch_size=50
)
X_train = train_tangent.view(-1, wins, TANGENT_DIM)

print(f"\nLSTM输入:")
print(f"  X_train 形状: {X_train.shape}")
print(f"  X_train 范围: [{X_train.min():.6f}, {X_train.max():.6f}]")
print(f"  标签范围: [{train_labels.min()}, {train_labels.max()}]")