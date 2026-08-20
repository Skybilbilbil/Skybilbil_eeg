import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# ========== 配置参数 ==========
INPUT_FILE = Path("trajectory_data") / "trajectories_500ms_win.pkl"
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEBUG_MODE = True
NUM_CLASSES = 4  # Left/Right/Feet/Hands
INCLUDE_TEST_IN_TRAIN = False  # 测试被试数据不混入训练（纯跨被试评估）
INTRA_SUBJECT_TEST_RATIO = 0.0  # 0=不用被试内划分，直接留 1-10 号被试做跨被试测试

# ========== 简单黎曼切空间映射（单个频带） ==========
def log_euclidean_mapping(cov_matrices, debug_name="mapping"):
    """
    单个频带的黎曼切空间映射
    Args: cov_matrices [N, 7, 7]
    Returns: tangent_vectors [N, 28]
    """
    C = cov_matrices.shape[-1]
    N = cov_matrices.shape[0]
    
    if DEBUG_MODE:
        print(f"   [LogEuclidean-{debug_name}] 输入: shape={cov_matrices.shape}")
    
    # 1. 特征值分解 + 矩阵对数
    evals, evecs = torch.linalg.eigh(cov_matrices)
    log_evals = torch.log(torch.clamp(evals, min=1e-7))
    log_D = torch.diag_embed(log_evals)
    log_C = torch.matmul(torch.matmul(evecs, log_D), evecs.mT)
    
    if DEBUG_MODE:
        print(f"     特征值: min={evals.min():.6f}, max={evals.max():.6f}")
    
    # 2. 提取上三角
    idx = torch.triu_indices(C, C)
    tangent_vectors = log_C[:, idx[0], idx[1]]
    
    # 3. 黎曼度量规范化
    off_diag_mask = (idx[0] != idx[1])
    tangent_vectors[:, off_diag_mask] *= np.sqrt(2.0)
    
    if DEBUG_MODE:
        print(f"     切空间向量: shape={tangent_vectors.shape}")
    
    return tangent_vectors.to(torch.float32)

# ========== FBCSP_RiemannianLSTM 模型（带时间注意力）==========
class FBCSP_RiemannianLSTM(nn.Module):
    def __init__(self, input_size=140, hidden_size=32, num_classes=4, seq_len=None):
        super(FBCSP_RiemannianLSTM, self).__init__()
        
        # 特征投影
        self.proj = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.5)
        )
        
        self.lstm = nn.LSTM(64, hidden_size, num_layers=1,
                            batch_first=True, bidirectional=True)
        
        # 时间注意力：可学习的注意力向量
        self.attention_vector = nn.Parameter(torch.randn(hidden_size * 2))
        nn.init.xavier_uniform_(self.attention_vector.unsqueeze(0))
        
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size * 2, num_classes)
        )
        
        self.seq_len = seq_len
    
    def forward(self, x, return_attention=False):
        batch_size, seq_len, _ = x.size()
        
        x_flat = x.contiguous().view(-1, x.size(-1))
        x_proj = self.proj(x_flat)
        x_proj_seq = x_proj.view(batch_size, seq_len, -1)
        
        out, _ = self.lstm(x_proj_seq)  # [batch, seq_len, hidden*2]
        
        # 时间注意力
        # attn_scores: [batch, seq_len]
        attn_scores = torch.matmul(out, self.attention_vector)
        attn_weights = torch.softmax(attn_scores, dim=1)  # [batch, seq_len]
        
        # 加权池化
        pooled = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)  # [batch, hidden*2]
        
        logits = self.fc(pooled)
        
        if return_attention:
            return logits, attn_weights
        return logits
    
    def get_temporal_attention(self, x):
        """获取时间注意力权重（用于可视化）"""
        with torch.no_grad():
            _, attn_weights = self.forward(x, return_attention=True)
        return attn_weights
    
    def forward_intermediate(self, x):
        """返回中间层输出，用于可视化"""
        batch_size, seq_len, _ = x.size()
        x_flat = x.contiguous().view(-1, x.size(-1))
        x_proj = self.proj(x_flat)
        x_proj_seq = x_proj.view(batch_size, seq_len, -1)
        out, (h_n, c_n) = self.lstm(x_proj_seq)
        
        # 时间注意力
        attn_scores = torch.matmul(out, self.attention_vector)
        attn_weights = torch.softmax(attn_scores, dim=1)
        pooled = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)
        
        logits = self.fc(pooled)
        return {
            'proj_out': x_proj_seq,
            'lstm_out': out,
            'hidden_final': h_n,
            'pooled': pooled,
            'logits': logits,
            'attention_weights': attn_weights
        }

# ========== 迹归一化 ==========
def normalize_trace(covs):
    """迹归一化：C_norm = C / trace(C)"""
    traces = torch.diagonal(covs, dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    return covs / traces

# ========== 主训练循环 ==========
def train_and_evaluate(save_results=True):
    print("🚀 启动 FBCSP 黎曼 LSTM 训练与评估")
    print("💡 固定 1-10 号被试作为测试集，其余被试全部用于训练")
    
    if DEBUG_MODE:
        print("\n" + "=" * 60)
        print("📊 数据加载诊断")
        print("=" * 60)
    
    with open(INPUT_FILE, 'rb') as f:
        dataset = pickle.load(f)
    
    subjects_data = dataset['subject_data']
    num_subjects = len(subjects_data)
    
    # 新数据格式：[trials, windows, 5, 7, 7]
    num_bands = 5
    num_channels = 7
    TANGENT_DIM_PER_BAND = num_channels * (num_channels + 1) // 2  # 28
    TANGENT_DIM = num_bands * TANGENT_DIM_PER_BAND  # 140
    
    if DEBUG_MODE:
        print(f"\n数据集信息:")
        print(f"  被试数量: {num_subjects}")
        print(f"  频带数: {num_bands}")
        print(f"  协方差矩阵维度: {num_channels}x{num_channels}")
        print(f"  每个频带切空间维度: {TANGENT_DIM_PER_BAND}")
        print(f"  总切空间维度: {TANGENT_DIM}")
        
        for i, subj in enumerate(subjects_data):
            covs = subj['trajectories']
            labels = subj['labels']
            subj_key = subj.get('subject_key', f"Subject_{i+1:02d}")
            print(f"\n被试 {subj_key}:")
            print(f"  协方差矩阵: shape={covs.shape}")
            print(f"  标签: shape={labels.shape}, 分布={dict(zip(*np.unique(labels, return_counts=True)))} ")
    
    # 只对 1-10 号被试进行验证（CSV 清洗数据格式为 SUB_001）
    test_subject_indices = []
    test_subject_keys = []
    for i, subj in enumerate(subjects_data):
        subj_key = subj.get('subject_key', '')
        if subj_key.startswith("SUB_"):
            try:
                subj_num = int(subj_key.split("_")[-1])
                if 1 <= subj_num <= 10:
                    test_subject_indices.append(i)
                    test_subject_keys.append(subj_key)
            except ValueError:
                pass
    
    # 兜底：如果没有找到，就使用前10个被试
    if not test_subject_indices:
        test_subject_indices = list(range(min(10, num_subjects)))
        test_subject_keys = [subjects_data[i].get('subject_key', f"Subject_{i+1:02d}") for i in test_subject_indices]
    
    print(f"\n📋 验证被试数量: {len(test_subject_indices)}")
    print(f"   验证被试: {test_subject_keys}")

    # 用于保存结果的数据结构
    all_fold_results = []
    all_training_histories = []

    print("\n" + "=" * 40)

    # 划分数据
    train_covs_list, train_labels_list = [], []
    test_covs_list, test_labels_list = [], []
    test_trials_per_subject = []

    if INTRA_SUBJECT_TEST_RATIO > 0:
        print(f"🔄 划分数据 | 每个被试内划分 {INTRA_SUBJECT_TEST_RATIO*100:.0f}% 做测试")
        for i, subj_dict in enumerate(subjects_data):
            covs = subj_dict['trajectories']
            labels = subj_dict['labels']
            n_trials = covs.shape[0]
            n_test = max(1, int(n_trials * INTRA_SUBJECT_TEST_RATIO))
            indices = np.random.RandomState(42 + i).permutation(n_trials)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]
            train_covs_list.append(covs[train_idx])
            train_labels_list.append(labels[train_idx])
            test_covs_list.append(covs[test_idx])
            test_labels_list.append(labels[test_idx])
            test_trials_per_subject.append(len(test_idx))
        # 按原来的 test_subject_indices 过滤测试结果
        test_subject_indices = list(range(num_subjects))
        test_subject_keys = [subjects_data[i].get('subject_key', f"Subject_{i+1:02d}") for i in range(num_subjects)]
    else:
        print(f"🔄 划分数据 | 测试被试: {test_subject_keys}")
        for i, subj_dict in enumerate(subjects_data):
            if i in test_subject_indices:
                test_covs_list.append(subj_dict['trajectories'])
                test_labels_list.append(subj_dict['labels'])
                test_trials_per_subject.append(subj_dict['trajectories'].shape[0])
                if INCLUDE_TEST_IN_TRAIN:
                    train_covs_list.append(subj_dict['trajectories'])
                    train_labels_list.append(subj_dict['labels'])
            else:
                train_covs_list.append(subj_dict['trajectories'])
                train_labels_list.append(subj_dict['labels'])
    
    # 🔧 统一窗口数：不同被试的试次长度可能不同，截取最小公共窗口数
    all_covs = train_covs_list + test_covs_list
    min_windows = min(c.shape[1] for c in all_covs)
    train_covs_list = [c[:, :min_windows, :, :, :] for c in train_covs_list]
    test_covs_list = [c[:, :min_windows, :, :, :] for c in test_covs_list]
    
    train_covs = np.concatenate(train_covs_list, axis=0)
    train_labels = np.concatenate(train_labels_list, axis=0)
    test_covs = np.concatenate(test_covs_list, axis=0)
    test_labels = np.concatenate(test_labels_list, axis=0)
    
    if DEBUG_MODE:
        print(f"   统一窗口数: {min_windows}")
        print(f"\n📊 数据划分:")
        print(f"  训练集: 被试数={len(train_covs_list)}, trials={train_covs.shape[0]}, wins={train_covs.shape[1]}")
        print(f"  标签分布: {dict(zip(*np.unique(train_labels, return_counts=True)))} ")
        print(f"  测试集: 被试数={len(test_covs_list)}, trials={test_covs.shape[0]}")
        print(f"  标签分布: {dict(zip(*np.unique(test_labels, return_counts=True)))} ")
    
    trials_train, wins = train_covs.shape[0], train_covs.shape[1]
    trials_test = test_covs.shape[0]
        
    # 分别处理每个频带
    print(f"\n🌌 处理每个频带...")
    
    band_tangents_train = []
    band_tangents_test = []
    
    for b in range(num_bands):
        print(f"   频带 {b+1}/{num_bands}")
        
        # 提取单个频带的协方差矩阵
        # shape: [trials, wins, 7, 7]
        band_train = train_covs[:, :, b, :, :]
        band_test = test_covs[:, :, b, :, :]
        
        # 展平为 [trials*wins, 7, 7]
        t_train_band = torch.tensor(band_train.reshape(-1, num_channels, num_channels), dtype=torch.float64, device=device)
        t_test_band = torch.tensor(band_test.reshape(-1, num_channels, num_channels), dtype=torch.float64, device=device)
        
        # 迹归一化
        t_train_band = normalize_trace(t_train_band)
        t_test_band = normalize_trace(t_test_band)
        
        # 切空间映射
        train_band_tangent = log_euclidean_mapping(t_train_band, debug_name=f"band{b+1}")
        test_band_tangent = log_euclidean_mapping(t_test_band, debug_name=f"band{b+1}")
        
        band_tangents_train.append(train_band_tangent)
        band_tangents_test.append(test_band_tangent)
        
    # 拼接所有频带的特征
    train_tangent = torch.cat(band_tangents_train, dim=1)
    test_tangent = torch.cat(band_tangents_test, dim=1)
    
    if DEBUG_MODE:
        print(f"\n✅ 总特征: shape={train_tangent.shape}")
    
    # 恢复序列形状
    X_train = train_tangent.view(trials_train, wins, TANGENT_DIM)
    X_test = test_tangent.view(trials_test, wins, TANGENT_DIM)
    
    y_train = torch.tensor(train_labels, dtype=torch.long, device=device)
    y_test = torch.tensor(test_labels, dtype=torch.long, device=device)
    
    # Z-score 标准化
    if DEBUG_MODE:
        print(f"\n📊 LSTM 输入 (标准化前):")
        print(f"  X_train: shape={X_train.shape}, min={X_train.min():.3f}, max={X_train.max():.3f}")
    
    flat_X_train = X_train.view(-1, TANGENT_DIM)
    feat_mean = flat_X_train.mean(dim=0, keepdim=True)
    feat_std = flat_X_train.std(dim=0, keepdim=True) + 1e-8
    
    X_train = (X_train - feat_mean.unsqueeze(0)) / feat_std.unsqueeze(0)
    X_test = (X_test - feat_mean.unsqueeze(0)) / feat_std.unsqueeze(0)
    
    if DEBUG_MODE:
        print(f"🔧 标准化后: mean={X_train.mean().item():.4f}, std={X_train.std().item():.4f}")
        
    # 划分训练集/验证集 (90/10) 用于监控
    n_train = int(0.9 * len(X_train))
    indices = torch.randperm(len(X_train))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train_split = X_train[train_idx]
    y_train_split = y_train[train_idx]
    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    
    train_loader = DataLoader(TensorDataset(X_train_split, y_train_split), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=64, shuffle=False)
    
    # 初始化模型
    model = FBCSP_RiemannianLSTM(input_size=TANGENT_DIM, hidden_size=32, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    
    # 余弦退火调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    
    max_epochs = 100
    
    if DEBUG_MODE:
        print(f"\n🤖 开始训练:")
        print(f"  模型: FBCSP_RiemannianLSTM, input={TANGENT_DIM}, hidden=32")
        print(f"  训练集: {len(X_train_split)} | 验证集: {len(X_val)}")
        print(f"  优化器: AdamW + CosineAnnealingLR | 固定轮次: {max_epochs}")
        
        # 记录训练历史
        training_history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'epoch': [], 'lr': []}
        best_val_acc = 0.0
        best_epoch = 0
        class_names = ["Left", "Right", "Feet", "Hands"]
        
        for epoch in range(max_epochs):
            # --- Training ---
            model.train()
            epoch_loss = 0.0
            total_grad_norm = 0.0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                
                # 梯度范数监控
                batch_grad_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        batch_grad_norm += p.grad.data.norm(2).item() ** 2
                total_grad_norm += batch_grad_norm ** 0.5
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / len(train_loader)
            avg_grad_norm = total_grad_norm / len(train_loader)
            
            # --- Validation ---
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            val_class_correct = np.zeros(NUM_CLASSES)
            val_class_total = np.zeros(NUM_CLASSES)
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    outputs = model(batch_x)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == batch_y).sum().item()
                    val_total += batch_y.size(0)
                    
                    for cls in range(NUM_CLASSES):
                        cls_mask = (batch_y == cls)
                        if cls_mask.sum() > 0:
                            val_class_correct[cls] += (predicted[cls_mask] == cls).sum().item()
                            val_class_total[cls] += cls_mask.sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_acc = val_correct / val_total
            current_lr = optimizer.param_groups[0]['lr']
            
            # 更新学习率
            scheduler.step()
            
            # 记录历史
            training_history['train_loss'].append(avg_train_loss)
            training_history['val_loss'].append(avg_val_loss)
            training_history['val_acc'].append(val_acc)
            training_history['epoch'].append(epoch + 1)
            training_history['lr'].append(current_lr)
            
            # 每类验证准确率
            val_class_acc = val_class_correct / np.maximum(val_class_total, 1)
            
            # 最佳验证准确率跟踪
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
            
            class_acc_str = ", ".join([f"{class_names[cls]}:{val_class_acc[cls]*100:.1f}%" for cls in range(NUM_CLASSES)])
            print(f"    Epoch {epoch+1:3d}/{max_epochs}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}, val_acc={val_acc*100:.1f}% [{class_acc_str}], grad_norm={avg_grad_norm:.4f}, lr={current_lr:.2e}, best={best_val_acc*100:.1f}%@{best_epoch}")
    
    # 测试
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
    
    if DEBUG_MODE:
        print(f"\n📊 测试结果:")
        print(f"  预测: {dict(zip(*np.unique(predicted.cpu().numpy(), return_counts=True)))} ")
        print(f"  真实: {dict(zip(*np.unique(test_labels, return_counts=True)))} ")
        
        y_true = test_labels
        y_pred = predicted.cpu().numpy()
        class_names = ["Left", "Right", "Feet", "Hands"]
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        print(f"\n  混淆矩阵 (行=True, 列=Pred):")
        header = "      " + " ".join([f"{n[:3]:>4}" for n in class_names])
        print(f"  {header}")
        for i, row in enumerate(cm):
            print(f"  {class_names[i][:3]:>3}   " + " ".join([f"{v:>4}" for v in row]))
        
        # 各类详细指标
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(NUM_CLASSES)), zero_division=0)
        print(f"\n  各类详细指标:")
        for cls in range(NUM_CLASSES):
            cls_mask = (y_true == cls)
            cls_acc = (y_pred[cls_mask] == cls).sum() / cls_mask.sum() if cls_mask.sum() > 0 else 0
            cls_probs = test_probs_all[cls_mask]
            avg_conf = cls_probs[:, cls].mean() if len(cls_probs) > 0 else 0
            print(f"    {class_names[cls]:>5}: acc={cls_acc*100:.1f}%, precision={precision[cls]*100:.1f}%, recall={recall[cls]*100:.1f}%, f1={f1[cls]*100:.1f}%, avg_confidence={avg_conf:.3f}")
        
        # 错误分析
        incorrect_mask = (y_pred != y_true)
        if incorrect_mask.sum() > 0:
            print(f"\n  错误样本分析: {incorrect_mask.sum()}/{len(y_true)} 个错误")
            for cls in range(NUM_CLASSES):
                true_cls_mask = (y_true == cls) & incorrect_mask
                if true_cls_mask.sum() > 0:
                    wrong_preds = y_pred[true_cls_mask]
                    from collections import Counter
                    wrong_dist = Counter(wrong_preds)
                    wrong_str = ", ".join([f"{class_names[w]}:{c}" for w, c in wrong_dist.items()])
                    print(f"    {class_names[cls]} 被错分为: {wrong_str}")
    
    print(f"🎯 测试集整体准确率: {acc * 100:.2f}%")

    # 保存测试集完整概率和中间数据（用于可视化分类过程）
    test_probs_all = probs.cpu().numpy()

    # 拆分每个测试被试的结果
    start = 0
    for idx, subj_key in enumerate(test_subject_keys):
        n_trials = test_trials_per_subject[idx]
        end = start + n_trials
        subj_y_true = test_labels[start:end]
        subj_y_pred = predicted.cpu().numpy()[start:end]
        subj_probs = test_probs_all[start:end]
        subj_acc = accuracy_score(subj_y_true, subj_y_pred)

        fold_result = {
            'subject': subj_key,
            'accuracy': subj_acc,
            'y_true': subj_y_true,
            'y_pred': subj_y_pred,
            'probs': subj_probs,
            'train_subjects': len(train_covs_list),
            'train_trials': len(train_labels)
        }
        all_fold_results.append(fold_result)
        start = end

    all_training_histories.append(training_history)

    # 保存模型权重（用于中间层可视化）
    model_save_path = OUTPUT_DIR / "lstm_model_weights.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': TANGENT_DIM,
        'hidden_size': 32,
        'num_classes': NUM_CLASSES,
        'feat_mean': feat_mean.cpu(),
        'feat_std': feat_std.cpu(),
        'min_windows': min_windows,
    }, model_save_path)
    print(f"💾 模型权重已保存到: {model_save_path}")
    
    # 汇总结果
    subject_accuracies = [r['accuracy'] for r in all_fold_results]
    mean_acc = np.mean(subject_accuracies)
    std_acc = np.std(subject_accuracies)
    
    class_names = ["Left", "Right", "Feet", "Hands"]
    print("\n" + "=" * 40)
    print("🏆 最终结果")
    print(f"  训练被试数: {len(train_covs_list)}")
    print(f"  训练样本数: {len(train_labels)}")
    print(f"  测试被试数: {len(test_subject_indices)}")
    print(f"  测试样本数: {len(test_labels)}")
    print(f"  测试集整体准确率: {acc*100:.2f}%")
    print(f"  各测试被试准确率: {[f'{a*100:.1f}%' for a in subject_accuracies]}")
    print(f"  测试被试平均准确率: {mean_acc*100:.2f}% ± {std_acc*100:.2f}%")
    print(f"  最高/最低被试准确率: {max(subject_accuracies)*100:.1f}% / {min(subject_accuracies)*100:.1f}%")
    
    print(f"\n  逐被试详细结果:")
    for idx, subj_key in enumerate(test_subject_keys):
        subj_res = all_fold_results[idx]
        subj_y_true = subj_res['y_true']
        subj_y_pred = subj_res['y_pred']
        print(f"    {subj_key}: acc={subj_res['accuracy']*100:.1f}%, n_trials={len(subj_y_true)}")
        for cls in range(NUM_CLASSES):
            cls_mask = (subj_y_true == cls)
            if cls_mask.sum() > 0:
                cls_acc = (subj_y_pred[cls_mask] == cls).sum() / cls_mask.sum()
                print(f"      {class_names[cls]}: {cls_acc*100:.1f}% ({cls_mask.sum()} trials)")
    
    # 保存所有结果用于可视化
    if save_results:
        final_results = {
            'subject_results': all_fold_results,
            'training_histories': all_training_histories,
            'overall_accuracy': float(acc),
            'subject_accuracies': [float(a) for a in subject_accuracies],
            'mean_accuracy': float(mean_acc),
            'std_accuracy': float(std_acc),
            'test_subjects': test_subject_keys,
            'data_info': {
                'num_train_subjects': len(subjects_data) - len(test_subject_indices),
                'num_test_subjects': len(test_subject_indices),
                'num_bands': num_bands,
                'num_channels': num_channels,
                'tangent_dim': TANGENT_DIM
            }
        }
        output_file = OUTPUT_DIR / "classification_results_full.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(final_results, f)
        print(f"\n💾 完整结果已保存到: {output_file}")
    
    return final_results


# 兼容旧调用
train_and_evaluate_loso = train_and_evaluate

if __name__ == "__main__":
    train_and_evaluate(save_results=True)