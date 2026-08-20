# -*- coding: utf-8 -*-
"""
动态流形曲线可分类性能全面分析
输出所有评价图到 figures/ 目录
"""

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from scipy.spatial.distance import cdist
from alignment import riemannian_distance, normalize_trace_np, log_euclidean_mapping_np
from Classification import FBCSP_RiemannianLSTM
import warnings
warnings.filterwarnings('ignore')

# ========== 中文显示配置 ==========
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
TRAJ_FILE = Path("trajectory_data") / "trajectories_500ms_win.pkl"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 4
CLASS_NAMES = ["左手", "右手", "双脚", "双手"]
BAND_NAMES = ["8-12μ", "12-14μ", "14-20μ", "20-25μ", "25-30μ"]

# ==================== 辅助函数 ====================

def load_data():
    with open(TRAJ_FILE, 'rb') as f:
        data = pickle.load(f)
    subjects_data = data['subject_data']
    metadata = data['metadata']
    return subjects_data, metadata

def split_train_test(subjects_data):
    train_subjects, test_subjects = [], []
    train_keys, test_keys = [], []
    for subj in subjects_data:
        key = subj.get('subject_key', '')
        num = int(key.split("_")[-1]) if key.startswith("SUB_") else 999
        if 1 <= num <= 10:
            test_subjects.append(subj)
            test_keys.append(key)
        else:
            train_subjects.append(subj)
            train_keys.append(key)
    return train_subjects, test_subjects, train_keys, test_keys

# （黎曼几何工具函数已从 alignment.py 导入；FBCSP_RiemannianLSTM 从 Classification.py 导入）

def flatten_covs_to_tangent(covs):
    """
    covs: [trials, wins, bands, 7, 7]
    返回: [trials, wins, 140]
    """
    n_trials, n_wins, n_bands, C, _ = covs.shape
    out = np.zeros((n_trials, n_wins, n_bands * (C*(C+1)//2)))
    for w in range(n_wins):
        band_feats = []
        for b in range(n_bands):
            bcovs = normalize_trace_np(covs[:, w, b, :, :])
            band_feats.append(log_euclidean_mapping_np(bcovs))
        out[:, w, :] = np.concatenate(band_feats, axis=1)
    return out

def fisher_ratio(X, y):
    """X: [N, D], y: [N]  返回标量 J"""
    classes = np.unique(y)
    overall_mean = X.mean(axis=0)
    SB, SW = 0, 0
    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        if nc == 0:
            continue
        mc = Xc.mean(axis=0)
        SB += nc * np.outer(mc - overall_mean, mc - overall_mean)
        SW += np.cov(Xc.T, bias=True) * nc
    return np.trace(SB) / (np.trace(SW) + 1e-12)

# （FBCSP_RiemannianLSTM 模型定义已从 Classification.py 导入）

# ==================== 图1: 逐窗口黎曼距离分析 ====================

def plot_riemann_distance_profile(test_covs, test_labels):
    """
    test_covs: [N, W, B, 7, 7]
    画每个窗口的类内平均距离 vs 类间平均距离
    """
    N, W, B, C, _ = test_covs.shape
    # 先对频带取平均（简化）
    mean_covs = normalize_trace_np(test_covs.mean(axis=2))  # [N, W, 7, 7]

    intra_mean, inter_mean = [], []
    for w in range(W):
        covs_w = mean_covs[:, w, :, :]
        intra_dists, inter_dists = [], []
        for i in range(N):
            for j in range(i+1, N):
                d = riemannian_distance(covs_w[i], covs_w[j])
                if np.isnan(d):
                    continue
                if test_labels[i] == test_labels[j]:
                    intra_dists.append(d)
                else:
                    inter_dists.append(d)
        intra_mean.append(np.mean(intra_dists) if intra_dists else 0)
        inter_mean.append(np.mean(inter_dists) if inter_dists else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(intra_mean, 'o-', label='类内平均黎曼距离', color='C0')
    ax.plot(inter_mean, 's-', label='类间平均黎曼距离', color='C1')
    ax.fill_between(range(W), intra_mean, alpha=0.3, color='C0')
    ax.fill_between(range(W), inter_mean, alpha=0.3, color='C1')
    ax.set_xlabel("时间窗口")
    ax.set_ylabel("平均黎曼距离")
    ax.set_title("逐窗口流形距离分析（测试集）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_riemann_distance_profile.png", dpi=200)
    plt.close(fig)
    print("✅ 图1已保存: 01_riemann_distance_profile.png")

# ==================== 图2: 逐窗口 Fisher Ratio ====================

def plot_fisher_ratio(test_covs, test_labels):
    N, W, B, C, _ = test_covs.shape
    X_tangent = flatten_covs_to_tangent(test_covs)  # [N, W, 140]
    frs = [fisher_ratio(X_tangent[:, w, :], test_labels) for w in range(W)]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(W), frs, color='steelblue', edgecolor='k')
    ax.set_xlabel("时间窗口")
    ax.set_ylabel("Fisher Ratio (tr(Sb)/tr(Sw))")
    ax.set_title("逐窗口切空间特征可分离性 (Fisher Ratio)")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_fisher_ratio.png", dpi=200)
    plt.close(fig)
    print("✅ 图2已保存: 02_fisher_ratio.png")
    return frs

# ==================== 图3: 时间-特征热力图 ====================

def plot_feature_heatmap(test_covs, test_labels):
    X_tangent = flatten_covs_to_tangent(test_covs)  # [N, W, 140]
    W = X_tangent.shape[1]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    for cls in range(NUM_CLASSES):
        mask = test_labels == cls
        if mask.sum() == 0:
            continue
        mean_feat = X_tangent[mask].mean(axis=0)  # [W, 140]
        # 取绝对值并归一化每行便于观察
        mean_feat = np.abs(mean_feat)
        mean_feat = (mean_feat - mean_feat.min()) / (mean_feat.max() - mean_feat.min() + 1e-12)

        ax = axes[cls]
        im = ax.imshow(mean_feat, aspect='auto', cmap='viridis', origin='lower')
        ax.set_title(f"{CLASS_NAMES[cls]} 平均特征热力图")
        ax.set_xlabel("特征维度 (5频带×28)")
        ax.set_ylabel("时间窗口")
        # 标记频带边界
        for b in range(1, 5):
            ax.axvline(x=b*28-0.5, color='white', linestyle='--', linewidth=0.8)
        plt.colorbar(im, ax=ax)
    plt.suptitle("逐类别切空间特征时间演化", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_feature_heatmap.png", dpi=200)
    plt.close(fig)
    print("✅ 图3已保存: 03_feature_heatmap.png")

# ==================== 图4: t-SNE 动态演化 ====================

def plot_tsne_evolution(test_covs, test_labels):
    X_tangent = flatten_covs_to_tangent(test_covs)  # [N, W, 140]
    W = X_tangent.shape[1]
    # 选5个关键窗口
    idxs = np.linspace(0, W-1, min(5, W), dtype=int)

    fig, axes = plt.subplots(1, len(idxs), figsize=(18, 4))
    if len(idxs) == 1:
        axes = [axes]
    colors = plt.cm.tab10(np.arange(NUM_CLASSES))

    for ax, w in zip(axes, idxs):
        Xw = X_tangent[:, w, :]
        # 用PCA预处理加速
        tsne = TSNE(n_components=2, perplexity=min(30, Xw.shape[0]-1), random_state=42, init='pca')
        emb = tsne.fit_transform(Xw)
        for cls in range(NUM_CLASSES):
            mask = test_labels == cls
            ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[cls]], label=CLASS_NAMES[cls], s=30, alpha=0.7)
        ax.set_title(f"窗口 {w}")
        ax.set_xticks([])
        ax.set_yticks([])
    axes[0].legend(loc='best', fontsize=8)
    plt.suptitle("切空间特征 t-SNE 动态演化（测试集）", fontsize=14)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_tsne_evolution.png", dpi=200)
    plt.close(fig)
    print("✅ 图4已保存: 04_tsne_evolution.png")

# ==================== 图5: 流形轨迹速度 ====================

def plot_riemann_speed(test_covs, test_labels):
    N, W, B, C, _ = test_covs.shape
    mean_covs = normalize_trace_np(test_covs.mean(axis=2))

    speeds = np.zeros((NUM_CLASSES, W-1))
    counts = np.zeros(NUM_CLASSES)
    for i in range(N):
        cls = test_labels[i]
        for w in range(W-1):
            d = riemannian_distance(mean_covs[i, w], mean_covs[i, w+1])
            if not np.isnan(d):
                speeds[cls, w] += d
                counts[cls] += 1 if w == 0 else 0  # 只计数一次
        # 修正：每trial独立计数
    # 重新计算：每个类别分别平均
    speeds = np.zeros((NUM_CLASSES, W-1))
    for cls in range(NUM_CLASSES):
        mask = test_labels == cls
        cls_speeds = []
        for i in np.where(mask)[0]:
            trial_speed = []
            for w in range(W-1):
                d = riemannian_distance(mean_covs[i, w], mean_covs[i, w+1])
                trial_speed.append(d if not np.isnan(d) else 0)
            cls_speeds.append(trial_speed)
        if cls_speeds:
            speeds[cls] = np.mean(cls_speeds, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in range(NUM_CLASSES):
        ax.plot(speeds[cls], 'o-', label=CLASS_NAMES[cls])
    ax.set_xlabel("时间窗口过渡")
    ax.set_ylabel("黎曼速度 (a.u.)")
    ax.set_title("流形轨迹速度曲线（测试集）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_riemann_speed.png", dpi=200)
    plt.close(fig)
    print("✅ 图5已保存: 05_riemann_speed.png")

# ==================== 图6: 流形轨迹曲率 ====================

def plot_riemann_curvature(test_covs, test_labels):
    N, W, B, C, _ = test_covs.shape
    mean_covs = normalize_trace_np(test_covs.mean(axis=2))

    curvatures = np.zeros((NUM_CLASSES, W-2))
    for cls in range(NUM_CLASSES):
        mask = test_labels == cls
        cls_curv = []
        for i in np.where(mask)[0]:
            trial_speed = []
            for w in range(W-1):
                d = riemannian_distance(mean_covs[i, w], mean_covs[i, w+1])
                trial_speed.append(d if not np.isnan(d) else 0)
            trial_curv = np.abs(np.diff(trial_speed))
            cls_curv.append(trial_curv)
        if cls_curv:
            curvatures[cls] = np.mean(cls_curv, axis=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    for cls in range(NUM_CLASSES):
        ax.plot(curvatures[cls], 's-', label=CLASS_NAMES[cls])
    ax.set_xlabel("时间窗口")
    ax.set_ylabel("黎曼曲率 (速度变化率)")
    ax.set_title("流形轨迹曲率曲线（测试集）")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "06_riemann_curvature.png", dpi=200)
    plt.close(fig)
    print("✅ 图6已保存: 06_riemann_curvature.png")

# ==================== 图7: 终点散布（最后窗口） ====================

def plot_endpoint_dispersion(test_covs, test_labels):
    N, W, B, C, _ = test_covs.shape
    last_covs = normalize_trace_np(test_covs[:, -1, :, :, :].mean(axis=1))  # [N, 7, 7]

    # 计算各类黎曼均值（简化欧氏均值）
    class_means = []
    for cls in range(NUM_CLASSES):
        mask = test_labels == cls
        if mask.sum() == 0:
            class_means.append(None)
            continue
        class_means.append(np.mean(last_covs[mask], axis=0))

    # 类内散度
    intra_disp = []
    for cls in range(NUM_CLASSES):
        mask = test_labels == cls
        if mask.sum() == 0 or class_means[cls] is None:
            intra_disp.append(0)
            continue
        dists = [riemannian_distance(c, class_means[cls]) for c in last_covs[mask]]
        intra_disp.append(np.nanmean(dists))

    # 类间中心距
    inter_disp = []
    for i in range(NUM_CLASSES):
        for j in range(i+1, NUM_CLASSES):
            if class_means[i] is None or class_means[j] is None:
                continue
            inter_disp.append(riemannian_distance(class_means[i], class_means[j]))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.bar(CLASS_NAMES, intra_disp, color='coral', edgecolor='k')
    ax1.set_ylabel("平均黎曼距离")
    ax1.set_title("终点类内散度（最后窗口）")
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(range(len(inter_disp)), sorted(inter_disp, reverse=True), color='steelblue', edgecolor='k')
    ax2.set_ylabel("黎曼距离")
    ax2.set_title("各类别中心间距离")
    ax2.set_xlabel("类别对")
    ax2.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_endpoint_dispersion.png", dpi=200)
    plt.close(fig)
    print("✅ 图7已保存: 07_endpoint_dispersion.png")

# ==================== 图8: 逐窗口分类准确率（静态分类器） ====================

def plot_window_wise_accuracy(train_covs, train_labels, test_covs, test_labels):
    X_train = flatten_covs_to_tangent(train_covs)  # [N, W, 140]
    X_test = flatten_covs_to_tangent(test_covs)
    W = X_train.shape[1]

    accs = []
    for w in range(W):
        clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
        clf.fit(X_train[:, w, :], train_labels)
        pred = clf.predict(X_test[:, w, :])
        accs.append(accuracy_score(test_labels, pred))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(accs, 'o-', color='darkgreen', markersize=6)
    ax.axhline(y=np.max(accs), color='red', linestyle='--', label=f"最高准确率: {np.max(accs)*100:.1f}%")
    ax.set_xlabel("时间窗口")
    ax.set_ylabel("分类准确率")
    ax.set_title("逐窗口静态分类准确率 (LogisticRegression)")
    ax.set_ylim(0, 1.0)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_window_wise_accuracy.png", dpi=200)
    plt.close(fig)
    print("✅ 图8已保存: 08_window_wise_accuracy.png")
    return accs

# ==================== LSTM 训练与高级分析 ====================

def prepare_lstm_data(train_covs, train_labels, test_covs, test_labels):
    """准备LSTM的140维时序输入"""
    N_train, W, B, C, _ = train_covs.shape
    N_test = test_covs.shape[0]

    # 训练集
    train_tangent = []
    for b in range(B):
        band_train = normalize_trace_np(train_covs[:, :, b, :, :])
        N, W_local, C, _ = band_train.shape
        flat = band_train.reshape(-1, C, C)
        # log映射
        log_flat = log_euclidean_mapping_np(flat)
        train_tangent.append(log_flat.reshape(N, W_local, -1))
    X_train = np.concatenate(train_tangent, axis=2)  # [N, W, 140]

    # 测试集
    test_tangent = []
    for b in range(B):
        band_test = normalize_trace_np(test_covs[:, :, b, :, :])
        N, W_local, C, _ = band_test.shape
        flat = band_test.reshape(-1, C, C)
        log_flat = log_euclidean_mapping_np(flat)
        test_tangent.append(log_flat.reshape(N, W_local, -1))
    X_test = np.concatenate(test_tangent, axis=2)

    # Z-score (基于训练集)
    flat_train = X_train.reshape(-1, X_train.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(train_labels, dtype=torch.long, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_t = torch.tensor(test_labels, dtype=torch.long, device=device)

    return X_train_t, y_train_t, X_test_t, y_test_t, mean, std

def train_lstm(X_train, y_train, X_test, y_test):
    model = FBCSP_RiemannianLSTM(input_size=140, hidden_size=32, num_classes=NUM_CLASSES).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.05)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

    # 90/10 训练验证划分
    n = len(X_train)
    idx = torch.randperm(n)
    n_train = int(0.9 * n)
    tr_idx, val_idx = idx[:n_train], idx[n_train:]

    train_loader = DataLoader(TensorDataset(X_train[tr_idx], y_train[tr_idx]), batch_size=64, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_train[val_idx], y_train[val_idx]), batch_size=64)

    print("\n🤖 训练 LSTM...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        for bx, by in train_loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        # val
        model.eval()
        with torch.no_grad():
            corr = 0
            tot = 0
            for bx, by in val_loader:
                out = model(bx)
                _, pred = torch.max(out, 1)
                corr += (pred == by).sum().item()
                tot += by.size(0)
        if (epoch+1) % 2 == 0:
            print(f"  Epoch {epoch+1}/10  loss={total_loss/len(train_loader):.4f}  val_acc={corr/tot:.3f}")

    # 测试
    model.eval()
    with torch.no_grad():
        outputs, attn_weights = model(X_test, return_attention=True)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs, 1)
        acc = accuracy_score(y_test.cpu().numpy(), predicted.cpu().numpy())
        print(f"🎯 LSTM 测试准确率: {acc*100:.2f}%")

    return model, predicted, probs, attn_weights, acc

# ==================== 图9: 注意力权重 + Fisher Ratio 叠图 ====================

def plot_attention_fisher(attn_weights, fisher_ratios, test_labels):
    W = len(fisher_ratios)
    attn_mean = attn_weights.cpu().numpy().mean(axis=0)  # [W]
    # 各类别的注意力
    attn_per_class = []
    for cls in range(NUM_CLASSES):
        mask = test_labels == cls
        if mask.sum() > 0:
            attn_per_class.append(attn_weights.cpu().numpy()[mask].mean(axis=0))
        else:
            attn_per_class.append(np.zeros(W))

    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax2 = ax1.twinx()

    ax1.bar(range(W), fisher_ratios, alpha=0.4, color='gray', label='Fisher Ratio')
    ax1.set_xlabel("时间窗口")
    ax1.set_ylabel("Fisher Ratio", color='gray')
    ax1.tick_params(axis='y', labelcolor='gray')

    for cls in range(NUM_CLASSES):
        ax2.plot(attn_per_class[cls], 'o-', label=f"{CLASS_NAMES[cls]} 注意力", linewidth=2)
    ax2.set_ylabel("注意力权重", color='black')
    ax2.tick_params(axis='y', labelcolor='black')
    ax2.legend(loc='upper left')
    ax1.set_title("时间注意力权重 vs Fisher Ratio")
    ax1.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "09_attention_fisher.png", dpi=200)
    plt.close(fig)
    print("✅ 图9已保存: 09_attention_fisher.png")

# ==================== 图10: 中间层特征 t-SNE ====================

def plot_hidden_tsne(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        h = model.proj(X_test)  # [B, T, 64]
        lstm_out, _ = model.lstm(h)  # [B, T, 64]
        # 取最后一个时间步作为代表
        last_hidden = lstm_out[:, -1, :].cpu().numpy()

    tsne = TSNE(n_components=2, perplexity=min(30, last_hidden.shape[0]-1), random_state=42, init='pca')
    emb = tsne.fit_transform(last_hidden)
    y = y_test.cpu().numpy()

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = plt.cm.tab10(np.arange(NUM_CLASSES))
    for cls in range(NUM_CLASSES):
        mask = y == cls
        ax.scatter(emb[mask, 0], emb[mask, 1], c=[colors[cls]], label=CLASS_NAMES[cls], s=50, alpha=0.8, edgecolors='k', linewidth=0.5)
    ax.set_title("LSTM 中间层特征 t-SNE (最后时间步)")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_hidden_tsne.png", dpi=200)
    plt.close(fig)
    print("✅ 图10已保存: 10_hidden_tsne.png")

# ==================== 图11: 静态基线 vs 动态序列 ====================

def plot_static_vs_dynamic(X_train, y_train, X_test, y_test, lstm_acc):
    # 静态：只用中间窗口
    mid = X_train.shape[1] // 2
    clf = LogisticRegression(max_iter=1000, C=1.0, multi_class='multinomial')
    clf.fit(X_train[:, mid, :].cpu().numpy(), y_train.cpu().numpy())
    static_pred = clf.predict(X_test[:, mid, :].cpu().numpy())
    static_acc = accuracy_score(y_test.cpu().numpy(), static_pred)

    # 动态：LSTM已训练好，取之前结果
    fig, ax = plt.subplots(figsize=(6, 5))
    methods = ["静态单窗口\n(Logistic)", "动态序列\n(LSTM)"]
    accs = [static_acc, lstm_acc]
    bars = ax.bar(methods, accs, color=['coral', 'steelblue'], edgecolor='k', width=0.5)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("准确率")
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{a*100:.1f}%", ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax.set_title("静态基线 vs 动态序列分类")
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    fig.savefig(FIG_DIR / "11_static_vs_dynamic.png", dpi=200)
    plt.close(fig)
    print("✅ 图11已保存: 11_static_vs_dynamic.png")

# ==================== 图12: 混淆矩阵 + 逐被试准确率 ====================

def plot_confusion_and_subject_acc(y_true, y_pred, test_subjects_data, test_keys):
    # 混淆矩阵
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax1)
    ax1.set_title("混淆矩阵 (LSTM)")
    ax1.set_ylabel("真实标签")
    ax1.set_xlabel("预测标签")

    # 逐被试准确率
    accs = []
    names = []
    start = 0
    for subj, key in zip(test_subjects_data, test_keys):
        n = len(subj['labels'])
        end = start + n
        yt = y_true[start:end]
        yp = y_pred[start:end]
        accs.append(accuracy_score(yt, yp))
        names.append(key)
        start = end

    ax2.barh(range(len(names)), accs, color='steelblue', edgecolor='k')
    ax2.set_yticks(range(len(names)))
    ax2.set_yticklabels(names)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("准确率")
    ax2.set_title("逐被试测试准确率")
    ax2.axvline(x=np.mean(accs), color='red', linestyle='--', label=f"平均: {np.mean(accs)*100:.1f}%")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    for i, a in enumerate(accs):
        ax2.text(a + 0.01, i, f"{a*100:.1f}%", va='center', fontsize=9)

    plt.tight_layout()
    fig.savefig(FIG_DIR / "12_confusion_subject_acc.png", dpi=200)
    plt.close(fig)
    print("✅ 图12已保存: 12_confusion_subject_acc.png")

# ==================== 主流程 ====================

def main():
    print("=" * 60)
    print("动态流形曲线可分类性能全面分析")
    print("=" * 60)

    subjects_data, metadata = load_data()
    train_subjects, test_subjects, train_keys, test_keys = split_train_test(subjects_data)

    # 拼接 train/test 的 covs 和 labels
    all_subjects = train_subjects + test_subjects
    min_wins = min(s['trajectories'].shape[1] for s in all_subjects)
    print(f"统一窗口数: {min_wins}（截断前范围: {[s['trajectories'].shape[1] for s in all_subjects]}）")

    for s in all_subjects:
        s['trajectories'] = s['trajectories'][:, :min_wins, :, :, :]

    train_covs = np.concatenate([s['trajectories'] for s in train_subjects], axis=0)
    train_labels = np.concatenate([s['labels'] for s in train_subjects], axis=0)
    test_covs = np.concatenate([s['trajectories'] for s in test_subjects], axis=0)
    test_labels = np.concatenate([s['labels'] for s in test_subjects], axis=0)

    print(f"训练集: {train_covs.shape[0]} trials, 测试集: {test_covs.shape[0]} trials")
    print(f"窗口数: {test_covs.shape[1]}, 频带数: {test_covs.shape[2]}")

    # ===== 前8张图（不依赖LSTM训练）=====
    plot_riemann_distance_profile(test_covs, test_labels)
    frs = plot_fisher_ratio(test_covs, test_labels)
    plot_feature_heatmap(test_covs, test_labels)
    plot_tsne_evolution(test_covs, test_labels)
    plot_riemann_speed(test_covs, test_labels)
    plot_riemann_curvature(test_covs, test_labels)
    plot_endpoint_dispersion(test_covs, test_labels)
    plot_window_wise_accuracy(train_covs, train_labels, test_covs, test_labels)

    # ===== LSTM 训练 + 后4张图 =====
    X_train_t, y_train_t, X_test_t, y_test_t, _, _ = prepare_lstm_data(train_covs, train_labels, test_covs, test_labels)
    model, predicted, probs, attn_weights, lstm_acc = train_lstm(X_train_t, y_train_t, X_test_t, y_test_t)

    plot_attention_fisher(attn_weights, frs, test_labels)
    plot_hidden_tsne(model, X_test_t, y_test_t)
    plot_static_vs_dynamic(X_train_t, y_train_t, X_test_t, y_test_t, lstm_acc)
    plot_confusion_and_subject_acc(y_test_t.cpu().numpy(), predicted.cpu().numpy(), test_subjects, test_keys)

    print(f"\n🎉 全部完成！图片保存在: {FIG_DIR.absolute()}")

if __name__ == "__main__":
    main()