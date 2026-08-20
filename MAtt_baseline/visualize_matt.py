# -*- coding: utf-8 -*-
"""
MAtt 算法全流程可视化（15 种图，每种独立一张）
读取 run_matt_physionetmi.py 保存的 npz 数据，生成：
  01: 训练曲线 (Loss + Acc)
  02: 原始 EEG 信号
  03: 卷积特征
  04: SPD 矩阵
  05: SPD 特征值谱
  06: 注意力热力图
  07: 逐类别注意力模式
  08: 类内/类间 Log-Euclidean 距离分布
  09: t-SNE 流形投影
  10: 混淆矩阵
  11: 头皮 Topomap
  12: 时频频谱图
  13: 梯度显著性图
  14: 通道重要性排序
  15: Silhouette 曲线
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import torch

# ---------- 中文支持 ----------
_font_paths = [
    'C:/Windows/Fonts/simhei.ttf',
    'C:/Windows/Fonts/msyh.ttc',
    'C:/Windows/Fonts/simsun.ttc',
]
for _fp in _font_paths:
    if os.path.exists(_fp):
        try:
            from matplotlib.font_manager import FontProperties
            _font = FontProperties(fname=_fp)
            plt.rcParams['font.family'] = _font.get_name()
            break
        except Exception:
            pass
plt.rcParams['axes.unicode_minus'] = False

# ---------- 路径 ----------
OUTPUT_DIR = Path(__file__).resolve().parent / 'results' / 'figures'
CLASS_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
CLASS_NAMES = ['左手', '右手', '双脚', '舌头/双手']

_CHANNEL_POS = {
    'Fz': (0.0, 0.75), 'FCz': (0.0, 0.50),
    'FC1': (-0.22, 0.50), 'FC2': (0.22, 0.50),
    'FC3': (-0.43, 0.45), 'FC4': (0.43, 0.45),
    'Cz': (0.0, 0.0), 'C1': (-0.22, 0.0), 'C2': (0.22, 0.0),
    'C3': (-0.45, 0.0), 'C4': (0.45, 0.0),
    'C5': (-0.70, 0.0), 'C6': (0.70, 0.0),
    'CPz': (0.0, -0.50), 'CP1': (-0.22, -0.50), 'CP2': (0.22, -0.50),
    'CP3': (-0.43, -0.45), 'CP4': (0.43, -0.45),
    'Pz': (0.0, -0.75), 'P1': (-0.22, -0.75), 'P2': (0.22, -0.75),
    'POz': (0.0, -0.90)
}


def load_data(data_npz):
    data = np.load(data_npz, allow_pickle=True)
    return {
        'history': {
            'train_loss': data['history_train_loss'].tolist(),
            'val_loss': data['history_val_loss'].tolist(),
            'train_acc': data['history_train_acc'].tolist(),
            'val_acc': data['history_val_acc'].tolist(),
            'silhouette': data.get('history_silhouette', np.array([])).tolist(),
        },
        'test_acc': float(data['test_acc']),
        'title_suffix': str(data['title_suffix']),
        'xb': torch.from_numpy(data['xb']),
        'yb': torch.from_numpy(data['yb']),
        'x2': torch.from_numpy(data['x2']),
        'spd_matrices': torch.from_numpy(data['spd_matrices']),
        'atten_prob': torch.from_numpy(data['atten_prob']),
        'logits': torch.from_numpy(data['logits']),
        'all_x_flat': torch.from_numpy(data['all_x_flat']),
        'all_y_true': torch.from_numpy(data['all_y_true']),
        'all_y_pred': torch.from_numpy(data['all_y_pred']),
        'all_spd': torch.from_numpy(data['all_spd']),
        'all_attn': torch.from_numpy(data['all_attn']),
        'conv1_weight': torch.from_numpy(data['conv1_weight']),
        'channel_names': data['channel_names'],
        'sfreq': float(data['sfreq']),
    }


_CURRENT_CRITERION = 'unknown'


def save_fig(fig, name):
    p = OUTPUT_DIR / _CURRENT_CRITERION / name
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  ✅ {p}')


# ==================== 01: 训练曲线 ====================
def fig01_training_curves(D):
    his = D['history']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    its = range(1, len(his['train_loss'])+1)
    ax1.plot(its, his['train_loss'], 'b-', alpha=0.6, label='训练 Loss')
    ax1.plot(its, his['val_loss'], 'r-', alpha=0.6, label='验证 Loss')
    ax1.set_title('Loss 曲线'); ax1.set_xlabel('迭代'); ax1.set_ylabel('Loss')
    ax1.legend(); ax1.grid(True, alpha=0.3)
    ax2.plot(its, his['train_acc'], 'b-', alpha=0.6, label='训练 Acc')
    ax2.plot(its, his['val_acc'], 'r-', alpha=0.6, label='验证 Acc')
    ax2.set_title('准确率曲线'); ax2.set_xlabel('迭代'); ax2.set_ylabel('准确率')
    ax2.legend(); ax2.grid(True, alpha=0.3)
    fig.suptitle(f'01 训练曲线\n{D["title_suffix"]}\n测试准确率: {D["test_acc"]*100:.2f}%', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '01_training_curves.png')


# ==================== 02: 原始 EEG ====================
def fig02_raw_eeg(D):
    x_np = D['xb'].squeeze(1).numpy()
    n = min(4, x_np.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    ch_idx = [0, 3, 7, 10, 14, 18, 21]
    for i, ax in enumerate(axes):
        for ch in ch_idx:
            ax.plot(x_np[i, ch, :], alpha=0.7, lw=0.6)
        cls = int(D['yb'][i])
        ax.set_ylabel(f'样本{i+1}\n{CLASS_NAMES[cls]}')
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel('时间点')
    axes[0].set_title('原始 EEG 信号（22通道中取7条典型通道）')

    # 右侧/下方加图例说明颜色对应的类别
    handles = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=2, label=CLASS_NAMES[c]) for c in range(4)]
    fig.legend(handles, CLASS_NAMES, loc='upper right', title='类别', fontsize=8)

    fig.suptitle(f'02 原始 EEG 信号\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '02_raw_eeg.png')


# ==================== 03: 卷积特征 ====================
def fig03_conv_features(D):
    x2_np = D['x2'].squeeze(2).numpy()
    n = min(4, x2_np.shape[0])
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.5 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        for ch in range(0, 20, 4):
            ax.plot(x2_np[i, ch, :], alpha=0.7, lw=0.6)
        cls = int(D['yb'][i])
        ax.set_ylabel(f'样本{i+1}\n{CLASS_NAMES[cls]}')
        ax.tick_params(labelsize=8)
    axes[-1].set_xlabel('时间点')
    axes[0].set_title('卷积特征提取后（20通道中取5条典型通道）')

    handles = [plt.Line2D([], [], color=CLASS_COLORS[c], lw=2, label=CLASS_NAMES[c]) for c in range(4)]
    fig.legend(handles, CLASS_NAMES, loc='upper right', title='类别', fontsize=8)

    fig.suptitle(f'03 卷积特征\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '03_conv_features.png')


# ==================== 04: SPD 矩阵 ====================
def fig04_spd_matrices(D):
    spd = D['spd_matrices'][0].numpy()
    fig, axes = plt.subplots(1, min(4, spd.shape[0]), figsize=(14, 4))
    if spd.shape[0] == 1:
        axes = [axes]
    vmin = min(spd.min(), -1); vmax = max(spd.max(), 1)
    for e, ax in enumerate(axes):
        im = ax.imshow(spd[e], cmap='RdBu_r', aspect='equal', vmin=vmin, vmax=vmax)
        ax.set_title(f'片段 {e+1}')
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'04 SPD 协方差矩阵\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '04_spd_matrices.png')


# ==================== 05: SPD 特征值谱 ====================
def fig05_spd_eigenvalues(D):
    spd = D['spd_matrices'][0].numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    for e in range(spd.shape[0]):
        ev = np.sort(np.linalg.eigvalsh(spd[e]))[::-1]
        ax.plot(range(1, len(ev)+1), ev, 'o-', lw=1.5, ms=4, label=f'片段 {e+1}')
    ax.set_xlabel('特征值序号'); ax.set_ylabel('特征值'); ax.set_title('SPD 矩阵特征值衰减曲线')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f'05 SPD 特征值谱\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '05_spd_eigenvalues.png')


# ==================== 06: 注意力热力图 ====================
def fig06_attention_heatmap(D):
    attn = D['atten_prob'][0].numpy()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(attn, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
    ax.set_title('注意力权重矩阵'); ax.set_xlabel('片段 j'); ax.set_ylabel('片段 i')
    for i in range(attn.shape[0]):
        for j in range(attn.shape[1]):
            ax.text(j, i, f'{attn[i,j]:.2f}', ha='center', va='center', fontsize=9,
                    color='black' if attn[i,j] < 0.6 else 'white')
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'06 注意力热力图\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '06_attention_heatmap.png')


# ==================== 07: 逐类别注意力模式 ====================
def fig07_per_class_attention(D):
    attn = D['all_attn'].numpy()  # (N, epochs, epochs)
    y = D['all_y_true'].numpy()
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    for cls in range(4):
        mask = y == cls
        if mask.sum() > 0:
            avg_attn = attn[mask].mean(axis=0)
            im = axes[cls].imshow(avg_attn, cmap='YlOrRd', aspect='equal', vmin=0, vmax=1)
            axes[cls].set_title(f'{CLASS_NAMES[cls]} (n={mask.sum()})')
            axes[cls].set_xlabel('片段 j'); axes[cls].set_ylabel('片段 i')
            plt.colorbar(im, ax=axes[cls], shrink=0.8)
        else:
            axes[cls].text(0.5, 0.5, '无数据', ha='center', va='center', transform=axes[cls].transAxes)
            axes[cls].set_title(CLASS_NAMES[cls])
    fig.suptitle(f'07 逐类别平均注意力模式\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '07_per_class_attention.png')


# ==================== 08: 类内/类间距离分布 ====================
def _spd_log_euclidean_matrix(spd):
    """计算 SPD 矩阵对的 Log-Euclidean 距离矩阵。spd: (N, C, C)"""
    s, u = np.linalg.eigh(spd)               # (N, C, C)
    s = np.clip(s, 1e-6, None)
    log_spd = u @ np.expand_dims(np.log(s), -1) * u.swapaxes(-2, -1)
    log_spd = log_spd.reshape(spd.shape[0], -1)
    # 距离矩阵
    diff = log_spd[:, None, :] - log_spd[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1))


def fig08_distance_distribution(D):
    spd = D['all_spd'].numpy()  # (N, epochs, 20, 20)
    y = D['all_y_true'].numpy()

    # 每个样本用第一个片段的 SPD 矩阵，计算真实 Log-Euclidean 距离
    N = min(spd.shape[0], 200)
    spd_first = spd[:N, 0]
    dists = _spd_log_euclidean_matrix(spd_first)

    intra, inter = [], []
    for i in range(N):
        for j in range(i+1, N):
            if y[i] == y[j]:
                intra.append(dists[i, j])
            else:
                inter.append(dists[i, j])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(intra, bins=30, alpha=0.6, label=f'类内距离 (n={len(intra)})', color='#2ca02c')
    ax.hist(inter, bins=30, alpha=0.6, label=f'类间距离 (n={len(inter)})', color='#d62728')
    if intra and inter:
        ax.axvline(np.mean(intra), color='#2ca02c', ls='--', lw=2, label=f'类内均值={np.mean(intra):.4f}')
        ax.axvline(np.mean(inter), color='#d62728', ls='--', lw=2, label=f'类间均值={np.mean(inter):.4f}')
    ax.set_xlabel('Log-Euclidean 黎曼距离')
    ax.set_ylabel('频次')
    ax.set_title('类内/类间 SPD 黎曼距离分布')
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.suptitle(f'08 距离分布\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '08_distance_distribution.png')


# ==================== 09: 流形投影 ====================
def fig09_manifold_projection(D):
    from sklearn.manifold import MDS
    spd = D['all_spd'].numpy()[:, 0]         # (N, 20, 20) 取第一个片段的 SPD
    y = D['all_y_true'].numpy()

    # 用 SPD Log-Euclidean 黎曼距离矩阵，再用 MDS 投影到 2D
    dists = _spd_log_euclidean_matrix(spd)
    n = dists.shape[0]
    n_init = 4 if n < 100 else 1
    feat = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=n_init, normalized_stress=False).fit_transform(dists)

    fig, ax = plt.subplots(figsize=(8, 6))
    for cls in range(4):
        mask = y == cls
        if mask.sum() > 0:
            ax.scatter(feat[mask, 0], feat[mask, 1], c=CLASS_COLORS[cls], label=CLASS_NAMES[cls],
                       alpha=0.7, edgecolors='black', lw=0.5)
    ax.set_xlabel('流形坐标 1')
    ax.set_ylabel('流形坐标 2')
    ax.set_title('基于 SPD Log-Euclidean 距离的 MDS 投影')
    ax.legend()
    fig.suptitle(f'09 流形投影\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '09_manifold_projection.png')


# ==================== 10: 混淆矩阵 ====================
def fig10_confusion_matrix(D):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(D['all_y_true'], D['all_y_pred'])
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(4)); ax.set_yticks(range(4))
    ax.set_xticklabels(CLASS_NAMES); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel('预测标签'); ax.set_ylabel('真实标签')
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=12,
                    color='white' if cm[i, j] > cm.max()/2 else 'black')
    plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'10 混淆矩阵\n{D["title_suffix"]}\n准确率: {D["test_acc"]*100:.2f}%', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.91])
    save_fig(fig, '10_confusion_matrix.png')


# ==================== 11: Topomap ====================
def fig11_topomap(D):
    fig, ax = plt.subplots(figsize=(7, 6))
    try:
        import mne
        chs = [str(c) for c in D['channel_names']]
        info = mne.create_info(ch_names=chs, sfreq=D['sfreq'], ch_types='eeg')
        info.set_montage('standard_1020', match_case=False, on_missing='ignore')
        power = np.var(D['xb'].squeeze(1).numpy(), axis=(0, 2))
        montage = info.get_montage()
        pos = np.array([montage.get_positions()['ch_pos'][name][:2] for name in chs])
        mne.viz.plot_topomap(power, pos, axes=ax, show=False, contours=0, cmap='RdBu_r')
        ax.set_title('通道功率 Topomap')
    except Exception:
        from scipy.interpolate import griddata
        chs = [str(c) for c in D['channel_names']]
        power = np.var(D['xb'].squeeze(1).numpy(), axis=(0, 2))
        pl, vl = [], []
        for ch, v in zip(chs, power):
            if ch in _CHANNEL_POS:
                pl.append(_CHANNEL_POS[ch]); vl.append(v)
        pos = np.array(pl); vals = np.array(vl)
        gx, gy = np.mgrid[-1:1:200j, -1:1:200j]
        gz = griddata(pos, vals, (gx, gy), method='cubic')
        gz[(gx**2+gy**2)>1.0] = np.nan
        im = ax.imshow(gz.T, extent=[-1,1,-1,1], origin='lower', cmap='RdBu_r', aspect='equal')
        theta = np.linspace(0, 2*np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5)
        ax.scatter(pos[:,0], pos[:,1], c='k', s=15, zorder=5)
        for (x,y), n in zip(pos, chs):
            if n in _CHANNEL_POS:
                ax.text(x, y+0.06, n, ha='center', va='bottom', fontsize=5)
        ax.set_xlim(-1.15, 1.15); ax.set_ylim(-1.15, 1.25); ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title('通道功率 Topomap（手动）')
    fig.suptitle(f'11 头皮拓扑图\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '11_topomap.png')


# ==================== 12: 时频频谱 ====================
def fig12_spectrogram(D):
    from scipy.signal import spectrogram
    x_np = D['xb'].squeeze(1).numpy()
    chs = [str(c) for c in D['channel_names']]
    preferred = ['C3', 'C4']
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, ch in zip(axes, preferred):
        ch_idx = chs.index(ch) if ch in chs else 7
        f, t, Sxx = spectrogram(x_np[0, ch_idx, :], fs=D['sfreq'], nperseg=128, noverlap=64)
        im = ax.pcolormesh(t, f, 10*np.log10(Sxx+1e-10), shading='gouraud', cmap='jet')
        ax.set_title(f'{ch}')
        ax.set_xlabel('时间 (s)'); ax.set_ylabel('频率 (Hz)'); ax.set_ylim(4, 40)
        plt.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle(f'12 时频频谱图\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '12_spectrogram.png')


# ==================== 13: 梯度显著性图 ====================
def fig13_gradient_saliency(D):
    # 需要模型，尝试自动加载 checkpoint
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from mAtt.mAtt import mAtt_bci
    model_path = Path(__file__).resolve().parent / 'results' / 'checkpoint'
    # 尝试找到对应的 checkpoint
    ckpts = sorted(model_path.glob('*_best_loss.pt'))
    if not ckpts:
        ckpts = sorted(model_path.glob('*.pt'))
    if not ckpts:
        print('  ⚠️ 未找到 checkpoint，跳过梯度显著性图')
        return
    ckpt = ckpts[0]
    net = torch.load(ckpt, map_location='cpu', weights_only=False)
    net.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    xb = D['xb'].clone().to(device).requires_grad_(True)
    out = net(xb)
    target = out.argmax(dim=1)
    grad = torch.zeros_like(xb)
    for i in range(xb.shape[0]):
        net.zero_grad()
        if xb.grad is not None:
            xb.grad.zero_()
        out_i = out[i:i+1]
        out_i[0, target[i]].backward(retain_graph=True)
        grad[i] = xb.grad[i].clone()
    saliency = grad.abs().squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(min(4, saliency.shape[0]), 1, figsize=(14, 10))
    if saliency.shape[0] == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        im = ax.pcolormesh(saliency[i], shading='auto', cmap='hot')
        ax.set_ylabel('通道')
        cls = CLASS_NAMES[int(D['yb'][i])]
        ax.set_title(f'样本 {i+1}（{cls}）')
        plt.colorbar(im, ax=ax, shrink=0.8)
    axes[-1].set_xlabel('时间点')
    fig.suptitle(f'13 梯度显著性图\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '13_gradient_saliency.png')


# ==================== 14: 通道重要性排序 ====================
def fig14_channel_importance(D):
    w = D['conv1_weight'].squeeze().numpy()  # (22, 22) or similar
    importance = np.abs(w).mean(axis=tuple(range(1, w.ndim)))
    # 如果形状不对，取第一个维度
    if importance.shape[0] != 22:
        importance = np.abs(w).reshape(22, -1).mean(axis=1)
    chs = [str(c) for c in D['channel_names']]
    idx = np.argsort(importance)[::-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['#d62728' if chs[i] in ['C3', 'C4', 'Cz'] else '#1f77b4' for i in idx]
    ax1.barh(range(22), importance[idx], color=colors)
    ax1.set_yticks(range(22)); ax1.set_yticklabels([chs[i] for i in idx])
    ax1.set_xlabel('重要性（|权重|均值）'); ax1.invert_yaxis()
    ax1.set_title('通道重要性排序')

    # Topomap 叠加
    from scipy.interpolate import griddata
    pl, vl = [], []
    for ch, v in zip(chs, importance):
        if ch in _CHANNEL_POS:
            pl.append(_CHANNEL_POS[ch]); vl.append(v)
    pos = np.array(pl); vals = np.array(vl)
    gx, gy = np.mgrid[-1:1:200j, -1:1:200j]
    gz = griddata(pos, vals, (gx, gy), method='cubic')
    gz[(gx**2+gy**2)>1.0] = np.nan
    im = ax2.imshow(gz.T, extent=[-1,1,-1,1], origin='lower', cmap='Reds', aspect='equal')
    theta = np.linspace(0, 2*np.pi, 200)
    ax2.plot(np.cos(theta), np.sin(theta), 'k-', lw=1.5)
    ax2.scatter(pos[:,0], pos[:,1], c='k', s=15, zorder=5)
    for (x,y), n in zip(pos, chs):
        if n in _CHANNEL_POS:
            ax2.text(x, y+0.06, n, ha='center', va='bottom', fontsize=5)
    ax2.set_xlim(-1.15, 1.15); ax2.set_ylim(-1.15, 1.25); ax2.axis('off')
    plt.colorbar(im, ax=ax2, shrink=0.7)
    ax2.set_title('通道重要性 Topomap')

    fig.suptitle(f'14 通道重要性\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '14_channel_importance.png')


# ==================== 15: Silhouette 曲线 ====================
def fig15_silhouette_curve(D):
    sil = D['history']['silhouette']
    # 只取非 NaN 的点
    valid = [i for i, s in enumerate(sil) if not np.isnan(s)]
    if not valid:
        print('  ⚠️ 无 Silhouette 数据，跳过')
        return
    vals = [sil[i] for i in valid]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([i+1 for i in valid], vals, 'g-o', lw=1.5, ms=4)
    ax.axhline(y=0, color='gray', ls='--', alpha=0.5)
    ax.set_xlabel('迭代'); ax.set_ylabel('Silhouette Score')
    ax.set_title('验证集 Silhouette Score 变化'); ax.grid(True, alpha=0.3)
    fig.suptitle(f'15 类别可分性曲线\n{D["title_suffix"]}', fontsize=11, fontweight='bold')
    fig.tight_layout(rect=[0,0,1,0.93])
    save_fig(fig, '15_silhouette_curve.png')


# ==================== 主入口 ====================
def main():
    # 只处理被试内脚本生成的两个 npz
    npz_files = []
    for crit in ['best_loss', 'best_acc']:
        p = OUTPUT_DIR / f'matt_visualization_data_{crit}.npz'
        if p.exists():
            npz_files.append(p)
    if not npz_files:
        print(f'❌ 未找到数据文件: {OUTPUT_DIR}/matt_visualization_data_best_{{loss,acc}}.npz')
        print('请先运行: python run_matt_physionetmi.py')
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for data_npz in npz_files:
        global _CURRENT_CRITERION
        _CURRENT_CRITERION = data_npz.stem.replace('matt_visualization_data_', '')
        print(f'\n===== 处理: {data_npz.name} =====')
        D = load_data(data_npz)
        print(f'  标题: {D["title_suffix"]}')
        print(f'  测试准确率: {D["test_acc"]*100:.2f}%')

        fig01_training_curves(D)
        fig02_raw_eeg(D)
        fig03_conv_features(D)
        fig04_spd_matrices(D)
        fig05_spd_eigenvalues(D)
        fig06_attention_heatmap(D)
        fig07_per_class_attention(D)
        fig08_distance_distribution(D)
        fig09_manifold_projection(D)
        fig10_confusion_matrix(D)
        fig11_topomap(D)
        fig12_spectrogram(D)
        fig13_gradient_saliency(D)
        fig14_channel_importance(D)
        fig15_silhouette_curve(D)

    print(f'\n✅ 全部图片已保存到: {OUTPUT_DIR}')


if __name__ == '__main__':
    main()