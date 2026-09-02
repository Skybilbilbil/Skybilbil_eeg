# -*- coding: utf-8 -*-
"""
跨被试问题 · 算法管道级深入分析
====================================
把每个被试自己的测试集全部样本分别送入:
  (a) 该被试的被试内模型 (within)
  (b) 跨被试模型 (cross)
在模型每一层采集完整特征分布，从 4 个层面量化"跨被试在算法过程中面临的问题"：

  [层面1] 逐层差异  : 范数比 / 质心偏移 / MMD / 方差比 / 流形测地距离
  [层面2] 逐被试差异: 哪些被试受影响最大
  [层面3] 逐类差异  : 哪一类在哪些环节被拉开（如脚类崩溃机制）
  [层面4] 管道传播  : 差异如何从输入层累积到输出层
                      + Conv2 时序动态滞后专项

不重新训练，只加载已有 checkpoint 前向推理。
"""

import os, sys, json, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from mAtt.mAtt import mAtt_bci
from utils.GetPhysionetMI import load_preprocessed_dataset, split_train_valid_test

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {DEVICE}')

OUT_DIR = BASE / 'results' / 'profile_10'
os.makedirs(OUT_DIR, exist_ok=True)

# ==================== 配置（与 profile_10_subjects.py 保持一致） ====================
DATASET = 'bci2a'
DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')
SUBJECTS = list(range(1, 10))
FS = 250
SAMPLE_RATIO = 0.5
N_CLASSES = 4
WIN_SEC, STEP_SEC = 1.5, 0.75
WIN_SAMPLES = int(round(WIN_SEC * FS))
STEP_SAMPLES = int(round(STEP_SEC * FS))
D_ATTN_IN, D_ATTN_OUT = 8, 8
BS = 16
CLASS_NAMES = ['左手', '右手', '脚', '舌头']

# 层顺序（管道传播顺序）+ 类型 + 显示名
# 类型: vector=卷积/信号层, spd=SPD流形层, flat=展平向量层
LAYER_ORDER = [
    ('channel_weight', 'vector', '[1] 通道加权'),
    ('conv1', 'vector', '[2] Conv1 空间'),
    ('In1', 'vector', '[3] InstanceNorm1'),
    ('conv2', 'vector', '[4] Conv2 时序'),
    ('In2', 'vector', '[5] InstanceNorm2'),
    ('ract1.signal2spd', 'spd', '[6] E2R协方差'),
    ('spdnorm', 'spd', '[7] SPDNorm'),
    ('att2.q_trans', 'spd', '[8a] 注意力Q变换'),
    ('att2.k_trans', 'spd', '[8b] 注意力K变换'),
    ('att2.v_trans', 'spd', '[8c] 注意力V变换'),
    ('ract2', 'spd', '[9] 平滑整流'),
    ('tangent.vec', 'flat', '[10] 切空间'),
    ('flat', 'flat', '[10b] Flatten'),
    ('dropout', 'flat', '[10c] Dropout'),
    ('linear', 'flat', '[11] 分类器'),
]
LAYER_DISP = {name: disp for name, kind, disp in LAYER_ORDER}
LAYER_KIND = {name: kind for name, kind, disp in LAYER_ORDER}


# ==================== 特征记录器（与 profile_10_subjects.py 一致） ====================
class FeatureRecorder:
    def __init__(self, model):
        self.hooks = []
        self.features = {}
        self._register_hooks(model)

    def _register_hooks(self, module, prefix=''):
        for name, child in module.named_children():
            full_name = f'{prefix}.{name}' if prefix else name
            if len(list(child.children())) == 0 or isinstance(child, (nn.Flatten, nn.Dropout)):
                self.hooks.append(child.register_forward_hook(
                    lambda m, i, o, n=full_name: self._record(n, o)
                ))
            else:
                self._register_hooks(child, full_name)

    def _record(self, name, output):
        if isinstance(output, tuple):
            for i, o in enumerate(output):
                if isinstance(o, torch.Tensor):
                    self.features[f'{name}[{i}]'] = o.detach().cpu()
        elif isinstance(output, torch.Tensor):
            self.features[name] = output.detach().cpu()

    def clear(self):
        self.features.clear()

    def remove(self):
        for h in self.hooks:
            h.remove()


# ==================== 特征 → 逐样本向量 ====================
def spd_logmean_matrices(t, M):
    """SPD 特征 (B,M,C,C) 或 (B*M,C,C) → 逐试次 log-欧氏均值矩阵 (B,C,C)"""
    if t.dim() == 4:
        B, M_, C, _ = t.shape
        t = t.reshape(B * M_, C, C)
    C = t.shape[-1]
    B = t.shape[0] // M
    t = t[:B * M].reshape(B, M, C, C)
    # 强数值加固：元素裁剪 + 对称化 + 正则扰动；eigh 失败(病态/重复特征值)降级为均值矩阵，
    #   避免早期随机权重下某层特征令 linalg.eigh 报 error 7 中断诊断(不影响训练损失)
    t = torch.nan_to_num(t, nan=0.0, posinf=20.0, neginf=-20.0)
    t = (t + t.transpose(-1, -2)) * 0.5
    t = t.clamp(-20.0, 20.0) + torch.eye(C, device=t.device) * 1e-4
    try:
        s, u = torch.linalg.eigh(t)
        s = s.abs().clamp(min=1e-8)
        logm = u @ torch.diag_embed(torch.log(s)) @ u.mT
        logmean = logm.mean(1)
        s2, u2 = torch.linalg.eigh(logmean)
        return u2 @ torch.diag_embed(torch.exp(s2)) @ u2.mT
    except Exception:
        return (t.mean(1) + torch.eye(C, device=t.device) * 1e-2).cpu()


def batch_to_vectors(name, t, M):
    """把一层 batch 特征转成 (B, d) 逐样本向量；SPD 层额外返回 (B,C,C) 矩阵"""
    kind = LAYER_KIND[name]
    if kind == 'spd':
        mats = spd_logmean_matrices(t, M).numpy()
        C = mats.shape[-1]
        r, c = np.triu_indices(C)
        vecs = mats[:, r, c]
        return vecs, mats
    if kind == 'flat':
        if name == 'tangent.vec':  # (B*M, d) → 逐试次 (B, M*d)，与 flat 层一致
            B = t.shape[0] // M
            return t.reshape(B, M * t.shape[1]).numpy(), None
        return t.numpy(), None
    # vector 卷积层: (B, C, 1, T) → 逐样本 [通道时间均值, 通道时间标准差] (2C,)
    t2 = t.reshape(t.shape[0], -1, t.shape[-1])
    feats = np.concatenate([t2.mean(axis=-1), t2.std(axis=-1)], axis=1)
    return feats, None


def extract_features(net, loader, M, device):
    """net 处理 loader 全部样本，返回逐层逐样本特征 + SPD矩阵 + Conv2时序轮廓 + 标签"""
    recorder = FeatureRecorder(net)
    collector = {}
    spd_mats = {}
    conv2_profiles = []
    labels = []
    net.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            recorder.clear()
            _ = net(xb)
            labels.append(yb.numpy())
            for name, kind, disp in LAYER_ORDER:
                if name not in recorder.features:
                    continue
                t = recorder.features[name].float()
                vecs, mats = batch_to_vectors(name, t, M)
                collector.setdefault(name, []).append(vecs)
                if mats is not None:
                    spd_mats.setdefault(name, []).append(mats)
                if name == 'conv2':
                    # (B,8,1,T) → 平均到通道 → (B,T) 时序轮廓
                    conv2_profiles.append(t.squeeze(2).mean(dim=1).numpy())
    feats = {k: np.concatenate(v, axis=0) for k, v in collector.items()}
    spd_mats = {k: np.concatenate(v, axis=0) for k, v in spd_mats.items()}
    conv2_profiles = np.concatenate(conv2_profiles, axis=0) if conv2_profiles else None
    labels = np.concatenate(labels)
    recorder.remove()
    return feats, spd_mats, conv2_profiles, labels


# ==================== 分布级差异指标 ====================
def mmd_rbf(X, Y):
    """RBF 核 MMD²（中位数带宽启发式），稳健的分布差异指标"""
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    m, n = X.shape[0], Y.shape[0]
    XX = X @ X.T
    YY = Y @ Y.T
    XY = X @ Y.T
    dXX = (X ** 2).sum(1)[:, None] + (X ** 2).sum(1)[None, :] - 2 * XX
    dYY = (Y ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * YY
    dXY = (X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None, :] - 2 * XY
    np.fill_diagonal(dXX, 0)
    np.fill_diagonal(dYY, 0)
    dXX = np.clip(dXX, 0, None); dYY = np.clip(dYY, 0, None); dXY = np.clip(dXY, 0, None)
    pooled = np.concatenate([dXX.ravel(), dYY.ravel(), dXY.ravel()])
    pos = pooled[pooled > 0]
    med = np.median(pos) if len(pos) else 1.0
    sigma = np.sqrt(med / 2) if med > 0 else 1.0
    KXX = np.exp(-dXX / (2 * sigma ** 2))
    KYY = np.exp(-dYY / (2 * sigma ** 2))
    KXY = np.exp(-dXY / (2 * sigma ** 2))
    return float(np.clip(KXX.sum() / (m * m) + KYY.sum() / (n * n) - 2 * KXY.sum() / (m * n), 0, None))


def mat_log(A):
    s, u = np.linalg.eigh(A)
    s = np.clip(s, 1e-10, None)
    return (u * np.log(s)) @ u.T


def mat_exp(A):
    s, u = np.linalg.eigh(A)
    return (u * np.exp(s)) @ u.T


def le_mean(mats):
    """SPD 矩阵集 (n,C,C) 的 Log-Euclidean 均值"""
    logs = np.stack([mat_log(m) for m in mats])
    return mat_exp(logs.mean(0))


def le_dist(A, B):
    """两个 SPD 矩阵的 Log-Euclidean 距离"""
    return float(np.sqrt(((mat_log(A) - mat_log(B)) ** 2).sum()))


def pair_metrics(vW, vC, kind, spd_W=None, spd_C=None):
    """within vs cross 特征集的分布级差异指标"""
    nW, nC = vW.shape[0], vC.shape[0]
    if nW < 5 or nC < 5:
        return None
    normW = np.linalg.norm(vW, axis=1)
    normC = np.linalg.norm(vC, axis=1)
    mW, mC = normW.mean(), normC.mean()
    muW, muC = vW.mean(0), vC.mean(0)
    res = {
        'norm_ratio': float(abs(mC - mW) / (abs(mW) + 1e-8)),
        'std_ratio': float((normC.std() + 1e-9) / (normW.std() + 1e-9)),
        'centroid_shift': float(np.linalg.norm(muC - muW) / (np.linalg.norm(muW) + 1e-8)),
        'mmd': mmd_rbf(vW, vC),
    }
    if kind == 'spd' and spd_W is not None and spd_C is not None:
        res['geo_dist'] = le_dist(spd_W, spd_C)
    return res


def gap_level(ratio):
    return 'HIGH' if ratio > 0.5 else 'MED' if ratio > 0.2 else 'LOW'


# ==================== 主流程 ====================
def main():
    t0 = time.time()
    print('=' * 70)
    print('加载 BCI2A 数据并复现与 profile_10 完全一致的数据划分')
    print('=' * 70)
    subject_data, metadata = load_preprocessed_dataset(DATA_PATH)
    T = subject_data[0][0].shape[-1]
    M = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    print(f'T={T}, n_windows M={M}, feat_per_window={D_ATTN_OUT * (D_ATTN_OUT + 1) // 2}')

    # 逐被试测试集（与 analyze_profile_results.py 完全一致）
    subject_tests = {}
    for sid in SUBJECTS:
        X, y = subject_data[sid - 1]
        from sklearn.model_selection import train_test_split
        _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
        subject_tests[sid] = (X_te, y_te)
        print(f'  S{sid:02d} 测试集: {X_te.shape}, 类别 {np.bincount(y_te, minlength=4)}')

    # ========== 逐被试提取 within / cross 特征 ==========
    print('\n' + '=' * 70)
    print('逐被试提取特征: 被试内模型 vs 跨被试模型')
    print('=' * 70)

    ckpt_pattern_w = 'profile10_repeat1_sub{sid}_epochs4_lr0.0005_wd0.5_best_loss.pt'
    ckpt_cross = OUT_DIR / 'profile10_cross_repeat1_sub0_epochs4_lr0.0005_wd0.5_best_loss.pt'

    # 先加载跨被试模型（共享给所有被试）
    if not ckpt_cross.exists():
        raise FileNotFoundError(f'找不到跨被试 checkpoint: {ckpt_cross}')
    net_cross = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                         d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT, n_windows=M).to(DEVICE)
    try:
        net_cross.load_state_dict(torch.load(ckpt_cross, map_location=DEVICE, weights_only=True))
    except Exception:
        net_cross.load_state_dict(torch.load(ckpt_cross, map_location=DEVICE))
    net_cross.eval()

    data = {}  # sid -> {'within': (feats, spd, profiles, labels), 'cross': ...}
    for sid in SUBJECTS:
        X_te, y_te = subject_tests[sid]
        X_t = torch.Tensor(X_te).unsqueeze(1)
        y_t = torch.Tensor(y_te).long()
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t), batch_size=BS, shuffle=False)

        # within 模型
        ckpt_w = OUT_DIR / ckpt_pattern_w.format(sid=sid)
        if not ckpt_w.exists():
            print(f'  [WARN] 缺少 {ckpt_w.name}, 跳过 S{sid:02d}')
            continue
        net_w = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                         d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT, n_windows=M).to(DEVICE)
        try:
            net_w.load_state_dict(torch.load(ckpt_w, map_location=DEVICE, weights_only=True))
        except Exception:
            net_w.load_state_dict(torch.load(ckpt_w, map_location=DEVICE))
        net_w.eval()

        fw = extract_features(net_w, loader, M, DEVICE)
        fc = extract_features(net_cross, loader, M, DEVICE)
        data[sid] = {'within': fw, 'cross': fc}
        n_used = fw[0]['linear'].shape[0]
        print(f'  S{sid:02d}: within={n_used} 样本, cross={fc[0]["linear"].shape[0]} 样本 [OK]')
        del net_w

    # ========== 层面1: 逐层差异（跨被试 pooled） ==========
    print('\n' + '=' * 70)
    print('层面1: 逐层跨被试差异（所有被试 pooled）')
    print('=' * 70)
    layer_stats = {}
    per_subj_layer = {}   # layer -> {sid: metrics}
    for name, kind, disp in LAYER_ORDER:
        layer_stats[name] = {'display': disp, 'kind': kind, 'per_subject': {}}
        vals = {m: [] for m in ['norm_ratio', 'centroid_shift', 'mmd', 'std_ratio', 'geo_dist']}
        for sid, d in data.items():
            fw, sw, pw, lw = d['within']
            fc, sc, pc, lc = d['cross']
            if name not in fw or name not in fc:
                continue
            m = pair_metrics(fw[name], fc[name], kind,
                             le_mean(sw[name]) if (kind == 'spd' and name in sw) else None,
                             le_mean(sc[name]) if (kind == 'spd' and name in sc) else None)
            if m is None:
                continue
            layer_stats[name]['per_subject'][sid] = m
            for k in vals:
                if k in m:
                    vals[k].append(m[k])
        # pooled = 逐被试均值
        entry = {}
        for k in vals:
            if vals[k]:
                entry[k] = float(np.mean(vals[k]))
                entry[k + '_std'] = float(np.std(vals[k]))
        if 'centroid_shift' in entry:
            entry['level'] = gap_level(entry['centroid_shift'])
        layer_stats[name].update(entry)

    # 打印逐层表
    print(f'\n{"层":<16s}{"范数比":<9s}{"质心偏移":<10s}{"MMD":<9s}{"方差比":<9s}{"测地":<9s}{"级别":<6s}')
    print('-' * 72)
    for name, kind, disp in LAYER_ORDER:
        e = layer_stats[name]
        if 'centroid_shift' not in e:
            print(f'{disp:<16s}{"缺失":<9s}')
            continue
        g = f'{e.get("geo_dist", float("nan")):.3f}' if 'geo_dist' in e else '-'
        print(f'{disp:<16s}{e["norm_ratio"]:<9.3f}{e["centroid_shift"]:<10.3f}'
              f'{e["mmd"]:<9.4f}{e["std_ratio"]:<9.3f}{g:<9s}{e.get("level","-"):<6s}')

    # ========== 层面2: 逐被试差异矩阵 ==========
    print('\n' + '=' * 70)
    print('层面2: 逐被试 × 逐层 质心偏移矩阵')
    print('=' * 70)
    subj_mat = np.full((len(SUBJECTS), len(LAYER_ORDER)), np.nan)
    for si, sid in enumerate(SUBJECTS):
        for li, (name, kind, disp) in enumerate(LAYER_ORDER):
            if sid in layer_stats[name]['per_subject']:
                subj_mat[si, li] = layer_stats[name]['per_subject'][sid]['centroid_shift']
    # 逐被试汇总（跨层平均）
    print(f'\n{"被试":<8s}{"平均质心偏移":<14s}{"最大层":<18s}{"最严重层值":<12s}')
    print('-' * 55)
    subj_summary = {}
    for si, sid in enumerate(SUBJECTS):
        row = subj_mat[si]
        if np.all(np.isnan(row)):
            continue
        mean_v = np.nanmean(row)
        li = int(np.nanargmax(row))
        subj_summary[sid] = {'mean_centroid': float(mean_v), 'max_layer': LAYER_DISP[LAYER_ORDER[li][0]],
                             'max_val': float(row[li])}
        print(f'  S{sid:02d}    {mean_v:<14.3f}{LAYER_DISP[LAYER_ORDER[li][0]]:<18s}{row[li]:<12.3f}')

    # ========== 层面3: 逐类差异（pooled per class） ==========
    print('\n' + '=' * 70)
    print('层面3: 逐类 × 逐层 质心偏移（跨被试 pooled）')
    print('=' * 70)
    class_mat = np.full((N_CLASSES, len(LAYER_ORDER)), np.nan)
    class_stats = {}
    for name, kind, disp in LAYER_ORDER:
        class_stats[name] = {}
        for c in range(N_CLASSES):
            vW = np.concatenate([d['within'][0][name][d['within'][3] == c]
                                 for d in data.values() if name in d['within'][0]])
            vC = np.concatenate([d['cross'][0][name][d['cross'][3] == c]
                                 for d in data.values() if name in d['cross'][0]])
            if len(vW) < 5 or len(vC) < 5:
                continue
            spdW = None; spdC = None
            if kind == 'spd':
                mW = np.concatenate([d['within'][1][name][d['within'][3] == c]
                                     for d in data.values() if name in d['within'][1]])
                mC = np.concatenate([d['cross'][1][name][d['cross'][3] == c]
                                     for d in data.values() if name in d['cross'][1]])
                if len(mW) and len(mC):
                    spdW = le_mean(mW); spdC = le_mean(mC)
            m = pair_metrics(vW, vC, kind, spdW, spdC)
            if m is None:
                continue
            class_stats[name][c] = m
            class_mat[c, [i for i, (n2, _, _) in enumerate(LAYER_ORDER) if n2 == name][0]] = m['centroid_shift']
    print(f'\n{"类":<6s}' + ''.join(f'{LAYER_DISP[n][:10]:<11s}' for n, _, _ in LAYER_ORDER))
    for c in range(N_CLASSES):
        row_str = '  '.join(f'{v:<10.3f}' if not np.isnan(v) else f'{"-":<10s}'
                            for v in class_mat[c])
        print(f'  {CLASS_NAMES[c]:<4s}{row_str}')

    # ========== 层面4a: Conv2 时序动态滞后 ==========
    print('\n' + '=' * 70)
    print('层面4a: Conv2 层时序动态差异（逐被试激活轮廓滞后）')
    print('=' * 70)
    lags = []
    for sid, d in data.items():
        pw, pc = d['within'][2], d['cross'][2]
        if pw is None or pc is None:
            continue
        prof_w = pw.mean(0)   # (T,)
        prof_c = pc.mean(0)
        prof_w = prof_w - prof_w.mean()
        prof_c = prof_c - prof_c.mean()
        c_full = np.correlate(prof_w, prof_c, mode='full')
        lags_all = np.arange(-len(prof_w) + 1, len(prof_w))
        lag = lags_all[int(np.argmax(np.abs(c_full)))]
        corr_at_0 = float(np.corrcoef(prof_w, prof_c)[0, 1])
        lags.append(lag)
        print(f'  S{sid:02d}: 最大互相关滞后={lag:+d} 采样点, 0滞后相关系数={corr_at_0:.3f}')
    if lags:
        lags = np.array(lags)
        print(f'\n  平均 |滞后| = {np.mean(np.abs(lags)):.1f} 采样点 = {np.mean(np.abs(lags)) / FS * 1000:.0f} ms')

    # ========== 输入层基线（数据级，无模型） ==========
    print('\n' + '=' * 70)
    print('输入层基线: 数据级协方差跨被试距离（Log-Euclidean）')
    print('=' * 70)
    n_sel = 30
    mean_covs = []
    for sid in SUBJECTS:
        X_te = subject_tests[sid][0]
        idxs = np.random.choice(len(X_te), min(n_sel, len(X_te)), replace=False)
        Xs = X_te[idxs] - X_te[idxs].mean(axis=-1, keepdims=True)
        covs = Xs @ Xs.transpose(0, 2, 1) / (Xs.shape[-1] - 1)
        covs = covs + 1e-6 * np.eye(Xs.shape[1])
        mean_covs.append(le_mean(covs))
    inter_d = [le_dist(mean_covs[i], mean_covs[j])
               for i in range(len(mean_covs)) for j in range(i + 1, len(mean_covs))]
    print(f'  被试间平均 Log-Euclidean 距离: {np.mean(inter_d):.4f}')

    # ========== 保存 ==========
    out = {
        'config': {'dataset': DATASET, 'subjects': SUBJECTS, 'fs': FS, 'n_windows': M,
                   'd_attn_in': D_ATTN_IN, 'd_attn_out': D_ATTN_OUT, 'bs': BS},
        'layer_order': [{'name': n, 'kind': k, 'display': d} for n, k, d in LAYER_ORDER],
        'layer_stats': layer_stats,
        'subj_summary': subj_summary,
        'subj_mat': subj_mat.tolist(),
        'class_mat': class_mat.tolist(),
        'class_stats': class_stats,
        'conv2_lags': [int(x) for x in lags],
        'input_inter_dist': float(np.mean(inter_d)),
    }
    json_path = OUT_DIR / 'pipeline_detail.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False, default=float)
    print(f'\n[OK] 结构化数据已保存: {json_path}')

    # ========== 可视化 ==========
    make_plots(out, subj_mat, class_mat, data, SUBJECTS, M)
    print(f'\n总耗时: {time.time() - t0:.1f}s')


def make_plots(out, subj_mat, class_mat, data, SUBJECTS, M):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    names = [e['name'] for e in out['layer_order']]
    disps = [e['display'] for e in out['layer_order']]
    stats = out['layer_stats']

    # ---- 图1: 逐层差异（质心偏移 + 范数比），含输入基线参考线 ----
    cs = [stats[n].get('centroid_shift', np.nan) for n in names]
    nr = [stats[n].get('norm_ratio', np.nan) for n in names]
    cols = ['#d62728' if stats[n].get('level') == 'HIGH' else
            '#ff7f0e' if stats[n].get('level') == 'MED' else '#2ca02c' for n in names]
    base = out['input_inter_dist']
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(names))
    ax.bar(x - 0.2, cs, 0.4, color=cols, alpha=0.85, label='质心偏移 centroid_shift')
    ax.bar(x + 0.2, nr, 0.4, color='#7f7f7f', alpha=0.45, label='范数比 norm_ratio')
    ax.axhline(base, color='blue', ls='--', lw=1.5,
               label=f'输入层数据级距离基线={base:.3f}')
    ax.set_xticks(x)
    ax.set_xticklabels(disps, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('归一化差异'); ax.set_title('跨被试差异 · 沿算法管道逐层变化（输入基线为数据级协方差距离）')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pipeline_divergence_by_layer.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- 图2: 逐被试 × 逐层 热力图 ----
    fig, ax = plt.subplots(figsize=(13, 6))
    im = ax.imshow(subj_mat, aspect='auto', cmap='Reds')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(disps, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(SUBJECTS)))
    ax.set_yticklabels([f'S{s}' for s in SUBJECTS])
    ax.set_title('逐被试 × 逐层 质心偏移（跨被试特征偏移强度）')
    for i in range(subj_mat.shape[0]):
        for j in range(subj_mat.shape[1]):
            if not np.isnan(subj_mat[i, j]):
                ax.text(j, i, f'{subj_mat[i, j]:.2f}', ha='center', va='center', fontsize=6, color='white' if subj_mat[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='centroid_shift')
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pipeline_subject_layer_heatmap.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- 图3: 逐类 × 逐层 热力图 ----
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(class_mat, aspect='auto', cmap='Purples')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(disps, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(N_CLASSES))
    ax.set_yticklabels(CLASS_NAMES)
    ax.set_title('逐类 × 逐层 质心偏移（定位哪一类在哪一环节被拉开）')
    for i in range(class_mat.shape[0]):
        for j in range(class_mat.shape[1]):
            if not np.isnan(class_mat[i, j]):
                ax.text(j, i, f'{class_mat[i, j]:.2f}', ha='center', va='center', fontsize=6,
                        color='white' if class_mat[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='centroid_shift')
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pipeline_class_layer_heatmap.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    # ---- 图4: Conv2 时序轮廓对比（前4个被试） ----
    sids = list(data.keys())[:4]
    fig, axes = plt.subplots(1, len(sids), figsize=(15, 3.5), squeeze=False)
    for ax, sid in zip(axes[0], sids):
        pw = data[sid]['within'][2].mean(0)
        pc = data[sid]['cross'][2].mean(0)
        t_axis = np.arange(len(pw)) / FS * 1000
        ax.plot(t_axis, pw, label='within', lw=1.2)
        ax.plot(t_axis, pc, label='cross', lw=1.2, alpha=0.8)
        ax.set_title(f'S{sid}'); ax.set_xlabel('时间(ms)'); ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle('Conv2 层时序激活轮廓: 被试内模型 vs 跨被试模型', fontsize=12)
    fig.tight_layout(); fig.savefig(OUT_DIR / 'pipeline_conv2_temporal.png', dpi=130, bbox_inches='tight')
    plt.close(fig)

    print('[OK] 可视化已保存到 results/profile_10/pipeline_*.png')


if __name__ == '__main__':
    main()
