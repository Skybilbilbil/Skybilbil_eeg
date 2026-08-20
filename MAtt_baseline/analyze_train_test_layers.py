# -*- coding: utf-8 -*-
"""
整个算法各个部分的「训练 vs 测试」特征对比
==========================================
把「训练集」与「测试集」分别送入同一模型，逐层采集特征，
对比每一层在训练阶段 vs 测试阶段看到的特征分布差异（训练-测试漂移），
并给出该模型的训练 acc / 测试 acc 泛化差距。

模型 × 数据组合：
  [within_S3]   S03 被试内模型 : S03 训练集 vs S03 测试集
  [within_S9]   S09 被试内模型 : S09 训练集 vs S09 测试集
  [cross_mix]   9人混合模型   : 混合训练集 vs 混合测试集
  [cross_S3]    9人混合模型   : S03 训练集 vs S03 测试集（S03数据在跨被试模型中的漂移）
  [cross_S9]    9人混合模型   : S09 训练集 vs S09 测试集

不重训，只加载已有 checkpoint 前向推理。
"""

import os, sys, json, time, pickle
import numpy as np
import torch
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

from analyze_cross_pipeline import (
    FeatureRecorder, batch_to_vectors, pair_metrics, le_mean,
    LAYER_ORDER, LAYER_DISP, LAYER_KIND, CLASS_NAMES,
    FS, WIN_SAMPLES, STEP_SAMPLES, DEVICE, OUT_DIR,
    mAtt_bci, load_preprocessed_dataset, split_train_valid_test,
)

SAMPLE_RATIO = 0.5
BS = 16
N_CLASSES = 4
D_ATTN_IN, D_ATTN_OUT = 8, 8
PAIR_SUBJECTS = [3, 9]

CKPT_PATTERN_W = 'profile10_repeat1_sub{sid}_epochs4_lr0.0005_wd0.5_best_loss.pt'
CKPT_CROSS = OUT_DIR / 'profile10_cross_repeat1_sub0_epochs4_lr0.0005_wd0.5_best_loss.pt'
DATA_PATH = str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl')


def load_model(ckpt):
    net = mAtt_bci(WIN_SAMPLES, STEP_SAMPLES, n_classes=N_CLASSES,
                   d_attn_in=D_ATTN_IN, d_attn_out=D_ATTN_OUT, n_windows=M).to(DEVICE)
    try:
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE, weights_only=True))
    except Exception:
        net.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    net.eval()
    return net


def extract(net, X, y, M, device):
    """把数据集全部样本送入 net，逐层采集原始特征 + 预测/标签"""
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1),
                                       torch.Tensor(y).long()),
        batch_size=BS, shuffle=False)
    recorder = FeatureRecorder(net)
    raw = {}
    labels, preds = [], []
    net.eval()
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            recorder.clear()
            logits = net(xb)
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(yb.numpy())
            for name, _, _ in LAYER_ORDER:
                if name in recorder.features:
                    raw.setdefault(name, []).append(recorder.features[name].float().numpy())
    raw = {k: np.concatenate(v) for k, v in raw.items()}
    return raw, np.concatenate(labels), np.concatenate(preds)


def to_vectors(name, t, M):
    return batch_to_vectors(name, torch.from_numpy(t), M)


def spd_mean_mat(name, t, M):
    if LAYER_KIND[name] != 'spd':
        return None
    _, mats = to_vectors(name, t, M)
    return le_mean(mats) if mats is not None else None


def vectorize(raw, M):
    """原始特征张量 → 逐层逐样本向量（SPD 层用 log-欧氏均值上三角），供保存/复现"""
    out = {}
    for name, _, _ in LAYER_ORDER:
        if name not in raw:
            continue
        v, mats = to_vectors(name, raw[name], M)
        out[name] = v
        if mats is not None:
            out[name + '__mats'] = mats
    return out


def layer_metrics(name, raw_tr, raw_te, M):
    if name not in raw_tr or name not in raw_te:
        return None
    vtr, _ = to_vectors(name, raw_tr[name], M)
    vte, _ = to_vectors(name, raw_te[name], M)
    m = pair_metrics(vtr, vte, LAYER_KIND[name],
                     spd_mean_mat(name, raw_tr[name], M),
                     spd_mean_mat(name, raw_te[name], M))
    return m


def main():
    global M
    t0 = time.time()
    print('=' * 70)
    print('整个算法各部分的「训练 vs 测试」特征对比')
    print('=' * 70)
    subject_data, _ = load_preprocessed_dataset(DATA_PATH)
    T = subject_data[0][0].shape[-1]
    M = (T - WIN_SAMPLES) // STEP_SAMPLES + 1
    print(f'T={T}, n_windows M={M}')

    from sklearn.model_selection import train_test_split

    # ---- 逐被试划分（与 profile_10 完全一致） ----
    splits = {}
    for sid in PAIR_SUBJECTS:
        X, y = subject_data[sid - 1]
        _, X, _, y = train_test_split(X, y, test_size=SAMPLE_RATIO, stratify=y, random_state=42)
        X_tr, y_tr, X_va, y_va, X_te, y_te = split_train_valid_test(X, y)
        splits[sid] = (X_tr, y_tr, X_te, y_te)
        print(f'  S{sid:02d}: train {len(y_tr)}, test {len(y_te)}  '
              f'类分布 train{np.bincount(y_tr, minlength=4)} test{np.bincount(y_te, minlength=4)}')

    # ---- 混合划分（与 profile_10 完全一致） ----
    # 注: 混合模型按 9 人训练, 这里 S3/S9 的合并数据仅用于观察其在跨被试模型下的训练-测试漂移
    X_all = np.concatenate([subject_data[s - 1][0] for s in PAIR_SUBJECTS], axis=0)
    y_all = np.concatenate([subject_data[s - 1][1] for s in PAIR_SUBJECTS], axis=0)
    X_cr_tr, y_cr_tr, X_cr_va, y_cr_va, X_cr_te, y_cr_te = split_train_valid_test(X_all, y_all)
    print(f'  S3+S9合并: train {len(y_cr_tr)}, test {len(y_cr_te)}')
    # cross_mix 用 9 人混合的训练划分 vs 混合测试划分（严格 train-test 对比）
    X9_all = np.concatenate([subject_data[s - 1][0] for s in range(1, 10)], axis=0)
    y9_all = np.concatenate([subject_data[s - 1][1] for s in range(1, 10)], axis=0)
    X9_tr, y9_tr, _, _, X9_te, y9_te = split_train_valid_test(X9_all, y9_all)
    print(f'  9人混合: train {len(y9_tr)}, test {len(y9_te)}')

    # ---- 加载模型 ----
    net_cross = load_model(CKPT_CROSS)
    print('已加载 9人混合跨被试模型')

    # ---- 各组合提取 ----
    combos = {}
    for sid in PAIR_SUBJECTS:
        X_tr, y_tr, X_te, y_te = splits[sid]
        ckpt_w = OUT_DIR / CKPT_PATTERN_W.format(sid=sid)
        net_w = load_model(ckpt_w)
        combos[f'within_S{sid}'] = {'net': net_w, 'X_tr': X_tr, 'y_tr': y_tr,
                                    'X_te': X_te, 'y_te': y_te, 'note': f'S{sid} 被试内模型'}
        combos[f'cross_S{sid}'] = {'net': net_cross, 'X_tr': X_tr, 'y_tr': y_tr,
                                   'X_te': X_te, 'y_te': y_te, 'note': f'S{sid} 数据在9人混合模型下'}
    combos['cross_mix'] = {'net': net_cross, 'X_tr': X9_tr, 'y_tr': y9_tr,
                           'X_te': X9_te, 'y_te': y9_te, 'note': '9人混合模型: 混合训练集 vs 混合测试集'}

    process = {'config': {'subjects': PAIR_SUBJECTS, 'n_windows': M, 'fs': FS}, 'combos': {}}
    report = {}
    for cname, c in combos.items():
        print(f'\n[{cname}] {c["note"]}')
        raw_tr, y_tr, p_tr = extract(c['net'], c['X_tr'], c['y_tr'], M, DEVICE)
        raw_te, y_te, p_te = extract(c['net'], c['X_te'], c['y_te'], M, DEVICE)
        acc_tr = float((y_tr == p_tr).mean())
        acc_te = float((y_te == p_te).mean())
        per_class = {}
        for cid in range(N_CLASSES):
            mask = y_te == cid
            per_class[CLASS_NAMES[cid]] = float((p_te[mask] == cid).mean()) if mask.sum() else float('nan')
        layers = {}
        for name, kind, disp in LAYER_ORDER:
            m = layer_metrics(name, raw_tr, raw_te, M)
            if m:
                layers[name] = m
        process['combos'][cname] = {
            'note': c['note'],
            'n_train': len(y_tr), 'n_test': len(y_te),
            'train_acc': acc_tr, 'test_acc': acc_te, 'gap': acc_tr - acc_te,
            'per_class_test_acc': per_class,
            'layers': layers,
            'raw': {'train': vectorize(raw_tr, M), 'test': vectorize(raw_te, M),
                    'y_train': y_tr, 'y_test': y_te},
        }
        report[cname] = {'note': c['note'], 'n_train': len(y_tr), 'n_test': len(y_te),
                         'train_acc': acc_tr, 'test_acc': acc_te, 'gap': acc_tr - acc_te,
                         'per_class_test_acc': per_class,
                         'layers': {k: v for k, v in layers.items()}}
        print(f'  train acc={acc_tr*100:.1f}%  test acc={acc_te*100:.1f}%  泛化差距={acc_tr-acc_te:+.3f}')
        print(f'  测试逐类: {per_class}')

    # ---- 保存完整过程数据 ----
    pkl_path = OUT_DIR / 'pair_train_test_process.pkl'
    with open(pkl_path, 'wb') as f:
        pickle.dump(process, f)
    json_path = OUT_DIR / 'pair_train_test.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=float)
    txt_path = OUT_DIR / 'pair_train_test_report.txt'
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(format_report(report))
    print(f'\n[OK] 完整过程数据: {pkl_path} ({pkl_path.stat().st_size/1e6:.1f} MB)')
    print(f'[OK] 结构化数据: {json_path}')
    print(f'[OK] 文字报告: {txt_path}')
    make_plot(report)
    print(f'总耗时: {time.time()-t0:.1f}s')


def format_report(report):
    L = []
    L.append('=' * 84)
    L.append('整个算法各部分的「训练 vs 测试」特征对比报告')
    L.append('=' * 84)
    for cname, r in report.items():
        L.append(f'\n{"-"*84}')
        L.append(f'[{cname}] {r["note"]}')
        L.append(f'训练集 {r["n_train"]} 样本 | 测试集 {r["n_test"]} 样本')
        L.append(f'train acc={r["train_acc"]*100:.1f}%  test acc={r["test_acc"]*100:.1f}%  '
                 f'泛化差距={r["gap"]:+.3f}')
        L.append(f'测试逐类 acc: ' + ', '.join(f'{k}={v*100:.0f}%' for k, v in r["per_class_test_acc"].items()))
        L.append(f'\n{"层":<14s}{"范数比":<9s}{"质心偏移":<10s}{"MMD":<9s}{"方差比":<9s}{"测地":<8s}')
        for name, kind, disp in LAYER_ORDER:
            if name not in r['layers']:
                continue
            m = r['layers'][name]
            g = f'{m.get("geo_dist", float("nan")):.3f}' if 'geo_dist' in m else '-'
            L.append(f'{disp:<14s}{m["norm_ratio"]:<9.3f}{m["centroid_shift"]:<10.3f}'
                     f'{m["mmd"]:<9.4f}{m["std_ratio"]:<9.3f}{g:<8s}')
    return '\n'.join(L)


def make_plot(report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    names = [n for n, _, _ in LAYER_ORDER if n in list(report.values())[0]['layers']]
    disps = [LAYER_DISP[n] for n in names]
    combos = list(report.keys())
    mat = np.full((len(combos), len(names)), np.nan)
    for i, cname in enumerate(combos):
        for j, name in enumerate(names):
            if name in report[cname]['layers']:
                mat[i, j] = report[cname]['layers'][name]['centroid_shift']
    fig, ax = plt.subplots(figsize=(15, 5.5))
    im = ax.imshow(mat, aspect='auto', cmap='Reds')
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(disps, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(combos)))
    ax.set_yticklabels([f'{c} ({report[c]["gap"]*100:+.0f}%)' for c in combos], fontsize=8)
    ax.set_title('训练 vs 测试 · 逐层特征质心偏移（括号为 train-test 准确率差距）')
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f'{mat[i, j]:.2f}', ha='center', va='center', fontsize=7,
                        color='white' if mat[i, j] > 0.5 else 'black')
    fig.colorbar(im, ax=ax, label='centroid_shift (train vs test)')
    fig.tight_layout()
    fig.savefig(OUT_DIR / 'pair_train_test_layer_heatmap.png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    print(f'[OK] 图: {OUT_DIR}/pair_train_test_layer_heatmap.png')


if __name__ == '__main__':
    main()
