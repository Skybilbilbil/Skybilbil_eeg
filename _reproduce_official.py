# -*- coding: utf-8 -*-
"""
原文复现验证：官方 BCIC-IV-2a 数据 + 跨 session 协议
====================================================
对照原文（NeurIPS 2022, MAtt）74.71% 的公平复现：
  - 官方 mat（0.5-100Hz+陷波+EOG回归清洗，无额外处理）
  - 跨 session：session T 训练 / session E 测试（原文协议）
  - 切片 [124:562] = [0.5, 2.25]s（原文 GetBci2a.py）
  - 现有模型：独立基 + 全连接无锚点（保留我们的实现主体）
跑 S01-S09 全部，报告平均 acc。
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'MAtt_baseline'))

from mAtt.mAtt import mAtt_bci, E2R
from mAtt.optimizer import MixOptimizer
from utils.GetPhysionetMI import load_preprocessed_dataset

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS = 250
N_CLASSES = 4
LR, BS = 5e-4, 64
WD = 0.02
GRAD_CLIP = 1.0
ITERATIONS = 350            # 原文值；消融证实 +5.6pp
N_SPATIAL_FILTERS = 22
DROP_PROB = 0.3
LOCAL_RADIUS = 6
USE_HANN = False             # 消融证实：等分3patch 下无 Hann 更好（原文亦无窗）

DATA = BASE / 'saved_data' / 'official_bciciv2a_matt_raw.pkl'
SLICE = slice(124, 562)   # 原文窗口 [0.5s, 2.25s]


def load_official(subject_idx):
    """返回 (X_tr, y_tr, X_te, y_te)，跨 session，已切片"""
    _, meta = load_preprocessed_dataset(str(DATA))
    m = meta[subject_idx]
    T_split, E_split = m['T_split'], m['E_split']
    X_tr = np.concatenate(T_split, axis=0)[:, :, SLICE].astype(np.float32)
    y_tr = np.concatenate([np.full(x.shape[0], c) for c, x in enumerate(T_split)])
    X_te = np.concatenate(E_split, axis=0)[:, :, SLICE].astype(np.float32)
    y_te = np.concatenate([np.full(x.shape[0], c) for c, x in enumerate(E_split)])
    return X_tr, y_tr, X_te, y_te


def load_official_cross_subject():
    """9 人 session T 混合 → 训练；session E 混合 → 测试（跨被试跨 session）"""
    _, meta = load_preprocessed_dataset(str(DATA))
    X_tr, y_tr, X_te, y_te = [], [], [], []
    for s in range(9):
        m = meta[s]
        T_split, E_split = m['T_split'], m['E_split']
        for c, arr in enumerate(T_split):
            X_tr.append(arr[:, :, SLICE].astype(np.float32))
            y_tr.append(np.full(arr.shape[0], c))
        for c, arr in enumerate(E_split):
            X_te.append(arr[:, :, SLICE].astype(np.float32))
            y_te.append(np.full(arr.shape[0], c))
    X_tr = np.concatenate(X_tr); y_tr = np.concatenate(y_tr)
    X_te = np.concatenate(X_te); y_te = np.concatenate(y_te)
    return X_tr, y_tr, X_te, y_te


def train_eval(X_tr, y_tr, X_te, y_te, tag='', seed=None):
    """官方协议训练+测试，返回 best test acc（可选固定 seed）"""
    if seed is not None:
        import random
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    T = X_tr.shape[-1]
    win = step = T // 3          # 原文均分 3 patch
    M = E2R.n_windows(T, win, step)
    print(f'{tag} T={T}（{T/FS:.2f}s） M={M}  tr={X_tr.shape[0]} te={X_te.shape[0]}', flush=True)

    def tl(Xi, yi, shuffle):
        ds = torch.utils.data.TensorDataset(torch.Tensor(Xi).unsqueeze(1), torch.Tensor(yi).long())
        return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle, pin_memory=True)

    tr_ld, te_ld = tl(X_tr, y_tr, True), tl(X_te, y_te, False)

    net = mAtt_bci(win, step, n_classes=N_CLASSES, n_windows=None,
                   n_spatial_filters=N_SPATIAL_FILTERS,
                   local_radius=LOCAL_RADIUS, use_hann=USE_HANN).to(DEVICE)
    net.dropout = nn.Dropout(DROP_PROB)

    CE = nn.CrossEntropyLoss()
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=LR, weight_decay=WD)
    opt = MixOptimizer(base_opt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=ITERATIONS)

    best_acc = 0.0
    net.train()
    for ite in range(ITERATIONS):
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = CE(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
            opt.step()
        sched.step()
        if ite % 10 == 0 or ite == ITERATIONS - 1:
            net.eval()
            correct = total = 0
            with torch.no_grad():
                for xb, yb in te_ld:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    correct += (net(xb).argmax(1) == yb).sum().item()
                    total += len(yb)
            acc = correct / total
            best_acc = max(best_acc, acc)
    return best_acc


def run_one(subject_idx):
    X_tr, y_tr, X_te, y_te = load_official(subject_idx)
    return train_eval(X_tr, y_tr, X_te, y_te, f'S{subject_idx+1:02d}')


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', type=str, default='both', choices=['within', 'cross', 'both'])
    args = ap.parse_args()

    print(f'Device: {DEVICE}  | 原文复现：官方数据 + 跨session + 独立基全连接 + 原文打分')

    if args.mode in ('within', 'both'):
        print('\n==== 被试内（单被试） ====')
        accs = []
        for s in range(9):
            acc = run_one(s)
            accs.append(acc)
        print('\n-- 被试内汇总 --')
        for s, a in enumerate(accs):
            print(f'S{s+1:02d}: {a*100:.2f}%')
        print(f'平均: {np.mean(accs)*100:.2f}%  （原文 MAtt 报告: 74.71% ± 5.01）')

    if args.mode in ('cross', 'both'):
        print('\n==== 跨被试（9人 sessionT 训 → sessionE 测） ====')
        X_tr, y_tr, X_te, y_te = load_official_cross_subject()
        acc = train_eval(X_tr, y_tr, X_te, y_te, 'cross(9人)')
        print(f'\n跨被试平均 acc: {acc*100:.2f}%')


if __name__ == '__main__':
    main()