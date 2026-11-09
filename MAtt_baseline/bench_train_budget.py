# -*- coding: utf-8 -*-
"""训练预算标定:在"几分钟跑完"的约束下,bs / iters / lr 该怎么配。

⚠ 口径(2026-10-30): 本脚本窗参数硬编码 375/187(250 Hz) + 全连接 conv2,是旧口径;
   train_pair 的新默认为 62.5 Hz + 深度可分离,故本脚本的耗时/精度曲线不与新默认可比。

固定条件:BCI-2a 留出被试 S08(训练 8 人 2764 / 验证 922 / 测试 576),
seed42、IN、wd0.02、cosine、grad_clip 1.0,只扫 (BS, ITERS, LR)。
基线 = bs64 / 150it / lr5e-4,GPU 上测试准确率 59.2%、单模型约 5 min。

为什么要同时看三者:
  总梯度步数 = ITERS x ceil(N_train / BS)。bs 变大 -> 每步更贵但步数更少;
  实测吞吐最优在 bs128(每 epoch 1.43s vs bs64 的 2.13s)。bs 变大后单步梯度
  噪声下降,按线性缩放律可以把 lr 放大到 bs 的倍数,用更少步数走同样远
  —— 这一条必须实测,不能想当然。

跑完输出「时间 vs 精度」表,用于挑默认档。
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import build_matt_model

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
HOLDOUT = 8
SEED = 42
WD, CLIP = 0.02, 1.0
# (BS, ITERS, LR):现状基线 + batch 加大 + lr 线性放大 + 轮次递减
CONFIGS = [(64, 150, 5e-4), (128, 150, 5e-4), (128, 150, 1e-3),
           (128, 75, 1e-3), (128, 40, 1e-3), (256, 150, 1e-3)]
CFG = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6, anchor_stride=None,
           n_heads=1, n_windows=None, conv2_separable=False, conv2_multiscale=False,
           conv2_ms_per_branch=None, conv2_ms_specs=None, n_spatial_filters=22,
           channel_gaussian=False, gaussian_sigma=0.62, attn_change=None, attn_prior=None,
           manifold_change=None, dropout=0.3)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def loader(X, y, bs, shuffle):
    ds = torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long())
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0)


def train(net, tl, iters, lr):
    CE = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=lr, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters)
    for _ in range(iters):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = CE(net(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
        sched.step()


def acc(net, X, y, bs=256):
    net.eval()
    c = 0
    with torch.no_grad():
        for xb, yb in loader(X, y, bs, False):
            c += (net(xb.to(DEV)).argmax(1) == yb.to(DEV)).sum().item()
    return c / len(y)


def main():
    print(f'设备={DEV}')
    sd, _ = load_preprocessed_dataset(str(DATA))
    tr_idx = [i for i in range(len(sd)) if i != HOLDOUT - 1]
    X = np.concatenate([np.asarray(sd[i][0], dtype=np.float32) for i in tr_idx], 0)
    y = np.concatenate([np.asarray(sd[i][1]) for i in tr_idx], 0)
    Xh = np.asarray(sd[HOLDOUT - 1][0], dtype=np.float32)
    yh = np.asarray(sd[HOLDOUT - 1][1])
    Xtr, ytr, Xva, yva, _, _ = split_train_valid_test(X, y)
    print(f'留出 S{HOLDOUT:02d}  训练 {len(ytr)}  验证 {len(yva)}  测试 {len(yh)}\n')

    print('   BS  ITERS     LR     梯度步数   训练耗时   测试acc   验证acc')
    for bs, iters, lr in CONFIGS:
        torch.manual_seed(SEED); np.random.seed(SEED)
        net = build_matt_model(375, 187, dict(CFG, norm_type='in')).to(DEV)
        tl = loader(Xtr, ytr, bs, True)
        steps = iters * len(tl)
        if DEV.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        train(net, tl, iters, lr)
        if DEV.type == 'cuda':
            torch.cuda.synchronize()
        dt = time.time() - t0
        a, av = acc(net, Xh, yh), acc(net, Xva, yva)
        print('  %4d  %4d  %6.0e   %8d   %6.0fs   %6.1f%%   %6.1f%%'
              % (bs, iters, lr, steps, dt, a * 100, av * 100), flush=True)
        del net, tl
        if DEV.type == 'cuda':
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
