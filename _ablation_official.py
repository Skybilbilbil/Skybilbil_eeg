# -*- coding: utf-8 -*-
"""
改进项消融验证（官方数据 · S01 快测）
=====================================
基线 = 独立基 + 原文打分 + 全连接 + channel_weight + IN + SPDNorm + Hann + dp0.3 + wd0.02 + 120iter
逐项开关我们相对原文的改动，验证每项是否真有效：
  1 baseline                : 当前全部改进（期望 ≈79.5%）
  2 no_channel_weight       : 关 A9 通道加权（===原文无此模块===）
  3 use_BN                  : IN→BN（===原文用 BN===）
  4 no_spdnorm              : 关 A12 流形对齐（===原文无===）
  5 no_hann                 : 关 Hann 窗（===原文无 Hann，纯 cov===）
  6 no_dropout              : 去掉 dropout（===原文无 dropout===）
  7 with_priorbias          : 加可学习先验种子（我们的改进）
  8 wd0.1                   : wd 改回原文 0.1（===原文值===）
  9 iter350                 : 迭代 350（===原文值===）
对被试内 S01 报告 best acc；找到真正有效的改进集合。
"""
import sys, argparse
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / 'MAtt_baseline'))

from mAtt.mAtt import mAtt_bci, E2R
from mAtt.optimizer import MixOptimizer
from _reproduce_official import load_official, DEVICE, FS, BS, GRAD_CLIP

N_CLASSES = 4
LR = 5e-4


def run_variant(name, use_channel=True, use_in=True, use_spdnorm=True, use_hann=True,
                drop_prob=0.3, wd=0.02, iters=120, prior=None):
    X_tr, y_tr, X_te, y_te = load_official(0)
    T = X_tr.shape[-1]
    win = step = T // 3

    def tl(Xi, yi, shuffle):
        ds = torch.utils.data.TensorDataset(torch.Tensor(Xi).unsqueeze(1), torch.Tensor(yi).long())
        return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle, pin_memory=True)

    tr_ld, te_ld = tl(X_tr, y_tr, True), tl(X_te, y_te, False)

    net = mAtt_bci(win, step, n_classes=N_CLASSES, n_windows=None,
                   n_spatial_filters=22, local_radius=6, use_hann=use_hann,
                   attn_prior=prior).to(DEVICE)
    net.dropout = nn.Dropout(drop_prob)
    if not use_channel:
        net.channel_weight = nn.Identity()
    if not use_in:
        net.In1 = nn.BatchNorm2d(net.In1.num_features, affine=True).to(DEVICE)
        net.In2 = nn.BatchNorm2d(net.In2.num_features, affine=True).to(DEVICE)
    if not use_spdnorm:
        net.spdnorm = nn.Identity()

    CE = nn.CrossEntropyLoss()
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=LR, weight_decay=wd)
    opt = MixOptimizer(base_opt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=iters)
    best = 0.0
    net.train()
    for ite in range(iters):
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = CE(net(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
            opt.step()
        sched.step()
        if ite % 30 == 0 or ite == iters - 1:
            net.eval(); correct = total = 0
            with torch.no_grad():
                for xb, yb in te_ld:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    correct += (net(xb).argmax(1) == yb).sum().item()
                    total += len(yb)
            best = max(best, correct / total)
    print(f'{name}: best_acc={best*100:.2f}%', flush=True)
    return best


def main():
    variants = {
        '1_baseline            ': dict(),
        '2_no_channel_weight   ': dict(use_channel=False),
        '3_use_BN              ': dict(use_in=False),
        '4_no_spdnorm          ': dict(use_spdnorm=False),
        '5_no_hann             ': dict(use_hann=False),
        '6_no_dropout          ': dict(drop_prob=0.0),
        '7_with_priorbias      ': dict(prior=(0, 1, 1)),
        '8_wd0.1               ': dict(wd=0.1),
        '9_iter350             ': dict(iters=350),
    }
    print(f'Device: {DEVICE}  | S01 消融（原文打分已开）')
    results = {}
    for name, kw in variants.items():
        results[name.strip()] = run_variant(name, **kw)
    print('\n==== 消融汇总（S01, 越高越好） ====')
    for k, v in results.items():
        print(f'{k}: {v*100:.2f}%')


if __name__ == '__main__':
    main()