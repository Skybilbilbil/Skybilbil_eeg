# -*- coding: utf-8 -*-
"""留一被试机制探针:在"真有域差"的设定下,检验零标签适配是否有效。

背景:当前 60/20/20 协议(测试集含同池被试)下,归一化统计量的域差恒为 0,
任何"目标域适配"方法都无法被检验 —— --bn_recalib 实测 -0.9pp(变差)。

本脚本只做 **1 折**:训练 8 个被试,测试第 9 个(完全没见过),对比三种设置
  ① InstanceNorm                      (历史默认)
  ② BatchNorm(不重算统计量)
  ③ BatchNorm + 用目标被试**无标签**试次重算统计量(BN trick 的可部署形式)
判据:③ 比 ② 高出 >2pp,才说明零标签适配在真有域差时有效。

独立于 train_pair,不改其任何开关;超参与 train_pair 的 cross_mix 保持一致
(lr 5e-4 / wd 0.02 / bs 64 / 150 it / cosine / grad_clip 1.0 / dropout 0.3)。
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
from utils.GetPhysionetMI import load_preprocessed_dataset, split_train_valid_test
from utils.functions import build_matt_model

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
HOLDOUTS = [8, 3]      # LDA 接收性排名 1、2(S08 36.1% / S03 35.4%),高于已测的 S09(33.1%)
SEED = 42
BS, ITERS, LR, WD, CLIP = 64, 150, 5e-4, 0.02, 1.0
CFG = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6, anchor_stride=None,
           n_heads=1, n_windows=None, conv2_separable=False, conv2_multiscale=False,
           conv2_ms_per_branch=None, conv2_ms_specs=None, n_spatial_filters=22,
           channel_gaussian=False, gaussian_sigma=0.62, attn_change=None, attn_prior=None,
           manifold_change=None, dropout=0.3)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_loader(X, y, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long())
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle,
                                       num_workers=0 if DEV.type == 'cpu' else 2)


def train(net, tl, vl):
    CE = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)
    for ite in range(ITERS):
        net.train()
        for xb, yb in tl:
            xb, yb = xb.to(DEV), yb.to(DEV)
            loss = CE(net(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
        sched.step()
    return net


def acc(net, X, y):
    net.eval()
    correct = 0
    with torch.no_grad():
        for xb, yb in make_loader(X, y, False):
            correct += (net(xb.to(DEV)).argmax(1) == yb.to(DEV)).sum().item()
    return correct / len(y)


def acc_bn_recalib(net, X, y):
    """用目标被试自己的**无标签**试次重算 BN running stats,再评估(权重不动)"""
    bns = [m for m in net.modules() if isinstance(m, nn.BatchNorm2d)]
    if not bns:
        return float('nan')
    saved = [(b.running_mean.clone(), b.running_var.clone(),
              b.num_batches_tracked.clone()) for b in bns]
    try:
        for b in bns:                      # 清空,只用目标域数据重建
            b.running_mean.zero_(); b.running_var.fill_(1.0)
            b.num_batches_tracked.zero_()
        net.train()                        # train 模式才会更新 running stats
        with torch.no_grad():
            for xb, yb in make_loader(X, y, False):
                net(xb.to(DEV))
    finally:
        pass
    a = acc(net, X, y)
    for b, (rm, rv, nt) in zip(bns, saved):      # 恢复,便于重复调用
        b.running_mean.copy_(rm); b.running_var.copy_(rv); b.num_batches_tracked.copy_(nt)
    net.eval()
    return a


def run_fold(sd, H, verbose=True):
    tr_idx = [i for i in range(len(sd)) if i != H - 1]
    X = np.concatenate([np.asarray(sd[i][0], dtype=np.float32) for i in tr_idx], 0)
    y = np.concatenate([np.asarray(sd[i][1]) for i in tr_idx], 0)
    Xh = np.asarray(sd[H - 1][0], dtype=np.float32)
    yh = np.asarray(sd[H - 1][1])
    Xtr, ytr, Xva, yva, _, _ = split_train_valid_test(X, y)
    if verbose:
        print(f'  训练被试 {[i+1 for i in tr_idx]}  留出 S{H:02d}  测试 {len(yh)} 试次')
    tl, vl = make_loader(Xtr, ytr, True), make_loader(Xva, yva, False)
    out = {}
    for tag, norm in [('IN', 'in'), ('BN', 'bn')]:
        torch.manual_seed(SEED); np.random.seed(SEED)   # 每折每模型同起点
        t0 = time.time()
        net = build_matt_model(int(1.5 * 250), int(0.75 * 250), dict(CFG, norm_type=norm))
        train(net, tl, vl)
        a_plain = acc(net, Xh, yh)
        a_recal = acc_bn_recalib(net, Xh, yh)
        out[tag] = (a_plain, a_recal)
        if verbose:
            print(f'   [{tag}] 不重算 {a_plain*100:5.1f}%'
                  + (f'   重算 {a_recal*100:5.1f}%   Δ={(a_recal-a_plain)*100:+.1f}pp'
                     if a_recal == a_recal else '')
                  + f'   ({time.time()-t0:.0f}s)')
        del net
        if DEV.type == 'cuda':
            torch.cuda.empty_cache()
    return out


def main():
    sd, _ = load_preprocessed_dataset(str(DATA))
    allr = {}
    for H in HOLDOUTS:
        print(f'\n===== 留出 S{H:02d} =====')
        allr[H] = run_fold(sd, H)
    print('\n' + '=' * 66)
    print('汇总(每折训练 8 被试、测试 1 个完全未见被试)')
    print('  留出   IN-不重算  IN-重算   BN-不重算  BN-重算   Δ(BN重算-不重算)')
    for H in HOLDOUTS:
        i_p, i_r = allr[H]['IN']
        b_p, b_r = allr[H]['BN']
        print(f'  S{H:02d}   {i_p*100:7.1f}%  {i_r*100:7.1f}%   {b_p*100:7.1f}%   {b_r*100:7.1f}%      {(b_r-b_p)*100:+.1f}pp')
    n = len(HOLDOUTS)
    ip = np.mean([allr[H]['IN'][0] for H in HOLDOUTS]) * 100
    ipr = np.mean([allr[H]['IN'][1] for H in HOLDOUTS]) * 100
    bp = np.mean([allr[H]['BN'][0] for H in HOLDOUTS]) * 100
    br = np.mean([allr[H]['BN'][1] for H in HOLDOUTS]) * 100
    print(f'  均值  {ip:7.1f}%  {ipr:7.1f}%   {bp:7.1f}%   {br:7.1f}%      {br-bp:+.1f}pp')
    print(f'\n  参照:同池协议 IN 54.6% / 留出 S09 时 IN 55.4%;固定基严格跨被试 30.5%')


if __name__ == '__main__':
    main()
