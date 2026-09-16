# -*- coding: utf-8 -*-
"""路线 B:域增广 / 合成虚拟被试,能否救回跨被试?

诊断结论(AGENTS.md):表示本身有内容(被试内 72.5%),垮在"坐标系不同"(跨被试 30.5%);
且跨被试梯度是**独立随机方向**,对齐比按 1/sqrt(N) 衰减 —— 说明共享梯度里没有可用方向。
路线 B 的假设:与其等梯度自己对齐,不如**人为造出连续的域**,把离散的"9 种坐标系"
变成连续统,逼前端学一个跨域连续的映射。

必须在**留出被试**协议下测(同池协议下域差恒为 0,任何域方法都测不出来)。

三个臂(同一 seed / 同一超参,只差增广):
  base     不增广(基线)
  mix_eeg  原始 EEG 逐试次"同类跨被试 mixup":x = lam*xi + (1-lam)*xj,同类别、不同被试
  mix_cov  类别条件协方差域内插(在 SPD 域插值再实现回时域):
           C_v = (1-a)C_i,c + a C_j,c  ->  A = C_v^{1/2} C_i,c^{-1/2},x_v = A x_i
           -> 直接合成"空间坐标被插值的虚拟被试",对准诊断出的瓶颈轴

判据:任一臂在 3 折留出被试上稳定超过 base >3pp 才算有效(单折 <5pp 不可信)。

用法(不传参数 = 跑全部 HOLDOUTS):
  python analyze_domain_aug.py            # 串行跑 8/3/9
  python analyze_domain_aug.py 8          # 只跑留出 S08(可多进程并行,每折一个进程)
每完成一个模型即写 results/domain_aug/ 下的断点 JSON;重启后自动跳过已完成的 (折,臂)。
"""
import json
import os
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
HOLDOUTS = [8, 3, 9]          # 用户指定的留一验证标准(接收性前 3)
ARMS = ['base', 'mix_eeg', 'mix_cov']
SEED = 42
# 历史档(与 AGENTS.md 里 GPU 数字可比):GPU ~5 min/模型,CPU 12-17 min。
# 只想快速筛时改成 (128, 40) -> CPU ~2 min/模型,但绝对值不可与历史比。
BS, ITERS, LR, WD, CLIP = 64, 150, 5e-4, 0.02, 1.0
C0, C1 = 375, 1126            # MI 窗
ALPHAS = [0.25, 0.5, 0.75, 1.0]
CFG = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6, anchor_stride=None,
           n_heads=1, n_windows=None, conv2_separable=False, conv2_multiscale=False,
           conv2_ms_per_branch=None, conv2_ms_specs=None, n_spatial_filters=22,
           channel_gaussian=False, gaussian_sigma=0.62, attn_change=None, attn_prior=None,
           manifold_change=None, dropout=0.3)
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(int(os.environ.get('DSH_THREADS', '16')))


# ---------- 增广算子 ----------
def mix_eeg(xb, yb, sb):
    """同类、不同被试的线性插值(域内插)。找不到合法伙伴的试次原样保留。"""
    perm = torch.randperm(len(yb), device=xb.device)
    ok = (yb[perm] == yb) & (sb[perm] != sb)
    lam = torch.rand(len(yb), 1, 1, 1, device=xb.device)
    xm = lam * xb + (1 - lam) * xb[perm]
    return torch.where(ok.view(-1, 1, 1, 1), xm, xb)


def spd_sqrt(C, inv=False):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    w = 1.0 / np.sqrt(w) if inv else np.sqrt(w)
    return (V * w) @ V.T


def class_covs(X, y, n_cls=4, ridge=1e-4):
    """每类平均协方差 (n_cls,22,22)。EA 后均值≈I,加相对 ridge 防条件数爆炸
    (A = C_v^{1/2} C_i^{-1/2} 在 C_i 病态时会放大噪声)。"""
    Cs = np.zeros((n_cls, 22, 22))
    for c in range(n_cls):
        S = X[y == c][:, :, C0:C1]
        C = np.einsum('nct,ndt->cd', S, S) / (len(S) * (C1 - C0))
        Cs[c] = C + ridge * np.trace(C) / 22 * np.eye(22)
    return Cs


def build_cov_table(X, y, sub_local, n_pool):
    """预计算所有 (i, j, alpha, c) 的 22x22 空间变换矩阵 A = C_v^{1/2} C_i^{-1/2}"""
    covs = {i: class_covs(X[sub_local == i], y[sub_local == i]) for i in range(n_pool)}
    table = {}
    for i in range(n_pool):
        inv_i = [spd_sqrt(covs[i][c], inv=True) for c in range(4)]
        for j in range(n_pool):
            if i == j:
                continue
            for ai, a in enumerate(ALPHAS):
                for c in range(4):
                    Cv = (1 - a) * covs[i][c] + a * covs[j][c]
                    table[(i, j, ai, c)] = (spd_sqrt(Cv) @ inv_i[c]).astype(np.float32)
    return table


def make_mix_cov_fn(table, subj_list, n_pool):
    """按 (本地被试, 类别) 随机取伙伴与 alpha,施加空间变换"""
    loc = {s: k for k, s in enumerate(subj_list)}

    def fn(xb, yb, sb):
        A = np.empty((len(yb), 22, 22), dtype=np.float32)
        for b in range(len(yb)):
            i = loc[int(sb[b])]
            j = int(torch.randint(0, n_pool - 1, (1,)).item())
            A[b] = table[(i, j if j < i else j + 1,          # 跳过自己,均匀取伙伴
                          int(torch.randint(0, len(ALPHAS), (1,)).item()), int(yb[b]))]
        At = torch.from_numpy(A).to(xb.device)
        return torch.einsum('bij,bjt->bit', At, xb[:, 0]).unsqueeze(1)
    return fn


# ---------- 训练/评估 ----------
def make_loader(X, y, s, shuffle):
    ds = torch.utils.data.TensorDataset(
        torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long(), torch.Tensor(s).long())
    return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle, num_workers=0)


def train(net, tl, aug):
    CE = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=ITERS)
    for _ in range(ITERS):
        net.train()
        for xb, yb, sb in tl:
            xb, yb, sb = xb.to(DEV), yb.to(DEV), sb.to(DEV)
            if aug is not None:
                xb = aug(xb, yb, sb)
            loss = CE(net(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), CLIP)
            opt.step()
        sched.step()
    return net


def acc(net, X, y):
    net.eval()
    c = 0
    with torch.no_grad():
        for xb, yb, _ in make_loader(X, y, np.zeros(len(y)), False):
            c += (net(xb.to(DEV)).argmax(1) == yb.to(DEV)).sum().item()
    return c / len(y)


# ---------- 断点续跑 ----------
def load_done(path):
    if path.exists():
        return {tuple(k.split('|')): tuple(v) for k, v in json.load(path.open()).items()}
    return {}


def save_done(path, res):
    with path.open('w') as f:
        json.dump({f'{k[0]}|{k[1]}': list(v) for k, v in res.items()}, f, indent=1)


def run_fold(sd, H, res, rpath):
    todo = [a for a in ARMS if (str(H), a) not in res]
    if not todo:
        print(f'  [S{H:02d}] 三个臂都已完成,跳过')
        return
    tr_idx = [i for i in range(len(sd)) if i != H - 1]
    subj_list = [i + 1 for i in tr_idx]
    X = np.concatenate([np.asarray(sd[i][0], dtype=np.float32) for i in tr_idx], 0)
    y = np.concatenate([np.asarray(sd[i][1]) for i in tr_idx], 0)
    s = np.concatenate([np.full(len(sd[i][1]), i + 1, dtype=np.int64) for i in tr_idx])
    Xh = np.asarray(sd[H - 1][0], dtype=np.float32)
    yh = np.asarray(sd[H - 1][1])
    Xtr, ytr, Xva, yva, _, _, str_, sva, _ = split_train_valid_test(X, y, others=[s])
    print(f'  训练被试 {subj_list}  留出 S{H:02d}  训练 {len(ytr)} / 验证 {len(yva)} / 测试 {len(yh)}'
          f'  待跑 {todo}', flush=True)

    table = None
    for arm in ARMS:
        if (str(H), arm) in res:
            continue
        torch.manual_seed(SEED); np.random.seed(SEED)
        aug = None
        if arm == 'mix_eeg':
            aug = mix_eeg
        elif arm == 'mix_cov':
            if table is None:
                table = build_cov_table(
                    Xtr, ytr, np.array([subj_list.index(v) for v in str_]), len(subj_list))
            aug = make_mix_cov_fn(table, subj_list, len(subj_list))
        t0 = time.time()
        net = build_matt_model(int(1.5 * 250), int(0.75 * 250), dict(CFG, norm_type='in'))
        train(net, make_loader(Xtr, ytr, str_, True), aug)
        a, a_tr = acc(net, Xh, yh), acc(net, Xtr, ytr)
        res[(str(H), arm)] = (a, a_tr)
        save_done(rpath, res)
        print(f'   [{arm:8s}] 留出 acc {a*100:5.1f}%   训练 acc {a_tr*100:5.1f}%'
              f'   ({time.time()-t0:.0f}s)', flush=True)
        del net


def main():
    folds = [int(x) for x in sys.argv[1:]] or HOLDOUTS
    rpath = BASE / 'results' / 'domain_aug' / f"partial_bs{BS}it{ITERS}_{'-'.join(map(str, folds))}.json"
    rpath.parent.mkdir(parents=True, exist_ok=True)
    res = load_done(rpath)
    if res:
        print(f'  断点续跑:已有 {sorted(res.keys())}')
    print(f'设备={DEV}  线程={torch.get_num_threads()}  折={folds}')
    sd, _ = load_preprocessed_dataset(str(DATA))
    for H in folds:
        print(f'\n===== 留出 S{H:02d} =====', flush=True)
        run_fold(sd, H, res, rpath)

    done = [H for H in folds if all((str(H), a) in res for a in ARMS)]
    if not done:
        return
    print('\n' + '=' * 72)
    print('汇总(留出被试)')
    print('   留出     ' + ''.join(f'{a:>12s}' for a in ARMS))
    for H in done:
        print(f'   S{H:02d}      ' + ''.join(f'{res[(str(H),a)][0]*100:11.1f}%' for a in ARMS))
    print('   ------' + ''.join('-------------' for _ in ARMS))
    print('   均值     ' + ''.join(
        f'{np.mean([res[(str(H),a)][0] for H in done])*100:11.1f}%' for a in ARMS))
    b = np.mean([res[(str(H), 'base')][0] for H in done])
    print('   相对 base ' + ''.join(
        f'{(np.mean([res[(str(H),a)][0] for H in done])-b)*100:+11.1f}pp' for a in ARMS))


if __name__ == '__main__':
    main()
