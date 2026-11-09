# -*- coding: utf-8 -*-
"""文献指向但本项尚未试过的两项预处理:基线校正 与 重参考。官方数据 + EA,主协议。

依据(2025 系统对比):「基线校正 + 带通」收益最稳定;CAR/Laplacian 提 SNR;ASR 不显著(跳过)。

臂:
  B0 基线(无校正)            -> 已测 63.6
  B1 基线校正(cue 前 1s 均值)
  B2 CAR 重参考(逐时刻减全通道均值)
  B3 B1 + B2
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import io
from sklearn.model_selection import train_test_split

BASE = Path('/home/lenovo/EEG_Analysis/Riemannian_Trajectories')
sys.path.insert(0, str(BASE / 'MAtt_baseline'))
sys.path.insert(0, str(BASE / 'preprocess'))
from utils.functions import build_matt_model          # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = BASE / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
NCUE = 128          # cue 前 1s @128Hz


def load_raw():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        per.append((np.concatenate(Xs, 0), np.concatenate(ys, 0)))
    return per


def baseline_correct(X):
    return X - X[:, :, :NCUE].mean(axis=-1, keepdims=True)


def car(X):
    return X - X.mean(axis=1, keepdims=True)


def run(Xtr, ytr, Xva, yva, Xte, yte, seed=42):
    torch.manual_seed(seed)
    net = build_matt_model(192, 96, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva).to(DEV); yv = torch.tensor(yva).to(DEV)
    rng = np.random.default_rng(seed)
    best = (1e10, None)
    for ep in range(100):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt) - 127, 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(Xt[idx].to(DEV).unsqueeze(1)), yt[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 99:
            net.eval()
            with torch.no_grad():
                vl = float(nn.functional.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        pr = []
        for i in range(0, len(Xte), 128):
            pr.append(net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)
                          ).argmax(1).cpu().numpy())
    return float((np.concatenate(pr) == yte).mean())


def main():
    raw = load_raw()
    arms = [('B0 基线', lambda X: X),
            ('B1 基线校正', baseline_correct),
            ('B2 CAR', car),
            ('B3 校正+CAR', lambda X: car(baseline_correct(X)))]
    for tag, fn in arms:
        per = []
        for X, y in raw:
            Xf = fn(X)
            per.append((apply_ea_to_numpy(Xf, compute_ea_matrix_from_numpy(Xf)), y))
        X = np.concatenate([p[0] for p in per], 0).astype(np.float32)
        y = np.concatenate([p[1] for p in per], 0)
        i_all = np.arange(len(y))
        tr, tmp = train_test_split(i_all, test_size=0.30, stratify=y, random_state=42)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
        a = run(X[tr], y[tr], X[va], y[va], X[te], y[te])
        print('  %-14s test %.1f%%' % (tag, 100 * a), flush=True)


if __name__ == '__main__':
    main()
