# -*- coding: utf-8 -*-
"""TTA(测试时增强):训练一个模型,测试时对输入做多个时间平移,softmax 平均后预测。
官方数据 + EA,主协议 128Hz。判据:比单次预测高 >5pp 才算;1-2pp 为噪声。
臂:
  T0 单次预测(基线)   -> 63.6
  T1 5 档平移平均(-2..+2 点 @128Hz)
  T2 9 档平移平均(-4..+4)
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


def load_ea():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        per.append((apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X)),
                    np.concatenate(ys, 0)))
    return per


def train(Xtr, ytr, Xva, yva, seed=42):
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
    return net


def predict(net, X, shifts):
    """对每个平移量预测 softmax 并平均。"""
    net.eval()
    acc = None
    with torch.no_grad():
        for s in shifts:
            Xs = torch.roll(torch.tensor(X), shifts=s, dims=-1)
            ps = []
            for i in range(0, len(Xs), 128):
                ps.append(torch.softmax(net(Xs[i:i + 128].to(DEV).unsqueeze(1)), 1
                                        ).cpu().numpy())
            p = np.concatenate(ps, 0)
            acc = p if acc is None else acc + p
    return acc / len(shifts)


def main():
    per = load_ea()
    X = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    y = np.concatenate([p[1] for p in per], 0)
    i_all = np.arange(len(y))
    tr, tmp = train_test_split(i_all, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    net = train(X[tr], y[tr], X[va], y[va])
    for tag, shifts in (('T0 单次', [0]),
                        ('T1 5 档(±2)', [-2, -1, 0, 1, 2]),
                        ('T2 9 档(±4)', list(range(-4, 5)))):
        prob = predict(net, X[te], shifts)
        acc = float((prob.argmax(1) == y[te]).mean())
        print('  %-14s test %.1f%%' % (tag, 100 * acc), flush=True)


if __name__ == '__main__':
    main()
