# -*- coding: utf-8 -*-
"""模型侧未试方向:数据增强(针对过拟合)。官方数据 + EA,主协议 128Hz,1 seed。

臂:
  E0 无增强(基线)                        -> 已测 63.6
  E1 时间平移 +-0.1s                     
  E2 高斯加噪(按试次 std 的 10%)
  E3 通道 dropout(随机丢 10% 通道)
  E4 三者联合
判据:若低四人上升且高五人不降(不同步平移) => 真改进;反之否。
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


def augment(x, rng, shift=True, noise=True, chdrop=True):
    """x: (B, 22, T) torch tensor。就地返回增强后的新张量。"""
    B, C, T = x.shape
    out = x.clone()
    if shift:
        s = int(rng.integers(-13, 14))          # +-0.1s @128Hz
        if s != 0:
            out = torch.roll(out, shifts=s, dims=-1)
    if noise:
        sd = out.std(dim=-1, keepdim=True) * 0.10
        out = out + torch.randn_like(out) * sd
    if chdrop:
        mask = (torch.rand(B, C, 1, device=out.device) > 0.10).float()
        out = out * mask
    return out


def run(Xtr, ytr, Xva, yva, Xte, yte, aug, seed=42):
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
            xb = Xt[idx].to(DEV)
            if aug:
                xb = augment(xb, rng, *aug)
            loss = nn.functional.cross_entropy(net(xb.unsqueeze(1)), yt[idx].to(DEV))
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
    per = load_ea()
    X = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    y = np.concatenate([p[1] for p in per], 0)
    i_all = np.arange(len(y))
    tr, tmp = train_test_split(i_all, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    arms = [('E0 无增强', None),
            ('E1 时间平移', (True, False, False)),
            ('E2 加噪', (False, True, False)),
            ('E3 通道dropout', (False, False, True)),
            ('E4 三者联合', (True, True, True))]
    for tag, aug in arms:
        a = run(X[tr], y[tr], X[va], y[va], X[te], y[te], aug)
        print('  %-16s test %.1f%%' % (tag, 100 * a), flush=True)


if __name__ == '__main__':
    main()
