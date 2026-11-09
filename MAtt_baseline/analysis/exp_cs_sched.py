# -*- coding: utf-8 -*-
"""跨 session(T->E)共享口径下,补上主入口的训练配置,看能否追上真共享 62.9。

臂:
  A0 基线(100ep,无调度,无裁剪)        -> 58.29(已测)
  A1 +CosineAnnealing +梯度裁剪(100ep)
  A2 +CosineAnnealing +梯度裁剪(300ep)
判据:若 A1/A2 显著高于 58.29(>5pp),说明之前的差距来自训练配置。
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
GRAD_CLIP = 1.0


def load_split():
    XT, yT, XE, yE = [], [], [], []
    for s in range(1, 10):
        mt = io.loadmat(str(DATA / ('BCIC_S%02d_T.mat' % s)))
        me = io.loadmat(str(DATA / ('BCIC_S%02d_E.mat' % s)))
        xt = np.asarray(mt['x_train'], dtype=np.float64)
        yt = np.asarray(mt['y_train']).reshape(-1).astype(np.int64)
        xe = np.asarray(me['x_test'], dtype=np.float64)
        ye = np.asarray(me['y_test']).reshape(-1).astype(np.int64)
        P = compute_ea_matrix_from_numpy(np.concatenate([xt, xe], 0))
        XT.append(apply_ea_to_numpy(xt, P)); yT.append(yt)
        XE.append(apply_ea_to_numpy(xe, P)); yE.append(ye)
    return (np.concatenate(XT, 0).astype(np.float32), np.concatenate(yT, 0),
            np.concatenate(XE, 0).astype(np.float32), np.concatenate(yE, 0))


def run(Xtr, ytr, Xva, yva, Xte, yte, epochs, sched, seed=42):
    torch.manual_seed(seed)
    net = build_matt_model(192, 96, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if sched else None
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva).to(DEV); yv = torch.tensor(yva).to(DEV)
    rng = np.random.default_rng(seed)
    best = (1e10, None)
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt) - 127, 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(Xt[idx].to(DEV).unsqueeze(1)), yt[idx].to(DEV))
            opt.zero_grad(); loss.backward()
            if sched:
                nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            opt.step()
        if sch is not None:
            sch.step()
        if ep % 5 == 4 or ep == epochs - 1:
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
    XT, yT, XE, yE = load_split()
    tr, va = train_test_split(np.arange(len(yT)), test_size=0.2, stratify=yT, random_state=42)
    for tag, ep, sc in (('A1 调度+裁剪 100ep', 100, True),
                        ('A2 调度+裁剪 300ep', 300, True)):
        a = run(XT[tr], yT[tr], XT[va], yT[va], XE, yE, ep, sc)
        print('  %-20s T->E %.1f%%' % (tag, 100 * a), flush=True)


if __name__ == '__main__':
    main()
