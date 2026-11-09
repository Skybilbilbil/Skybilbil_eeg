# -*- coding: utf-8 -*-
"""检验用户的推理:逐人时「跨 session vs 随机切分」的落差,是否显著小于共享时的落差。

已知:
  共享  跨 session 58.29 / 随机切分 64.40  -> 落差 6.1pp
  逐人  跨 session 72.1(官方代码,3 人)
本实验补:逐人 随机切分(70/15/15),同一模型、同一数据,3 人(S01/S03/S06)。
判据:
  若 逐人落差 << 6.1pp  => 共享模型对 session 漂移更敏感 => 有可补空间
  若 逐人落差 ≈ 6.1pp  => 落差是数据本身的性质,与共享无关
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


def load(subj):
    mt = io.loadmat(str(DATA / ('BCIC_S%02d_T.mat' % subj)))
    me = io.loadmat(str(DATA / ('BCIC_S%02d_E.mat' % subj)))
    xt = np.asarray(mt['x_train'], dtype=np.float64)
    yt = np.asarray(mt['y_train']).reshape(-1).astype(np.int64)
    xe = np.asarray(me['x_test'], dtype=np.float64)
    ye = np.asarray(me['y_test']).reshape(-1).astype(np.int64)
    P = compute_ea_matrix_from_numpy(np.concatenate([xt, xe], 0))
    return (apply_ea_to_numpy(xt, P).astype(np.float32), yt,
            apply_ea_to_numpy(xe, P).astype(np.float32), ye)


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
    for subj in (1, 3, 6):
        XT_, yT_, XE_, yE_ = load(subj)
        # 跨 session: T 训练(切 20% val) -> E 测试
        tr, va = train_test_split(np.arange(len(yT_)), test_size=0.2, stratify=yT_, random_state=42)
        a_cs = run(XT_[tr], yT_[tr], XT_[va], yT_[va], XE_, yE_)
        # 随机切分: T+E 合并按类分层 70/15/15
        X = np.concatenate([XT_, XE_], 0); y = np.concatenate([yT_, yE_], 0)
        i_all = np.arange(len(y))
        tr2, tmp = train_test_split(i_all, test_size=0.30, stratify=y, random_state=42)
        va2, te2 = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
        a_rs = run(X[tr2], y[tr2], X[va2], y[va2], X[te2], y[te2])
        print('  S%02d  跨session %.1f%%  随机切分 %.1f%%  落差 %+.1fpp'
              % (subj, 100 * a_cs, 100 * a_rs, 100 * (a_rs - a_cs)), flush=True)


if __name__ == '__main__':
    main()
