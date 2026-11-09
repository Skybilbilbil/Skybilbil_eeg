# -*- coding: utf-8 -*-
"""前端差距(5.4pp)的攻击:用 9 个逐人模型蒸馏共享前端。

2×2 分解:共享前端+共享头 55.59 / 逐人前端+共享头 60.96 ⇒ 前端差 5.4pp。
本实验训练一个共享学生,损失 = CE + λ·MSE(学生后端特征, 对应人的教师特征)。
训练时用 subject_id 选教师(合法:前向不接收 ID);测试时只用学生。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'distill_frontend'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN, STEP = 192, 96
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)


def load_split():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        H = apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X))
        per.append((H[:288], ys[0], H[288:], ys[1]))
    return per


def grp(per):
    return (float(np.mean([per[SUBJ[i]] for i in LOW])),
            float(np.mean([per[SUBJ[i]] for i in HIGH])))


def train_teacher(X, y, seed=42, epochs=40):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    nva = max(1, int(0.15 * len(X)))
    iva, itr = idx[:nva], idx[nva:]
    torch.manual_seed(seed)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(np.asarray(X, np.float32)), torch.tensor(y)
    best = (1e10, None)
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 31, 32):
            j = pm[i:i + 32]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(Fn.cross_entropy(net(Xt[iva].to(DEV).unsqueeze(1)), yt[iva].to(DEV)))
        if vl < best[0]:
            best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return net


def feats(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(net._features(torch.tensor(np.asarray(X[i:i + 128], np.float32)).to(DEV).unsqueeze(1)).cpu())
    return torch.cat(out, 0)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    t0 = time.time()
    Tf = torch.zeros(len(Xtr), 684)
    for i in range(9):
        net = train_teacher(per[i][0], per[i][1])
        m = sb == i
        Tf[m] = feats(net, per[i][0])
        print('  教师 %s 完成 %.0fs' % (SUBJ[i], time.time() - t0), flush=True)
    print('教师全部完成 %.0fs' % (time.time() - t0), flush=True)

    for lam in (0.0, 0.1, 0.5):
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(Xtr))
        nva = int(0.15 * len(Xtr))
        iva, itr = idx[:nva], idx[nva:]
        torch.manual_seed(42)
        net = build_matt_model(WIN, STEP, ARGS).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
        Tn = float(Tf.std())
        best = (1e10, None)
        for ep in range(60):
            net.train()
            pm = rng.permutation(itr)
            for i in range(0, len(pm) - 127, 128):
                j = pm[i:i + 128]
                f = net._features(Xt[j].to(DEV).unsqueeze(1))
                loss = Fn.cross_entropy(net.linear(f), yt[j].to(DEV))
                if lam > 0:
                    loss = loss + lam * ((f - Tf[j].to(DEV)) ** 2).mean() / (Tn ** 2)
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 4 or ep == 59:
                net.eval()
                with torch.no_grad():
                    vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
                if vl < best[0]:
                    best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
        net.load_state_dict(best[1]); net.eval()
        with torch.no_grad():
            pr = torch.cat([net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)).argmax(1).cpu()
                            for i in range(0, len(Xte), 128)]).numpy()
        ok = pr == yte
        ps = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
        lo, hi = grp(ps)
        print('λ=%.1f  %.2f%%  (低四 %.1f / 高五 %.1f)' % (lam, 100 * ok.mean(), 100 * lo, 100 * hi), flush=True)


if __name__ == '__main__':
    main()
