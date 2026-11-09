# -*- coding: utf-8 -*-
"""便宜的架构扫描：窗数 与 注意力输出维。全部只改构造参数，不动模型代码。

动机：官方数据 + 跨 session 口径下，这两个维度从没系统扫过（此前的"增加窗数不掉分"
是 pkl 250Hz 口径；d_attn_out 一直固定 18）。两者都不改代码，各跑一次训练即可，
属于"捡漏"性质：如果有 1–2pp 就在，没有就干净地收口。

扫描臂（每组只变一个变量，其余同基线）：
  窗数   ：192/96 → 4 窗（基线） | 160/80 → 6 窗 | 144/56 → 8 窗
  注意力维：d_attn_out 18（基线） | 12 | 24

其余配置同基线（跨 session、逐被试 EA、MixOptimizer、60 epoch、seed 42、local_radius=6）。
注意 local_radius=6 在 4/6/8 窗下分别覆盖 全部/全部/近全部，窗序稀疏结构基本不构成混杂。
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
from utils.functions import build_matt_model                        # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402
from mAtt.optimizer import MixOptimizer                              # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'arch_sweep'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
T_OUT = 562


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


def grp(ps):
    return (float(np.mean([ps[SUBJ[i]] for i in LOW])),
            float(np.mean([ps[SUBJ[i]] for i in HIGH])))


def run_arm(per, win, step, d_out, tag, epochs=60):
    args = dict(ARGS, d_attn_out=d_out)
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]
    torch.manual_seed(42)
    net = build_matt_model(win, step, args)
    net.train()
    with torch.no_grad():
        net(torch.zeros(2, 1, 22, T_OUT, device=DEV))
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_win = (T_OUT - win) // step + 1
    base = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=5e-4, weight_decay=0.02)
    opt = MixOptimizer(base)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
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
    print('%-24s %6.2f%%  (低四 %5.1f / 高五 %5.1f)  %d窗 %d参数 %.0fs'
          % (tag, 100 * ok.mean(), 100 * lo, 100 * hi, n_win, n_par, time.time() - t0), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi, 'per_subject': ps,
            'win': win, 'step': step, 'd_attn_out': d_out,
            'n_windows': int(n_win), 'n_params': int(n_par)}


ARMS = [
    # 第二轮：d=12 在首轮里最好（60.76 vs 基线 60.22，参数少 20%），沿这个方向探趋势。
    # d=24 首轮报错：spd.py SPDIncreaseDim.forward 用输入 shape 做 view，升维路径从未被用过。
    ('4窗 192/96 d=8', 192, 96, 8),
    ('4窗 192/96 d=10', 192, 96, 10),
    ('4窗 192/96 d=14', 192, 96, 14),
    ('4窗 192/96 d=16', 192, 96, 16),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {}
    for tag, win, step, d in ARMS:
        rec[tag] = run_arm(per, win, step, d, tag)
    (OUT / 'arch_sweep_d.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;MixOptimizer;60 epoch;1 seed(42);'
                 '每组只变一个变量(窗数 或 注意力输出维)',
         **rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
