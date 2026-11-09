# -*- coding: utf-8 -*-
"""验证 Stiefel 优化器是否是「设计意图未实现」的差距。

论文 §3.3:q/k/v 的变换矩阵 W 必须在 Stiefel 流形上更新(行满秩/正交)。
上游:StiefelParameter + MixOptimizer(重映射到 Stiefel 空间)。
主入口 train_pair_with_recording.py 用了 MixOptimizer;
但今天所有实验脚本(58.29/64.12 等)用的是普通 AdamW —— 正交约束失效。

本脚本同一份数据、同一超参,只换优化器:
  A: AdamW(现状)
  B: Adam + MixOptimizer(论文设计)
并检查训练后 q/k/v 权重的正交性 ||W^T W - I||。
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
from mAtt.optimizer import MixOptimizer                            # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'stiefel_optimizer'
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


def orth_err(net):
    errs = []
    for name in ('q_trans', 'k_trans', 'v_trans'):
        mod = getattr(net.att2, name, None)
        if mod is None or not hasattr(mod, 'weight'):
            continue
        W = mod.weight.detach().float().cpu()
        # 本地 SGDTransform: output = W^T x W  (weight 形状 in x out)
        Wn = W / (W.norm() + 1e-12)
        G = Wn.t() @ Wn
        errs.append(float((G - torch.eye(G.shape[0])).abs().mean()))
    return float(np.mean(errs)) if errs else float('nan')


def run_arm(per, opt_name, epochs=60):
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
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    if opt_name == 'adamw':
        opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
        base = opt
    else:
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
    oe = orth_err(net)
    with torch.no_grad():
        pr = torch.cat([net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)).argmax(1).cpu()
                        for i in range(0, len(Xte), 128)]).numpy()
    ok = pr == yte
    ps = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(ps)
    print('%-22s %.2f%%  (低四 %.1f / 高五 %.1f)  正交误差 %.4f  %.0fs'
          % (opt_name, 100 * ok.mean(), 100 * lo, 100 * hi, oe, time.time() - t0), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi, 'orth_err': oe}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {}
    rec['AdamW(现状)'] = run_arm(per, 'adamw')
    rec['Adam+MixOptimizer(论文设计)'] = run_arm(per, 'mix')
    (OUT / 'stiefel_optimizer.json').write_text(json.dumps(
        {'note': '跨 session;1 seed;60 epoch;只换优化器', **rec}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print('[OK]')


if __name__ == '__main__':
    main()
