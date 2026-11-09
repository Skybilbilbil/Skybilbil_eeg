# -*- coding: utf-8 -*-
"""Stiefel 专家池（SCAP 式）：把注意力里的单个 Stiefel 双线性映射换成 K 个专家的样本自适应凸组合。

动机（"Escaping the Capacity Ceiling: Routing on the Stiefel Manifold for Bilinear SPD
Layers", arXiv 2605.31043）：固定滤波器的 SPD 网络有容量上限——当域之间没有共享的判别
方向时，**单个滤波器无法同时对齐所有域**。项目已独立证明同一件事（"被试特异的线性算子
无法被共享读法参数表达"），2×2 分解也量化了逐人前端的价值（+4.75pp）。

做法：
  out = Σ_k a_k(x) · (W_k^T x W_k)，  a(x) = softmax(gate(spec(x)))
  - K 个 SPDTransform（各自独立的 Stiefel 约束）并联，凸组合保证输出仍 SPD；
  - gate 只用输入 SPD 的**前 8 个特征值对数**（成本极低，且有物理含义：谱形状就能
    区分不同被试/状态——项目实测输入端身份可分性 0.9992–1.0000）；
  - 只替换注意力里 q/k/v 三处投影，其余全部不动。

警告（SCAP 原文）："Naively trained, however, this routing can collapse to a fixed filter"。
故本脚本额外报告门控熵（1 = 均匀即塌缩，越小越个性化）与各专家被选中比例。

其余配置与基线一致（跨 session、逐被试 EA、MixOptimizer、60 epoch、seed 42、192/96 四窗），
基线 = 60.22（同配置，见 exp_attn_score）。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                        # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402
from mAtt.spd import SPDTransform                                    # noqa: E402
from mAtt.optimizer import MixOptimizer                              # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'stiefel_experts'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
WIN, STEP, T_OUT = 192, 96, 562
K = 4


class ExpertSPDTransform(nn.Module):
    """K 个 Stiefel 双线性映射的样本自适应凸组合。"""

    def __init__(self, c_in, c_out, k=4, hidden=16, k_spec=8):
        super().__init__()
        self.experts = nn.ModuleList([SPDTransform(c_in, c_out) for _ in range(k)])
        self.k_spec = k_spec
        self.gate = nn.Sequential(nn.Linear(k_spec, hidden), nn.ReLU(), nn.Linear(hidden, k))
        self.last_alpha = None

    def forward(self, x):
        s = torch.linalg.eigvalsh(x).clamp(min=1e-8)[..., -self.k_spec:]
        a = torch.softmax(self.gate(torch.log(s)), dim=-1)          # (..., K)
        self.last_alpha = a.detach()
        outs = torch.stack([e(x) for e in self.experts], dim=-3)     # (..., K, c_out, c_out)
        return (a.unsqueeze(-1).unsqueeze(-1) * outs).sum(-3)


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


def gate_stats(net):
    ents, shares = [], []
    for name in ('q_trans', 'k_trans', 'v_trans'):
        mod = getattr(net.att2, name, None)
        if not isinstance(mod, ExpertSPDTransform) or mod.last_alpha is None:
            continue
        a = mod.last_alpha.reshape(-1, K).clamp(min=1e-9)
        ent = float((-(a * a.log()).sum(-1)).mean() / np.log(K))
        ents.append(ent)
        shares.append((a.mean(0) / a.mean(0).sum()).tolist())
    if not ents:
        return float('nan'), []
    return float(np.mean(ents)), shares


def run_arm(per, tag, use_experts=True, epochs=60):
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
    net = build_matt_model(WIN, STEP, ARGS)
    if use_experts:
        for name in ('q_trans', 'k_trans', 'v_trans'):
            net.att2.__setattr__(name,
                                 ExpertSPDTransform(ARGS['d_attn_in'], ARGS['d_attn_out'], k=K).to(DEV))
    net.train()
    with torch.no_grad():
        net(torch.zeros(2, 1, 22, T_OUT, device=DEV))
    n_par = sum(p.numel() for p in net.parameters() if p.requires_grad)
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
    ent, shares = gate_stats(net)
    print('%-16s %6.2f%%  (低四 %5.1f / 高五 %5.1f)  参数 %d  门控熵 %.3f  %.0fs'
          % (tag, 100 * ok.mean(), 100 * lo, 100 * hi, n_par, ent, time.time() - t0), flush=True)
    if shares:
        for nm, sh in zip(('q', 'k', 'v'), shares):
            print('    %s 专家占比 %s' % (nm, ['%.2f' % x for x in sh]), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi, 'per_subject': ps,
            'n_params': int(n_par), 'gate_entropy': ent, 'expert_shares': shares}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {'基线(单滤波器,取自 exp_attn_score)': {'pooled': 0.6022, 'low4': 0.491, 'high5': 0.691}}
    rec['Stiefel 专家池 K=%d' % K] = run_arm(per, '专家池 K=%d' % K, use_experts=True)
    (OUT / 'stiefel_experts.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;MixOptimizer;60 epoch;1 seed(42);'
                 '唯一变量 = 注意力 q/k/v 由单个 Stiefel 映射换成 K 个专家的样本自适应凸组合',
         'K': K, **rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
