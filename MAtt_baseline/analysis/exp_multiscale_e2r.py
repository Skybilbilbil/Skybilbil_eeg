# -*- coding: utf-8 -*-
"""多尺度 E2R：把 E2R 的单一时间窗长换成多尺度并联。

「多尺度子流形块」是 SPD 流形网络的标准构件之一（Riemannian Local Mechanism for SPD
Neural Networks, AAAI 2023）：单一尺度只能看到一种时间粒度上的协方差结构，多尺度并联
让下游（这里是流形注意力）在不同时间尺度之间分配权重。我们此前只试过「同一窗长下增加
窗数 / 加细栅格」，从未试过「不同窗长并联」。

基线 = 192/96 单一尺度 4 窗；多尺度 = 多个 (win, step) 各出一段 SPD 序列后沿窗维拼接。
其余配置（跨 session 主协议、逐被试 EA、MixOptimizer、60 epoch、1 seed）与基线完全一致，
唯一变量是 E2R 的窗规格。
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
from mAtt.mAtt import E2R                                            # noqa: E402
from mAtt.optimizer import MixOptimizer                              # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'multiscale_e2r'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
WIN, STEP, T_OUT = 192, 96, 562


class MultiScaleE2R(torch.nn.Module):
    """多个 E2R 并联，各自 (win, step) 不同，输出沿窗维拼接 (B, sum M_i, C, C)。"""

    def __init__(self, specs):
        super().__init__()
        self.parts = torch.nn.ModuleList([E2R(w, s) for w, s in specs])
        self.n_windows = sum((T_OUT - w) // s + 1 for w, s in specs)

    def forward(self, x):
        return torch.cat([p(x) for p in self.parts], dim=1)


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


def grp(per_sub):
    return (float(np.mean([per_sub[SUBJ[i]] for i in LOW])),
            float(np.mean([per_sub[SUBJ[i]] for i in HIGH])))


def run_arm(per, specs, tag, epochs=60):
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
    if specs is not None:
        net.ract1 = MultiScaleE2R(specs).to(DEV)
    # LazyLinear 分类头需要一次前向才知道输入维度；必须在建优化器之前完成
    net.train()
    with torch.no_grad():
        nw = net(torch.zeros(2, 1, 22, T_OUT, device=DEV)).shape[1]
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
    print('%-26s %6.2f%%  (低四 %5.1f / 高五 %5.1f)  窗数 %d  %.0fs'
          % (tag, 100 * ok.mean(), 100 * lo, 100 * hi, nw, time.time() - t0), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi,
            'n_windows': int(nw), 'per_subject': ps}


ARMS = [
    ('单一 192/96 (基线)', None),
    ('双尺度 192/96+288/144', [(192, 96), (288, 144)]),
    ('三尺度 +96/48', [(192, 96), (288, 144), (96, 48)]),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {}
    for tag, specs in ARMS:
        rec[tag] = run_arm(per, specs, tag)
        rec[tag]['specs'] = specs
    (OUT / 'multiscale_e2r.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;MixOptimizer;60 epoch;1 seed(42);'
                 '唯一变量 = E2R 窗规格(多尺度并联 vs 单一尺度)',
         **rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]')


if __name__ == '__main__':
    main()
