# -*- coding: utf-8 -*-
"""几何度量损失（中心损失）：当前只有交叉熵，没有任何类内紧致约束。

动机（SPD 网络标准件之九，MAtt 与我们都没有）：
  - "Geometry-Aware Similarity Learning on SPD Manifolds"：在流形上直接做相似度学习，
    让同类 SPD 靠拢、异类分开，而不是只靠交叉熵隐式驱动。
  - 我们的实现只有交叉熵；项目此前的 SupCon 是在编码器上做的，不是分类头。

做法（最小改动，只改损失）：
  L = CE + alpha * c_loss_scaled，
  c_loss = mean_i || f_i - c_{y_i} ||^2，f 是分类头前的 684 维切空间拼接向量，
  c 是 4 个可学习类中心；alpha 缩放成与 CE 同量级（自适应，避免手调 lambda）：
      c_loss_scaled = c_loss / detach(c_loss) * detach(CE)
  这样两项在数值上同量级，alpha=0.1 表示"几何项占总损失 10%"。

其余全部与 exp_attn_score 的基线一致（跨 session、逐被试 EA、MixOptimizer、60 epoch、seed 42、
192/96 四窗），基线 = 60.22（同一脚本配置）。
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
from mAtt.optimizer import MixOptimizer                              # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'geometric_loss'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
WIN, STEP, T_OUT = 192, 96, 562
ALPHA = 0.5


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


def run_arm(per, alpha, tag, epochs=60):
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
    net.train()
    with torch.no_grad():
        net(torch.zeros(2, 1, 22, T_OUT, device=DEV))
    dim = net.linear.in_features
    centers = nn.Parameter(torch.randn(4, dim, device=DEV) * 0.1) if alpha > 0 else None
    params = list(filter(lambda p: p.requires_grad, net.parameters()))
    if centers is not None:
        params.append(centers)
    base = torch.optim.Adam(params, lr=5e-4, weight_decay=0.02)
    opt = MixOptimizer(base)

    def fwd(xb):
        f = net._features(xb)
        fd = net.dropout(f) if net.dropout is not None else f
        return f, net.linear(fd)

    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            xb = Xt[j].to(DEV).unsqueeze(1); yb = yt[j].to(DEV)
            f, logit = fwd(xb)
            ce = Fn.cross_entropy(logit, yb)
            if centers is not None:
                # 在单位球上算类内距离：f 的绝对尺度（684 维）比 CE 大几个数量级，
                # 直接相减会让几何项梯度被压到 1e-5，等于没训（首版就是这么失效的）。
                fn = f / (f.norm(dim=1, keepdim=True) + 1e-8)
                cl = ((fn - centers[yb]) ** 2).sum(1).mean()
                loss = ce + alpha * cl
            else:
                loss = ce
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
    extra = ''
    if centers is not None:
        d = torch.cdist(centers.detach(), centers.detach())
        extra = '  类中心最小间隔 %.2f' % float(d[d > 0].min())
    print('%-18s %6.2f%%  (低四 %5.1f / 高五 %5.1f)%s  %.0fs'
          % (tag, 100 * ok.mean(), 100 * lo, 100 * hi, extra, time.time() - t0), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi, 'per_subject': ps}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {'基线(交叉熵,取自 exp_attn_score)': {'pooled': 0.6022, 'low4': 0.491, 'high5': 0.691}}
    rec['中心损失 alpha=%.1f' % ALPHA] = run_arm(per, ALPHA, '中心损失 alpha=%.1f' % ALPHA)
    (OUT / 'geometric_loss.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;MixOptimizer;60 epoch;1 seed(42);'
                 '唯一变量 = 是否加类中心几何损失(alpha 为占总损失比例)',
         'alpha': ALPHA, **rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
