# -*- coding: utf-8 -*-
"""流形注意力打分函数：原文打分把注意力冻成均匀分布，换成能真正区分距离的形式。

实测（exp_attn_score 之前的零训练检查）：E2R 输出协方差的 LE 距离平方 d² 在
P5–P95 约 16.7–32.0 之间，而原文打分 sim = 1/(1+log(1+d²)) 在该区间只从 0.258
变到 0.222（14%）。softmax 之后各窗权重几乎相同 ⇒ 时间注意力在数学上被冻结成均匀分布，
这与"注意力集中度 0.201–0.202 ≈ 1/5"的实测完全一致。

本脚本只改打分函数，其余（跨 session 主协议、逐被试 EA、MixOptimizer、60 epoch、
1 seed、192/96 四窗）全部与基线一致：
  orig  : 原文 1/(1+log(1+d²))，softmax(dim=-2)            —— 基线，已由 exp_multiscale_e2r 测得 60.22
  neg   : 仓库已有的 score_original=False 分支，-d²         —— 项目 docstring 里写的"原意"
  temp  : -d²/τ，τ=20（按实测 d² 中位数 21.8 定的尺度）      —— 保留距离次序、只调锐度
"""
import json
import sys
import time
import types
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
OUT = ROOT / 'MAtt_baseline' / 'results' / 'attn_score'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
WIN, STEP, T_OUT = 192, 96, 562
TAU = 20.0


def _head_prob_temp(self, Q, K):
    """把原文 1/(1+log(1+d²)) 换成 -d²/τ：距离次序不变，只是让 softmax 真正能分开。"""
    bs, m = Q.shape[0], Q.shape[1]
    qi, kj = self._pairs(m)
    qi, kj = qi.to(Q.device), kj.to(Q.device)
    logQ = self.tensor_log(Q)
    logK = self.tensor_log(K)
    inner = logQ[:, qi] - logK[:, kj]
    energy = (inner ** 2).sum(dim=(-2, -1))
    scores = Q.new_full((bs, m, m), float('-inf'))
    scores[:, qi, kj] = -energy / self.tau
    if self.prior_bias is not None:
        scores = scores + self.prior_bias.view(1, 1, -1).to(scores.device)
    return torch.softmax(scores, dim=-1)


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


def attn_entropy(net, Xsample):
    """注意力集中度：1 = 完全均匀，越小越集中（相对最大熵）。"""
    net.eval()
    store = {}
    orig = net.att2._head_prob

    def spy(self, Q, K):
        p = orig(Q, K)
        store['p'] = p.detach()
        return p
    net.att2._head_prob = types.MethodType(spy, net.att2)
    with torch.no_grad():
        net(Xsample.unsqueeze(1))
    net.att2._head_prob = orig
    p = store['p'].clamp(min=1e-12)
    m = p.shape[-1]
    ent = -(p * p.log()).sum(-1).mean() / np.log(m)
    return float(ent), float(p.max(-1).values.mean())


def run_arm(per, mode, tag, epochs=60):
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
    if mode == 'neg':
        net.att2.score_original = False
    elif mode == 'temp':
        net.att2.tau = TAU
        net.att2._head_prob = types.MethodType(_head_prob_temp, net.att2)
    net.train()
    with torch.no_grad():
        net(torch.zeros(2, 1, 22, T_OUT, device=DEV))
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
    ent, pmax = attn_entropy(net, torch.tensor(Xte[:64]).to(DEV))
    print('%-8s %6.2f%%  (低四 %5.1f / 高五 %5.1f)  注意力熵 %.3f  最大权重 %.3f  %.0fs'
          % (tag, 100 * ok.mean(), 100 * lo, 100 * hi, ent, pmax, time.time() - t0), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi,
            'attn_entropy': ent, 'attn_max': pmax, 'per_subject': ps}


ARMS = [('neg', '负距离 -d²'), ('temp', '温度 -d²/20')]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {'orig(基线,取自 exp_multiscale_e2r)': {'pooled': 0.6022, 'low4': 0.491, 'high5': 0.691}}
    for mode, tag in ARMS:
        rec[tag] = run_arm(per, mode, tag)
    (OUT / 'attn_score.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;MixOptimizer;60 epoch;1 seed(42);'
                 '唯一变量 = 流形注意力打分函数',
         **rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]')


if __name__ == '__main__':
    main()
