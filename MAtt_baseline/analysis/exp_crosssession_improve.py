# -*- coding: utf-8 -*-
"""跨 session 主协议下的改进搜索(1 seed)。

基线:官方数据 + EA + 128Hz,跨 session 58.29(3 seed)/ 58.5(seed42)。
目标:超过经典零训练读出 58.76。

一次训练,评估多个读出变体:
  V0 baseline
  V1 TTA 时间平移(±2,±4 点,5 档 softmax 平均)
  V2 TTA 加噪(0.1σ,5 次平均)
  V3 输入逐通道 z-score(用 E 的统计量;transductive)
  V4 V3 + TTA
  V5 EA 后再 z-score + TTA
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
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'crosssession_improve'
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


def evaluate(net, Xte, yte, sub, rng):
    """返回多个变体的 pooled / 逐人。"""
    out = {}

    def pred(X):
        with torch.no_grad():
            return torch.cat([net(torch.tensor(X[i:i + 128], dtype=torch.float32).to(DEV).unsqueeze(1))
                              for i in range(0, len(X), 128)]).cpu()

    def rec(tag, prob):
        p = prob.argmax(1).numpy()
        ok = p == yte
        per = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        out[tag] = {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi}
        print('  %-28s %.2f%%  (低四 %.1f / 高五 %.1f)' % (tag, 100 * ok.mean(), 100 * lo, 100 * hi),
              flush=True)

    logits0 = pred(Xte)
    rec('V0 baseline', Fn.softmax(logits0, 1))

    # V1 TTA 时间平移
    probs = []
    for sh in (-4, -2, 0, 2, 4):
        Xs = np.roll(Xte, sh, axis=-1) if sh else Xte
        probs.append(Fn.softmax(pred(Xs), 1))
    rec('V1 TTA-shift(±4,5档)', torch.stack(probs, 0).mean(0))

    # V2 TTA 加噪
    probs = [Fn.softmax(logits0, 1)]
    sd = Xte.std()
    for _ in range(4):
        Xn = Xte + 0.10 * sd * rng.standard_normal(Xte.shape).astype(np.float32)
        probs.append(Fn.softmax(pred(Xn), 1))
    rec('V2 TTA-noise(0.1σ,5次)', torch.stack(probs, 0).mean(0))

    # V3 输入逐通道 z-score(用 E 统计)
    mu = Xte.mean(axis=(0, 2), keepdims=True)
    sg = Xte.std(axis=(0, 2), keepdims=True) + 1e-8
    Xz = ((Xte - mu) / sg).astype(np.float32)
    rec('V3 输入z-score(E统计)', Fn.softmax(pred(Xz), 1))

    # V4 V3 + TTA
    probs = []
    for sh in (-4, -2, 0, 2, 4):
        Xs = np.roll(Xz, sh, axis=-1) if sh else Xz
        probs.append(Fn.softmax(pred(Xs), 1))
    rec('V4 z-score + TTA', torch.stack(probs, 0).mean(0))

    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sub = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    print('T', Xtr.shape, 'E', Xte.shape, flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]

    torch.manual_seed(42)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV)
    yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(100):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 99:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    print('训练完成 %.0fs' % (time.time() - t0), flush=True)

    res = evaluate(net, Xte, yte, sub, rng)
    res['note'] = '跨 session 主协议,seed 42;基线 3-seed 58.29 / seed42 58.5;经典 58.76'
    (OUT / 'crosssession_improve.json').write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/crosssession_improve.json' % OUT)


if __name__ == '__main__':
    main()
