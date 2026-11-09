# -*- coding: utf-8 -*-
"""本地实现上模拟上游的「均分 E2R」(无重叠) vs 现状「滑动窗」(重叠 50%)。

上游 mAtt: E2R 把时间轴均分成 epochs 段(默认 3),不重叠。
本地实现: 192 点窗 / 96 步长(重叠 50%) → 4 窗。
本脚本在同一实现上只改窗参数,隔离「均分 vs 滑动」这一个变量,跨 session 主协议。
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
OUT = ROOT / 'MAtt_baseline' / 'results' / 'erp_window_mode'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
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


def run_arm(per, win, step, tag):
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
    net = build_matt_model(win, step, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(60):
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
    with torch.no_grad():
        pr = torch.cat([net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)).argmax(1).cpu()
                        for i in range(0, len(Xte), 128)]).numpy()
    ok = pr == yte
    ps = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(ps)
    print('%-26s %.2f%%  (低四 %.1f / 高五 %.1f)  %.0fs'
          % (tag, 100 * ok.mean(), 100 * lo, 100 * hi, time.time() - t0), flush=True)
    return {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi, 'win': win, 'step': step}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    rec = {}
    for tag, win, step in (('滑动 192/96 (现状)', 192, 96),
                           ('均分 3 段 187/187', 187, 187)):
        rec[tag] = run_arm(per, win, step, tag)
    (OUT / 'erp_window_mode.json').write_text(
        json.dumps({'note': '本地实现,只改窗参数;跨 session;1 seed;基线 58.29(3seed)/58.5(seed42)',
                    **rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]')


if __name__ == '__main__':
    main()
