# -*- coding: utf-8 -*-
"""上游原始 mAtt 设计在跨 session 共享口径下的表现。

本地实现与上游的差异:
  本地: 4 窗滑动(192/96) + InstanceNorm + 通道先验 + SPDNorm
  上游: 3 段均分 E2R + BatchNorm + 无通道加权 + 无 SPDNorm（mAtt_bci(epochs=3)）
本脚本直接跑上游实现,数据/EA/切分与主协议完全一致。
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
sys.path.insert(0, str(ROOT / 'reproduce' / 'official_matt_src'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402
import mAtt.spd as _spd                                                # noqa: E402
from mAtt.spd import SPDRectifiedFunction as _RectFn                   # noqa: E402
from mAtt.mAtt import mAtt_bci                                         # noqa: E402


def _rect_forward(self, input):
    eps = self.epsilon.to(input.device)
    return _RectFn.apply(input, eps)


_spd.SPDRectified.forward = _rect_forward

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'upstream_design'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    print('T', Xtr.shape, 'E', Xte.shape, flush=True)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]
    # 注意: 上游实现内部把 q/k/v 的 SPDTransform 放在 .cpu();这里整体搬到 GPU 后
    # 若报设备错误则回退 CPU。
    def move_idx(mod, target):
        for mm in mod.modules():
            for k, v in list(vars(mm).items()):
                if isinstance(v, torch.Tensor) and v.dtype in (torch.long, torch.int64):
                    setattr(mm, k, v.to(target))

    try:
        torch.manual_seed(42)
        net = mAtt_bci(epochs=3).to(DEV)
        move_idx(net, DEV)
        _ = net(torch.zeros(2, 1, 22, 562).to(DEV))
        print('device', DEV, flush=True)
    except Exception as e:
        print('GPU 失败,退回 CPU:', str(e)[:120], flush=True)
        DEV_USE = 'cpu'
        torch.manual_seed(42)
        net = mAtt_bci(epochs=3)
        dev = 'cpu'
    else:
        dev = DEV

    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(dev); yv = torch.tensor(ytr[iva]).to(dev)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(100):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            loss = Fn.cross_entropy(net(Xt[j].to(dev).unsqueeze(1)), yt[j].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 99:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    print('训练 %.0fs' % (time.time() - t0), flush=True)
    with torch.no_grad():
        pr = torch.cat([net(torch.tensor(Xte[i:i + 128]).to(dev).unsqueeze(1)).argmax(1).cpu()
                        for i in range(0, len(Xte), 128)]).numpy()
    ok = pr == yte
    per_s = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(per_s)
    print('上游设计 mAtt_bci(epochs=3) 跨 session: %.2f%%  (低四 %.1f / 高五 %.1f)'
          % (100 * ok.mean(), 100 * lo, 100 * hi), flush=True)
    print('对照 本地设计: 58.29 (3-seed)/ 58.5 (seed42) | 经典 58.76', flush=True)
    (OUT / 'upstream_design.json').write_text(json.dumps(
        {'pooled': float(ok.mean()), 'low4': lo, 'high5': hi, 'per_subject': per_s,
         'device': str(dev), 'note': '上游 mAtt_bci(epochs=3);跨 session;1 seed'},
        indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
