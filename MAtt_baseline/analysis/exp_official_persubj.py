# -*- coding: utf-8 -*-
"""补 64.40 的逐人分布(判据「低四人是否被单独抬起」需要)。
主协议,官方数据 + EA,128Hz,seed=42(与 official_classic 同切分),保存逐人 test 预测。
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy import io
from sklearn.model_selection import train_test_split

BASE = Path('/home/lenovo/EEG_Analysis/Riemannian_Trajectories')
sys.path.insert(0, str(BASE / 'MAtt_baseline'))
sys.path.insert(0, str(BASE / 'preprocess'))
from utils.functions import build_matt_model          # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DATA = BASE / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = BASE / 'MAtt_baseline' / 'results' / 'official_best'
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
LOW4 = {2, 4, 5, 6}


def load_ea():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        per.append((apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X)),
                    np.concatenate(ys, 0)))
    return per


def run(Xtr, ytr, Xva, yva, Xte, yte, seed=42):
    torch.manual_seed(seed)
    net = build_matt_model(192, 96, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt = torch.tensor(Xtr); yt = torch.tensor(ytr)
    Xv = torch.tensor(Xva).to(DEV); yv = torch.tensor(yva).to(DEV)
    rng = np.random.default_rng(seed)
    best = (1e10, None)
    for ep in range(100):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt) - 127, 128):
            idx = perm[i:i + 128]
            loss = nn.functional.cross_entropy(net(Xt[idx].to(DEV).unsqueeze(1)), yt[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 99:
            net.eval()
            with torch.no_grad():
                vl = float(nn.functional.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        pr = []
        for i in range(0, len(Xte), 128):
            pr.append(net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)
                          ).argmax(1).cpu().numpy())
    return np.concatenate(pr)


def main():
    per = load_ea()
    X = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    y = np.concatenate([p[1] for p in per], 0)
    sid = np.concatenate([np.full(len(p[1]), i + 1) for i, p in enumerate(per)])
    i_all = np.arange(len(y))
    tr, tmp = train_test_split(i_all, test_size=0.30, stratify=y, random_state=42)
    va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=42)
    pred = run(X[tr], y[tr], X[va], y[va], X[te], y[te])
    correct = (pred == y[te])
    print('pooled %.2f%%' % (100 * correct.mean()), flush=True)
    per_acc = {}
    for s in range(1, 10):
        m = sid[te] == s
        if m.sum() > 0:
            per_acc['S%02d' % s] = float(correct[m].mean())
    low = np.mean([per_acc['S%02d' % s] for s in sorted(LOW4)])
    high = np.mean([v for k, v in per_acc.items() if int(k[1:]) not in LOW4])
    print('per_subject:', {k: round(100 * v, 1) for k, v in per_acc.items()}, flush=True)
    print('low4 %.2f%%  high5 %.2f%%' % (100 * low, 100 * high), flush=True)
    import json
    d = json.loads((OUT / 'official_best.json').read_text(encoding='utf-8'))
    d['per_subject_seed42'] = {k: round(v, 4) for k, v in per_acc.items()}
    d['low4_seed42'] = round(float(low), 4)
    d['high5_seed42'] = round(float(high), 4)
    d['note_per_subject'] = 'seed=42 单折；逐人 test 约 86 试次，Wilson 95% CI 半宽约 ±10.6pp，组内排序不可解读'
    (OUT / 'official_best.json').write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding='utf-8')
    print('已写回 official_best.json')


if __name__ == '__main__':
    main()
