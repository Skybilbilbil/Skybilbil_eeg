# -*- coding: utf-8 -*-
"""跨 session + 共享：九人合并，**session T 训练 -> session E 测试**（不混 session）。
官方数据 + EA，128Hz。这才是「跨 session 口径」的正确版本。

对照：
  同一模型、同一数据，70/15/15 随机切分 = 64.40（此前误用的口径）
  官方代码逐人 T->E = 72.1（逐人、非共享）
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
OUT = BASE / 'MAtt_baseline' / 'results' / 'official_crosssession'
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
LOW4 = {2, 4, 5, 6}


def load_split():
    """返回 T 段(九人合并)与 E 段(九人合并)，EA 在每人 576 试次上估计。"""
    XT, yT, ST, XE, yE, SE = [], [], [], [], [], []
    for s in range(1, 10):
        mt = io.loadmat(str(DATA / ('BCIC_S%02d_T.mat' % s)))
        me = io.loadmat(str(DATA / ('BCIC_S%02d_E.mat' % s)))
        xt = np.asarray(mt['x_train'], dtype=np.float64)
        yt = np.asarray(mt['y_train']).reshape(-1).astype(np.int64)
        xe = np.asarray(me['x_test'], dtype=np.float64)
        ye = np.asarray(me['y_test']).reshape(-1).astype(np.int64)
        # EA 用该人全部 576 试次（与官方/项目定义一致）
        X_all = np.concatenate([xt, xe], 0)
        P = compute_ea_matrix_from_numpy(X_all)
        XT.append(apply_ea_to_numpy(xt, P)); yT.append(yt); ST.append(np.full(len(yt), s))
        XE.append(apply_ea_to_numpy(xe, P)); yE.append(ye); SE.append(np.full(len(ye), s))
    return (np.concatenate(XT, 0).astype(np.float32), np.concatenate(yT, 0), np.concatenate(ST, 0),
            np.concatenate(XE, 0).astype(np.float32), np.concatenate(yE, 0), np.concatenate(SE, 0))


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
    XT, yT, ST, XE, yE, SE = load_split()
    print('[data] T 训练 %s | E 测试 %s' % (XT.shape, XE.shape), flush=True)
    accs, persubj = [], {}
    for seed in (42, 2024, 7):
        tr, va = train_test_split(np.arange(len(yT)), test_size=0.2, stratify=yT, random_state=seed)
        pred = run(XT[tr], yT[tr], XT[va], yT[va], XE, yE, seed)
        acc = float((pred == yE).mean())
        accs.append(acc)
        print('  seed %-5d T->E test %.1f%%' % (seed, 100 * acc), flush=True)
    # 逐人用 seed42 的模型（重跑一次取预测）—— 简化：用最后一个 seed 的预测按人拆
    pred = run(XT[tr], yT[tr], XT[va], yT[va], XE, yE, 42)
    ok = (pred == yE)
    for s in range(1, 10):
        m = SE == s
        persubj['S%02d' % s] = round(float(ok[m].mean()), 4)
    low = float(np.mean([persubj['S%02d' % s] for s in sorted(LOW4)]))
    high = float(np.mean([v for k, v in persubj.items() if int(k[1:]) not in LOW4]))
    print('')
    print('=== 跨 session(T->E) + 九人共享 ===')
    print('  %.2f ± %.2f  (3 seed: %s)' % (100 * np.mean(accs), 100 * np.std(accs),
                                           '/'.join('%.1f' % (100 * a) for a in accs)))
    print('  逐人(seed42): %s' % {k: round(100 * v, 1) for k, v in persubj.items()})
    print('  低四 %.2f%%  高五 %.2f%%' % (100 * low, 100 * high))
    OUT.mkdir(parents=True, exist_ok=True)
    import json
    (OUT / 'official_crosssession.json').write_text(json.dumps({
        'title': '跨 session(T 训练 -> E 测试) + 九人共享,官方数据 + EA,128Hz',
        'seeds': {str(s): round(a, 4) for s, a in zip((42, 2024, 7), accs)},
        'mean': round(float(np.mean(accs)), 4),
        'sd': round(float(np.std(accs)), 4),
        'per_subject_seed42': persubj,
        'low4_seed42': round(low, 4),
        'high5_seed42': round(high, 4),
        'note': 'T/E 不混;EA 用该人全部 576 试次估计;逐人 test 288 试次,CI 半宽约 ±5.8pp',
    }, indent=2, ensure_ascii=False), encoding='utf-8')
    print('已写 official_crosssession.json')


if __name__ == '__main__':
    main()
