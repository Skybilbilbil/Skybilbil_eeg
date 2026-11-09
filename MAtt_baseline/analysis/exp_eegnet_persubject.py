# -*- coding: utf-8 -*-
"""EEGNet 的被试内读数(每人一个模型,70/15/15)—— 与 MAtt 原生场景同口径。

## 为什么
主协议(共享混合)下 EEGNet 64.14 > MAtt 58.82。用户问「对比它我的优势是什么」,
唯一没说清的格子是**被试内**:那是 MAtt 的原生场景(原文 74.71% 即逐人读数),
而 EEGNet 的被试内从未在本项目测过。本脚本补这一格:
  每人 576 试次 → 70/15/15(rs=42) → 训练 EEGNet → test 只读一次
  9 人各一个模型;250Hz(EEGNet 的最优采样率,主协议测过 250 > 62.5)
对照:同类经典读出(LE253+LDA)逐人 70/15/15 = 70.1%、被试内 5 折 72.4%。

产物 results/eegnet_persubject/
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from exp_eegnet_baseline import EEGNet                                    # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

LAWHERN = '--lawhern' in sys.argv   # 原文配置:128Hz / 500 epoch / dropout 0.5 / val-loss 早停
N_EPOCH = 500 if LAWHERN else 100
DROP = 0.5 if LAWHERN else 0.25
OUT = BASE / 'results' / ('eegnet_persubject_lawhern' if LAWHERN else 'eegnet_persubject')
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
SUBJ = ['S%02d' % i for i in range(1, 10)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


def run(Xtr, ytr, Xva, yva, Xte, yte, seed=42):
    torch.manual_seed(seed)
    net = EEGNet(drop=DROP).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(DEV)
    rng = np.random.default_rng(seed)
    best = (None, None, 0.0)
    for ep in range(N_EPOCH):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt) - 31, 32):
            idx = perm[i:i + 32]
            loss = nn.functional.cross_entropy(net(Xt[idx].to(DEV).unsqueeze(1)),
                                               yt[idx].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == N_EPOCH - 1:
            net.eval()
            with torch.no_grad():
                logits = net(Xv.unsqueeze(1))
                pr = logits.argmax(1).cpu().numpy()
                vl = float(nn.functional.cross_entropy(
                    logits, torch.tensor(yva, dtype=torch.long).to(DEV)))
            v = float((pr == yva).mean())
            score = -vl if LAWHERN else v
            if best[0] is None or score > best[0]:
                best = (score, {k: t.clone() for k, t in net.state_dict().items()}, v)
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        pr = net(torch.tensor(Xte, dtype=torch.float32).to(DEV).unsqueeze(1)
                 ).argmax(1).cpu().numpy()
    return float((pr == yte).mean()), best[2]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    rec = {'note': 'EEGNet 被试内:每人一个模型,70/15/15 rs=42,250Hz', 'per_subject': {}}
    accs = []
    for i, (X, y) in enumerate(subject_data):
        t0 = time.time()
        X = np.asarray(X, dtype=np.float32)
        if LAWHERN:
            X = resample_poly(X, 128, 250, axis=-1).astype(np.float32)
        X = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-7)
        y = y.astype(np.int64)
        ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42)
        acc, vb = run(ret[0], ret[1], ret[2], ret[3], ret[4], ret[5])
        rec['per_subject'][SUBJ[i]] = {'acc': acc, 'val': vb,
                                       'sec': round(time.time() - t0, 1)}
        accs.append(acc)
        print('  %s: val %.1f -> test %.1f%% (%.0fs)'
              % (SUBJ[i], 100 * vb, 100 * acc, time.time() - t0), flush=True)
    rec['summary'] = {'mean': float(np.mean(accs)), 'sd': float(np.std(accs)),
                      'low4': float(np.mean([accs[i] for i in LOW])),
                      'high5': float(np.mean([accs[i] for i in HIGH]))}
    (OUT / 'eegnet_persubject.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== EEGNet 被试内(每人一个模型) ===')
    print('均值 %.1f ±%.1f | 低四 %.1f | 高五 %.1f | 对照:经典 LE+LDA 逐人 70.1%%'
          % (100 * rec['summary']['mean'], 100 * rec['summary']['sd'],
             100 * rec['summary']['low4'], 100 * rec['summary']['high5']))
    print('[OK] %s/eegnet_persubject.json' % OUT)


if __name__ == '__main__':
    main()
