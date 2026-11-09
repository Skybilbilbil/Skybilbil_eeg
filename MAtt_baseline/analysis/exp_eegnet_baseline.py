# -*- coding: utf-8 -*-
"""现代基线:EEGNet-8,2 在主协议下的读数(一次性训练实验,exp_ 前缀)。

## 为什么
对照表不能只有 CSP+LDA(2008) 这种最低基线。EEGNet(Lawhern et al. 2018)是 BCI 领域
被引用最多的 CNN 基线,是审稿人必问的一格。本脚本在**完全相同的口径**下训练它:
主协议 subject-dependent 70/15/15(rs=42)、输入 62.5 Hz、不接收 subject_id、3 seed。

## 结构(EEGNet-8,2 原版,适配 22 通道)
  Conv2d(1→8, (1,64), same) → BN → DepthwiseConv2d(8→16, (22,1)) → BN → ELU
  → AvgPool(1,4) → Dropout(0.25) → SeparableConv2d(16→16, (1,16), same) → BN → ELU
  → AvgPool(1,8) → Dropout(0.25) → Flatten → Linear(→4)
训练:Adam 1e-3,100 epoch(与项目惯例同口径),bs=128;val 选模,test 只读一次。

产物 results/eegnet_baseline/
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

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
HZ250 = '--hz250' in sys.argv   # 与经典读出(LE/多切图/核)同数据源:250Hz 原始
RAW = '--raw' in sys.argv       # 逆 EA 还原(配对消融另一臂)
SAMPLE = float(next((a.split('=')[1] for a in sys.argv if a.startswith('--sample=')), 1.0))
LAWHERN = '--lawhern' in sys.argv   # 原文配置:128Hz / 500 epoch / dropout0.5(被试内) / val-loss 早停
SUF = '' if SAMPLE >= 1.0 else ('_s%02d' % int(round(SAMPLE * 100)))
DOWN = 1 if HZ250 else 4
OUT = Path(str(BASE / 'results' / (
    ('eegnet_raw_%s' % ('250hz' if HZ250 else '62hz')) if RAW
    else ('eegnet_baseline_250hz' if HZ250 else 'eegnet_baseline'))) + SUF)
SUBJ = ['S%02d' % i for i in range(1, 10)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
SEEDS = (42,) if (RAW or LAWHERN) else (42, 2024, 7)
N_EPOCH = 500 if LAWHERN else 100
DROP = 0.5 if LAWHERN else 0.25
if LAWHERN:
    OUT = Path(str(OUT) + '_lawhern')
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


class EEGNet(nn.Module):
    def __init__(self, n_ch=22, n_cls=4, F1=8, D=2, F2=16, kt=64, drop=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(1, F1, (1, kt), padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.dw = nn.Conv2d(F1, F1 * D, (n_ch, 1), groups=F1, bias=False)
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(drop)
        self.sep = nn.Conv2d(F1 * D, F2, (1, 16), padding='same', bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(drop)
        self.out = nn.LazyLinear(n_cls)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = torch.nn.functional.elu(self.bn2(self.dw(x)))
        x = self.drop1(self.pool1(x))
        x = torch.nn.functional.elu(self.bn3(self.sep(x)))
        x = self.drop2(self.pool2(x))
        return self.out(torch.flatten(x, 1))


def pack(ok, sub_te):
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    acc = float(ok.mean())
    return {'acc': acc, 'kappa': (acc - 0.25) / 0.75, 'per_subject': per,
            'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([per[SUBJ[i]] for i in HIGH]))}


def run(Xtr, ytr, Xva, yva, Xte, yte, seed):
    torch.manual_seed(seed)
    net = EEGNet(drop=DROP).to(DEV)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    Xv = torch.tensor(Xva, dtype=torch.float32).to(DEV)
    rng = np.random.default_rng(seed)
    best = (None, None, 0.0, 0.0)
    for ep in range(N_EPOCH):
        net.train()
        perm = rng.permutation(len(Xt))
        for i in range(0, len(Xt) - 127, 128):
            idx = perm[i:i + 128]
            xb = Xt[idx].to(DEV).unsqueeze(1)
            yb = yt[idx].to(DEV)
            loss = nn.functional.cross_entropy(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if ep % 5 == 4 or ep == N_EPOCH - 1:
            net.eval()
            with torch.no_grad():
                logits = net(Xv.unsqueeze(1))
                pr = logits.argmax(1).cpu().numpy()
                vloss = float(nn.functional.cross_entropy(
                    logits, torch.tensor(yva, dtype=torch.long).to(DEV)))
            v = float((pr == yva).mean())
            score = -vloss if LAWHERN else v      # 原文按 val-loss 早停
            if best[0] is None or score > best[0]:
                best = (score, {k: t.clone() for k, t in net.state_dict().items()},
                        v, vloss)
    net.load_state_dict(best[1])
    net.eval()
    with torch.no_grad():
        pr = net(torch.tensor(Xte, dtype=torch.float32).to(DEV).unsqueeze(1)
                 ).argmax(1).cpu().numpy()
    return pr, best[2], best[3]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, meta = load_preprocessed_dataset(str(DATA), sample_ratio=SAMPLE)
    if RAW:
        conv = []
        for i, (X, yy) in enumerate(subject_data):
            P = np.asarray(meta[i]['ea_matrix'], dtype=np.float64)
            conv.append((np.einsum('ij,njt->nit', np.linalg.inv(P),
                                   np.asarray(X, dtype=np.float64)).astype(np.float32), yy))
        subject_data = conv
        print('[input_align] raw: X <- P^{-1}X', flush=True)
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    if LAWHERN:                       # 原文 128Hz(250 -> 128 有理重采样)
        X62 = np.concatenate([resample_poly(np.asarray(s[0], dtype=np.float32), 128, 250,
                                            axis=-1) for s in subject_data], 0)
    elif DOWN > 1:
        X62 = np.concatenate([resample_poly(np.asarray(s[0], dtype=np.float32), 1, DOWN,
                                            axis=-1) for s in subject_data], 0)
    else:
        X62 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    # 逐试次 z-score(EEGNet 的标准输入处理;与 LE 的 trace 归一不同,但都是无标签变换)
    X62 = (X62 - X62.mean(-1, keepdims=True)) / (X62.std(-1, keepdims=True) + 1e-7)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X62, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, iva, ite = ret[9], ret[10], ret[11]
    ytr, yva, yte, sub_te = ret[1], ret[3], ret[5], ret[8]
    print('[data] %s | train %d val %d test %d'
          % (X62.shape, len(itr), len(iva), len(ite)), flush=True)

    rec = {'note': 'EEGNet-8,2;62.5Hz;主协议 rs=42;3 seed;val 选模', 'per_seed': {}}
    accs, ks, l4, h5 = [], [], [], []
    for seed in SEEDS:
        t0 = time.time()
        pr, vb, vlb = run(X62[itr], ytr, X62[iva], yva, X62[ite], yte, seed)
        r = pack(pr == yte, sub_te)
        r['val'] = vb
        r['sec'] = round(time.time() - t0, 1)
        rec['per_seed'][str(seed)] = r
        accs.append(r['acc'])
        ks.append(r['kappa'])
        l4.append(r['low4'])
        h5.append(r['high5'])
        print('  seed%d: val %.1f -> test %.1f%% | kappa %.3f | 低四 %.1f 高五 %.1f | %.0fs'
              % (seed, 100 * vb, 100 * r['acc'], r['kappa'], 100 * r['low4'],
                 100 * r['high5'], r['sec']), flush=True)

    ps = {s: float(np.mean([rec['per_seed'][str(x)]['per_subject'][s] for x in SEEDS]))
          for s in SUBJ}
    rec['summary'] = {'acc': float(np.mean(accs)), 'acc_sd': float(np.std(accs)),
                      'kappa': float(np.mean(ks)), 'low4': float(np.mean(l4)),
                      'high5': float(np.mean(h5)), 'per_subject': ps}
    (OUT / 'eegnet_baseline.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== EEGNet-8,2 主协议(3 seed) ===')
    print('acc %.2f ±%.2f | kappa %.3f | 低四 %.1f 高五 %.1f'
          % (100 * rec['summary']['acc'], 100 * rec['summary']['acc_sd'],
             100 * rec['summary']['kappa'], 100 * rec['summary']['low4'],
             100 * rec['summary']['high5']))
    print('[OK] %s/eegnet_baseline.json' % OUT)


if __name__ == '__main__':
    main()
