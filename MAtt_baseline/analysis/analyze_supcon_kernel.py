# -*- coding: utf-8 -*-
"""T-1 续②:SupCon 前端表示接核读出 —— 它有没有增加信息总量?

## 判据
历史:LE 253 维 + RBF-SVM = 73.5(val 74.7);SupCon 只在 LE 上做过,且 SUP+MC4 的
LDA 与 supcon64 逐位相同 ⇒ 「把非线性信息线性化,总量未增」。
本脚本问:把 SupCon 挪到**深度前端**之后,(a) 该表示自身的核读出是多少?
(b) 与 LE 拼接能否超过 73.5? 若都不超 ⇒ SupCon 仍未增加信息总量。

## 臂(全部 RBF-SVM,超参在 val 折 3×3 网格选,test 只报一次)
  LE      250Hz 任务窗协方差 log 切向量 253 维      —— 锚点,须复现 73.5
  S5      64 维投影(CE 前端 + SupCon 微调)
  S0      64 维投影(CE 前端 + 未训练投影,对照)
  S5+LE   317 维
  S0+LE   317 维

产物 results/supcon_kernel/
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import build_matt_model                                # noqa: E402

OUT = BASE / 'results' / 'supcon_kernel'
CKPT = BASE / 'results' / 'supcon_frontend' / 'ckpt'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
WIN, STEP, DOWN = 94, 47, 4
TASK0 = 375                     # 250Hz 任务窗起点(与历史 LE 同口径)
SUBJ = ['S%02d' % i for i in range(1, 10)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
SEEDS = (42, 2024, 7)
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ARGS = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=True)


class Proj(nn.Module):
    def __init__(self, d_in=180, d=128, z=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(),
                                 nn.Linear(d, z))

    def forward(self, x):
        return self.net(x)


def pack(ok, sub):
    per = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
    return {'pooled': float(ok.mean()),
            'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([per[SUBJ[i]] for i in HIGH])),
            'per_subject': per}


def svm_eval(Ftr, Fva, Fte, ytr, yva, yte, sub_tr, sub_te, tag=''):
    sub_ = np.random.default_rng(1).choice(len(ytr), 300, replace=False)
    d2 = ((Ftr[sub_][:, None, :] - Ftr[sub_][None, :]) ** 2).sum(-1)
    med2 = float(np.median(d2[np.triu_indices(len(sub_), 1)]))
    best = None
    grid = {}
    for C_ in (10.0, 100.0, 1000.0):
        for gm in (0.5, 1.0, 2.0):
            m = SVC(kernel='rbf', C=C_, gamma=gm / (2 * med2)).fit(Ftr, ytr)
            v = float((m.predict(Fva) == yva).mean())
            grid['C%.0f_g%.1f' % (C_, gm)] = v
            if best is None or v > best[0]:
                best = (v, C_, gm, m)
    v, C_, gm, m = best
    ok = m.predict(Fte) == yte
    r = pack(ok, sub_te)
    r.update({'dim': int(Ftr.shape[1]), 'val': v, 'C': C_, 'gamma_mult': gm,
              'grid_val_max_is_edge': (C_ == 1000.0 or gm == 2.0)})
    okp = np.zeros(len(yte), dtype=bool)
    for i in range(9):
        tr, te = sub_tr == i, sub_te == i
        mm = SVC(kernel='rbf', C=C_, gamma=gm / (2 * med2)).fit(Ftr[tr], ytr[tr])
        okp[te] = mm.predict(Fte[te]) == yte[te]
    r['per_subject_lda_pooled'] = float(okp.mean())
    print('  %-10s %4dD val %5.1f -> test %5.1f | 低四 %5.1f 高五 %5.1f | 拆9个 %5.1f '
          '(C=%.0f g=%.1f)'
          % (tag, r['dim'], 100 * v, 100 * r['pooled'], 100 * r['low4'],
             100 * r['high5'], 100 * r['per_subject_lda_pooled'], C_, gm), flush=True)
    return r, ok


def front64(arm, seed, X):
    ck = torch.load(str(CKPT / ('%s_s%d.pt' % (arm, seed))), map_location='cpu')
    net = build_matt_model(WIN, STEP, ARGS)
    net.load_state_dict(ck['net'])
    proj = Proj().to(DEV)
    proj.load_state_dict(ck['proj'])
    net.eval()
    proj.eval()
    F = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            t = torch.Tensor(X[i:i + 128]).unsqueeze(1).to(DEV)
            F.append(net._features(t).cpu().numpy())
    F = np.concatenate(F, 0)
    with torch.no_grad():
        Z = np.concatenate([proj(torch.tensor(F[i:i + 1024], dtype=torch.float32).to(DEV)
                                 ).cpu().numpy() for i in range(0, len(F), 1024)])
    return Z


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X250 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    from scipy.signal import resample_poly
    X62 = np.concatenate([resample_poly(np.asarray(s[0], dtype=np.float32), 1, DOWN,
                                        axis=-1) for s in subject_data], 0)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X250, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, iva, ite = ret[9], ret[10], ret[11]
    ytr, yva, yte = ret[1], ret[3], ret[5]
    sub_tr, sub_te = ret[6], ret[8]
    print('[data] train %d / val %d / test %d' % (len(itr), len(iva), len(ite)), flush=True)

    L = {k: raw_cov_vec(v, TASK0) for k, v in
         (('tr', X250[itr]), ('va', X250[iva]), ('te', X250[ite]))}
    rec = {'note': 'RBF-SVM on SupCon-frontend 64D vs LE253; 超参 val 网格 3x3',
           'arms': {}}
    r, _ = svm_eval(L['tr'], L['va'], L['te'], ytr, yva, yte, sub_tr, sub_te, 'LE')

    for arm, key in (('S0_ceinit_0ep', 'S0'), ('S5_ceinit_lr1e-4', 'S5')):
        rows = []
        for seed in SEEDS:
            Ztr = front64(arm, seed, X62[itr])
            Zva = front64(arm, seed, X62[iva])
            Zte = front64(arm, seed, X62[ite])
            for vname, (a, b, c) in ((key, (Ztr, Zva, Zte)),
                                     (key + '+LE', (np.hstack([Ztr, L['tr']]),
                                                    np.hstack([Zva, L['va']]),
                                                    np.hstack([Zte, L['te']])))):
                r2, ok = svm_eval(a, b, c, ytr, yva, yte, sub_tr, sub_te,
                                  '%s s%d' % (vname, seed))
                r2['seed'] = seed
                rows.append((vname, r2))
        for vname in (key, key + '+LE'):
            sel = [x[1] for x in rows if x[0] == vname]
            ps = {s: float(np.mean([x['per_subject'][s] for x in sel])) for s in SUBJ}
            rec['arms'][vname] = {
                'dim': sel[0]['dim'],
                'pooled': float(np.mean([x['pooled'] for x in sel])),
                'low4': float(np.mean([ps[SUBJ[i]] for i in LOW])),
                'high5': float(np.mean([ps[SUBJ[i]] for i in HIGH])),
                'per_subject_lda_pooled': float(np.mean(
                    [x['per_subject_lda_pooled'] for x in sel])),
                'val': float(np.mean([x['val'] for x in sel])),
                'per_subject': ps,
                'per_seed': [x['pooled'] for x in sel],
            }
        # LE 锚点只算一次
    rec['arms']['LE'] = r
    (OUT / 'supcon_kernel.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== summary ===')
    for k, v in rec['arms'].items():
        print('%-10s %4dD test %5.1f | 低四 %5.1f 高五 %5.1f | 拆9个 %5.1f'
              % (k, v['dim'], 100 * v['pooled'], 100 * v['low4'], 100 * v['high5'],
                 100 * v['per_subject_lda_pooled']))
    print('[OK] %s/supcon_kernel.json' % OUT)


if __name__ == '__main__':
    main()
