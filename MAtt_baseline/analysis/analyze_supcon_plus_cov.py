# -*- coding: utf-8 -*-
"""T-1 续:SupCon 前端(64 维)⊕ 经典协方差旁路(253 维),零训练。

依据:E-1 已证「深度后端 180 维 + 250 Hz 协方差旁路」= 63.5(不显著超 62.9);
T-1 已证「CE 前端 + SupCon 微调 100ep/lr1e-4 → 64 维投影 + 闭式 LDA」= 60.8 ±0.8,
比同起点未训练投影(56.7 ±0.3)显著高 4.1pp(3/3 seed p<0.05),且拆成 9 个只涨 1.8pp。
本脚本问:把这套 64 维表示接上经典协方差旁路,能否超过经典零训练基线 62.9?

臂(全部共享闭式 LW-LDA,rs=42 同一 train/test 折):
  R250  250Hz 任务窗协方差 253 维       —— 锚点,须复现 62.9
  R62   62.5Hz 同构造                    —— 须复现 61.4
  S5    64 维投影(CE 前端 + SupCon)      —— 须复现 60.8
  S0    64 维投影(CE 前端 + 未训练投影)  —— 须复现 56.7
  P1/P2/P3 拼接臂                       Q1  S0 ⊕ R250 对照
产物 results/supcon_plus_cov/
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import resample_poly
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import build_matt_model                                # noqa: E402

OUT = BASE / 'results' / 'supcon_plus_cov'
CKPT = BASE / 'results' / 'supcon_frontend' / 'ckpt'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
WIN, STEP, DOWN = 94, 47, 4
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


def fit_eval(Ftr, Fte, ytr, yte, sub_tr, sub_te):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
    ok = clf.predict(Fte) == yte
    per = pack(ok, sub_te)
    okp = np.zeros(len(yte), dtype=bool)
    for i in range(9):
        tr, te = sub_tr == i, sub_te == i
        c = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr[tr], ytr[tr])
        okp[te] = c.predict(Fte[te]) == yte[te]
    per['per_subject_lda_pooled'] = float(okp.mean())
    return per, ok


def load_front(arm, seed, Xtr, Xte):
    """加载 ckpt → (64 维投影特征, 180 维前端特征)。"""
    ck = torch.load(str(CKPT / ('%s_s%d.pt' % (arm, seed))), map_location='cpu')
    net = build_matt_model(WIN, STEP, ARGS)
    net.load_state_dict(ck['net'])
    proj = Proj().to(DEV)
    proj.load_state_dict(ck['proj'])
    net.eval()
    proj.eval()

    def f180(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                t = torch.Tensor(X[i:i + 128]).unsqueeze(1).to(DEV)
                out.append(net._features(t).cpu().numpy())
        return np.concatenate(out, 0)

    def f64(F):
        with torch.no_grad():
            return np.concatenate([proj(torch.tensor(F[i:i + 1024],
                                                     dtype=torch.float32).to(DEV)
                                        ).cpu().numpy()
                                   for i in range(0, len(F), 1024)])
    Ftr, Fte = f180(Xtr), f180(Xte)
    return f64(Ftr), f64(Fte), Ftr, Fte


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X250 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    X62 = np.concatenate([resample_poly(np.asarray(s[0], dtype=np.float32), 1, DOWN,
                                        axis=-1) for s in subject_data], 0)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X250, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, ite = ret[9], ret[11]
    ytr, yte, sub_tr, sub_te = ret[1], ret[5], ret[6], ret[8]
    print('[data] 250Hz %s / 62.5Hz %s | train %d test %d'
          % (X250.shape, X62.shape, len(itr), len(ite)), flush=True)

    R250_tr = raw_cov_vec(X250[itr], WIN * DOWN)
    R250_te = raw_cov_vec(X250[ite], WIN * DOWN)
    R62_tr, R62_te = raw_cov_vec(X62[itr], WIN), raw_cov_vec(X62[ite], WIN)
    rec = {'note': 'SupCon frontend 64D (+) classic covariance side-channel, zero-train LW-LDA',
           'arms': {}, 'paired': {}}
    pred = {}
    for name, (a, b) in (('R250', (R250_tr, R250_te)), ('R62', (R62_tr, R62_te))):
        r, ok = fit_eval(a, b, ytr, yte, sub_tr, sub_te)
        r['dim'] = int(a.shape[1])
        rec['arms'][name] = r
        pred[name] = ok
        print('[%s] %d维 pooled %.1f | 低四 %.1f 高五 %.1f | 拆9个 %.1f'
              % (name, a.shape[1], 100 * r['pooled'], 100 * r['low4'], 100 * r['high5'],
                 100 * r['per_subject_lda_pooled']), flush=True)

    for arm, key in (('S0_ceinit_0ep', 'S0'), ('S5_ceinit_lr1e-4', 'S5')):
        rows = []
        for seed in SEEDS:
            Ztr, Zte, Ftr, Fte = load_front(arm, seed, X62[itr], X62[ite])
            variants = {
                key: (Ztr, Zte),
                key + '+R250': (np.hstack([Ztr, R250_tr]), np.hstack([Zte, R250_te])),
                key + '+R62': (np.hstack([Ztr, R62_tr]), np.hstack([Zte, R62_te])),
                key + '+R250+R62': (np.hstack([Ztr, R250_tr, R62_tr]),
                                    np.hstack([Zte, R250_te, R62_te])),
            }
            for vname, (A_tr, A_te) in variants.items():
                r, ok = fit_eval(A_tr, A_te, ytr, yte, sub_tr, sub_te)
                r['dim'] = int(A_tr.shape[1])
                r['seed'] = seed
                rows.append((vname, r, ok))
                print('  [%s s%d] %d维 pooled %.1f | 低四 %.1f 高五 %.1f | 拆9个 %.1f'
                      % (vname, seed, r['dim'], 100 * r['pooled'], 100 * r['low4'],
                         100 * r['high5'], 100 * r['per_subject_lda_pooled']), flush=True)
        for vname in [key, key + '+R250', key + '+R62', key + '+R250+R62']:
            sel = [x for x in rows if x[0] == vname]
            ps = {s: float(np.mean([x[1]['per_subject'][s] for x in sel])) for s in SUBJ}
            rec['arms'][vname] = {
                'dim': sel[0][1]['dim'],
                'pooled': float(np.mean([x[1]['pooled'] for x in sel])),
                'low4': float(np.mean([ps[SUBJ[i]] for i in LOW])),
                'high5': float(np.mean([ps[SUBJ[i]] for i in HIGH])),
                'per_subject_lda_pooled': float(np.mean(
                    [x[1]['per_subject_lda_pooled'] for x in sel])),
                'per_subject': ps,
                'per_seed': {str(x[1]['seed']): x[1]['pooled'] for x in sel},
            }
            if vname != key:
                mcs = []
                for x in sel:
                    a_, b_ = x[2], pred['R250']
                    n01, n10 = int(np.sum(~a_ & b_)), int(np.sum(a_ & ~b_))
                    p = 1.0 if n01 + n10 == 0 else float(
                        binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
                    mcs.append({'seed': x[1]['seed'], 'a_only': n10, 'b_only': n01, 'p': p})
                rec['paired'][vname] = mcs

    (OUT / 'supcon_plus_cov.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== summary (3 seed) ===')
    for k, v in rec['arms'].items():
        print('%-16s %4dD pooled %5.1f | low4 %5.1f high5 %5.1f | split9 %5.1f'
              % (k, v['dim'], 100 * v['pooled'], 100 * v['low4'], 100 * v['high5'],
                 100 * v['per_subject_lda_pooled']))
    print('')
    print('vs R250 (62.9) paired McNemar:')
    for k, v in rec['paired'].items():
        txt = '  '.join('s%d p=%.3f' % (x['seed'], x['p']) for x in v)
        print('  %-16s %s' % (k, txt))
    print('[OK] %s/supcon_plus_cov.json' % OUT)


if __name__ == '__main__':
    main()
