# -*- coding: utf-8 -*-
"""A(新):SupCon **深度前端**表示(64 维)⊕ 多切图(MC8 / MC32),零训练。

## 与 10-25 那个 analyze_supcon_multichart.py 的区别
那个:SupCon 训在 **LE 253** 上,拼 MC4/MC8 ⇒ 结论「不可叠加」(SUP+MC4 与 supcon64 逐位相同)。
本脚本:SupCon 训在 **mAtt 前端**(CE 前端续训)的投影头上,身份纯度 0.501(比 0.713 更低),
拼 **MC8/MC32**。问的是:深度前端能否给「当前最强共享线性特征 MC32」加东西。

## 臂(rs=42 同折;S5 部分 3 seed;读出 = 历史同款闭式 LW-LDA)
  L0 单切图 253(锚点 62.9)   MC8 2024   MC32 8096
  S5 64(锚点 60.8)            S5+MC8     S5+MC32

## 判据
S5+MCk − MCk > 5pp 才算「深度前端有正贡献」;否则与历史「深度后端对经典读出零增益」一致。
产物 results/supcon_front_multichart/
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from analyze_multichart_ceiling import cov_tn, chart_feats, fit_lda, TASK  # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import build_matt_model                                # noqa: E402

OUT = BASE / 'results' / 'supcon_front_multichart'
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


def eval_arm(Ftr, Fte, ytr, yte, sub_tr, sub_te, do_per=True):
    clf = fit_lda(Ftr, ytr)
    ok = clf.predict(Fte) == yte
    r = pack(ok, sub_te)
    if do_per:
        okp = np.zeros(len(yte), dtype=bool)
        for i in range(9):
            tr, te = sub_tr == i, sub_te == i
            c = fit_lda(Ftr[tr], ytr[tr])
            okp[te] = c.predict(Fte[te]) == yte[te]
        r['per_subject_lda_pooled'] = float(okp.mean())
    else:
        r['per_subject_lda_pooled'] = float('nan')
    return r


def front64(seed, X):
    ck = torch.load(str(CKPT / ('S5_ceinit_lr1e-4_s%d.pt' % seed)), map_location='cpu')
    net = build_matt_model(WIN, STEP, ARGS)
    net.load_state_dict(ck['net'])
    proj = Proj().to(DEV)
    proj.load_state_dict(ck['proj'])
    net.eval()
    proj.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(net._features(torch.Tensor(X[i:i + 128]).unsqueeze(1).to(DEV)
                                     ).cpu().numpy())
    F = np.concatenate(out, 0)
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
    itr, ite = ret[9], ret[11]
    ytr, yte, sub_tr, sub_te = ret[1], ret[5], ret[6], ret[8]
    C_tr = cov_tn(X250[itr][:, :, TASK[0]:TASK[1]])
    C_te = cov_tn(X250[ite][:, :, TASK[0]:TASK[1]])
    print('[data] train %d test %d' % (len(ytr), len(yte)), flush=True)

    cache = OUT / 'mc_cache.npz'
    if cache.exists():
        z = np.load(cache)
        MC = {8: (z['m8tr'], z['m8te']), 32: (z['m32tr'], z['m32te'])}
        print('[cache] MC 已缓存', flush=True)
    else:
        MC = {}
        for k in (8, 32):
            rng = np.random.default_rng(0)
            refs = [C_tr[i] for i in rng.choice(len(C_tr), k, replace=False)]
            a = np.concatenate([chart_feats(C_tr, R)[0] for R in refs], 1).astype(np.float32)
            b = np.concatenate([chart_feats(C_te, R)[0] for R in refs], 1).astype(np.float32)
            MC[k] = (a, b)
            print('  [MC%d] %s' % (k, a.shape), flush=True)
        np.savez_compressed(cache, m8tr=MC[8][0], m8te=MC[8][1],
                            m32tr=MC[32][0], m32te=MC[32][1])

    rec = {'note': 'SupCon 深度前端 64D + 多切图;读出 = 历史同款闭式 LW-LDA', 'arms': {}}
    L0_tr, L0_te = raw_cov_vec(X250[itr], TASK[0]), raw_cov_vec(X250[ite], TASK[0])
    for name, (a, b) in (('L0', (L0_tr, L0_te)), ('MC8', MC[8]), ('MC32', MC[32])):
        r = eval_arm(a, b, ytr, yte, sub_tr, sub_te, do_per=(name != 'MC32'))
        r['dim'] = int(a.shape[1])
        rec['arms'][name] = r
        print('[%s] %d维 pooled %.1f | 低四 %.1f 高五 %.1f | 拆9个 %.1f'
              % (name, r['dim'], 100 * r['pooled'], 100 * r['low4'], 100 * r['high5'],
                 100 * r['per_subject_lda_pooled']), flush=True)

    for seed in SEEDS:
        Ztr, Zte = front64(seed, X62[itr]), front64(seed, X62[ite])
        cand = {'S5': (Ztr, Zte)}
        for k in (8, 32):
            cand['S5+MC%d' % k] = (np.hstack([Ztr, MC[k][0]]), np.hstack([Zte, MC[k][1]]))
        for name, (a, b) in cand.items():
            r = eval_arm(a, b, ytr, yte, sub_tr, sub_te, do_per=(name != 'S5+MC32'))
            r['dim'] = int(a.shape[1])
            rec['arms'].setdefault(name, {'per_seed': {}})['per_seed'][str(seed)] = r
            print('  [%s s%d] %d维 pooled %.1f | 低四 %.1f 高五 %.1f | 拆9个 %.1f'
                  % (name, seed, r['dim'], 100 * r['pooled'], 100 * r['low4'],
                     100 * r['high5'], 100 * r['per_subject_lda_pooled']), flush=True)

    for name, v in list(rec['arms'].items()):
        if 'per_seed' in v:
            sel = list(v['per_seed'].values())
            agg = {s: float(np.mean([x['per_subject'][s] for x in sel])) for s in SUBJ}
            rec['arms'][name] = {
                'dim': sel[0]['dim'],
                'pooled': float(np.mean([x['pooled'] for x in sel])),
                'low4': float(np.mean([agg[SUBJ[i]] for i in LOW])),
                'high5': float(np.mean([agg[SUBJ[i]] for i in HIGH])),
                'per_subject_lda_pooled': float(np.nanmean(
                    [x['per_subject_lda_pooled'] for x in sel])),
                'per_subject': agg,
                'per_seed': {k: x['pooled'] for k, x in v['per_seed'].items()},
            }
    (OUT / 'supcon_front_multichart.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== A:SupCon 深度前端 ⊕ 多切图 ===')
    for k, v in rec['arms'].items():
        print('%-9s %5d维 pooled %5.1f | 低四 %5.1f 高五 %5.1f | 拆9个 %5.1f'
              % (k, v['dim'], 100 * v['pooled'], 100 * v['low4'], 100 * v['high5'],
                 100 * v['per_subject_lda_pooled']))
    for k in (8, 32):
        d = 100 * (rec['arms']['S5+MC%d' % k]['pooled'] - rec['arms']['MC%d' % k]['pooled'])
        print('S5+MC%d - MC%d = %+.2fpp' % (k, k, d))
    print('[OK] %s/supcon_front_multichart.json' % OUT)


if __name__ == '__main__':
    main()
