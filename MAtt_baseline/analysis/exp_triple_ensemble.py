# -*- coding: utf-8 -*-
"""三路集成：d_attn_out=12 深度 ⊕ d_attn_out=18 深度 ⊕ 经典 LE+LDA。

动机：单维度的两支在不同分组上偏科——d=12 在 seed42 上高五 76.5（比 d=18 的 75.2 高
1.3pp），而 d=18 的低四 48.7（比 d=12 的 47.5 高 1.2pp）。若两支的错误不高度相关，
三路组合可能同时吃到两边的好处。

权重只在 T 的验证集（389 样本，两模型同一 rng 切分）上搜，E 仅前向。
同时报告两路对照，避免把"三路"的收益归因错。
"""
import json
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MAtt_baseline' / 'results' / 'triple_ensemble'
D12 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d12' / 'pred_42.npz'
D18 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18pred' / 'pred_42.npz'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


def grp(ps):
    return (float(np.mean([ps[SUBJ[i]] for i in LOW])),
            float(np.mean([ps[SUBJ[i]] for i in HIGH])))


def score(p, y, sb):
    ok = p.argmax(1) == y
    ps = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    l4, h5 = grp(ps)
    return float(ok.mean()), l4, h5, ps


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    a = np.load(D12)
    b = np.load(D18)
    assert np.allclose(a['yva'], b['yva']), '两模型的验证集划分不一致'
    yva, yte, sb = a['yva'], a['yte'], a['sb']
    pv = {'d12': a['p_d_va'], 'd18': b['p_d_va'], 'cls': a['p_c_va']}
    pe = {'d12': a['p_deep'], 'd18': b['p_deep'], 'cls': a['p_cls']}

    rec = {}
    # 单路
    for k in pv:
        ov, l4, h5, ps = score(pe[k], yte, sb)
        rec['单路 ' + k] = {'pooled': ov, 'low4': l4, 'high5': h5, 'per_subject': ps}
        print('单路 %-4s %.2f%%  (低四 %.1f / 高五 %.1f)' % (k, 100 * ov, 100 * l4, 100 * h5), flush=True)

    # 两路（验证集搜 w）
    for pair in (('d12', 'cls'), ('d18', 'cls'), ('d12', 'd18')):
        best = None
        for w in np.arange(0.0, 1.001, 0.05):
            pvv = w * pv[pair[0]] + (1 - w) * pv[pair[1]]
            acc = float((pvv.argmax(1) == yva).mean())
            if best is None or acc > best[1]:
                best = (float(w), acc)
        p = best[0] * pe[pair[0]] + (1 - best[0]) * pe[pair[1]]
        ov, l4, h5, ps = score(p, yte, sb)
        rec['两路 %s+%s' % pair] = {'pooled': ov, 'low4': l4, 'high5': h5,
                                    'w': best[0], 'val_acc': best[1], 'per_subject': ps}
        print('两路 %-4s+%-4s w=%.2f  %.2f%%  (低四 %.1f / 高五 %.1f)'
              % (pair[0], pair[1], best[0], 100 * ov, 100 * l4, 100 * h5), flush=True)

    # 三路（单纯形网格 0.05 步长）
    best3 = None
    grid = np.arange(0.0, 1.001, 0.05)
    for w1 in grid:
        for w2 in grid:
            if w1 + w2 > 1.0 + 1e-9:
                continue
            w3 = 1 - w1 - w2
            pvv = w1 * pv['d12'] + w2 * pv['d18'] + w3 * pv['cls']
            acc = float((pvv.argmax(1) == yva).mean())
            if best3 is None or acc > best3[1]:
                best3 = (float(w1), float(w2), float(w3), acc)
    p = best3[0] * pe['d12'] + best3[1] * pe['d18'] + best3[2] * pe['cls']
    ov, l4, h5, ps = score(p, yte, sb)
    rec['三路 d12+d18+cls'] = {'pooled': ov, 'low4': l4, 'high5': h5,
                               'w': best3[:3], 'val_acc': best3[3], 'per_subject': ps}
    print('三路 d12+d18+cls w=(%.2f, %.2f, %.2f)  %.2f%%  (低四 %.1f / 高五 %.1f)'
          % (best3[0], best3[1], best3[2], 100 * ov, 100 * l4, 100 * h5), flush=True)

    # 等权三路（不看验证集，避免选权重过拟合）作为稳健对照
    p = (pe['d12'] + pe['d18'] + pe['cls']) / 3
    ov, l4, h5, ps = score(p, yte, sb)
    rec['三路等权 (1/3 each)'] = {'pooled': ov, 'low4': l4, 'high5': h5, 'per_subject': ps}
    print('三路等权       %.2f%%  (低四 %.1f / 高五 %.1f)' % (100 * ov, 100 * l4, 100 * h5), flush=True)

    (OUT / 'triple_ensemble.json').write_text(json.dumps(
        {'note': '跨 session;seed 42;权重在 T 验证集上搜,E 仅前向', **rec},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
