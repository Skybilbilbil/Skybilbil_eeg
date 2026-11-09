# -*- coding: utf-8 -*-
"""留一分析：把 6 支深度模型逐一/逐 seed 剔除，看融合结果对 seed 与维度的敏感度。

不训练，直接用已落盘的 pred_*.npz。固定 w = 0.70（T 验证集选出的值）不变，
只改「参与平均的模型集合」，因此测的纯粹是模型集合的抽样波动。
"""
import json
import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MAtt_baseline' / 'results' / 'leave_one_out'
D12 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d12'
D14 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d14'
D18 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18pred'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
W = 0.70


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
    pool = {}
    for seed in (42, 2024, 7):
        for tag, d in (('d12', D12), ('d14', D14), ('d18', D18)):
            f = d / ('pred_%d.npz' % seed)
            if f.exists():
                v = np.load(f)
                pool['%s_s%d' % (tag, seed)] = v['p_deep']
                ref = v
    yte, sb, p_cls = ref['yte'], ref['sb'], ref['p_cls']

    rec = {}
    def rep(name, keys):
        p = np.mean([pool[k] for k in keys], 0)
        ov, l4, h5, ps = score(W * p + (1 - W) * p_cls, yte, sb)
        rec[name] = {'n_models': len(keys), 'pooled': ov, 'low4': l4, 'high5': h5}
        print('%-42s %d 支  %.2f%%  (低四 %.1f / 高五 %.1f)'
              % (name, len(keys), 100 * ov, 100 * l4, 100 * h5), flush=True)
        return ov

    base = [k for k in pool if k.startswith(('d12', 'd18'))]
    print('=== 基准（6 支 d12+d18，w=0.70）===', flush=True)
    rep('全部 6 支', base)

    print('\n=== 逐个剔除单模型（5 支）===', flush=True)
    for k in base:
        rep('去掉 ' + k, [x for x in base if x != k])

    print('\n=== 逐个剔除整个 seed（4 支）===', flush=True)
    for s in (42, 2024, 7):
        rep('去掉全部 s%d' % s, [x for x in base if not x.endswith('_s%d' % s)])

    print('\n=== 只用单个 seed 的 2 支 ===', flush=True)
    for s in (42, 2024, 7):
        rep('只用 s%d' % s, [x for x in base if x.endswith('_s%d' % s)])

    print('\n=== 只用单个维度 ===', flush=True)
    for tag in ('d12', 'd18'):
        rep('只用 %s 的 3 个 seed' % tag, [x for x in base if x.startswith(tag)])

    print('\n=== 加入 d14 后的 9 支 ===', flush=True)
    allk = list(pool)
    rep('全部 9 支（d12+d14+d18）', allk)
    rep('去掉 d14_s2024（8 支）', [x for x in allk if x != 'd14_s2024'])

    vals = [v['pooled'] for k, v in rec.items() if k.startswith(('去掉', '只用'))]
    print('\n【汇总】剔除/替换后的 pooled 范围 %.2f–%.2f，极差 %.2fpp'
          % (100 * min(vals), 100 * max(vals), 100 * (max(vals) - min(vals))), flush=True)

    (OUT / 'leave_one_out.json').write_text(json.dumps(
        {'note': '固定 w=0.70；只改参与平均的模型集合；跨 session', 'w': W, 'results': rec},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
