# -*- coding: utf-8 -*-
"""全新 seed 的外部验证：换三个从未参与过任何选择的 seed，看融合读数是否还在同一水平。

已有 6 支用 seed ∈ {42,2024,7}，而 w=0.70 是在 seed42 那两次运行的验证集上选出来的——
因此那 6 支对 seed42 有轻微依赖。本脚本用全新的 seed ∈ {123,456,789} 训练 d_attn_out=18，
考察：① 新 seed 单独水平；② 纯新 seed 集成；③ 新旧混合；④ 全部 9 支。
固定 w=0.70（沿用原选择，不再重新调），测的是纯粹的抽样波动。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MAtt_baseline' / 'results' / 'seed_robustness'
DIRS = {
    'd12_old': ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d12',
    'd18_old': ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18pred',
    'd18_new': ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18new',
}
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
    pool, ref = {}, None
    for tag, d in DIRS.items():
        for f in sorted(d.glob('pred_*.npz')):
            v = np.load(f)
            pool['%s_%s' % (tag, f.stem.replace('pred_', 's'))] = v['p_deep']
            ref = ref or v
    if ref is None:
        raise SystemExit('没有预测文件')
    yte, sb, p_cls = ref['yte'], ref['sb'], ref['p_cls']

    rec = {}
    def rep(name, keys):
        p = np.mean([pool[k] for k in keys], 0)
        ov, l4, h5, ps = score(W * p + (1 - W) * p_cls, yte, sb)
        rec[name] = {'n_models': len(keys), 'models': keys,
                     'pooled': ov, 'low4': l4, 'high5': h5, 'per_subject': ps}
        print('%-40s %2d 支  %.2f%%  (低四 %.1f / 高五 %.1f)'
              % (name, len(keys), 100 * ov, 100 * l4, 100 * h5), flush=True)
        return ov

    old = [k for k in pool if '_old_' in k or k.startswith('d12_old')]
    new = [k for k in pool if k.startswith('d18_new')]
    old = [k for k in pool if not k.startswith('d18_new')]

    print('=== 单支（新 seed，d=18）===', flush=True)
    for k in new:
        ov, l4, h5, _ = score(W * pool[k] + (1 - W) * p_cls, yte, sb)
        print('%-40s      %.2f%%  (低四 %.1f / 高五 %.1f)' % (k, 100 * ov, 100 * l4, 100 * h5), flush=True)
        rec['单路 ' + k] = {'pooled': ov, 'low4': l4, 'high5': h5}

    print('\n=== 组合（w 固定 0.70）===', flush=True)
    rep('原 6 支（seed 42/2024/7）', old)
    rep('全新 3 支（seed 123/456/789）', new)
    rep('新旧混合 6 支（d12 旧3 + d18 新3）', [k for k in old if 'd12' in k] + new)
    rep('全部 9 支', list(pool))

    vals = [v['pooled'] for k, v in rec.items() if not k.startswith('单路')]
    print('\n【汇总】四种集合 pooled = %s，极差 %.2fpp'
          % (['%.2f' % (100 * v) for v in vals], 100 * (max(vals) - min(vals))), flush=True)

    (OUT / 'seed_robustness.json').write_text(json.dumps(
        {'note': 'w 固定 0.70；新 seed 从未参与任何选择；跨 session', 'w': W,
         'runs': list(pool), 'results': rec}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
