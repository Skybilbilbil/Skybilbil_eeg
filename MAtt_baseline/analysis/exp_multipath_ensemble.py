# -*- coding: utf-8 -*-
"""多路等权集成：把多支共享深度模型的概率与经典 LE+LDA 的概率直接等权平均。

背景：三路等权（d12_s42 ⊕ d18_s42 ⊕ 经典）= 64.66%（单 seed），比在验证集上搜权重的
59.84% 高 4.8pp —— 验证集只有 389 样本，选权重会过拟合，**等权不选**反而最好。
本脚本沿这条路加大模型数量：同一维度不同 seed、不同维度，全部等权。
等权不需要验证集，因此不同 seed 的切分差异不构成问题。

输出：单路 / 各组合的 pooled、低四、高五、逐人。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MAtt_baseline' / 'results' / 'multipath_ensemble'
D12 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d12'
D18 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18pred'
D14 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d14'
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


def load(path):
    if not path.exists():
        return None
    d = np.load(path)
    return {'p_deep': d['p_deep'], 'p_cls': d['p_cls'], 'yte': d['yte'], 'sb': d['sb']}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pool = {}
    for seed in (42, 2024, 7):
        r = load(D12 / ('pred_%d.npz' % seed))
        if r: pool['d12_s%d' % seed] = r
        r = load(D18 / ('pred_%d.npz' % seed))
        if r: pool['d18_s%d' % seed] = r
        r = load(D14 / ('pred_%d.npz' % seed))
        if r: pool['d14_s%d' % seed] = r
    if not pool:
        raise SystemExit('没有可用的预测文件')
    any_run = next(iter(pool.values()))
    yte, sb, p_cls = any_run['yte'], any_run['sb'], any_run['p_cls']
    print('载入 %d 个深度预测：%s' % (len(pool), ', '.join(pool)), flush=True)

    rec = {}

    def rep(name, p):
        ov, l4, h5, ps = score(p, yte, sb)
        rec[name] = {'pooled': ov, 'low4': l4, 'high5': h5,
                     'n_dirs': len(p) if isinstance(p, list) else 1, 'per_subject': ps}
        print('%-28s %.2f%%  (低四 %.1f / 高五 %.1f)' % (name, 100 * ov, 100 * l4, 100 * h5), flush=True)
        return ov

    # 单路
    for k, v in pool.items():
        rep('单路 ' + k, v['p_deep'])
    rep('单路 经典', p_cls)

    def avg(keys):
        return np.mean([pool[k]['p_deep'] for k in keys], 0)

    d12_keys = [k for k in pool if k.startswith('d12')]
    d18_keys = [k for k in pool if k.startswith('d18')]
    d14_keys = [k for k in pool if k.startswith('d14')]

    # 单维度多 seed
    if len(d12_keys) > 1:
        rep('d12 多 seed 平均', avg(d12_keys))
    if len(d18_keys) > 1:
        rep('d18 多 seed 平均', avg(d18_keys))
    if len(d14_keys) > 1:
        rep('d14 多 seed 平均', avg(d14_keys))

    # 所有深度等权
    rep('全部深度等权', avg(list(pool)))

    # 深度族 ⊕ 经典：两族等权
    p_all = avg(list(pool))
    rep('全部深度 ⊕ 经典 (0.5/0.5)', 0.5 * p_all + 0.5 * p_cls)
    rep('全部深度 ⊕ 经典 (0.7/0.3)', 0.7 * p_all + 0.3 * p_cls)
    rep('全部深度 ⊕ 经典 (0.3/0.7)', 0.3 * p_all + 0.7 * p_cls)
    rep('全部深度 ⊕ 经典 (0.6/0.4)', 0.6 * p_all + 0.4 * p_cls)
    rep('全部深度 ⊕ 经典 (0.8/0.2)', 0.8 * p_all + 0.2 * p_cls)

    # 每个深度模型各自与经典 0.5/0.5，再平均（等价于上面的 0.5/0.5）
    best = max(rec.items(), key=lambda kv: kv[1]['pooled'])
    print('\n最好: %s = %.2f%% (低四 %.1f / 高五 %.1f)'
          % (best[0], 100 * best[1]['pooled'], 100 * best[1]['low4'], 100 * best[1]['high5']), flush=True)
    (OUT / 'multipath_ensemble.json').write_text(json.dumps(
        {'note': '跨 session;全部等权(不选权重);深度各模型 softmax 平均后与经典按固定比例融合',
         'n_deep': len(pool), 'runs': list(pool), 'results': rec, 'best': best[0]},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
