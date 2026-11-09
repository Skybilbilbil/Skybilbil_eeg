# -*- coding: utf-8 -*-
"""把「深度族 ⊕ 经典」的融合权重放回 T 的验证集选，确认它能合法地选到高点。

背景：多路集成里 6 个深度模型等权 ⊕ 经典，固定比例 0.7/0.3 时 test 64.93%、0.5/0.5 时
63.00%、0.3/0.7 时 60.80%。但这三个比例是直接看 test 试的，**不能作为报数依据**。

合法做法：权重只许用 T 内部信息决定。这里用 seed 42 那两次运行共享的验证集（389 样本，
d12_s42 与 d18_s42 切分相同）：
  - 深度侧验证概率 = 两个模型在验证集上的 softmax 平均
  - 经典侧验证概率 = 同一切分下 LDA 的 predict_proba
在其中搜 w，再把该 w 用到 test。
同时给出「按模型数量加权」的无信息先验（6 深度 : 1 经典 = 0.857/0.143）作对照。
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'MAtt_baseline' / 'results' / 'fusion_weight'
D12 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d12'
D18 = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18pred'
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
    a = np.load(D12 / 'pred_42.npz')
    b = np.load(D18 / 'pred_42.npz')
    assert np.allclose(a['yva'], b['yva'])

    # 验证集：两深度模型平均 ⊕ 经典
    pv_deep = (a['p_d_va'] + b['p_d_va']) / 2
    pv_cls = a['p_c_va']
    yva = a['yva']
    curve = {}
    for w in np.round(np.arange(0.0, 1.001, 0.05), 2):
        acc = float(((w * pv_deep + (1 - w) * pv_cls).argmax(1) == yva).mean())
        curve['%.2f' % w] = acc
    w_best = float(max(curve.items(), key=lambda kv: kv[1])[0])
    print('验证集（389 样本）上搜权重，曲线：', flush=True)
    for k in ('0.30', '0.40', '0.50', '0.60', '0.70', '0.80', '0.90'):
        if k in curve:
            print('   w=%s -> %.2f%%' % (k, 100 * curve[k]), flush=True)
    print('   -> 验证集选中 w = %.2f (%.2f%%)' % (w_best, 100 * curve['%.2f' % w_best]), flush=True)

    # E 上：6 个深度模型等权
    keys = []
    for seed in (42, 2024, 7):
        keys.append(np.load(D12 / ('pred_%d.npz' % seed)))
        keys.append(np.load(D18 / ('pred_%d.npz' % seed)))
    pe_deep = np.mean([k['p_deep'] for k in keys], 0)
    pe_cls = keys[0]['p_cls']
    yte, sb = keys[0]['yte'], keys[0]['sb']

    res = {}
    def rep(name, w):
        ov, l4, h5, ps = score(w * pe_deep + (1 - w) * pe_cls, yte, sb)
        res[name] = {'w_deep': w, 'pooled': ov, 'low4': l4, 'high5': h5, 'per_subject': ps}
        print('%-38s w=%.3f  %.2f%%  (低四 %.1f / 高五 %.1f)'
              % (name, w, 100 * ov, 100 * l4, 100 * h5), flush=True)

    rep('验证集选出的 w（合法）', w_best)
    rep('两族等权 0.5', 0.5)
    rep('按模型数量 6:1', 6 / 7)
    rep('纯深度 (w=1)', 1.0)
    rep('纯经典 (w=0)', 0.0)

    (OUT / 'fusion_weight.json').write_text(json.dumps(
        {'note': '跨 session;权重用 T 内验证集（seed42 两次运行共享切分,389 样本）选,E 仅前向',
         'val_curve': curve, 'w_selected_by_val': w_best, 'results': res},
        indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
