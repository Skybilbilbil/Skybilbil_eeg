# -*- coding: utf-8 -*-
"""议题 D：验证集选模是不是负收益（零训练，只读 pair_training.json）。

背景：项目所有报数都用 val_acc 峰值选出的 ckpt，但训练日志显示 val_acc 峰值系统性高于
同一模型的 test_acc。本脚本把三件事分开：
  ① 训练内记录的 test_acc（21 个评估点，0-5-...-99）→ 参数选择的 test 上界
  ② val_acc / val_loss 峰值出现的迭代 → 现行选模点
  ③ 最终迭代（it=99）的 test_acc → 不选模
判据：① − ③ 是否 ≥ 5pp（项目可测地板）。若 ≥，则「按验证集早停」损失了可测量的精度。

同时给出 val 峰值迭代相对 test 峰值迭代的位置（早/晚），以及逐类 test 准确率在第 ①③ 两点的差异。
"""
import json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parent.parent
RUNS = {'seed42':   ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024',        'v70s2024'),
        'seed7':    ('verify70_seed7',           'v70s7')}
OUT = BASE / 'results' / 'val_selection_audit'
CLS = ['左', '右', '脚', '舌']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    rows = []
    for name, (d, tag) in RUNS.items():
        j = json.load(open(BASE / 'results' / d / 'pair_training.json'))
        m = j['models']['cross_mix']
        te = np.array(m['test_acc'], dtype=float)
        ti = np.array(m['test_iter'], dtype=int)
        va = np.array(m['val_acc'], dtype=float)
        vl = np.array(m['val_loss'], dtype=float)
        it_best_te = int(ti[int(np.argmax(te))])
        it_best_va = int(np.argmax(va))
        it_best_vl = int(np.argmin(vl))
        peak_te, final_te = float(te.max()), float(te[-1])
        pc_peak = m['per_class_test'][int(np.argmax(te))]
        pc_final = m['per_class_test'][-1]
        d_sel = (peak_te - final_te) * 100
        res[name] = {'it_best_test': it_best_te, 'it_best_val_acc': it_best_va,
                     'it_best_val_loss': it_best_vl,
                     'val_peak': float(va.max()) * 100, 'val_final': float(va[-1]) * 100,
                     'val_loss_min': float(vl.min()),
                     'test_peak': peak_te * 100, 'test_final': final_te * 100,
                     'test_at_valpeak_iter': float(te[it_best_va // 5]) * 100,
                     'gap_peak_minus_final': d_sel,
                     'per_class_peak': [float(x) * 100 for x in pc_peak],
                     'per_class_final': [float(x) * 100 for x in pc_final],
                     'test_curve': [float(x) * 100 for x in te],
                     'test_iter': [int(x) for x in ti]}
        rows.append((name, it_best_va, it_best_te, float(va.max()) * 100, peak_te * 100,
                     final_te * 100, d_sel))
    print(f"{'seed':9s} {'val峰it':>8s} {'test峰it':>9s} {'val峰%':>7s} {'test峰%':>8s} {'test终%':>8s} {'峰-终':>7s}")
    for r in rows:
        print(f'{r[0]:9s} {r[1]:8d} {r[2]:9d} {r[3]:7.2f} {r[4]:8.2f} {r[5]:8.2f} {r[6]:+7.2f}')
    gaps = np.array([r[6] for r in rows])
    print(f'\n3-seed 平均 峰-终 = {gaps.mean():+.2f}pp (逐 seed: ' +
          ', '.join(f'{g:+.2f}' for g in gaps) + ')')
    print(f'判据：≥5pp 才算可测 -> {"成立" if gaps.mean() >= 5 else "不成立（低于地板）"}')
    print(f'val 峰值迭代平均 = {np.mean([r[1] for r in rows]):.1f}  (总 100 迭代)')
    print('\n逐类 test 准确率  峰 / 终 (%):')
    for name in RUNS:
        r = res[name]
        print(f"  {name:9s} " + '  '.join(f'{c}:{p:.1f}/{f:.1f}' for c, p, f in
                                          zip(CLS, r['per_class_peak'], r['per_class_final'])))
    (OUT / 'val_selection_audit.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f'written {OUT}/val_selection_audit.json')


if __name__ == '__main__':
    main()