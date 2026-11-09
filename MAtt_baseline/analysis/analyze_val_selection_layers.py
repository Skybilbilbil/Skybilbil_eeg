# -*- coding: utf-8 -*-
"""议题 D 复核:验证集选模的代价,三层分开算。

第 0 层(本脚本新增,此前没做过):**同一批 run 的 val_acc 轨迹**,直接看
  「val 峰值迭代的 test_acc」 与 「test 轨迹峰值」 差多少 —— 这是选模代价的真身。
第 1 层(既有 evaluate70_persubject 已算):best_acc 与 best_loss 两个 ckpt 的 test 精度。
第 2 层(既有 analyze_val_selection_audit):峰值−终值。
只读既有产物,不训练。"""
import json, sys
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parent.parent
RUNS = {'seed42': ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024', 'v70s2024'),
        'seed7': ('verify70_seed7', 'v70s7')}
OUT = BASE / 'results' / 'val_selection_audit'

def main():
    print('=' * 74)
    print('第 0 层:val_acc 轨迹本身(选模到底在看什么)')
    print('=' * 74)
    print(f"{'seed':9s} {'val峰值':>7s} {'峰值it':>6s} {'val@100':>8s} | "
          f"{'该it的test':>10s} {'test峰值':>8s} {'test峰值it':>10s} {'选模代价':>8s}")
    rows = {}
    for name, (d, tag) in RUNS.items():
        j = json.load(open(BASE / 'results' / d / 'pair_training.json'))
        m = j['models']['cross_mix']
        va = np.array(m['val_acc'], float)
        te = np.array(m['test_acc'], float)
        ti = np.array(m['test_iter'], int)
        it_va = int(np.argmax(va))
        it_te = int(ti[int(np.argmax(te))])
        at = float(te[it_va // 5])
        cost = (float(te.max()) - at) * 100
        rows[name] = {'it_val_peak': it_va, 'val_peak': float(va.max()) * 100,
                      'val_final': float(va[-1]) * 100, 'test_at_val_peak': at * 100,
                      'test_peak': float(te.max()) * 100, 'it_test_peak': it_te,
                      'cost_pp': cost, 'val_curve': [float(x)*100 for x in va],
                      'test_curve': [float(x)*100 for x in te]}
        print(f'{name:9s} {va.max()*100:7.2f} {it_va:6d} {va[-1]*100:8.2f} | '
              f'{at*100:10.2f} {te.max()*100:8.2f} {it_te:10d} {cost:+8.2f}')
    c = np.array([rows[k]['cost_pp'] for k in RUNS])
    print(f'\n  3-seed 平均选模代价 = {c.mean():+.2f}pp (逐 seed ' +
          ', '.join(f'{x:+.2f}' for x in c) + ')')
    print(f'  判据:≥5pp 才算可测 ⇒ {"成立" if c.mean() >= 5 else "不成立(低于地板)"}')
    print(f'  平均 val 峰值迭代 = {np.mean([rows[k]["it_val_peak"] for k in RUNS]):.1f}/100 '
          f'(test 峰值迭代 = {np.mean([rows[k]["it_test_peak"] for k in RUNS]):.1f})')
    print('  ⇒ 两条轨迹的峰值迭代**不是同一个地方**,说明"验证集选模"在替测试集做一个')
    print('     它自己也不擅长的选择(验证集只有 864 试次、且不含测试被试之外的信息)。')

    print('\n' + '=' * 74)
    print('第 1 层:best_acc 与 best_loss 两个 ckpt 的真实 test 精度(既有产物)')
    print('=' * 74)
    p = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
    ev = json.load(open(p))
    print(f"{'seed':9s} {'best_acc':>9s} {'best_loss':>10s} {'差':>7s}")
    dl = []
    for name in RUNS:
        if name not in ev:
            continue
        a = ev[name].get('best_acc', {}).get('overall')
        b = ev[name].get('best_loss', {}).get('overall')
        if a is None or b is None:
            print(f'{name:9s}  缺一侧'); continue
        dl.append((b - a) * 100)
        print(f'{name:9s} {a*100:8.2f}% {b*100:9.2f}% {(b-a)*100:+6.2f}pp')
        rows[name]['best_acc_overall'] = a * 100
        rows[name]['best_loss_overall'] = b * 100
        rows[name]['per_subject'] = {k: v for k, v in ev[name].items()}
    if dl:
        print(f'\n  best_loss − best_acc 平均 = {np.mean(dl):+.2f}pp (逐 seed ' +
              ', '.join(f'{x:+.2f}' for x in dl) + ')')
        print(f'  判据:≥5pp 才算可测 ⇒ {"成立" if np.mean(dl) >= 5 else "不成立(低于地板)"}')
        print('  ⇒ 但**这不是"选模有代价"**的证据,而是"val_acc 选模比 val_loss 选模差一点"')
        print('     的证据:两者都只看验证集,都不含测试作弊。')

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'val_selection_audit2.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/val_selection_audit2.json')

if __name__ == '__main__':
    main()
