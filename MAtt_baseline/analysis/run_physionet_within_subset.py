# -*- coding: utf-8 -*-
"""PhysionetMI 适配 · 深度被试内分层子集(6 人, 每人独立训练, GPU 轻量)。

问题: 单人仅 ~63 train / ~14 val / ~14 test 试次时, 逐人独立训练深度模型能到多少?
选人: 按零训练分析的逐人 CSP 5 折 CV(physionet_adapt.json)分层 —— 最高 2 人 /
中位 2 人 / 最低 2 人 ⇒ 覆盖可分性分布两端, 与 BCI 被试内读数(S3 82.8%)对照。
实现: 逐个 subject 驱动 run_matt_physionetmi.main()(只改 CONFIG.sub / model_path,
不改其代码); 该入口会把逐人结果 json 写到 results/ 根, 本脚本立即搬进自己的目录。
产物: results/physionet_within_subset/(ckpt + 可视化 + within_subset_summary.json)。
"""
import sys, json, shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

import run_matt_physionetmi as R

OUT = BASE / 'results' / 'physionet_within_subset'
ADAPT = BASE / 'results' / 'physionet_adapt' / 'physionet_adapt.json'
N_CLASSES = 4


def pick_subjects():
    accs = np.array(json.load(open(ADAPT))['csp_within']['acc_by_subject'])
    order = np.argsort(accs)          # 升序; index = 被试号-1
    picks = [order[0], order[1],                      # 低 2
             order[len(order) // 2], order[len(order) // 2 + 1],   # 中 2
             order[-1], order[-2]]                    # 高 2
    return [(int(i) + 1, float(accs[i])) for i in picks]   # (1-based 被试号, CSP CV)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for sub, csp in pick_subjects():
        print(f'\n{"="*70}\n被试 S{sub:03d}(CSP CV {csp*100:.1f}%)\n{"="*70}', flush=True)
        R.CONFIG.update({'dataset': 'physionetmi', 'sub': sub, 'repeat': 1,
                         'model_path': './results/physionet_within_subset/'})
        try:
            R.main()
        except Exception as e:
            print(f'[S{sub:03d}] 失败: {e!r}')
            continue
        # 入口把结果 json 写到 results/ 根(硬编码), 搬回本目录
        row = {'subject': sub, 'csp_cv': csp}
        for kind in ('best_loss', 'best_acc'):
            src = BASE / 'results' / f'physionetmi_sub{sub:03d}_{kind}_results.json'
            if src.exists():
                d = json.load(open(src))
                row[f'test_acc_{kind}'] = d['test_acc']
                row[f'n_windows_{kind}'] = d.get('n_windows')
                shutil.move(str(src), str(OUT / src.name))
        rows.append(row)
        print(f'[S{sub:03d}] best_loss={row.get("test_acc_best_loss", float("nan"))*100:.1f}% '
              f'best_acc={row.get("test_acc_best_acc", float("nan"))*100:.1f}%', flush=True)
    with open(OUT / 'within_subset_summary.json', 'w', encoding='utf-8') as fh:
        json.dump({'config': R.CONFIG, 'rows': rows}, fh, indent=2, ensure_ascii=False)
    print('\n[OK] within_subset_summary.json')
    for r in rows:
        print(f"S{r['subject']:03d} CSP {r['csp_cv']*100:5.1f}% → "
              f"deep within: loss {r.get('test_acc_best_loss', float('nan'))*100:5.1f}% "
              f"/ acc {r.get('test_acc_best_acc', float('nan'))*100:5.1f}%")


if __name__ == '__main__':
    main()
