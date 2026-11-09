# -*- coding: utf-8 -*-
"""PhysionetMI 强可分被试组(用户 2026-10-24 指定开发协议):定义 + 零训练组读数。

筛选: 按已有逐人 62ch CSP 5 折 CV(csp62.json, 最强线性读出)取**上四分位** n≈26 人
为「强组」; 中位数分组作对照。名单落盘 subjects.json(含 id 与依据), 保证可复算。

零训练部分: 把已有全部臂的逐人精度(22ch/62ch × A共享/B调头/C微调×预算/D同切分CSP)
切到强组统计 —— 回答「全 103 人训练出的模型与逐人微调, 在强被试身上深度是否已胜线性」。
新训练(强组池共享训练)由 run_physionet_62ch.py 的 SUBSET_FILE / train_pair --subjects 承担。
产物: results/physionet_strong/{subjects.json, strong_readouts.json}
口径警告: 这是子集开发协议, 不得当作主协议报数(与「筛掉低 ERD 只换总体」的禁令的关系:
此处不改主协议结论, 只回答算法在信号强时能否工作)。
"""
import sys, json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

OUT = BASE / 'results' / 'physionet_strong'
CSP62 = BASE / 'results' / 'physionet_62ch_s42' / 'csp62.json'
CSP22 = BASE / 'results' / 'physionet_adapt' / 'physionet_adapt.json'


def load_rows():
    """(name, {subject: acc}) 序列 —— 全部已有臂的逐人精度。"""
    out = []
    # 22ch A(persubject best_acc)
    d = json.load(open(BASE / 'results' / 'physionet_persubject' / 'physionet_persubject.json'))
    out.append(('A_22ch', {r['subject']: r['acc'] for r in d['best_acc']['per_subject']}))
    # 22ch ladder C 50it / 150it(扫描) + D csp
    for tag, name in ((BASE / 'results' / 'physionet_finetune' / 'physionet_finetune.json', 'C50_22ch'),
                      (BASE / 'results' / 'physionet_finetune' / 'physionet_finetune_lr1e-4it150.json', 'C150_22ch')):
        d = json.load(open(tag))
        out.append((name, {r['subject']: r['full_final'] for r in d['rows']}))
        out.append((name.replace('C', 'D_csp'), {r['subject']: r['csp'] for r in d['rows']}))
    # 62ch A
    d = json.load(open(BASE / 'results' / 'physionet_62ch_s42' / 'physionet_62ch.json'))
    out.append(('A_62ch', {r['subject']: r['acc'] for r in d['best_acc']['per_subject']}))
    # 62ch ladder 50it / 150it
    for tag, name in ((BASE / 'results' / 'physionet_finetune_62ch' / 'physionet_finetune_62ch.json', 'C50_62ch'),
                      (BASE / 'results' / 'physionet_finetune_62ch' / 'physionet_finetune_62ch_it150.json', 'C150_62ch')):
        d = json.load(open(tag))
        out.append((name, {r['subject']: r['full_final'] for r in d['rows']}))
        out.append((name.replace('C', 'D_csp'), {r['subject']: r['csp'] for r in d['rows']}))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    csp62 = json.load(open(CSP62))['acc_by_subject']
    csp22 = json.load(open(CSP22))['csp_within']['acc_by_subject']
    order = np.argsort(csp62)[::-1]
    n_strong = 103 // 4                     # 上四分位 n=25
    strong_ids = [int(i) + 1 for i in order[:n_strong]]      # 1-based 被试号
    med = float(np.median(csp62))
    hi_all = [int(i) + 1 for i in np.where(np.array(csp62) >= med)[0]]
    lo = min(csp62[i] for i in order[:n_strong]) * 100
    hi = max(csp62[i] for i in order[:n_strong]) * 100
    print(f'强组(62ch CSP 上四分位) n={n_strong}, CSP 范围 {lo:.1f}-{hi:.1f}% | 中位分组 n={len(hi_all)}')
    json.dump({'strong_top_quartile': strong_ids,
               'strong_csp62_range': [float(min(csp62[i] for i in order[:n_strong])),
                                      float(max(csp62[i] for i in order[:n_strong]))],
               'median_split_hi': hi_all,
               'criterion': 'csp62 5-fold CV top quartile (results/physionet_62ch_s42/csp62.json)'},
              open(OUT / 'subjects.json', 'w'), indent=2)

    strong = set(s - 1 for s in strong_ids)   # 0-based
    rows = load_rows()
    res = {}
    print(f'\n{"臂":14s} {"强组 pooled":>10s} {"强组 macro":>10s} {"全103 pooled参考":>14s}')
    for name, per in rows:
        ids = [s for s in sorted(per) if s in strong]
        if len(ids) < n_strong:
            continue
        a = np.array([per[s] for s in ids])
        res[name] = {'n': len(ids), 'macro': float(a.mean()), 'ids': ids}
        print(f'{name:14s} {a.mean()*100:9.1f}% {a.mean()*100:9.1f}%')
    json.dump(res, open(OUT / 'strong_readouts.json', 'w'), indent=2, ensure_ascii=False)
    print(f'[OK] {OUT}/strong_readouts.json')


if __name__ == '__main__':
    main()
