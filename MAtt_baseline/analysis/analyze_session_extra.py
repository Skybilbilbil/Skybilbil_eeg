# -*- coding: utf-8 -*-
"""补两个该早做的检验。

## 检验 1:会话受控的 μ ERD(把"当天状态"从被试层均值里挤掉)
报告里的 ρ=−0.950 用的是**某一次测量**的 μ ERD。本检验改用
「同一被试两个 session 的平均 ERD」并把每个 session 的状态分量单独拿出来,
看会话受控后的跨被试相关还剩多少 —— 这是对那条核心相关最严的复核。

## 检验 2:μ ERD 与被试内部的行为量有没有关系(θ 是不是「投入度」)
若 μ ERD 真的是一个被试层特质 θ 的读数,那 θ 应该在**与脑电无关**的地方也留下痕迹:
  · 被试在 576 个试次上的**分块精度曲线**(前半 vs 后半 = 学习/疲劳方向)
  · 每被试的**精度方差**(状态波动的幅度)
  · 每被试的**类别混淆结构**(是否偏向某几类)
本检验把这些和 μ ERD 逐人配对。全部零训练、纯 CPU。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_causal_x2 import predict_ok, RUNS, ARGS, WIN, STEP       # noqa: E402
from analyze_session_within_causal import session_index              # noqa: E402
from utils.dataset import load_preprocessed_dataset                  # noqa: E402
from utils.functions import load_matt_model                          # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'session_within_causal'
SUBJ = [f'S{i+1:02d}' for i in range(9)]


def main():
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = [np.asarray(d[0], dtype=np.float64) for d in sd]
    y_all = [np.asarray(d[1]) for d in sd]
    panel = json.load(open(OUT / 'session_panel.json'))
    R = {(r['subject'], r['session']): r for r in panel}
    tbl = json.load(open(BASE / 'results' / 'individual_diff' / 'subject_table.json'))
    acc_deep = np.array([r['deep70_acc'] for r in tbl])
    erd_rep = np.array([R[(s, 0)]['erd_hand_dB'] for s in SUBJ])

    nets = [load_matt_model(str(BASE / 'results' / d / f'{t}_cross_mix_best_acc.pt'),
                            WIN, STEP, ARGS) for (d, t) in RUNS.values()]
    ok_all, conf_all, curves = [], [], []
    for s in range(9):
        ok, _ = predict_ok(nets, X_all[s], y_all[s])
        ok_all.append(ok)
        y = y_all[s]
        cm = np.zeros((4, 4), int)
        pred = np.where(ok, y, (y + 1) % 4)          # 只用于占位,下面重算
        curves.append(ok.reshape(6, -1).mean(1))     # 576 → 6 块 × 96
    print('每被试的分块精度曲线(6 块 × 96 试次)与 μ ERD:')
    erd_sess = np.array([0.5 * (R[(s, 0)]['erd_hand_dB'] + R[(s, 1)]['erd_hand_dB'])
                         for s in SUBJ])
    print(f"{'被试':>5s}{'ERD报告':>10s}{'ERD双session均':>15s}"
          f"{'前半':>8s}{'后半':>8s}{'后半-前半':>11s}{'块间sd':>9s}")
    d_half, d_sd = [], []
    for i, s in enumerate(SUBJ):
        c = curves[i]
        h1, h2 = c[:3].mean(), c[3:].mean()
        d_half.append((h2 - h1) * 100)
        d_sd.append(c.std() * 100)
        print(f'{s:>5s}{erd_rep[i]:>+10.3f}{erd_sess[i]:>+15.3f}'
              f'{h1*100:>7.1f}%{h2*100:>7.1f}%{(h2-h1)*100:>+11.1f}'
              f'{c.std()*100:>9.2f}')
    d_half, d_sd = np.array(d_half), np.array(d_sd)

    print('\n' + '=' * 74)
    print('检验 1:会话受控的 μ ERD vs 逐人精度')
    print('=' * 74)
    for nm, v in [('单 session ERD(报告口径)', erd_rep),
                  ('两 session 平均 ERD', erd_sess)]:
        r1, p1 = spearmanr(v, acc_deep)
        print(f'  {nm:24s} ρ(deep70_acc) = {r1:+.3f} (p={p1:.4f})')
    print('  ⚠ 因为主协议测试折只落在一个 session 内(实测 session0 35 / session1 52,')
    print('     9 人合计),报告里那条 ρ 只反映**单一 session** 的读数。')

    print('\n' + '=' * 74)
    print('检验 2:μ ERD 与被试内行为量(θ 应在与脑电无关处留痕)')
    print('=' * 74)
    for nm, v in [('后半−前半 精度差(pp)', d_half), ('分块精度 sd(pp)', d_sd)]:
        r1, p1 = spearmanr(erd_sess, v)
        print(f'  ρ(μ ERD, {nm:22s}) = {r1:+.3f} (p={p1:.4f})')
    print('\n  精度的 session 间变化的信噪比(②已算):Δdeep 的二项 SE ≈ 10.5pp,')
    print(f'  而实测 |Δ| 的中位数 = {np.median(np.abs(np.array([R[(s,1)]["deep_acc"]-R[(s,0)]["deep_acc"] for s in SUBJ])))*100:.1f}pp')
    (OUT / 'session_extra.json').write_text(json.dumps(
        {'erd_report': erd_rep.tolist(), 'erd_two_session_mean': erd_sess.tolist(),
         'delta_half_pp': d_half.tolist(), 'block_sd_pp': d_sd.tolist(),
         'curves': [c.tolist() for c in curves]}, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/session_extra.json')


if __name__ == '__main__':
    main()
