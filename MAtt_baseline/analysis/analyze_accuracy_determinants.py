# -*- coding: utf-8 -*-
"""不同被试的准确率到底由什么决定:【类间信号】还是【类内噪声】?

## 问题
逐人精度 33.3% ~ 71.8%(深度模型),极差 38.5pp。判别精度必然只由
    可分性 ≈ 类间分离度 / 类内散布
决定,所以这个 38.5pp 必须归到其中一项或两项:
  · 类间分离度小 ⇒ **信号弱**(类别模式本身不清楚)
  · 类内散布大   ⇒ **噪声大**(模式清楚但每次不一样)
两者处置完全不同 ⇒ 必须分开量。

## 口径(为什么用逐人 CV,不复现主协议切分)
之前的脚本反复被 "split_train_valid_test 的 others 返回的是切分结果而不是行号" 绊住
(全池口径下被试是混合切的,行号对不回去)。本问题不需要那个切分:**逐人被试内
5 折分层 CV** 就是干净的口径,而且它对应的正是报告里 csp_within / tangent_lda_within
那一族读数,可直接与逐人精度并列。

## 三个量(全部在切空间,22 通道,任务窗 [0.5,3.5]s)
  ① d_b  类间:4 个类的类心之间的平均两两距离
  ② w    类内:试次到本类心的 RMS 距离
  ③ Fisher = d_b / w —— 无量纲,直接决定线性分类器能到什么精度
  ④ margin:test 试次的「最近错误类心距离 − 正确类心距离」的均值 —— 决策边界的余量
每个量都按「train 估类心 / test 量距离」的折外口径算,避免样本内互相污染。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_spd_separability import tangent                          # noqa: E402
from utils.dataset import load_preprocessed_dataset                    # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'accuracy_determinants'
TASK = (375, 1126)
SUBJ = [f'S{i+1:02d}' for i in range(9)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    tbl = json.load(open(BASE / 'results' / 'individual_diff' / 'subject_table.json'))
    acc = np.array([r['deep70_acc'] for r in tbl])
    csp = np.array([r['csp_within'] for r in tbl])

    rows = []
    print(f"{'被试':>5s}{'类间 d_b':>11s}{'类内 w':>10s}{'Fisher':>9s}"
          f"{'margin':>10s}{'折外LDA':>9s}{'深度模型':>9s}")
    for s in range(9):
        X = np.asarray(sd[s][0], dtype=np.float64)
        y = np.asarray(sd[s][1])
        seg = X[:, :, TASK[0]:TASK[1]]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        C = np.einsum('nct,ndt->ncd', seg, seg) / seg.shape[-1]
        F = tangent(C)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        db_l, w_l, mg_l, nc_l = [], [], [], []
        for tr, te in skf.split(F, y):
            cen = np.stack([F[tr][y[tr] == c].mean(0) for c in range(4)])
            wr = np.concatenate([F[tr][y[tr] == c] - cen[c] for c in range(4)])
            w = float(np.sqrt((wr ** 2).sum(1).mean()))
            pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
            db = float(np.mean([np.linalg.norm(cen[i] - cen[j]) for i, j in pairs]))
            D = np.stack([np.sqrt(((F[te] - cen[c]) ** 2).sum(1)) for c in range(4)], 1)
            pred = D.argmin(1)
            dc = D[np.arange(len(te)), y[te]]
            D2 = D.copy(); D2[np.arange(len(te)), y[te]] = np.inf
            mg_l.append(float((D2.min(1) - dc).mean()))
            db_l.append(db); w_l.append(w); nc_l.append(float((pred == y[te]).mean()))
        db_m, w_m = float(np.mean(db_l)), float(np.mean(w_l))
        rows.append({'subject': SUBJ[s], 'd_between': db_m, 'w_within': w_m,
                     'fisher': db_m / w_m, 'margin': float(np.mean(mg_l)),
                     'cv_acc': float(np.mean(nc_l)), 'deep_acc': float(acc[s]),
                     'csp_within': float(csp[s])})
        print(f'{SUBJ[s]:>5s}{db_m:>11.4f}{w_m:>10.4f}{db_m/w_m:>9.4f}'
              f'{np.mean(mg_l):>10.4f}{np.mean(nc_l)*100:>8.1f}%{acc[s]*100:>8.1f}%')

    db = np.array([r['d_between'] for r in rows])
    ww = np.array([r['w_within'] for r in rows])
    fi = np.array([r['fisher'] for r in rows])
    mg = np.array([r['margin'] for r in rows])
    cv = np.array([r['cv_acc'] for r in rows])
    a = np.array([r['deep_acc'] for r in rows])
    cs = np.array([r['csp_within'] for r in rows])

    print('\n' + '=' * 76)
    print('★ 类间信号 vs 类内噪声,分别与精度的关系(n=9,Spearman)')
    print('=' * 76)
    for nm, v in [('类间分离度 d_b', db), ('类内散布 w', ww),
                  ('Fisher d_b/w', fi), ('决策边界 margin', mg)]:
        r1, p1 = spearmanr(v, a)
        r2, p2 = spearmanr(v, cs)
        print(f'  {nm:16s} ρ(深度模型)={r1:+.3f}(p={p1:.4f})   '
              f'ρ(CSP_LDA)={r2:+.3f}(p={p2:.4f})')
    print('\n  逐人范围(极差倍数):')
    print(f'    类间分离度 {db.min():.4f} ~ {db.max():.4f}   ⇒ {db.max()/db.min():.2f}x')
    print(f'    类内散布   {ww.min():.4f} ~ {ww.max():.4f}   ⇒ {ww.max()/ww.min():.2f}x')
    print(f'    Fisher     {fi.min():.4f} ~ {fi.max():.4f}   ⇒ {fi.max()/fi.min():.2f}x')
    print('\n  线性解释力(r²,对深度模型精度):')
    for nm, v in [('类间分离度', db), ('类内散布', ww), ('Fisher', fi), ('margin', mg)]:
        r = pearsonr(v, a)[0]
        print(f'    {nm:12s} r² = {r**2:.3f}')
    print(f'\n  交叉校验:折外最近类心 CV 与深度模型精度的相关:'
          f'r = {pearsonr(cv, a)[0]:+.3f}')
    (OUT / 'accuracy_determinants.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/accuracy_determinants.json')


if __name__ == '__main__':
    main()
