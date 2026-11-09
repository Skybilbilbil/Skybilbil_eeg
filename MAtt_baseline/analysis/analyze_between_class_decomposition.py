# -*- coding: utf-8 -*-
"""类间信号强 5.46 倍,强在哪:【幅度】还是【形状(空间特异性)】?

## 问题
d_b(类心平均两两距离)是被试间差异的主载。但 d_b 大可以有两种完全不同的来源:
  · **幅度**:每个类的均值都离全体均值很远,但四个类的偏移方向**接近共线** ⇒
    大类对比度、小的空间特异性。可能只是"这个人的任务期协方差整体被改了"。
  · **形状**:四个类的偏移方向**互不成比例** ⇒ 真正的类别特异性空间模式。
把 d_b 拆成这两块,才能说清"类间信号强"到底指什么。

## 做法(切空间,22 通道,任务窗 [0.5,3.5]s;逐人被试内 5 折,折外口径)
对被试 s、类 c,记类心 μ_c ∈ R^253,全体质心 μ。
  · 每类偏移 v_c = μ_c − μ,幅度 a_c = ‖v_c‖
  · 幅度总和 A = mean_c a_c                    —— "类间幅度"
  · 方向一致性 coh = ‖mean_c v_c/‖v_c‖‖ ∈ [0,1] —— 1 表示四类偏移完全同向
  · 形状部分 S = ‖v_c − (v_c·u)u‖ 的均值(u = 主方向) —— 去掉共线后的残余
再与 d_b、Fisher、逐人精度并列。
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
    rows = []
    print(f"{'被试':>5s}{'幅度A':>9s}{'方向一致性':>11s}{'形状S':>9s}"
          f"{'d_b':>9s}{'类内w':>9s}{'精度':>8s}")
    for s in range(9):
        X = np.asarray(sd[s][0], dtype=np.float64)
        y = np.asarray(sd[s][1])
        seg = X[:, :, TASK[0]:TASK[1]]
        seg = seg - seg.mean(axis=-1, keepdims=True)
        C = np.einsum('nct,ndt->ncd', seg, seg) / seg.shape[-1]
        F = tangent(C)
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        A_l, coh_l, S_l, db_l, w_l = [], [], [], [], []
        for tr, te in skf.split(F, y):
            cen = np.stack([F[tr][y[tr] == c].mean(0) for c in range(4)])
            mu = cen.mean(0)
            V = cen - mu                                  # (4, 253)
            a = np.linalg.norm(V, axis=1)
            u = V / np.maximum(a[:, None], 1e-30)
            A_l.append(float(a.mean()))
            coh_l.append(float(np.linalg.norm(u.mean(0))))
            # 形状:去掉 V 的主方向后残余的均值
            w_, U_ = np.linalg.eigh(V.T @ V)
            top = U_[:, -1]
            Vp = V - np.outer(V @ top, top)
            S_l.append(float(np.linalg.norm(Vp, axis=1).mean()))
            pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
            db_l.append(float(np.mean([np.linalg.norm(cen[i] - cen[j])
                                       for i, j in pairs])))
            wr = np.concatenate([F[tr][y[tr] == c] - cen[c] for c in range(4)])
            w_l.append(float(np.sqrt((wr ** 2).sum(1).mean())))
        rows.append({'subject': SUBJ[s], 'amp': float(np.mean(A_l)),
                     'coherence': float(np.mean(coh_l)), 'shape': float(np.mean(S_l)),
                     'd_between': float(np.mean(db_l)), 'w_within': float(np.mean(w_l)),
                     'deep_acc': float(acc[s])})
        print(f'{SUBJ[s]:>5s}{np.mean(A_l):>9.4f}{np.mean(coh_l):>11.4f}'
              f'{np.mean(S_l):>9.4f}{np.mean(db_l):>9.4f}{np.mean(w_l):>9.4f}'
              f'{acc[s]*100:>7.1f}%')

    def col(k):
        return np.array([r[k] for r in rows])

    print('\n' + '=' * 74)
    print('★ 幅度 / 一致性 / 形状 与精度的关系(n=9)')
    print('=' * 74)
    print(f"{'量':>14s}{'范围':>22s}{'极差':>8s}{'ρ(精度)':>10s}{'p':>8s}{'r²':>8s}")
    for nm, k in [('幅度 A', 'amp'), ('方向一致性', 'coherence'), ('形状 S', 'shape'),
                  ('d_b', 'd_between'), ('类内 w', 'w_within')]:
        v = col(k)
        r1, p1 = spearmanr(v, acc)
        r2 = pearsonr(v, acc)[0]
        print(f'{nm:>14s}{v.min():>10.4f} ~ {v.max():9.4f}'
              f'{v.max()/max(v.min(), 1e-9):>8.2f}x{r1:>+10.3f}{p1:>8.3f}{r2**2:>8.3f}')
    amp, coh, shp, db = col('amp'), col('coherence'), col('shape'), col('d_between')
    print(f'\n  幅度与 d_b 的相关:r = {pearsonr(amp, db)[0]:+.3f}')
    print(f'  形状与 d_b 的相关:r = {pearsonr(shp, db)[0]:+.3f}')
    print(f'  幅度与形状的相关:r = {pearsonr(amp, shp)[0]:+.3f}')
    (OUT / 'between_class_decomposition.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/between_class_decomposition.json')


if __name__ == '__main__':
    main()
