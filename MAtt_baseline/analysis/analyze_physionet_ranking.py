# -*- coding: utf-8 -*-
"""PhysionetMI 选人信号的排序有效性（零训练，CPU）。

问题（用户 2026-11-08）：BCI-2a 上「逐人 CSP+LDA 5 折 CV」与主协议逐人精度的
Spearman 是 0.917（analyze_individual_diff::csp_within vs deep70_acc）。同一方法在
Physionet 上只有 0.455。差异在试次数：BCI 每人 576 试次/5 折（每折 115），
Physionet 训练折每人 63 试次/5 折（每折 13）。

本脚本检验：把重复次数从 1 提高到 20（即 100 折取平均），排序信度能否恢复。
同时对照 BCI 上另一个逐人指标 tangent+LDA。

判据（全部只碰训练折，合法）：
  rho(排序分数, 深度 test acc)；前 52 人的 test 均值；抓取率 = (v-floor)/(ceil-floor)
  其中 floor = 全体 test 均值（随机取 52 人的期望），ceil = 完美排序（oracle）。

产物：results/physionet_ranking/physionet_ranking.json
"""
import sys, json
from pathlib import Path
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from analyze_physionet_gap import cov, csp_acc
from utils.dataset import load_preprocessed_dataset, split_train_valid_test

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
OUT = BASE / 'results' / 'physionet_ranking'
RESAMPLE = 2
T0_SEC, T1_SEC = 1.0, 10.0


def tan_feats(C):
    """trace 归一 → logm → 上三角 253 维（与 gap 脚本 C2 同口径）。"""
    C = C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    if RESAMPLE > 1:
        from scipy.signal import resample_poly
        sd = [(resample_poly(np.asarray(X, np.float32), 1, RESAMPLE, axis=-1).astype(np.float32), y)
              for X, y in sd]
    X = np.concatenate([s[0] for s in sd], 0)
    y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_ = ret[0], ret[1], ret[6]
    fs = 160 // RESAMPLE
    t0, t1 = int(T0_SEC * fs), min(int(T1_SEC * fs), X.shape[-1])
    Ctr = cov(Xtr[:, :, t0:t1].astype(np.float64))
    print(f'[数据] {len(sd)} 人; train {len(ytr)}; 任务窗 {t0}:{t1}', flush=True)

    g = json.load(open(BASE / 'results' / 'physionet_gap' / 'physionet_gap_gap80.json'))
    te = np.array([p['acc'] for p in g['depth']['test']['per_subject']])
    keep = int(np.ceil(len(te) * 0.5))
    floor, ceil_ = float(te.mean()), float(np.sort(te)[::-1][:keep].mean())
    print(f'[参照] 随机 {floor*100:.2f}% | 完美 {ceil_*100:.2f}% | 区间 {(ceil_-floor)*100:.2f}pp',
          flush=True)

    rec = {'floor': floor, 'ceiling': ceil_, 'keep': keep, 'signals': {}}

    def evaluate(name, scores, reps):
        best = np.argsort(-scores)[:keep]
        v = float(te[best].mean())
        rho = float(spearmanr(scores, te)[0])
        grab = (v - floor) / (ceil_ - floor)
        print(f'[{name:22s} x{reps:2d}] rho={rho:+.3f} | 前52 test {v*100:5.2f}% | '
              f'抓取 {grab*100:5.1f}%', flush=True)
        rec['signals'][f'{name}_x{reps}'] = {
            'rho': rho, 'top52_test': v, 'grab': grab,
            'cohort_1based': sorted(int(i) + 1 for i in best),
            'scores': scores.tolist()}

    # CSP 每次都要重算 fit_csp + csp_feats，做 20 次重复太慢 ⇒ 只在 x1 上算一次作基线；
    # tangent 特征是预计算的，重复只多花 LDA 拟合，故用 x10 检验「重复能否降方差」。
    for reps in (1, 10):
        csp_s = np.full(len(sd), np.nan)
        tan_s = np.full(len(sd), np.nan)
        for k in np.unique(str_):
            m = str_ == k
            yk, Ck = ytr[m], Ctr[m]
            Fk = tan_feats(Ck)
            ta = []
            for r in range(reps):
                for a, b in StratifiedKFold(5, shuffle=True, random_state=r).split(Ck, yk):
                    ta.append(LDA(solver='lsqr', shrinkage='auto')
                              .fit(Fk[a], yk[a]).score(Fk[b], yk[b]))
            if reps == 1:
                ca = [csp_acc(Ck, yk, a, b) for a, b in
                      StratifiedKFold(5, shuffle=True, random_state=0).split(Ck, yk)]
                csp_s[k] = float(np.mean(ca))
            tan_s[k] = float(np.mean(ta))
        if reps == 1:
            evaluate('CSP+LDA', csp_s, reps)
        evaluate('tangent+LDA', tan_s, reps)

    (OUT / 'physionet_ranking.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/physionet_ranking.json', flush=True)


if __name__ == '__main__':
    main()
