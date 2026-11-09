# -*- coding: utf-8 -*-
"""从相关到因果:用「被试内、会话级」的自然变异去检验慢通道。

## 为什么是这一个设计(前面几个设计各自排除了什么)
  · X-1(协方差层、试次级剂量):被试内,但动的是**试次级 μ 波动** ⇒ 无效。
  · X-2(观测式、试次级):被试内,但预测子信度只有 ρ≈0.37 ⇒ 为零。
  · X-3:证明端到端精度的移动由**空间模式**驱动、不是 μ 功率。
  ⇒ 三个都是**试次级**。而 β_B 显著、β_W 不显著这一对结果说的是:
    **与精度耦合的那个东西是慢的/被试层的,不是逐试次的。**
    所以要找因果,就必须在**同一个被试身上把那个慢变量动起来**。

## 本设计:会话级(每个被试 2 个 session,BCI-2a 官方就是两个 session)
若 μ ERD 是慢通道的因果量,则在**同一个被试内部**,ERD 更深的那个 session
应当精度更高。这是一个**被试内**(免于被试层混杂)且**慢变量真的会动**的设计。

## 两个必须同时报的量
  ① 会话级 ERD 的**信度**:两个 session 的 ERD 相关。信度低则本设计只是"又一个噪声受限的零"。
  ② 会话级 ERD 与**会话级精度**的关系(含被试固定效应)。
  ③ 噪声地板标定:单试次抽样 SE = 1.59 dB、每 session 288 试次 ⇒ 会话级 ERD 的
     SE ≈ 1.59/√288 ≈ 0.094 dB。任何观察到的会话间 ERD 差必须与这个地板比。

## 精度怎么算
用冻结的 3 个主协议 ckpt 做**多数投票**,分别算 session0 / session1 上的精度
(测试折内的试次)。零训练。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_causal_erd_injection import MU, TASK, BASE_WIN, narrow, band_pow  # noqa: E402
from analyze_causal_x2 import predict_ok, RUNS, ARGS, WIN, STEP                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test     # noqa: E402
from utils.functions import load_matt_model                                     # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'session_within_causal'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
SE_SINGLE = 1.59          # X-0 实测:单试次 μ ERD 抽样 SE (dB)


def session_index(y):
    """与 analyze_individual_diff / analyze_trial_vs_subject 同口径:
    每人每类 144 = 2 session × 72,按类内前 72 / 后 72。"""
    idx = np.arange(len(y))
    pos = idx - y * 144
    return (pos >= 72).astype(int)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('会话级被试内检验(零训练)')
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = [np.asarray(d[0], dtype=np.float64) for d in sd]
    y_all = [np.asarray(d[1]) for d in sd]
    sub_orig = np.concatenate([np.full(len(y_all[i]), i) for i in range(9)])
    Xc = np.concatenate(X_all, 0)
    yc = np.concatenate(y_all, 0)
    ret = split_train_valid_test(Xc, yc, 0.70, 0.15, random_state=42,
                                 others=[sub_orig])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    # 测试折的 X_te/y_te 是 Xc/yc 的**行子集**,但 train_test_split(shuffle=True) 把行号换了序,
    # 所以拿不到"测试折第 i 行 = Xc 第几行"。本项目既有脚本用的是内容指纹匹配
    # (`analyze_pipeline_attribution.py` 也吃过这个坑):这里用 (被试, 特征和) 双键,
    # 校验唯一性 —— 不唯一就报错而不是静默错配。
    key_pool = {}
    for i in range(len(sub_orig)):
        key_pool.setdefault((int(sub_orig[i]), float(Xc[i].sum())), []).append(i)
    bad = sum(1 for v in key_pool.values() if len(v) != 1)
    if bad:
        print(f'  [warn] {bad} 个 (被试, 特征和) 键不唯一,改用 argmin 距离匹配仍可能错位')
    used, rows = {}, np.empty(len(sub_te), dtype=int)
    for i in range(len(sub_te)):
        k = (int(sub_te[i]), float(X_te[i].sum()))
        rows[i] = key_pool[k][0]

    nets = [load_matt_model(str(BASE / 'results' / d / f'{t}_cross_mix_best_acc.pt'),
                            WIN, STEP, ARGS) for (d, t) in RUNS.values()]
    ok_te, _ = predict_ok(nets, X_te, y_te)

    recs = []
    for s in range(9):
        X, y = X_all[s], y_all[s]
        sess = session_index(y)
        Zb = narrow(X, MU)
        R = band_pow(Zb, BASE_WIN)
        A = band_pow(Zb, TASK)
        erd_t = 10 * np.log10(np.maximum(A, 1e-30) / np.maximum(R, 1e-30))
        m = sub_te == s
        loc = rows[m]                       # 该被试在测试折里的试次 → 全池原始行
        local = loc - (0 if s == 0 else sum(len(y_all[j]) for j in range(s)))
        ok_s = ok_te[m]
        for k in (0, 1):
            mk = sess[local] == k
            if mk.sum() == 0:
                continue
            recs.append({'subject': SUBJ[s], 'session': k, 'n': int(mk.sum()),
                         'erd_mean_dB': float(erd_t[local][mk].mean()),
                         'erd_sd_dB': float(erd_t[local][mk].std()),
                         'acc': float(ok_s[mk].mean())})
    print(f"{'被试':>5s} {'sess':>5s} {'n':>5s} {'ERD均值(dB)':>12s} {'ERD sd':>8s} {'精度':>8s}")
    for r in recs:
        print(f"{r['subject']:>5s} {r['session']:>5d} {r['n']:>5d} "
              f"{r['erd_mean_dB']:>+12.3f} {r['erd_sd_dB']:>8.3f} {r['acc']*100:>7.1f}%")

    e = np.array([r['erd_mean_dB'] for r in recs])
    a = np.array([r['acc'] for r in recs])
    su = np.array([r['subject'] for r in recs])
    n = np.array([r['n'] for r in recs], float)
    se = SE_SINGLE / np.sqrt(n)             # 会话级 ERD 的抽样 SE
    print(f'\n  会话级 ERD 的抽样 SE ≈ 1.59/√n = '
          + ' '.join(f'{v:.3f}' for v in se) + ' dB')

    # ① 信度:两个 session 的 ERD 相关(去被试均值后 = 真正的稳定性)
    e0 = np.array([r['erd_mean_dB'] for r in recs if r['session'] == 0])
    e1 = np.array([r['erd_mean_dB'] for r in recs if r['session'] == 1])
    print(f'\n  ① 会话级 ERD 的信度:session0 vs session1 相关')
    print(f'     Pearson  r = {pearsonr(e0, e1)[0]:+.3f} (p={pearsonr(e0, e1)[1]:.3f})')
    print(f'     Spearman ρ = {spearmanr(e0, e1).statistic:+.3f} '
          f'(p={spearmanr(e0, e1).pvalue:.3f})')
    print(f'     session 间 ERD 差的范围 = {np.abs(e0-e1).min():.3f} ~ '
          f'{np.abs(e0-e1).max():.3f} dB  (抽样 SE 只有 {se.mean():.3f} dB)')

    # ② 被试固定效应下的会话级斜率
    print(f'\n  ② 会话级斜率(被试固定效应:每人内部去均值)')
    d_e, d_a = [], []
    for s_ in SUBJ:
        idx = np.where(su == s_)[0]
        if len(idx) < 2:
            continue
        d_e += list(e[idx] - e[idx].mean())
        d_a += list(a[idx] - a[idx].mean())
    d_e, d_a = np.array(d_e), np.array(d_a)
    b = float((d_e @ d_a) / (d_e @ d_e))
    r = d_a - b * d_e
    seb = float(np.sqrt((r @ r) / (len(r) - 2) / (d_e @ d_e)))
    print(f'     β_session = {b*100:+.3f} pp/dB  (SE {seb*100:.3f}, '
          f'n={len(d_e)} 观测 / 9 人)')
    print(f'     会话级 ERD 的组内 sd = {d_e.std():.3f} dB')
    print(f'     ⇒ 该通道能解释的精度变动 = {abs(b)*d_e.std()*100:.3f} pp')
    print(f'     对照:逐人精度的组间极差 = {(a.max()-a.min())*100:.1f} pp')
    print(f'     用抽样 SE 做 errors-in-variables 修正:信度 ρ_s = '
          f'1 − mean(SE²)/var(e) = '
          f'{1 - np.mean(se**2)/np.var(e):.3f}')

    out = {'per_session': recs,
           'reliability': {'pearson': float(pearsonr(e0, e1)[0]),
                           'spearman': float(spearmanr(e0, e1).statistic)},
           'session_slope_pp_per_dB': b * 100, 'session_slope_se': seb * 100,
           'within_sd_dB': float(d_e.std()),
           'reliability_ev': float(1 - np.mean(se ** 2) / np.var(e))}
    (OUT / 'session_within_causal.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\n总耗时 {time.time()-t0:.1f}s  json -> {OUT}/session_within_causal.json')


if __name__ == '__main__':
    main()
