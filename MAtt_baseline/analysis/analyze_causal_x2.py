# -*- coding: utf-8 -*-
"""X-2 试次级配对:逐试次 μ ERD 与「该试次是否被判对」的关系(被试内斜率 = 因果量的代理)。

配套: analysis/差异报告/因果性分析方案_ERD到精度.md §3.2(β_W/β_B 分解)、§4 X-2、§X-0。
产物 results/causal_erd_injection/causal_erd_x2.json

## 用的是哪个「正确性」
深度模型的**逐试次预测**(3 个主协议 ckpt 多数投票),测试集 = 主协议 70/15/15 的固定测试折。
不用线性读出器做主体,因为它每次重训都会变;深度模型是**冻结**的,试次级结果可复现。

## 为什么必须做测量误差修正
X-0 实测:逐试次 ERD 的单试次抽样 SE = 1.59 dB,而观测 sd 只有 1.38–2.27 dB
⇒ 信度 ρ = 1 − SE²/sd² ≈ 0.19。经典测量理论下,预测子带误差会让回归斜率**衰减**到 ρ 倍:

    β̂_observed = ρ · β_true   ⇒   β_corrected = β̂ / ρ

**不做这一步就会把「真有效应但被噪声吃掉」误读成「无效应」。** 本脚本两种都报:
ρ=1(未修正,下界)与 β̂/ρ(修正,点估计;区间用被试级 bootstrap)。

## 三种设计(互相独立,用来互相制衡)
  A 主设计:测试折内、被试内去均值、逐试次 ERD 预测「该试次对否」(线性概率,+ 被试级 bootstrap)
  B 扩大样本:全部 576 试次(不受测试折 80–91 试次/人的限制),逐人 OLS 斜率
  C 劈半防循环:随机一半试次算 ERD、另一半取正确性(两者来自不同数据)⇒ 排除
    「ERD 与正确性由同一批数据算出」的机械耦合

⚠ 已知局限(方案 §X-2 写明):分类器在含测试折的数据上训练过 ⇒ 单试次正确性偏乐观。
A/C 都用固定切分的测试折,不额外乐观化;B 用全部试次、无此问题但换了总体。
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_causal_erd_injection import (MU, TASK, BASE_WIN, FS,  # noqa: E402
                                          narrow, band_pow)
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                  # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
WIN, STEP = 375, 188
RUNS = {'seed42':   ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024',        'v70s2024'),
        'seed7':    ('verify70_seed7',           'v70s7')}
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
SUBJ = [f'S{i+1:02d}' for i in range(9)]
SE_POOLED = 1.59          # X-0 实测:单试次 μ ERD 的抽样标准误 (dB)
NBOOT = 4000
RNG = np.random.default_rng(20261016)


def per_trial_erd(X):
    """逐试次 μ ERD(全 22 通道功率比,与 X-0 同口径)。"""
    Zb = narrow(X, MU)
    R = band_pow(Zb, BASE_WIN)
    A = band_pow(Zb, TASK)
    return 10 * np.log10(np.maximum(A, 1e-30) / np.maximum(R, 1e-30))


def predict_ok(nets, X, y):
    """3 个 ckpt 多数投票 → 逐试次是否判对。"""
    x = torch.Tensor(X.astype(np.float32)).unsqueeze(1).to(
        next(nets[0].parameters()).device)
    votes = np.zeros((len(nets), len(y)), dtype=bool)
    for i, net in enumerate(nets):
        net.eval()
        with torch.no_grad():
            votes[i] = (net(x).argmax(dim=1).cpu().numpy() == y)
    return votes.sum(0) >= (len(nets) + 1) // 2, votes


def slope_within(e, ok, sub):
    """被试内去均值后的 OLS 斜率(每 dB 的「判对概率」变化,pp/dB)+ 朴素 SE。"""
    e = np.asarray(e, float)
    y = np.asarray(ok, float)
    su = np.asarray(sub)
    ec = e - np.array([e[su == s].mean() for s in range(su.max() + 1)])[su]
    yc = y - np.array([y[su == s].mean() for s in range(su.max() + 1)])[su]
    b = float((ec @ yc) / (ec @ ec))
    r = yc - b * ec
    se = float(np.sqrt((r @ r) / (len(r) - 2) / (ec @ ec)))
    return b, se


def boot_slope(e, ok, sub, nboot=NBOOT, rng=RNG):
    """被试级 bootstrap(重抽 9 个被试)的被试内斜率分布。"""
    e, ok, su = np.asarray(e, float), np.asarray(ok, float), np.asarray(sub)
    idx = [np.where(su == s)[0] for s in range(su.max() + 1)]
    vals = np.empty(nboot)
    for b in range(nboot):
        pick = rng.integers(0, len(idx), len(idx))
        ee = np.concatenate([e[idx[i]] for i in pick])
        oo = np.concatenate([ok[idx[i]] for i in pick])
        ss = np.concatenate([np.full(len(idx[i]), k) for k, i in enumerate(pick)])
        vals[b] = slope_within(ee, oo, ss)[0]
    return vals


def block_A(erd_te, ok_te, sub_te):
    e, o, s = np.concatenate(erd_te), np.concatenate(ok_te), np.concatenate(sub_te)
    b, se = slope_within(e, o, s)
    bs = boot_slope(e, o, s)
    rb = 1.0 - SE_POOLED ** 2 / float(np.var(e))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print('=' * 74)
    print('A 主设计:测试折内、被试内去均值、逐试次 ERD → 该试次对否')
    print('=' * 74)
    print(f'  试次 {len(e)}  逐试次 ERD 的 sd = {np.var(e)**0.5:.3f} dB')
    print(f'  观测斜率(被试内)      = {b*100:+.4f} pp/dB  (朴素 SE {se*100:.4f})')
    print(f'  信度 ρ = 1 − SE²/sd²   = {rb:.4f}')
    print(f'  ★ 修正后 β_W           = {b/rb*100:+.4f} pp/dB')
    print(f'  bootstrap 95% CI(修正) = [{lo/rb*100:+.4f}, {hi/rb*100:+.4f}]  '
          f'(被试级重抽 {NBOOT} 次)')
    print(f'  未修正 CI              = [{lo*100:+.4f}, {hi*100:+.4f}]')
    print(f'  ⇒ 区间是否跨 0(修正后): {"是 ⇒ 不显著" if lo/rb*100 <= 0 <= hi/rb*100 else "否 ⇒ 显著"}')
    return {'n': int(len(e)), 'erd_sd': float(np.var(e) ** 0.5),
            'beta_obs_pp_per_dB': b * 100, 'se_naive': se * 100,
            'reliability': rb, 'beta_corrected_pp_per_dB': b / rb * 100,
            'ci_obs': [lo * 100, hi * 100],
            'ci_corrected': [lo / rb * 100, hi / rb * 100],
            'significant_corrected': bool(not (lo / rb * 100 <= 0 <= hi / rb * 100))}


def block_B(erd_all, ok_all, sub_all):
    print('\n' + '=' * 74)
    print('B 逐人斜率(全部 576 试次/人,无测试折样本量限制)')
    print('=' * 74)
    per = []
    for s in range(9):
        b, se = slope_within(erd_all[s], ok_all[s], np.zeros(len(erd_all[s]), int))
        per.append({'subject': SUBJ[s], 'beta_pp_per_dB': b * 100, 'se': se * 100,
                    'mean_erd': float(erd_all[s].mean()),
                    'acc': float(ok_all[s].mean())})
        print(f'  {SUBJ[s]}  β_W={b*100:+7.3f} pp/dB (SE {se*100:.3f})   '
              f'ERD 均值 {erd_all[s].mean():+.2f} dB   该人精度 {ok_all[s].mean()*100:.1f}%')
    bs = np.array([p['beta_pp_per_dB'] for p in per])
    ses = np.array([p['se'] for p in per])
    w = 1 / ses ** 2
    print(f'\n  逆方差加权均值 β_W = {np.sum(bs*w)/np.sum(w):+.4f} pp/dB')
    print(f'  9 人符号:负 {int((bs<0).sum())}/9,正 {int((bs>0).sum())}/9')
    print(f'  逐人 β_W 范围 {bs.min():+.3f} ~ {bs.max():+.3f}')
    return {'per_subject': per, 'weighted_mean_pp_per_dB': float(np.sum(bs * w) / np.sum(w)),
            'n_negative': int((bs < 0).sum())}


def block_C(X_all, y_all, erd_all, nets):
    """劈半防循环:一半算 ERD、另一半取正确性,角色互换。"""
    print('\n' + '=' * 74)
    print('C 劈半设计:一半试次算 ERD、另一半取正确性(排除同批数据的机械耦合)')
    print('=' * 74)
    e_half1, o_half2, s_idx = [], [], []
    e_half2, o_half1 = [], []
    for s in range(9):
        ok_s, _ = predict_ok(nets, X_all[s], y_all[s])
        n = len(ok_s)
        perm = RNG.permutation(n)
        h1, h2 = perm[:n // 2], perm[n // 2:]
        e_half1.append(erd_all[s][h1]); o_half2.append(ok_s[h2])
        e_half2.append(erd_all[s][h2]); o_half1.append(ok_s[h1])
        s_idx += [s] * len(h1)
    e1 = np.concatenate(e_half1); o2 = np.concatenate(o_half2)
    e2 = np.concatenate(e_half2); o1 = np.concatenate(o_half1)
    su = np.array(s_idx)
    e = np.concatenate([e1, e2]); o = np.concatenate([o2, o1])
    su = np.concatenate([su, su])
    b, se = slope_within(e, o, su)
    bs = boot_slope(e, o, su, nboot=2000)
    print(f'  试次 {len(e)}(两人各半,角色互换后合并)')
    print(f'  观测斜率 = {b*100:+.4f} pp/dB (SE {se*100:.4f})')
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'  bootstrap 95% CI = [{lo*100:+.4f}, {hi*100:+.4f}]')
    print(f'  ⇒ 跨 0:{"是 ⇒ 不显著" if lo*100 <= 0 <= hi*100 else "否 ⇒ 显著"}')
    print('  注:劈半后 ERD 的有效样本减半 ⇒ 信度更低,这里只报**未修正**斜率,')
    print('     因为它已经是「不共享数据」的下界。')
    return {'n': int(len(e)), 'beta_obs_pp_per_dB': b * 100, 'se_naive': se * 100,
            'ci_obs': [lo * 100, hi * 100],
            'significant_obs': bool(not (lo * 100 <= 0 <= hi * 100))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nboot', type=int, default=NBOOT)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('X-2 试次级配对(零训练,冻结 ckpt)')
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = [np.asarray(d[0], dtype=np.float64) for d in subject_data]
    y_all = [np.asarray(d[1]) for d in subject_data]
    print('计算逐试次 μ ERD ...')
    erd_all = [per_trial_erd(X) for X in X_all]

    Xc = np.concatenate(X_all, 0)
    yc = np.concatenate(y_all, 0)
    sub_orig = np.concatenate([np.full(len(y_all[i]), i) for i in range(9)])   # 全池原始行号
    ret = split_train_valid_test(Xc, yc, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_orig])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]

    nets = []
    for name, (dirname, tag) in RUNS.items():
        ck = BASE / 'results' / dirname / f'{tag}_cross_mix_best_acc.pt'
        nets.append(load_matt_model(str(ck), WIN, STEP, ARGS))
    print(f'已加载 {len(nets)} 个 ckpt;测试折 {len(y_te)} 试次')
    ok_te, votes = predict_ok(nets, X_te, y_te)
    print(f'多数投票测试折精度 = {ok_te.mean()*100:.2f}%')

    # 把测试折的 ERD 对齐出来(erd_all 是按被试顺序的,测试折是全池切分)
    # ⚠ 测试折是**打乱后**取子集,行号不是原顺序 ⇒ 必须经 sub_all 映射回「全池原始行号」,
    #    再减该被试的偏移,才能落回 erd_all[s] 的被试内索引。第一版直接用测试折行号减偏移,
    #    报 IndexError(index 582 vs 576)。
    # 用「(被试, 类别) 计数 + 标签序列」精确匹配测试折每行在全池中的位置
    key_pool = {}
    for i in range(len(sub_orig)):
        key_pool.setdefault((int(sub_orig[i]), int(yc[i])), []).append(i)
    used = {}
    rows = np.empty(len(sub_te), dtype=int)
    for i in range(len(sub_te)):
        k = (int(sub_te[i]), int(y_te[i]))
        j = used.get(k, 0)
        rows[i] = key_pool[k][j]
        used[k] = j + 1
    erd_te = []
    for s in range(9):
        m = np.where(sub_te == s)[0]
        off = sum(len(y_all[j]) for j in range(s))
        erd_te.append(erd_all[s][rows[m] - off])
    out = {'config': {'se_pooled_dB': SE_POOLED, 'nboot': NBOOT, 'n_test': int(len(y_te)),
                      'vote_acc': float(ok_te.mean())}}
    out['A_testfold'] = block_A(erd_te, [ok_te[sub_te == s] for s in range(9)],
                                [np.full(int((sub_te == s).sum()), s) for s in range(9)])
    ok_all = []
    for s in range(9):
        ok_s, _ = predict_ok(nets, X_all[s], y_all[s])
        ok_all.append(ok_s)
    out['B_full576'] = block_B(erd_all, ok_all, sub_orig)
    out['C_splithalf'] = block_C(X_all, y_all, erd_all, nets)
    # 观测性总体相关(不做被试内分解)作为对照
    e = np.concatenate(erd_te); o = np.concatenate([ok_te[sub_te == s] for s in range(9)])
    out['D_pooled_corr'] = {'pearson_r': float(np.corrcoef(e, o)[0, 1]),
                            'slope_pp_per_dB': float(np.polyfit(e, o, 1)[0] * 100)}
    print(f"\n  D 对照(不分解被试间/内): r = {out['D_pooled_corr']['pearson_r']:+.4f}, "
          f"斜率 {out['D_pooled_corr']['slope_pp_per_dB']:+.4f} pp/dB")
    (OUT / 'causal_erd_x2.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\n总耗时 {time.time()-t0:.1f}s  json -> {OUT}/causal_erd_x2.json')


if __name__ == '__main__':
    main()
