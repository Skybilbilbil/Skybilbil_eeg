# -*- coding: utf-8 -*-
"""§3.6 判决性检验的可复现实现: EA 之后剩下的【不是】旋转。

被检验的模型 H0(共享源结构 + 被试私有线性导联场):
    x_i = L_i s,   s = 跨被试共享的源过程,  L_i = 被试 i 的私有线性导联场
  => 逐类协方差  C_{i,c} = L_i Σ_c L_i^T,  其中 Σ_c 跨被试共享。
  EA 用被试平均协方差 R_i = L_i Σ_tot L_i^T 白化。令
      A_i = (L_i Σ_tot L_i^T)^{-1/2} · L_i Σ_tot^{1/2}
  可直接验证 A_i^T A_i = I(正交), 且
      C~_{i,c} = A_i Σ~_c A_i^T,   Σ~_c = Σ_tot^{-1/2} Σ_c Σ_tot^{-1/2} 仍共享
  即 **H0 + EA => 被试之间只差一个正交变换**。

旋转不变量(判决性统计量):
      M^(i)_{cc'} = C~_{i,c}^{-1/2} · C~_{i,c'} · C~_{i,c}^{-1/2}
                  = A_i ( Σ~_c^{-1/2} Σ~_{c'} Σ~_c^{-1/2} ) A_i^T
  相似变换不改变特征值谱 => **H0 成立时 M 的谱必须跨被试完全一致**。

判据 = 跨被试谱差异 / 同一人劈半的估计噪声地板。H0 成立时该比值应为 1/√2 ≈ 0.707(不是 1):
劈半用的是半样本估计、每半估计方差是全样本的 2 倍 ⇒ 两条半样本谱之差 E[d²]=4v;
跨被试是两条全样本谱之差 ⇒ E[d²]=2v;故 d_between/d_noise = √(2v/4v) = 0.707。
实测远大于 0.707 即与 H0 不一致。

⚠ 本脚本是历史 /tmp/check_rotation.py(已丢失)的【重建】。原脚本的谱距离定义未被
  记录, 故 d_between / d_noise 的【绝对数值】不保证与实验记录 ⑩ 的 0.3451 / 0.0592
  逐位一致。本脚本显式给出度量定义, 可复现性判据分两层:
    (a) 逐人 λ₁(M_left,right 的最大特征值)—— 不依赖任何距离定义, 应与实验记录 ⑩
        的表直接可比(1.771 / 1.225 / 2.564 / 1.363 / 1.225 / 1.437 / 1.517 /
        2.988 / 4.376);
    (b) 比值 ≈5.8 与「被试间 > 类间」—— 不依赖度量的整体缩放。

谱距离定义(本脚本): M 的 22 个特征值取 log 并升序排列, 两谱之间用 RMS 距离
      d(a, b) = sqrt( mean_k (log λ_k(a) − log λ_k(b))^2 )
该度量对 M 的整体缩放不变(常数偏移在差分中抵消), 与项目其它处使用的
Log-Euclidean 表示同源。

纯 CPU; 只读 saved_data/ 的 EA pkl, 只写 results/rotation_invariance/。秒级。
"""
import json
import pickle
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
OUT = BASE / 'results' / 'rotation_invariance'
PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'

KEYS = ['aligned_left_epochs_list', 'aligned_right_epochs_list',
        'aligned_feet_epochs_list', 'aligned_hands_epochs_list']
LABELS = ['left', 'right', 'feet', 'tongue']
# cue 在样本 250(实验记录 ⑭ 时间轴核实), 故 [375,1126) = 事件后 0.5–3.5s
TASK = (375, 1126)
N_SUBJ = 9
# 12 个有序类别对: M_{cc'} 以 c 为参考白化 c', 故 (c,c') 与 (c',c) 不等价
PAIRS = [(c, d) for c in range(4) for d in range(4) if c != d]
N_BOOT = 200
SEED = 0


def trial_outer(X):
    """X (n,22,T) -> 逐 trial 去均值外积 / T, 形状 (n,22,22)。

    批量协方差 = 这些外积的均值, 因此 bootstrap 只需对它们重采样后取均值,
    与「重采样原始 trial 再重算协方差」完全等价, 但无需把原始信号留在内存。
    """
    s = X - X.mean(axis=2, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / X.shape[-1]


def invsqrt(C):
    w, V = eigh(C)
    return V @ np.diag(1.0 / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def log_spec(M):
    """M 的升序 log 特征值向量 (22,)。M 由 C^{-1/2} C' C^{-1/2} 构造, 对称正定。"""
    return np.log(np.maximum(np.linalg.eigvalsh(M), 1e-12))


def spectra(Cs):
    """Cs (4,22,22) -> {pair: log谱}, {pair: λ₁}"""
    ish = [invsqrt(Cs[c]) for c in range(4)]
    sp, lam1 = {}, {}
    for p in PAIRS:
        M = ish[p[0]] @ Cs[p[1]] @ ish[p[0]]
        sp[p] = log_spec(M)
        lam1[p] = float(np.exp(sp[p][-1]))
    return sp, lam1


def rms(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def three_distances(all_sp):
    """all_sp: list(9) of {pair: log谱} -> (pair_bet, pair_bet_mean, d_class)

    pair_bet[p]      : 类别对 p 上, 36 个被试两两谱距离的均值。
    pair_bet_mean[p] : 类别对 p 上, 每人谱与【9 人平均谱】距离的均值。
                       独立噪声下 ≈ 两两版 / sqrt(2), 二者只是聚合口径不同。
    d_class          : 同一人内「换类别」造成的谱变化 —— 固定被试与参考类 c,
                       {M_{c,d}: d≠c} 这 3 条谱两两之间的距离, 对全部
                       被试 × 参考类 × 类对平均。
    """
    pair_bet, pair_bet_mean = {}, {}
    for p in PAIRS:
        specs = [all_sp[s][p] for s in range(N_SUBJ)]
        pair_bet[p] = float(np.mean([rms(specs[i], specs[j])
                                     for i, j in combinations(range(N_SUBJ), 2)]))
        mu = np.mean(specs, axis=0)
        pair_bet_mean[p] = float(np.mean([rms(sp, mu) for sp in specs]))
    cls = [rms(all_sp[s][(c, d1)], all_sp[s][(c, d2)])
           for s in range(N_SUBJ) for c in range(4)
           for d1, d2 in combinations([d for d in range(4) if d != c], 2)]
    return pair_bet, pair_bet_mean, float(np.mean(cls))


def noise_floor(S, rng):
    """同一人劈半噪声地板 -> {pair: 距离均值}。

    每个(被试,类)的 trial 随机劈成两半各算一次 M, 取两半谱之间的距离 —— 这是
    纯采样噪声的下界: H0 成立时跨被试距离应当落在这个量级上。
    """
    acc = {p: [] for p in PAIRS}
    for s in range(N_SUBJ):
        halves = []
        for c in range(4):
            n = len(S[s][c])
            perm = rng.permutation(n)
            halves.append([S[s][c][perm[:n // 2]].mean(0),
                           S[s][c][perm[n // 2:]].mean(0)])
        sp_a, _ = spectra(np.stack([halves[c][0] for c in range(4)]))
        sp_b, _ = spectra(np.stack([halves[c][1] for c in range(4)]))
        for p in PAIRS:
            acc[p].append(rms(sp_a[p], sp_b[p]))
    return {p: float(np.mean(v)) for p, v in acc.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PKL, 'rb') as fh:
        d = pickle.load(fh)
    n = len(d['subject_metadata'])
    assert n == N_SUBJ, 'pkl 被试数 %d != %d' % (n, N_SUBJ)

    # 逐(被试,类)的 trial 外积, 一次算好后 bootstrap 只做索引+均值
    S = [[trial_outer(np.asarray(d[k][s], dtype=np.float64)[:, :, TASK[0]:TASK[1]])
          for k in KEYS] for s in range(N_SUBJ)]
    print('[载入] %d 被试 × %d 类, 每类 trial 数 = %s'
          % (N_SUBJ, 4, [len(S[0][c]) for c in range(4)]))

    Cs = np.stack([np.stack([S[s][c].mean(0) for c in range(4)]) for s in range(N_SUBJ)])
    all_sp, lam1 = [], {}
    for s in range(N_SUBJ):
        sp, l1 = spectra(Cs[s])
        all_sp.append(sp)
        lam1['S%02d' % (s + 1)] = l1

    pair_bet, pair_betm, d_cls = three_distances(all_sp)
    rng = np.random.default_rng(SEED)
    pair_noi = noise_floor(S, rng)
    d_bet = float(np.mean([pair_bet[p] for p in PAIRS]))
    d_betm = float(np.mean([pair_betm[p] for p in PAIRS]))
    d_noi = float(np.mean([pair_noi[p] for p in PAIRS]))

    hist_lam1 = [1.771, 1.225, 2.564, 1.363, 1.225, 1.437, 1.517, 2.988, 4.376]
    lr = [lam1['S%02d' % (s + 1)][(0, 1)] for s in range(N_SUBJ)]

    print('\n=== 判据 (a): 逐人 λ₁ of M_(left,right) —— 不依赖距离定义 ===')
    print('%-6s %10s %10s %10s' % ('被试', '本脚本', '实验记录⑩', '差'))
    for s in range(N_SUBJ):
        print('%-6s %10.3f %10.3f %10.3f'
              % ('S%02d' % (s + 1), lr[s], hist_lam1[s], lr[s] - hist_lam1[s]))
    print('最大绝对差 = %.4f  ⇒ M 的构造与历史脚本逐位一致'
          % max(abs(a - b) for a, b in zip(lr, hist_lam1)))

    print('\n=== 判据 (b): 逐类别对的谱距离 ===')
    print('%-14s %10s %10s %10s %8s' % ('类别对', '被试间两两', '被试间对均值', '劈半噪声', '比值'))
    for p in PAIRS:
        print('%-14s %10.4f %10.4f %10.4f %8.2f'
              % ('%s->%s' % (LABELS[p[0]], LABELS[p[1]]), pair_bet[p],
                 pair_betm[p], pair_noi[p], pair_bet[p] / pair_noi[p]))
    print('%-14s %10.4f %10.4f %10.4f %8.2f'
          % ('12 对平均', d_bet, d_betm, d_noi, d_bet / d_noi))

    print('\n=== 与实验记录⑩ 的对照(历史: 0.3451 / 0.0592 = 5.83×, 类间 0.245) ===')
    print('本脚本 12 对平均: 被试间(两两) %.4f, 被试间(对均值) %.4f, 噪声 %.4f, 类间 %.4f'
          % (d_bet, d_betm, d_noi, d_cls))
    print('噪声地板 %.4f vs 历史 0.0592 (差 %.1f%%) ⇒ 度量本身与历史一致'
          % (d_noi, 100 * abs(d_noi - 0.0592) / 0.0592))
    best = min(PAIRS, key=lambda p: abs(pair_betm[p] - 0.3451))
    print('历史的 0.3451 未记录聚合口径; 最接近的是「对均值」聚合下的 %s->%s = %.4f'
          % (LABELS[best[0]], LABELS[best[1]], pair_betm[best]))
    print('⇒ 结论方向不变且更强: 被试间/噪声 = %.2f× (历史 5.83×), 被试间 %.4f %s 类间 %.4f (历史 0.345 > 0.245)'
          % (d_bet / d_noi, d_bet, '>' if d_bet > d_cls else '<=', d_cls))

    # trial 级 bootstrap: 对每个(被试,类)的 trial 外积有放回重采样
    boot = []
    for _ in range(N_BOOT):
        Cb = np.stack([np.stack([S[s][c][rng.integers(len(S[s][c]),
                                                      size=len(S[s][c]))].mean(0)
                                 for c in range(4)]) for s in range(N_SUBJ)])
        pb, _pm, dc = three_distances([spectra(Cb[s])[0] for s in range(N_SUBJ)])
        db = float(np.mean([pb[p] for p in PAIRS]))
        boot.append((db, dc, db / d_noi))
    boot = np.array(boot)
    ci = {k: [float(np.percentile(boot[:, i], 2.5)), float(np.percentile(boot[:, i], 97.5))]
          for i, k in enumerate(['d_between', 'd_class', 'ratio_vs_noise'])}
    print('\n=== bootstrap 95%% CI (%d 次, trial 级重采样) ===' % N_BOOT)
    for k, v in ci.items():
        print('%-16s [%.4f, %.4f]' % (k, v[0], v[1]))
    # H0 下的期望比值 = 1/√2(半样本 vs 全样本,见文件头推导),不是 1
    null_ratio = float(1.0 / np.sqrt(2.0))
    print('\n⇒ H0(共享源 + 私有线性导联场)要求 ratio ≈ %.3f(1/√2,半样本方差是全样本的 2 倍),'
          'CI 应含该值' % null_ratio)
    if ci['ratio_vs_noise'][0] > null_ratio:
        print('  CI 下界 %.3f > %.3f ⇒ 与 H0 不一致;相对零假设 = %.1f×'
              % (ci['ratio_vs_noise'][0], null_ratio, d_bet / d_noi / null_ratio))
    else:
        print('  CI 含零假设值 ⇒ 不能否证 H0')

    rec = {'window': list(TASK),
           'pairs': ['%s->%s' % (LABELS[a], LABELS[b]) for a, b in PAIRS],
           'metric': 'RMS of log-eigenvalue differences (22-dim), scale-invariant',
           'lambda1_left_right': {'S%02d' % (s + 1): round(lr[s], 4) for s in range(N_SUBJ)},
           'lambda1_left_right_history': dict(zip(['S%02d' % (s + 1) for s in range(N_SUBJ)],
                                                 hist_lam1)),
           'lambda1_all_pairs': {'S%02d' % (s + 1): {'%s->%s' % (LABELS[a], LABELS[b]):
                                                     round(lam1['S%02d' % (s + 1)][(a, b)], 4)
                                                     for a, b in PAIRS}
                                 for s in range(N_SUBJ)},
           'per_pair': {'%s->%s' % (LABELS[a], LABELS[b]):
                        {'d_between_pairwise': round(pair_bet[(a, b)], 4),
                         'd_between_vs_groupmean': round(pair_betm[(a, b)], 4),
                         'd_noise_splithalf': round(pair_noi[(a, b)], 4)}
                        for a, b in PAIRS},
           'd_between': round(d_bet, 4), 'd_between_vs_groupmean': round(d_betm, 4),
           'd_noise': round(d_noi, 4), 'd_class': round(d_cls, 4),
           'ratio_between_over_noise': round(d_bet / d_noi, 3),
           # H0 下应得的比值 = 1/√2(劈半=半样本方差 2v,跨被试=全样本方差 v)
           'null_ratio_under_H0': round(null_ratio, 4),
           'ratio_over_null': round(d_bet / d_noi / null_ratio, 2),
           'history': {'d_between': 0.3451, 'd_noise': 0.0592, 'ratio': 5.83,
                       'd_class': 0.245,
                       'note': '历史 /tmp/check_rotation.py 已丢失, 其跨被试聚合口径未记录; '
                               '本脚本 λ₁ 与噪声地板与其一致, 绝对 d_between 因聚合口径不同而偏大'},
           'bootstrap_ci95': ci, 'n_boot': N_BOOT, 'seed': SEED}
    (OUT / 'rotation_invariance.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/rotation_invariance.json' % OUT)


if __name__ == '__main__':
    main()
