# -*- coding: utf-8 -*-
"""§3.7 session 落差的【等预算 / 单方向 / 配对】重做。

背景: 报告 §3.7 引用「同池随机半半 63.6% vs 跨 session 59.0% = +4.7pp」, 出自已丢失的
临时脚本 /tmp/sessgap3.py; 而 subject_table 里的 `session_gap = csp_within − csp_cross_sess`
是【另一个口径】的差, 且被 §3.5 的相关分析使用。本脚本核实两者并给出无混杂版本。

⚠ 已发现的口径缺陷(本脚本要修的正是它):
  `csp_within`      = StratifiedKFold(5) 在【混合两个 session】的 576 试次上 => 每折约
                      460 训练 / 115 测试。附录 A 写作「5 折(同 session)」是【错的】。
  `csp_cross_sess`  = session0->1 与 1->0 的平均, 各 288 训练 / 288 测试。
  => 二者相减【同时】改变了训练样本量(460 vs 288)与 session 构成, 不是纯 session 效应。

四个臂(全部沿用 analyze_individual_diff 的 CSP 配方: one-vs-rest k=2 + reg=0.05 +
log 方差 + LDA(shrinkage='auto'), 任务窗 [0.5,3.5]s, 故与历史「落差」列可比):
  A 同池随机 288/288  : 每类 144 试次随机劈 72/72, 重复 R 次取均值(等预算的干净对照)
  B 单方向 session0->1: 训练 288(全 session0) / 测试 288(全 session1)
  C 单方向 session1->0: 反向
  D 混合 5 折(= 现表 `csp_within`, 用于确认本脚本能复现该列)

session 归属沿用 analyze_individual_diff 的约定: 每个类别内【前一半 = session 0(A0T),
后一半 = session 1(A0E)】, 依据 preprocess/moabb_bci2a.py 用 sorted(sessions) 拼接
0train->1test(实验记录 ⑫/⑬)。

纯 CPU; 只读 EA pkl, 只写 results/session_gap/。秒级。
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
sys.path.insert(0, str(BASE))
from analyze_individual_diff import KEYS, TASK, cov, csp_acc  # noqa: E402

OUT = BASE / 'results' / 'session_gap'
PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
N_SUBJ = 9
N_REPEAT = 20          # A 臂的随机半半重复次数(降低单次切分的抽样波动)
SEED = 0


def session_index(n_per_class):
    """每类前一半 = session 0, 后一半 = session 1; 按 KEYS 顺序拼接成 (576,)"""
    return np.concatenate([np.r_[np.zeros(n // 2, int), np.ones(n - n // 2, int)]
                           for n in n_per_class])


def random_half_split(y, rng):
    """每类随机劈一半作训练 -> (tr_idx, te_idx), 严格 288/288 且类别平衡"""
    tr, te = [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        h = len(idx) // 2
        tr.append(idx[:h]); te.append(idx[h:])
    return np.concatenate(tr), np.concatenate(te)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PKL, 'rb') as fh:
        d = pickle.load(fh)
    n = len(d['subject_metadata'])
    assert n == N_SUBJ, 'pkl 被试数 %d != %d' % (n, N_SUBJ)

    rng = np.random.default_rng(SEED)
    res = {k: [] for k in 'ABCD'}
    for s in range(N_SUBJ):
        X = np.concatenate([np.asarray(d[k][s]) for k in KEYS], 0).astype(np.float64)
        y = np.concatenate([np.full(len(d[k][s]), c) for c, k in enumerate(KEYS)])
        sess = session_index([len(d[k][s]) for k in KEYS])
        Ct = cov(X[:, :, TASK[0]:TASK[1]])
        m0, m1 = np.where(sess == 0)[0], np.where(sess == 1)[0]
        assert len(m0) == len(m1) == 288, '两 session 试次数不等: %d/%d' % (len(m0), len(m1))

        # A: 同池随机 288/288, 重复 N_REPEAT 次
        res['A'].append(float(np.mean([csp_acc(Ct, y, *random_half_split(y, rng))
                                       for _ in range(N_REPEAT)])))
        # B / C: 单方向跨 session
        res['B'].append(float(csp_acc(Ct, y, m0, m1)))
        res['C'].append(float(csp_acc(Ct, y, m1, m0)))
        # D: 混合 5 折(= 现表 csp_within)
        res['D'].append(float(np.mean([csp_acc(Ct, y, tr, te) for tr, te in
                                       StratifiedKFold(5, shuffle=True,
                                                       random_state=0).split(Ct, y)])))
        print('  S%02d  A=%.4f B=%.4f C=%.4f D=%.4f'
              % (s + 1, res['A'][-1], res['B'][-1], res['C'][-1], res['D'][-1]), flush=True)

    A, B, C, D = (np.array(res[k]) for k in 'ABCD')
    cross_avg = (B + C) / 2

    print('\n=== 逐被试(等预算 288 训练 / 288 测试) ===')
    print('%-6s %10s %10s %10s %10s %10s %10s'
          % ('被试', 'A同池随机', 'B s0->s1', 'C s1->s0', 'BC平均', 'A-BC', 'D混合5折'))
    for s in range(N_SUBJ):
        print('%-6s %10.4f %10.4f %10.4f %10.4f %+10.4f %10.4f'
              % ('S%02d' % (s + 1), A[s], B[s], C[s], cross_avg[s],
                 A[s] - cross_avg[s], D[s]))
    print('%-6s %10.4f %10.4f %10.4f %10.4f %+10.4f %10.4f'
          % ('均值', A.mean(), B.mean(), C.mean(), cross_avg.mean(),
             (A - cross_avg).mean(), D.mean()))

    gap = A - cross_avg
    pos = int((gap > 0).sum())
    bt = binomtest(pos, N_SUBJ, 0.5, alternative='greater')
    print('\n=== 配对比较 ===')
    print('等预算 session 落差 A − (B+C)/2 = %+.4f  (逐人 %+.3f ~ %+.3f)'
          % (gap.mean(), gap.min(), gap.max()))
    print('符号检验: %d/%d 为正, 单侧 p = %.5f' % (pos, N_SUBJ, bt.pvalue))
    print('单方向不对称: B(s0->s1) %.4f vs C(s1->s0) %.4f, 差 %+.4f'
          % (B.mean(), C.mean(), B.mean() - C.mean()))
    print('  逐人 B−C: %s' % np.array2string(B - C, precision=3, suppress_small=True))
    print('\n=== 口径缺陷核实 ===')
    print('D(混合5折, 每折约460训练) 均值 = %.4f  <- 这才是 subject_table 的 csp_within'
          % D.mean())
    print('A(同池随机, 严格288训练)  均值 = %.4f' % A.mean())
    print('⇒ 训练样本量 460->288 本身就值 %+.4f; 故 `session_gap = csp_within − csp_cross_sess`'
          % (D.mean() - A.mean()))
    print('   = %+.4f 中混入了 %+.4f 的样本量效应, 纯 session 效应只有 %+.4f'
          % ((D - cross_avg).mean(), (D - A).mean(), gap.mean()))

    rec = {'recipe': 'CSP one-vs-rest k=2, reg=0.05, log-var, LDA(shrinkage=auto), '
                     'task window [0.5,3.5]s, 288 train / 288 test',
           'n_repeat_armA': N_REPEAT, 'seed': SEED,
           'subjects': ['S%02d' % (s + 1) for s in range(N_SUBJ)],
           'A_pooled_random': [round(v, 4) for v in A],
           'B_session0_to_1': [round(v, 4) for v in B],
           'C_session1_to_0': [round(v, 4) for v in C],
           'D_mixed_5fold': [round(v, 4) for v in D],
           'mean': {'A': round(float(A.mean()), 4), 'B': round(float(B.mean()), 4),
                    'C': round(float(C.mean()), 4), 'BC': round(float(cross_avg.mean()), 4),
                    'D': round(float(D.mean()), 4)},
           'gap_equal_budget': round(float(gap.mean()), 4),
           'gap_range': [round(float(gap.min()), 4), round(float(gap.max()), 4)],
           'sign_test_positive': pos, 'sign_test_p_one_sided': round(float(bt.pvalue), 6),
           'train_size_effect_D_minus_A': round(float((D - A).mean()), 4),
           'session_gap_column_confounded': round(float((D - cross_avg).mean()), 4)}
    (OUT / 'session_gap.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
    print('\n[OK] %s/session_gap.json' % OUT)


if __name__ == '__main__':
    main()
