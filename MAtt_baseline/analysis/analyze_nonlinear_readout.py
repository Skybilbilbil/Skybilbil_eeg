# -*- coding: utf-8 -*-
"""非线性读出对照扫描:多切图增益是流形特有还是通用非线性;K 继续爬;「注意力剖面」。

## 三问(模型方向,承接 multichart_ceiling)
① 家族爬到哪:randK K=48/64(2 抽样)是否继续升(32 处未饱和);
② 机制对照:单切图 RBF-SVM、度-2 单项式(随机 4096 子集)、幂函数族 C^p —— 若这些通用
   非线性读出也能到 ~68,则「多切图」本质 = 核/非线性展开,流形几何不特殊;
   若到不了,切图族携带独特内容,模型应对准它;
③ 读出层:rand32 特征上共享 MLP(非线性)能否超线性 68.6;
   LDA 系数按切图分块的范数剖面 = 「注意力剖面」(哪些切图承担读出权重),
   前 5 + 末 1 切图逐一消融(全体/低四/高五 + McNemar)——检验「切图选择」有没有价值。

口径:主协议 rs=42 同一 test 折;SVM/MLP 超参在 val 折选;除 MLP 外零训练。
只写 results/nonlinear_readout/。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from analyze_multichart_ceiling import (                             # noqa: E402
    DIM, chart_feats, cov_tn, eval_split, fit_lda, grp, mcnemar)
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'nonlinear_readout'
TASK = ED.TASK
SUBS = ['S%02d' % i for i in range(1, 10)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_va, y_va, sub_va = ret[2], ret[3], ret[7]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    C_tr, C_va, C_te = (cov_tn(X[:, :, TASK[0]:TASK[1]]) for X in (X_tr, X_va, X_te))
    print('train %d val %d test %d' % (len(y_tr), len(y_va), len(y_te)))

    rec = {'rungs': {}}

    def report(name, ov, ovv, per, ok, note=''):
        lo, hi = grp(per)
        rec['rungs'][name] = {'val': ovv, 'test': ov, 'low4': lo, 'high5': hi,
                              'per_subject': per, 'note': note,
                              '_ok': [bool(v) for v in ok]}
        print('%-16s val %.1f  test %.1f (低%.1f 高%.1f)  %s'
              % (name, ovv * 100, ov * 100, lo * 100, hi * 100, note))
        return ok

    def run_lda(name, F_tr, F_va, F_te, note=''):
        clf = fit_lda(F_tr, y_tr)
        ovv, _, _ = eval_split(clf, F_va, y_va, sub_va)
        ov, per, ok = eval_split(clf, F_te, y_te, sub_te)
        return report(name, ov, ovv, per, ok, note), clf

    # ---- 锚:identity 单切图 + rand32(draw0) ----
    F_id = [chart_feats(C, None)[0] for C in (C_tr, C_va, C_te)]
    okA, _ = run_lda('A_identity', *F_id, note='旧天花板 62.9')

    def get_draw(draw, K_max=32):
        rng = np.random.default_rng(draw)
        idx = rng.choice(len(C_tr), K_max, replace=False)
        return [[chart_feats(C, C_tr[i])[0] for i in idx]
                for C in (C_tr, C_va, C_te)]

    print('切图特征计算中(draw0 × 32 参考点)...')
    D0 = get_draw(0)
    print('完成')

    def concat_draw(D, K):
        return [np.concatenate(D[i][:K], 1) for i in range(3)]

    okR32, clf32 = run_lda('R32_draw0', *concat_draw(D0, 32), note='家族当前最优锚 68.6')

    # ---- ① K 饱和已在 multichart/本脚本前两次运行定论(48 两抽 67.4、64 两抽 67.5,见 log),不再重拟合 ----

    # ---- ③ 注意力剖面(K=32 draw0,已拟合):分块范数 + 前5/末1 系数置零消融 ----
    F32 = concat_draw(D0, 32)
    blocks = np.array([np.linalg.norm(clf32.coef_[:, k * DIM:(k + 1) * DIM])
                       for k in range(32)])
    top5 = np.argsort(blocks)[::-1][:5]
    bot1 = int(np.argmin(blocks))
    print('切图块范数剖面: max %.2f min %.2f 中位 %.2f; top5=%s bottom=%d'
          % (blocks.max(), blocks.min(), np.median(blocks), top5.tolist(), bot1))
    sc_full = F32[2] @ clf32.coef_.T + clf32.intercept_
    ok64 = sc_full.argmax(1) == y_te
    rec['attn_profile'] = {'block_norms': [round(float(b), 4) for b in blocks],
                           'top5': top5.tolist(), 'bottom': bot1}
    for k in list(top5) + [bot1]:
        coef_k = clf32.coef_.copy()
        coef_k[:, k * DIM:(k + 1) * DIM] = 0.0
        ok = (F32[2] @ coef_k.T + clf32.intercept_).argmax(1) == y_te
        ov = float(ok.mean())
        per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        lo, hi = grp(per)
        p, b, c = mcnemar(ok, ok64)
        rec['attn_profile']['ablate_%d' % k] = {
            'test': ov, 'd_vs_full': float(ov - ok64.mean()), 'p': p, 'b': b, 'c': c,
            'note': 'coef 置零(读出级),非重拟合'}
        print('  消融切图 %2d: test %.1f (Δ%+.1f, 低%.1f 高%.1f, McNemar p=%.3f)'
              % (k, ov * 100, 100 * (ov - ok64.mean()), lo * 100, hi * 100, p))

    # ---- ②a 幂函数族 C^p ----
    def pow_feats(p):
        out = []
        for C in (C_tr, C_va, C_te):
            w, V = np.linalg.eigh(C.astype(np.float64))
            w = np.maximum(w, 1e-6)
            out.append(np.stack([_spdvec((V[i] * w[i] ** p) @ V[i].T)
                                 for i in range(len(C))]))
        return out

    def _spdvec(L):
        iu = np.triu_indices(22)
        v = L[iu].copy()
        v[iu[0] != iu[1]] *= np.sqrt(2)
        return v

    P = {p: pow_feats(p) for p in (-0.5, -0.25, 0.25, 0.5, 2.0)}
    pow_only = [np.concatenate([P[p][i] for p in (-0.5, -0.25, 0.25, 0.5, 2.0)], 1)
                for i in range(3)]
    run_lda('power5_only', *pow_only, note='C^p 五个幂,5×253')
    pow_le = [np.concatenate([F_id[i]] + [P[p][i] for p in (-0.5, -0.25, 0.25, 0.5, 2.0)], 1)
              for i in range(3)]
    run_lda('power5_plus_LE', *pow_le, note='幂族⊕log(identity),6×253')

    # ---- ②b 度-2 单项式(LE 253 维的 v_i v_j,随机 4096) ----
    v = np.concatenate([F_id[i] for i in range(3)], 0)
    iu = np.triu_indices(253)
    rng_m = np.random.default_rng(0)
    sel = rng_m.choice(len(iu[0]), 4096, replace=False)
    mono = np.stack([v[:, iu[0][s]] * v[:, iu[1][s]] for s in sel], 1)
    mono_tr, mono_va, mono_te = (mono[:len(y_tr)], mono[len(y_tr):len(y_tr) + len(y_va)],
                                 mono[len(y_tr) + len(y_va):])
    run_lda('mono2_4096', mono_tr, mono_va, mono_te, note='单项式 4096(不含一次项)')
    mono_le = [np.concatenate([F_id[i], m], 1) for i, m in
               enumerate((mono_tr, mono_va, mono_te))]
    run_lda('mono2_plus_LE', *mono_le, note='单项式 4096 ⊕ LE')

    # ---- ②c 单切图 RBF-SVM ----
    from sklearn.svm import SVC
    sub_ = np.random.default_rng(1).choice(len(y_tr), 1000, replace=False)
    d2 = ((F_id[0][sub_][:, None, :] - F_id[0][sub_][None, :]) ** 2).sum(-1)
    med2 = float(np.median(d2[np.triu_indices(len(sub_), 1)]))
    best = None
    for C_ in (1.0, 10.0, 100.0):
        for gm in (0.5, 1.0, 2.0):
            svm = SVC(kernel='rbf', C=C_, gamma=gm / (2 * med2))
            svm.fit(F_id[0], y_tr)
            ovv = float((svm.predict(F_id[1]) == y_va).mean())
            if best is None or ovv > best[0]:
                best = (ovv, C_, gm, svm)
    ovv, C_, gm, svm = best
    ok = svm.predict(F_id[2]) == y_te
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    pA, b_, c_ = mcnemar(ok, okA)
    p32, _, _ = mcnemar(ok, okR32)
    lo, hi = grp(per)
    rec['rungs']['svm_rbf_single'] = {
        'val': ovv, 'test': float(ok.mean()), 'low4': lo, 'high5': hi,
        'per_subject': per, 'note': 'C=%.0f γ=%.1f/2med², val 选' % (C_, gm),
        'p_vs_A': pA, 'p_vs_R32': p32}
    print('%-16s val %.1f  test %.1f (低%.1f 高%.1f)  p_vsA=%.3f p_vsR32=%.3f'
          % ('svm_rbf_single', ovv * 100, ok.mean() * 100, lo * 100, hi * 100, pA, p32))

    # ---- ③ MLP 共享非线性读出(rand32 draw0) ----
    from sklearn.neural_network import MLPClassifier
    mlp = MLPClassifier(hidden_layer_sizes=(256,), max_iter=300,
                        early_stopping=True, random_state=0)
    mlp.fit(F32[0], y_tr)
    ok = mlp.predict(F32[2]) == y_te
    ovv = float((mlp.predict(F32[1]) == y_va).mean())
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    p32, _, _ = mcnemar(ok, okR32)
    rec['rungs']['mlp_on_rand32'] = {
        'val': ovv, 'test': float(ok.mean()), 'low4': lo, 'high5': hi,
        'per_subject': per, 'note': '(256,) early-stop;非线性共享读出(有训练)',
        'p_vs_R32': p32}
    print('%-16s val %.1f  test %.1f (低%.1f 高%.1f)  p_vsR32=%.3f'
          % ('mlp_on_rand32', ovv * 100, ok.mean() * 100, lo * 100, hi * 100, p32))

    # ---- 汇总 McNemar ----
    for name in ('power5_plus_LE', 'mono2_plus_LE', 'mlp_on_rand32'):
        ok_ = np.array(rec['rungs'][name]['_ok']) if '_ok' in rec['rungs'][name] \
            else None
        if ok_ is None:
            continue
        p32, b_, c_ = mcnemar(ok_, okR32)
        rec['rungs'][name]['p_vs_R32'] = p32
        print('McNemar %-16s vs R32 p=%.3f (b=%d c=%d)' % (name, p32, b_, c_))

    (OUT / 'nonlinear_readout.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/nonlinear_readout.json' % OUT)


if __name__ == '__main__':
    main()
