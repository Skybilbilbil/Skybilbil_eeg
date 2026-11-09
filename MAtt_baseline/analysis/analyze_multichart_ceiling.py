# -*- coding: utf-8 -*-
"""多切图(landmark)读法的零训练检验:SPD 流形上「换/加切点」能否超过单切图 62.9 天花板。

## 动机(数学 + 文献)
- 现行天花板 = 单切图:任务窗协方差(22×22 迹归一)在**单一参考点**做 log 映射
  (ceiling_decomposition 的 L0 实为 identity 切图,253 维)+ 共享收缩 LDA = 62.9。
  MAtt 的注意力相似度也在单一切空间里做内积,逐级追踪已证 ≈ 线性读出(闭合)。
- 数学:SPD 流形非平坦 ⇒ log 映射在不同参考点 R_k 给出**本质不同**的坐标(曲率使切图不等价)。
  多切图拼接 = landmark 展开,函数类比「任何单切图的线性函数」严格更大;
  到参考点的 AIRM 距离 d_A(P,R_k)=‖log_{R_k}P‖_F 在任何单切图里是非线性量。
  若类边界含曲率成分,多切图共享读出应超过单切图,且**离参考点远的被试受益更多**
  ——直接对准主线②分组判据(低四单独抬起)。
- 文献:BCI/pyriemann 的切空间法全用单一参考点(全体几何均值);注意力侧已有
  kernel 相似度替代点积(Elliptical Attention 用 Mahalanobis 距离),但 SPD 流形上
  用测地距离当相似度、对一组参考点(切图)做注意力的形式没有先例;SPD 上高斯 RBF
  核的正定性有证明(Jayasumana et al. 2015)。本实验 = 可训练版之前的零训练阶梯。

## 阶梯(全部共享、无 subject_id、零训练)
A L0_identity(=旧天花板,须复现 62.9)  B chart_pooled(训练折 LE 均值单切图,标准 pyriemann 口径)
C class4_concat(4 类均值切图拼接)     D8/D16 km8/km16 切图拼接(切空间 k-means 参考点)
E8 rand8_concat(随机试次参考点,同维对照:判「切图几何」还是「任意非线性容量」)
M MDRM_AIRM(类均值最近测地距离,纯经典)  Fd dist13⊕B(仅距离特征)  Fr rbf13⊕B(高斯核特征)
X mix13(逐切图 LDA 打分 × 测地邻近度 softmax 混合 —— 即「切图注意力」的零训练形态)

## 口径
主协议切分(rs=42,同测试折);σ/τ/K 在 val 折选,报 test;单折 <5pp 按噪声,
关键读数配 McNemar。只写 results/multichart_ceiling/。纯 CPU。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import chi2, spearmanr
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH          # noqa: E402
import analyze_erd_decomposition as ED                              # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                         # noqa: E402

OUT = BASE / 'results' / 'multichart_ceiling'
TASK = ED.TASK
D = 22
DIM = D * (D + 1) // 2
SUBS = ['S%02d' % i for i in range(1, 10)]
VERIFY70 = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'


def _sym_fun(M, f):
    """对称矩阵函数 V f(w) V^T,支持 (d,d) 与 (n,d,d)。"""
    w, V = np.linalg.eigh(M.astype(np.float64))
    fw = f(w)
    if M.ndim == 2:
        return (V * fw) @ V.T
    return np.einsum('nij,nj,nkj->nik', V, fw, V)


def logm_pd(M):
    return _sym_fun(M, lambda w: np.log(np.maximum(w, 1e-12)))


def expm_sym(S):
    return _sym_fun(S, np.exp)


def spdvec(L):  # 对称矩阵 -> vech(上三角, 非对角 ×√2, 与 ceiling 脚本一致)
    iu = np.triu_indices(D)
    v = L[iu].copy()
    v[iu[0] != iu[1]] *= np.sqrt(2)
    return v


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (D / np.trace(C, axis1=1, axis2=2)[:, None, None])


def chart_feats(Cs, R):
    """切图特征 + AIRM 距离。R=I 时退化为 LE 坐标(=ceiling 的 L0)。

    S = logm(R^{-1/2} C R^{-1/2});d_A = ‖S‖_F = sqrt(Σ log²λ_gen(C,R))。
    """
    if R is None:
        out = np.empty((len(Cs), DIM))
        dist = np.empty(len(Cs))
        for i, C in enumerate(Cs):
            w, V = np.linalg.eigh(C.astype(np.float64))
            w = np.maximum(w, 1e-12)
            out[i] = spdvec((V * np.log(w)) @ V.T)
            dist[i] = np.sqrt((np.log(w) ** 2).sum())
        return out, dist
    wR, VR = np.linalg.eigh(R.astype(np.float64))
    wR = np.maximum(wR, 1e-12)
    Ri = (VR / np.sqrt(wR)) @ VR.T
    out = np.empty((len(Cs), DIM))
    dist = np.empty(len(Cs))
    for i, C in enumerate(Cs):
        w, V = np.linalg.eigh((Ri @ C @ Ri).astype(np.float64))
        w = np.maximum(w, 1e-12)
        out[i] = spdvec((V * np.log(w)) @ V.T)
        dist[i] = np.sqrt((np.log(w) ** 2).sum())
    return out, dist


def fit_lda(F_tr, y_tr):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(F_tr, y_tr)
    return clf


def eval_split(clf, F, y, sub):
    ok = clf.predict(F) == y
    per = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
    return float(ok.mean()), per, ok


def grp(per):
    lo = np.mean([per[SUBJ[i]] for i in LOW])
    hi = np.mean([per[SUBJ[i]] for i in HIGH])
    return lo, hi


def mcnemar(ok1, ok2):
    b = int(np.sum(ok1 & ~ok2))
    c = int(np.sum(~ok1 & ok2))
    if b + c == 0:
        return 1.0, b, c
    chi = (abs(b - c) - 1) ** 2 / (b + c)
    return float(chi2.sf(chi, 1)), b, c


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

    C_tr = cov_tn(X_tr[:, :, TASK[0]:TASK[1]])
    C_va = cov_tn(X_va[:, :, TASK[0]:TASK[1]])
    C_te = cov_tn(X_te[:, :, TASK[0]:TASK[1]])
    print('train %d val %d test %d, 任务窗 %s' % (len(y_tr), len(y_va), len(y_te), TASK))

    # ---- 参考点(全部只用训练折,无 subject_id) ----
    R_pooled = expm_sym(logm_pd(C_tr).mean(0))
    R_class = [expm_sym(logm_pd(C_tr[y_tr == c]).mean(0)) for c in range(4)]
    LE_tr = np.stack([spdvec(logm_pd(C)) for C in C_tr])
    km_refs = {}
    for k in (8, 16):
        lab = KMeans(n_clusters=k, n_init=10, random_state=0).fit_predict(LE_tr)
        km_refs[k] = [expm_sym(logm_pd(C_tr[lab == j]).mean(0)) for j in range(k)]
    rng = np.random.default_rng(0)
    R_rand = [C_tr[i] for i in rng.choice(len(C_tr), 8, replace=False)]

    # ---- 各切图特征 + 距离(一次算好缓存) ----
    def batch(R):
        f_tr, d_tr = chart_feats(C_tr, R)
        f_va, d_va = chart_feats(C_va, R)
        f_te, d_te = chart_feats(C_te, R)
        return (f_tr, f_va, f_te), (d_tr, d_va, d_te)

    F_id = batch(None)
    F_pl = batch(R_pooled)
    F_cl = [batch(R) for R in R_class]
    F_km8 = [batch(R) for R in km_refs[8]]
    F_km16 = [batch(R) for R in km_refs[16]]
    F_rd = [batch(R) for R in R_rand]
    print('特征缓存完成')

    rec = {'refs': 'train 折 LE 均值 / 类均值×4 / km8 / km16 / rand8;无 subject_id',
           'rungs': {}}

    def add(name, Ftr, Fva, Fte, note=''):
        clf = fit_lda(Ftr, y_tr)
        ov_va, per_va, _ = eval_split(clf, Fva, y_va, sub_va)
        ov, per, ok = eval_split(clf, Fte, y_te, sub_te)
        lo, hi = grp(per)
        rec['rungs'][name] = {'note': note, 'val': ov_va, 'test': ov,
                              'per_subject': per,
                              'low4': lo, 'high5': hi, '_ok': None}
        rec['rungs'][name]['_ok'] = [bool(v) for v in ok]  # json 可存,供事后 McNemar
        print('%-14s val %.1f  test %.1f (低%.1f 高%.1f)  %s'
              % (name, ov_va * 100, ov * 100, lo * 100, hi * 100, note))
        return ok

    okA = add('A_L0_identity', *F_id[0], note='复现旧天花板 62.9')
    okB = add('B_chart_pooled', *F_pl[0], note='标准 pyriemann 单切图(中心化)')

    def concat(fs):
        return (np.concatenate([f[0][i] for f in fs], 1) for i in range(3))

    okC = add('C_class4_concat', *concat(F_cl), note='4 类均值切图 ×253')
    okD8 = add('D8_km8_concat', *concat(F_km8), note='km8 切图 ×253')
    okD16 = add('D16_km16_concat', *concat(F_km16), note='km16 切图 ×253')
    okE8 = add('E8_rand8_concat', *concat(F_rd), note='随机参考切图(同维对照)')

    # MDRM(AIRM 类均值最近距离)
    dcls = [np.stack([F_cl[c][1][i] for c in range(4)], 1) for i in range(3)]  # 各折 (n,4)
    mdrm = [dcls[i].argmin(1) for i in range(3)]
    ok_mdrm = {i: (mdrm[i] == [y_tr, y_va, y_te][i]) for i in range(3)}
    per = {SUBJ[j]: float(ok_mdrm[2][sub_te == j].mean()) for j in range(9)}
    lo, hi = grp(per)
    rec['rungs']['M_MDRM_AIRM'] = {'note': 'argmin_c d_A(P,R_c)',
                                   'val': float(ok_mdrm[1].mean()),
                                   'test': float(ok_mdrm[2].mean()),
                                   'per_subject': per, 'low4': lo, 'high5': hi,
                                   '_ok': [bool(v) for v in ok_mdrm[2]]}
    print('%-14s val %.1f  test %.1f (低%.1f 高%.1f)' % ('M_MDRM_AIRM',
          ok_mdrm[1].mean() * 100, ok_mdrm[2].mean() * 100, lo * 100, hi * 100))

    # 13 参考点距离族:pooled + class4 + km8
    dsets = [(F_pl[1],)] + [(F_cl[c][1],) for c in range(4)] + [(F_km8[j][1],) for j in range(8)]
    d13 = [np.stack([dd[0][i] for dd in dsets], 1) for i in range(3)]  # (n,13)
    med2 = float(np.median(d13[0] ** 2))
    add('Fd_dist13_pooled', *[np.concatenate([F_pl[0][i], d13[i]], 1) for i in range(3)],
        note='13 距离 ⊕ 单切图')

    best_rbf = None
    for smul in (0.5, 1.0, 2.0):
        s2 = smul * med2
        Fs = [np.concatenate([F_pl[0][i], np.exp(-d13[i] ** 2 / (2 * s2))], 1) for i in range(3)]
        clf = fit_lda(Fs[0], y_tr)
        ovv = float((clf.predict(Fs[1]) == y_va).mean())
        print('  Fr σ²=%.1f×median² → val %.1f' % (smul, ovv * 100))
        if best_rbf is None or ovv > best_rbf[0]:
            best_rbf = (ovv, smul, Fs)
    add('Fr_rbf13_pooled', *best_rbf[2], note='RBF(exp(−d²/2σ²)) σ²=%.1f×med², val 选' % best_rbf[1])

    # X mix13:逐切图 LDA 打分 × 测地邻近 softmax(切图注意力的零训练形态)
    charts = [F_pl] + [F_cl[c] for c in range(4)] + F_km8
    clfs = [fit_lda(ch[0][0], y_tr) for ch in charts]
    sc = [np.stack([c.decision_function(ch[0][i]) for c, ch in zip(clfs, charts)], 1)
          for i in range(3)]  # (n,13,4);与 d13 的参考点同序(pooled, class4, km8)
    best_mix = None
    for tmul in (0.5, 1.0, 2.0, 4.0, np.inf):
        if np.isinf(tmul):
            w = [np.ones_like(d13[i]) / 13 for i in range(3)]
        else:
            w = [np.exp(-d13[i] ** 2 / (tmul * med2)) for i in range(3)]
            w = [wi / wi.sum(1, keepdims=True) for wi in w]
        accs = []
        for i in (1, 2):
            pred = np.einsum('nk,nkc->nc', w[i], sc[i]).argmax(1)
            accs.append(float((pred == [y_va, y_te][i - 1]).mean()))
        print('  X τ=%.1f×median² → val %.1f' % (tmul, accs[0] * 100))
        if best_mix is None or accs[0] > best_mix[0]:
            best_mix = (accs[0], accs[1], tmul)
    tmul = best_mix[2]
    w_te = (np.ones_like(d13[2]) / 13 if np.isinf(tmul)
            else (lambda z: z / z.sum(1, keepdims=True))(np.exp(-d13[2] ** 2 / (tmul * med2))))
    pred_te = np.einsum('nk,nkc->nc', w_te, sc[2]).argmax(1)
    okX = pred_te == y_te
    per = {SUBJ[j]: float(okX[sub_te == j].mean()) for j in range(9)}
    lo, hi = grp(per)
    w_va = (np.ones_like(d13[1]) / 13 if np.isinf(tmul)
            else (lambda z: z / z.sum(1, keepdims=True))(np.exp(-d13[1] ** 2 / (tmul * med2))))
    ovv = float((np.einsum('nk,nkc->nc', w_va, sc[1]).argmax(1) == y_va).mean())
    rec['rungs']['X_mix13'] = {'note': '逐切图LDA打分×测地softmax, τ=%.1f×med², val 选' % tmul,
                               'val': ovv, 'test': float(okX.mean()),
                               'per_subject': per, 'low4': lo, 'high5': hi,
                               '_ok': [bool(v) for v in okX]}
    print('%-14s val %.1f  test %.1f (低%.1f 高%.1f)' % ('X_mix13', ovv * 100,
          okX.mean() * 100, lo * 100, hi * 100))

    # ---- ① 味道的前置检查:逐人离参考点的距离 & 与精度的相关 ----
    base = json.loads(VERIFY70.read_text(encoding='utf-8'))
    deep70 = {s: float(np.mean([base['seed%d' % sd]['best_acc']['per_subject'][s][1]
                                for sd in (42, 2024, 7)])) for s in SUBS}
    d_sub = {SUBJ[i]: float(F_pl[1][2][sub_te == i].mean()) for i in range(9)}
    rho, pv = spearmanr([d_sub[s] for s in SUBS], [deep70[s] for s in SUBS])
    rec['distance_check'] = {'per_subject_test_d_to_pooled': d_sub,
                             'deep70': deep70, 'spearman_d_vs_deep70': [rho, pv]}
    print('\n逐人 test 折到 pooled 参考点 AIRM 距离:',
          ' '.join('%s %.2f' % (s, d_sub[s]) for s in SUBS))
    print('距离 × deep70 精度 Spearman ρ=%.3f (p=%.3f)' % (rho, pv))

    # ---- McNemar(对 A、B) ----
    rec['mcnemar'] = {}
    for name, ok in (('B_chart_pooled', okB), ('C_class4_concat', okC),
                     ('D8_km8_concat', okD8), ('D16_km16_concat', okD16),
                     ('E8_rand8_concat', okE8), ('X_mix13', okX)):
        pa, b, c = mcnemar(ok, okA)
        pb, _, _ = mcnemar(ok, okB)
        rec['mcnemar'][name] = {'vs_A': [pa, b, c], 'vs_B': [pb]}
        print('McNemar %-14s vs A p=%.3f (b=%d c=%d)  vs B p=%.3f' % (name, pa, b, c, pb))

    # ---- 稳定性:随机参考点族在不同抽样(5 抽)× K(4/8/16/32) 下的读数 ----
    rec['stability'] = {}
    for draw in range(5):
        rng_d = np.random.default_rng(draw)
        idx = rng_d.choice(len(C_tr), 32, replace=False)
        feats32 = [batch(C_tr[i]) for i in idx]
        for K in (4, 8, 16, 32):
            FK = list(concat(feats32[:K]))
            clf = fit_lda(FK[0], y_tr)
            ov, per, ok = eval_split(clf, FK[2], y_te, sub_te)
            ovv = float((clf.predict(FK[1]) == y_va).mean())
            lo, hi = grp(per)
            pA, _, _ = mcnemar(ok, okA)
            rec['stability']['draw%d_K%d' % (draw, K)] = {
                'val': ovv, 'test': ov, 'low4': lo, 'high5': hi,
                'per_subject': per, 'p_vs_A': pA}
            print('rand%d draw%d val %.1f test %.1f (低%.1f 高%.1f) p_vsA=%.3f'
                  % (K, draw, ovv * 100, ov * 100, lo * 100, hi * 100, pA))
    for K in (4, 8, 16, 32):
        keys = [k for k in rec['stability'] if k.startswith('draw') and k.endswith('_K%d' % K)]
        accs = [rec['stability'][k]['test'] for k in keys]
        los = [rec['stability'][k]['low4'] for k in keys]
        his = [rec['stability'][k]['high5'] for k in keys]
        print('K=%d: test %.1f±%.2f  低 %.1f±%.2f  高 %.1f±%.2f'
              % (K, 100 * np.mean(accs), 100 * np.std(accs, ddof=1),
                 100 * np.mean(los), 100 * np.std(los, ddof=1),
                 100 * np.mean(his), 100 * np.std(his, ddof=1)))
        rec['stability']['summary_K%d' % K] = {
            'test_mean': float(np.mean(accs)), 'test_sd': float(np.std(accs, ddof=1)),
            'low4_mean': float(np.mean(los)), 'low4_sd': float(np.std(los, ddof=1)),
            'high5_mean': float(np.mean(his)), 'high5_sd': float(np.std(his, ddof=1))}

    # ---- 深度后端 180 维在多切图之上还有没有剩余增益(rcs_s42 后端,draw0) ----
    ARGS_RCS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
                    local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
                    raw_cov_stream=True)
    net = load_matt_model(str(BASE / 'results' / 'rcs_s42' /
                              'rcs42_cross_mix_best_acc.pt'), 375, 188, ARGS_RCS)
    net.eval()
    dev = next(net.parameters()).device

    def backend180(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                out.append(net._features(
                    torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy()[:, :180])
        return np.concatenate(out, 0)

    B_tr, B_va, B_te = backend180(X_tr), backend180(X_va), backend180(X_te)
    rng_d = np.random.default_rng(0)
    feats_d0 = [batch(C_tr[i]) for i in rng_d.choice(len(C_tr), 32, replace=False)]
    B = (B_tr, B_va, B_te)
    rec['backend_fusion'] = {'note': 'rcs_s42 后端180 ⊕ randK(draw0),单 seed 预检'}
    for K in (16, 32):
        FK = list(concat(feats_d0[:K]))
        for tag, cols in (('pure', None), ('with_backend', 'B')):
            Fs = [np.concatenate([B[i], FK[i]], 1) if cols else FK[i] for i in range(3)]
            clf = fit_lda(Fs[0], y_tr)
            ov, per, ok = eval_split(clf, Fs[2], y_te, sub_te)
            lo, hi = grp(per)
            pA, _, _ = mcnemar(ok, okA)
            rec['backend_fusion']['K%d_%s' % (K, tag)] = {
                'test': ov, 'low4': lo, 'high5': hi, 'p_vs_A': pA}
            print('K=%d %-12s test %.1f (低%.1f 高%.1f) p_vsA=%.3f'
                  % (K, tag, ov * 100, lo * 100, hi * 100, pA))

    (OUT / 'multichart_ceiling.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/multichart_ceiling.json' % OUT)


if __name__ == '__main__':
    main()
