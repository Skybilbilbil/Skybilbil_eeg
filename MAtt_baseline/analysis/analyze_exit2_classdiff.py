# -*- coding: utf-8 -*-
"""出口 2:把「类差分估计」做成共享对象 —— 有监督的方向一致性正则(零训练,闭式)。

## 为什么是这个形式
论文 §10.3 出口 2 = 「不再让网络直接读协方差特征,而是先估计『类差分方向』这一对象,
再在对象层面做跨人共享/对齐」。它赌的前提是「架构/表示变换能让 Δ 变得可共享」。
目前只有表 11c 的**无监督白化**预检(答:否);**有监督的方向一致性**从未检验。

## 形式化
在固定表示(LE 253)上,训练折给每被试 i、每类 c 估计 one-vs-rest 类差分
  Δ_ic = m_ic − mean_{c'} m_{ic'}   (m_ic = 该被试该类均值)
把「方向可共享」写成对**度量**的约束:LDA 的解只依赖类内度量 S_w,
我们把它换成 S_w + ρ·S_var,其中
  S_var = 跨被试方向变异 = cov_i(Δ_ic)  (对每个类 c,9 个方向相对其均值的散布)
ρ=0 就是标准共享 LDA(=62.9,锚点);ρ>0 表示「宁可牺牲一点类内紧致,也要让九人的
Δ 方向一致」。等价实现:用 M = S_w + ρ·scale·S_var 的白化把特征搬到「Δ 方向对齐」的
坐标系,再跑标准收缩 LDA。

## 判据(事先登记)
  ① ρ 增大时,Δ 方向的跨被试余弦一致性是否真的上升(操作检查);
  ② test pooled 是否在某个 ρ 上 > 62.9(标准 LDA);
  ③ 分组(低四/高五)是否出现响应。
三条必须一起看:一致性升而精度不升 ⇒ 「可共享方向不携带判别的额外信息」⇒ 出口 2 关闭。

产物 results/exit2_classdiff/
"""
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import binomtest

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))

from analyze_lda_head_62 import raw_cov_vec                                 # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402

OUT = BASE / 'results' / 'exit2_classdiff'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
TASK0 = 375
SUBJ = ['S%02d' % i for i in range(1, 10)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
NCLS = 4
RHOS = (0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


def pack(ok, sub):
    per = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
    return {'pooled': float(ok.mean()),
            'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([per[SUBJ[i]] for i in HIGH])),
            'per_subject': per}, ok


def deltas(F, y, s, nsub=9):
    """每个被试的 one-vs-rest 类差分方向 Δ_ic:(nsub, NCLS, d)。"""
    d = F.shape[1]
    D = np.zeros((nsub, NCLS, d))
    M = np.zeros((nsub, NCLS, d))
    for i in range(nsub):
        for c in range(NCLS):
            m = F[(s == i) & (y == c)].mean(0)
            M[i, c] = m
        D[i] = M[i] - M[i].mean(0, keepdims=True)
    return D, M


def direction_cosine(D):
    """跨被试方向一致性:每个类 c,9 个单位方向两两余弦的均值(全体平均)。"""
    cs = []
    for c in range(NCLS):
        U = D[:, c, :]
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12)
        G = U @ U.T
        iu = np.triu_indices(len(U), 1)
        cs.append(float(G[iu].mean()))
    return float(np.mean(cs)), [round(x, 3) for x in cs]


def within_scatter(F, y, s):
    d = F.shape[1]
    Sw = np.zeros((d, d))
    n = 0
    for i in range(9):
        for c in range(NCLS):
            X = F[(s == i) & (y == c)]
            if len(X) == 0:
                continue
            Xc = X - X.mean(0, keepdims=True)
            Sw += Xc.T @ Xc
            n += len(X)
    return Sw / max(n, 1)


def whiten(M):
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-8 * max(w.max(), 1e-12), None)
    return (V / np.sqrt(w)) @ V.T


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X250 = np.concatenate([np.asarray(s[0], dtype=np.float32) for s in subject_data], 0)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X250, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, ite = ret[9], ret[11]
    ytr, yte, sub_tr, sub_te = ret[1], ret[5], ret[6], ret[8]
    F = raw_cov_vec(X250, TASK0)
    Ftr, Fte = F[itr], F[ite]
    print('[data] LE %d 维 | train %d / test %d' % (F.shape[1], len(itr), len(ite)),
          flush=True)

    D, _ = deltas(Ftr, ytr, sub_tr)
    Sw = within_scatter(Ftr, ytr, sub_tr)
    dbar = D.mean(0)                                   # (NCLS, d)
    Svar = np.zeros_like(Sw)
    for i in range(9):
        for c in range(NCLS):
            e = D[i, c] - dbar[c]
            Svar += np.outer(e, e)
    Svar /= (9 * NCLS)
    scale = float(np.trace(Sw) / max(np.trace(Svar), 1e-12))
    cos0, cos_per0 = direction_cosine(D)
    print('[基线] 类差分方向跨被试余弦一致性 %.3f (逐类 %s)' % (cos0, cos_per0), flush=True)

    rec = {'note': '出口2 有监督方向一致性正则:LDA 的类内度量换成 Sw + rho*scale*Svar',
           'base_cosine': cos0, 'base_cosine_per_class': cos_per0, 'rho': {}}
    preds = {}
    for rho in RHOS:
        M = Sw + rho * scale * Svar
        # 加一点 shrinkage 保证可逆,再白化 ⇒ 在新坐标系里跑标准收缩 LDA
        M = 0.99 * M + 0.01 * np.trace(M) / M.shape[0] * np.eye(M.shape[0])
        W = whiten(M)
        Ztr, Zte = Ftr @ W, Fte @ W
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ztr, ytr)
        ok = clf.predict(Zte) == yte
        r, okv = pack(ok, sub_te)
        Dz, _ = deltas(Ztr, ytr, sub_tr)
        cosz, cosz_per = direction_cosine(Dz)
        r['cosine'] = cosz
        r['cosine_per_class'] = cosz_per
        rec['rho']['%.1f' % rho] = r
        preds['%.1f' % rho] = okv
        print('rho=%5.1f | pooled %5.1f | 低四 %5.1f 高五 %5.1f | 方向一致性 %.3f'
              % (rho, 100 * r['pooled'], 100 * r['low4'], 100 * r['high5'], cosz),
              flush=True)

    # 配对:每个 rho 对 rho=0
    rec['paired_vs_rho0'] = {}
    for k, okv in preds.items():
        if k == '0.0':
            continue
        n01, n10 = int(np.sum(~okv & preds['0.0'])), int(np.sum(okv & ~preds['0.0']))
        p = 1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
        rec['paired_vs_rho0'][k] = {'a_only': n10, 'b_only': n01, 'p': p}

    (OUT / 'exit2_classdiff.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('')
    print('=== 出口2 扫 rho(锚点 rho=0 = 标准共享 LDA) ===')
    for k, r in rec['rho'].items():
        pp = rec['paired_vs_rho0'].get(k, {})
        print('rho=%-5s pooled %5.1f | 低四 %5.1f 高五 %5.1f | 一致性 %.3f | vs rho0 p=%s'
              % (k, 100 * r['pooled'], 100 * r['low4'], 100 * r['high5'],
                 r['cosine'], ('%.3f' % pp['p']) if pp else '-'))
    print('[OK] %s/exit2_classdiff.json' % OUT)


if __name__ == '__main__':
    main()
