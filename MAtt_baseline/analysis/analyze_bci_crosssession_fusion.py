# -*- coding: utf-8 -*-
"""BCI 跨 session 主协议：核读出 + 深度融合（零训练新读数 + 一次推理）。

## 为什么做
用户 2026-11-08：Physionet 侧发现共享核在 103 人上是真共享，反哺 BCI 需在**同一个口径**下
比较核与深度。BCI 的两条已有读数口径不兼容：核 73.5% 是 samepool（250Hz pkl），
深度 64.93%（族融合）/ 58.5%（单支）是跨 session。本脚本把两者都放到**跨 session 主协议**
下：核在跨 session 切分上重算，深度用已有 ckpt 推理，再在 val 上搜融合权重。

## 口径
- 数据：saved_data/preprocessed_data_bci2a_raw_ica.pkl（250Hz、逐被试 EA、576 试次/人）
- 切分：_session_index（类内前 72 = session0、后 72 = session1）+ split_by_session
  （session1 全部作 test；session0 内 75/25 分 train/valid；random_state=42）
  —— 与 train_pair --split cross_session 逐位一致。
- 核：LE 253 维（trace 归一 + logm，任务窗 375:1126）+ RBF-SVM（γ 在 **val** 上选，C=100）
- 深度：results/sep62_full_s42/sep62f42_cross_mix_best_acc.pt（62.5Hz：1/4 抽取、WIN=94/STEP=47、
  conv2 深度可分离）—— 与上述切分同一批试次。
- 融合：val 上搜一维权重（test 只读一次）。

产物：results/bci_crosssession_fusion/bci_crosssession_fusion.json
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from utils.dataset import load_preprocessed_dataset
from utils.functions import load_matt_model

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
CKPT = BASE / 'results' / 'sep62_full_s42' / 'sep62f42_cross_mix_best_acc.pt'
OUT = BASE / 'results' / 'bci_crosssession_fusion'
TASK = (375, 1126)
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=True)
WIN, STEP, DOWN = 94, 47, 4          # 62.5 Hz


def session_index(y):
    """单被试内的 session 标签（类内前 72 = session0，后 72 = session1）。"""
    sess = np.zeros(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        sess[idx[len(idx) // 2:]] = 1
    return sess


def split_by_session(X, y, sess, valid_ratio=0.25, random_state=42, others=None):
    from sklearn.model_selection import train_test_split
    assert len(sess) == len(y)
    m0, m1 = np.where(sess == 0)[0], np.where(sess == 1)[0]
    parts = [X] + (list(others) if others else [])
    o = train_test_split(*[p[m0] for p in parts], y[m0], test_size=valid_ratio,
                         stratify=y[m0], random_state=random_state)
    ret = [o[0], o[-2], o[1], o[-1], X[m1], y[m1]]
    for k, ot in enumerate(o[2:-2:2]):
        ret += [ot, o[3 + 2 * k], (others[k][m1] if others else None)]
    return ret


def cov_tn(S):
    s = S.astype(np.float64)
    s = s - s.mean(axis=2, keepdims=True)
    C = np.einsum('nct,ndt->ncd', s, s) / s.shape[-1]
    return C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)


def le_feats(C):
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]]


def med2(F, n=800, seed=1):
    r = np.random.default_rng(seed).choice(len(F), min(n, len(F)), replace=False)
    d2 = ((F[r][:, None, :] - F[r][None, :, :]) ** 2).sum(-1)
    return float(np.median(d2[np.triu_indices(len(r), 1)]))


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    Xs, ys, ss = [], [], []
    for i, (X, y) in enumerate(sd):
        Xs.append(X); ys.append(y); ss.append(session_index(y) + 0 * i)
    X = np.concatenate(Xs, 0); y = np.concatenate(ys, 0)
    sess = np.concatenate([session_index(yy) for _, yy in sd])
    sub = np.concatenate([np.full(len(yy), i, np.int64) for i, (_, yy) in enumerate(sd)])
    ret = split_by_session(X, y, sess, others=[sub])
    X_tr, y_tr, X_va, y_va, X_te, y_te = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
    sub_tr, sub_va, sub_te = ret[6], ret[7], ret[8]
    print(f'[切分] train {len(y_tr)} / val {len(y_va)} / test {len(y_te)}（跨 session）', flush=True)

    # ---- 核读出：LE 253 维 + RBF-SVM（γ 在 val 上选）----
    F = [le_feats(cov_tn(S[:, :, TASK[0]:TASK[1]])) for S in (X_tr, X_va, X_te)]
    Ftr, Fva, Fte = F
    m2 = med2(Ftr)
    best = (-1, None, None)
    for gm in (1.0, 2.0, 3.0, 4.0, 6.0, 8.0):
        clf = SVC(kernel='rbf', C=100., gamma=gm / (2 * m2), decision_function_shape='ovr').fit(Ftr, y_tr)
        v = float((clf.predict(Fva) == y_va).mean())
        print(f'  [核 gm={gm:4.1f}] val {v*100:5.2f}', flush=True)
        if v > best[0]:
            best = (v, gm, clf)
    v_k, gm_best, kclf = best
    pte_k = float((kclf.predict(Fte) == y_te).mean())
    print(f'[核] 选定 gm={gm_best} | val {v_k*100:.2f} | test {pte_k*100:.2f}', flush=True)

    # 拆开检验（逐人 LDA on 同特征）
    per = []
    for s in np.unique(sub_te):
        m_tr, m_te = sub_tr == s, sub_te == s
        clf_s = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr[m_tr], y_tr[m_tr])
        per.append(float((clf_s.predict(Fte[m_te]) == y_te[m_te]).mean()))
    print(f'[拆开] 逐人 LDA test {np.mean(per)*100:.2f}', flush=True)

    # ---- 深度融合（62.5 Hz ckpt）----
    net = load_matt_model(str(CKPT), WIN, STEP, ARGS).cpu().eval()
    from scipy.signal import resample_poly

    def infer(S):
        Sr = resample_poly(np.asarray(S, np.float32), 1, DOWN, axis=-1).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(Sr), 256):
                out.append(net(torch.Tensor(Sr[i:i + 256]).unsqueeze(1)).numpy())
        return np.concatenate(out)

    pv_d, pt_d = softmax(infer(X_va)), softmax(infer(X_te))
    acc_dv = float((pv_d.argmax(1) == y_va).mean())
    acc_dt = float((pt_d.argmax(1) == y_te).mean())
    print(f'[深度] val {acc_dv*100:.2f} | test {acc_dt*100:.2f}', flush=True)

    # ---- 融合（val 搜 w）----
    pv_k = softmax(kclf.decision_function(Fva))
    pt_k = softmax(kclf.decision_function(Fte))
    bestw, bestv = 0.5, -1.0
    for w in np.arange(0, 1.0001, 0.02):
        v = float(((w * pv_d + (1 - w) * pv_k).argmax(1) == y_va).mean())
        if v > bestv:
            bestv, bestw = v, float(w)
    testf = float(((bestw * pt_d + (1 - bestw) * pt_k).argmax(1) == y_te).mean())
    print(f'[融合] w_deep={bestw:.2f} | val {bestv*100:.2f} | test {testf*100:.2f}', flush=True)

    rec = {'split': 'cross_session (session1 test)',
           'n': {'train': len(y_tr), 'val': len(y_va), 'test': len(y_te)},
           'kernel': {'gm': gm_best, 'val': v_k, 'test': pte_k,
                      'per_subject_lda': float(np.mean(per))},
           'deep': {'val': acc_dv, 'test': acc_dt},
           'fusion': {'w_deep': bestw, 'val': bestv, 'test': testf}}
    (OUT / 'bci_crosssession_fusion.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/bci_crosssession_fusion.json', flush=True)


if __name__ == '__main__':
    main()
