# -*- coding: utf-8 -*-
"""BCI 跨 session 主协议：多切图核 + 深度融合（零训练，追更好的读数）。

## 背景
2026-11-08 已有读数（pkl 250Hz、跨 session、session1 全测试）：
  单切图 LE 核 60.49 / 深度 sep62 64.20 / 两路融合 66.36（w_deep=0.56）。
瓶颈在核：单切图线性读出在同 session 只有 62.9，而多切图可到 68.6（论文表 12）。
跨 session 下多切图从未测过。本脚本补这一格并重做融合。

## 做法
  · 多切图 = 在 K 个随机训练折参考点 R_k 处展开切向量：
      log_{R_k}(C) = logm(R_k^{-1/2} C R_k^{-1/2}) 的上三角（253 维），拼接 K 份
  · 参考点取训练折随机试次（论文口径），K ∈ {1, 4, 8}
  · 核 = RBF-SVM（C=100，γ 在 val 上选）
  · 深度 = sep62_full_s42 ckpt（62.5Hz）
  · 融合 = 深度与最优核按 val 搜一维权重，test 只读一次

产物：results/bci_crosssession_multichart/bci_crosssession_multichart.json
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from utils.dataset import load_preprocessed_dataset
from utils.functions import load_matt_model

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
CKPT = BASE / 'results' / 'sep62_full_s42' / 'sep62f42_cross_mix_best_acc.pt'
OUT = BASE / 'results' / 'bci_crosssession_multichart'
TASK = (375, 1126)
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=True)
WIN, STEP, DOWN = 94, 47, 4


def session_index(y):
    sess = np.zeros(len(y), dtype=np.int64)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        sess[idx[len(idx) // 2:]] = 1
    return sess


def split_by_session(X, y, sess, valid_ratio=0.25, random_state=42, others=None):
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


def isqrtm(R):
    w, V = np.linalg.eigh(R)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def tangent_at(C, Ris):
    """在参考点 R 处展开：logm(R^{-1/2} C R^{-1/2}) 上三角。"""
    M = Ris @ C @ Ris
    w, V = np.linalg.eigh(M)
    L = (V * np.log(np.clip(w, 1e-12, None))) @ V.T
    iu = np.triu_indices(C.shape[0])
    return L[iu]


def multichart(Cs, refs):
    """Cs: 样本协方差列表；refs: 参考点协方差列表 ⇒ 拼接切向量。"""
    out = []
    for R in refs:
        Ris = isqrtm(R)
        out.append(np.stack([tangent_at(c, Ris) for c in Cs]))
    return np.concatenate(out, axis=1)


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
    X = np.concatenate([s[0] for s in sd], 0)
    y = np.concatenate([s[1] for s in sd], 0)
    sess = np.concatenate([session_index(yy) for _, yy in sd])
    sub = np.concatenate([np.full(len(yy), i, np.int64) for i, (_, yy) in enumerate(sd)])
    ret = split_by_session(X, y, sess, others=[sub])
    X_tr, y_tr, X_va, y_va, X_te, y_te = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
    print(f'[切分] train {len(y_tr)} / val {len(y_va)} / test {len(y_te)}', flush=True)

    C = [cov_tn(S[:, :, TASK[0]:TASK[1]]) for S in (X_tr, X_va, X_te)]
    Ctr, Cva, Cte = C
    rng = np.random.default_rng(0)

    def pick_refs(k):
        idx = rng.choice(len(Ctr), k, replace=False)
        return [Ctr[i] for i in idx]

    rec = {'split': 'cross_session', 'n_test': int(len(y_te)), 'arms': {}}
    best = (-1, None, None, None)
    for K in (1, 4, 8):
        if K > 1:
            refs = pick_refs(K)                  # 一次选定，train/val/test 共用同一组参考点
            F = multichart(Ctr, refs)
            Fva = multichart(Cva, refs)
            Fte = multichart(Cte, refs)
        else:
            def le_(Cs):                          # 单切图 = trace 归一 + logm（无参考点，同已有 60.49 口径）
                out = []
                for c in Cs:
                    w, V = np.linalg.eigh(c)
                    L = (V * np.log(np.clip(w, 1e-12, None))) @ V.T
                    iu = np.triu_indices(c.shape[0])
                    out.append(L[iu])
                return np.stack(out)
            F, Fva, Fte = le_(Ctr), le_(Cva), le_(Cte)
        m2 = med2(F)
        bv, bg, bc = -1, None, None
        for gm in (1.0, 2.0, 3.0, 4.0, 6.0):
            clf = SVC(kernel='rbf', C=100., gamma=gm / (2 * m2)).fit(F, y_tr)
            v = float((clf.predict(Fva) == y_va).mean())
            if v > bv:
                bv, bg, bc = v, gm, clf
        pt = float((bc.predict(Fte) == y_te).mean())
        rec['arms'][f'K{K}'] = {'dim': int(F.shape[1]), 'gm': bg, 'val': bv, 'test': pt}
        print(f'[多切图 K={K}] dim {F.shape[1]:5d} | gm={bg} | val {bv*100:5.2f} | test {pt*100:5.2f}',
              flush=True)
        if bv > best[0]:
            best = (bv, pt, K, (bg, bc, Fva, Fte))
    v_k, t_k, K_best, (gm_b, kclf, Fva_b, Fte_b) = best
    print(f'[选定] K={K_best}, gm={gm_b} | val {v_k*100:.2f} | test {t_k*100:.2f}', flush=True)

    # 深度：3 支 seed 等权成族（论文 §5.2 的口径：多支等权降方差）
    CKPTS = [BASE / 'results' / d / f for d, f in (
        ('sep62_full_s42', 'sep62f42_cross_mix_best_acc.pt'),
        ('sep62_full_s2024', 'sep62f2024_cross_mix_best_acc.pt'),
        ('sep62_full_s7', 'sep62f7_cross_mix_best_acc.pt'),
    )]
    from scipy.signal import resample_poly

    Sr_va = resample_poly(np.asarray(X_va, np.float32), 1, DOWN, axis=-1).astype(np.float32)
    Sr_te = resample_poly(np.asarray(X_te, np.float32), 1, DOWN, axis=-1).astype(np.float32)

    def infer(net, S):
        out = []
        with torch.no_grad():
            for i in range(0, len(S), 256):
                out.append(net(torch.Tensor(S[i:i + 256]).unsqueeze(1)).numpy())
        return np.concatenate(out)

    moms = {'val': [], 'test': []}
    for ck in CKPTS:
        if not ck.exists():
            print(f'  [跳过] 缺 {ck}', flush=True)
            continue
        net = load_matt_model(str(ck), WIN, STEP, ARGS).cpu().eval()
        pv_, pt_ = softmax(infer(net, Sr_va)), softmax(infer(net, Sr_te))
        moms['val'].append(pv_); moms['test'].append(pt_)
        print(f'  [单支 {ck.parent.name}] val {float((pv_.argmax(1)==y_va).mean())*100:5.2f} | '
              f'test {float((pt_.argmax(1)==y_te).mean())*100:5.2f}', flush=True)
    pv_d = np.mean(moms['val'], axis=0)
    pt_d = np.mean(moms['test'], axis=0)
    acc_dv = float((pv_d.argmax(1) == y_va).mean())
    acc_dt = float((pt_d.argmax(1) == y_te).mean())
    print(f'[深度族 x{len(moms["val"])}] val {acc_dv*100:.2f} | test {acc_dt*100:.2f}', flush=True)

    pv_k = softmax(kclf.decision_function(Fva_b))
    pt_k = softmax(kclf.decision_function(Fte_b))
    bw, bvv = 0.5, -1.0
    for w in np.arange(0, 1.0001, 0.02):
        v = float(((w * pv_d + (1 - w) * pv_k).argmax(1) == y_va).mean())
        if v > bvv:
            bvv, bw = v, float(w)
    tf = float(((bw * pt_d + (1 - bw) * pt_k).argmax(1) == y_te).mean())
    print(f'[融合] w_deep={bw:.2f} | val {bvv*100:.2f} | test {tf*100:.2f}', flush=True)

    rec['deep'] = {'val': acc_dv, 'test': acc_dt}
    rec['fusion'] = {'K': K_best, 'w_deep': bw, 'val': bvv, 'test': tf}
    (OUT / 'bci_crosssession_multichart.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/bci_crosssession_multichart.json', flush=True)


if __name__ == '__main__':
    main()
