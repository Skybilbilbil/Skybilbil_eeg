# -*- coding: utf-8 -*-
"""BCI pkl 口径（跨 session 主协议）：3 支深度族 + 单切图核的交叉组合。

已有读数（同一 pkl、同一跨 session 切分）：
  1 支 + 单切图核 → 66.36（w=0.56）
  3 支族        → 66.01
  3 支族 + 多切图核 K=4 → 66.13（w=0.52）
本脚本补齐 3×2 组合里缺的格子：1支+K4、3支+单切图、3支+K8，并在 val 上选最优配置。

产物：results/bci_crosssession_grid/bci_crosssession_grid.json
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
CK = [BASE / 'results' / d / f for d, f in (
    ('sep62_full_s42', 'sep62f42_cross_mix_best_acc.pt'),
    ('sep62_full_s2024', 'sep62f2024_cross_mix_best_acc.pt'),
    ('sep62_full_s7', 'sep62f7_cross_mix_best_acc.pt'))]
OUT = BASE / 'results' / 'bci_crosssession_grid'
TASK = (375, 1126)
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=True)
WIN, STEP, DOWN = 94, 47, 4


def session_index(y):
    s = np.zeros(len(y), np.int64)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        s[idx[len(idx) // 2:]] = 1
    return s


def split_by_session(X, y, sess, vr=0.25, rs=42, others=None):
    m0, m1 = np.where(sess == 0)[0], np.where(sess == 1)[0]
    parts = [X] + (list(others) if others else [])
    o = train_test_split(*[p[m0] for p in parts], y[m0], test_size=vr,
                         stratify=y[m0], random_state=rs)
    ret = [o[0], o[-2], o[1], o[-1], X[m1], y[m1]]
    for k, ot in enumerate(o[2:-2:2]):
        ret += [ot, o[3 + 2 * k], (others[k][m1] if others else None)]
    return ret


def cov_tn(S):
    s = S.astype(np.float64); s = s - s.mean(axis=2, keepdims=True)
    C = np.einsum('nct,ndt->ncd', s, s) / s.shape[-1]
    return C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)


def isqrtm(R):
    w, V = np.linalg.eigh(R)
    return (V / np.sqrt(np.maximum(w, 1e-12))) @ V.T


def tangent_at(C, Ris):
    M = Ris @ C @ Ris
    w, V = np.linalg.eigh(M)
    L = (V * np.log(np.clip(w, 1e-12, None))) @ V.T
    iu = np.triu_indices(C.shape[0])
    return L[iu]


def multichart(Cs, refs):
    return np.concatenate([np.stack([tangent_at(c, isqrtm(R)) for c in Cs]) for R in refs], 1)


def le_(Cs):
    out = []
    for c in Cs:
        w, V = np.linalg.eigh(c)
        L = (V * np.log(np.clip(w, 1e-12, None))) @ V.T
        out.append(L[np.triu_indices(c.shape[0])])
    return np.stack(out)


def med2(F, seed=1):
    r = np.random.default_rng(seed).choice(len(F), min(800, len(F)), replace=False)
    d2 = ((F[r][:, None, :] - F[r][None, :, :]) ** 2).sum(-1)
    return float(np.median(d2[np.triu_indices(len(r), 1)]))


def softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X = np.concatenate([s[0] for s in sd], 0); y = np.concatenate([s[1] for s in sd], 0)
    sess = np.concatenate([session_index(yy) for _, yy in sd])
    sub = np.concatenate([np.full(len(yy), i, np.int64) for i, (_, yy) in enumerate(sd)])
    r = split_by_session(X, y, sess, others=[sub])
    X_tr, y_tr, X_va, y_va, X_te, y_te = r[0], r[1], r[2], r[3], r[4], r[5]
    print(f'[切分] {len(y_tr)}/{len(y_va)}/{len(y_te)}', flush=True)

    C = [cov_tn(S[:, :, TASK[0]:TASK[1]]) for S in (X_tr, X_va, X_te)]
    Ctr, Cva, Cte = C
    rng = np.random.default_rng(0)
    # 核：单切图 + K4 + K8，γ 在 val 上选
    arms = {}
    for K in (1, 4, 8):
        if K == 1:
            F1, F2, F3 = le_(Ctr), le_(Cva), le_(Cte)
        else:
            refs = [Ctr[i] for i in rng.choice(len(Ctr), K, replace=False)]
            F1, F2, F3 = multichart(Ctr, refs), multichart(Cva, refs), multichart(Cte, refs)
        m2 = med2(F1)
        bv, bg, bc = -1, None, None
        for gm in (1.0, 2.0, 3.0, 4.0, 6.0):
            clf = SVC(kernel='rbf', C=100., gamma=gm / (2 * m2)).fit(F1, y_tr)
            v = float((clf.predict(F2) == y_va).mean())
            if v > bv:
                bv, bg, bc = v, gm, clf
        arms[K] = {'clf': bc, 'val': bv, 'test': float((bc.predict(F3) == y_te).mean()),
                   'pva': softmax(bc.decision_function(F2)),
                   'pte': softmax(bc.decision_function(F3))}
        print(f'[核 K={K}] gm={bg} val {bv*100:5.2f} test {arms[K]["test"]*100:5.2f}', flush=True)

    # 深度：1 支与 3 支
    from scipy.signal import resample_poly
    Sv = resample_poly(np.asarray(X_va, np.float32), 1, DOWN, axis=-1).astype(np.float32)
    St = resample_poly(np.asarray(X_te, np.float32), 1, DOWN, axis=-1).astype(np.float32)
    pv, pt = [], []
    for ck in CK:
        if not ck.exists():
            continue
        net = load_matt_model(str(ck), WIN, STEP, ARGS).cpu().eval()

        def inf(S):
            o = []
            with torch.no_grad():
                for i in range(0, len(S), 256):
                    o.append(net(torch.Tensor(S[i:i + 256]).unsqueeze(1)).numpy())
            return np.concatenate(o)
        pv.append(softmax(inf(Sv))); pt.append(softmax(inf(St)))
        print(f'  [单支 {ck.parent.name}] val {float((pv[-1].argmax(1)==y_va).mean())*100:5.2f} '
              f'test {float((pt[-1].argmax(1)==y_te).mean())*100:5.2f}', flush=True)
    deep = {1: (pv[0], pt[0]), len(pv): (np.mean(pv, 0), np.mean(pt, 0))}

    best = (-1, None)
    rec = {}
    for nd, (dv, dt) in deep.items():
        for K, ka in arms.items():
            bw, bvv = 0.5, -1.0
            for w in np.arange(0, 1.0001, 0.02):
                v = float(((w * dv + (1 - w) * ka['pva']).argmax(1) == y_va).mean())
                if v > bvv:
                    bvv, bw = v, float(w)
            t = float(((bw * dt + (1 - bw) * ka['pte']).argmax(1) == y_te).mean())
            rec[f'deep{nd}_K{K}'] = {'w': bw, 'val': bvv, 'test': t}
            print(f'[深度 {nd} 支 + 核 K={K}] w={bw:.2f} | val {bvv*100:5.2f} | test {t*100:5.2f}',
                  flush=True)
            if bvv > best[0]:
                best = (bvv, (nd, K, t, bw))
    print(f'[val 最优] 深度 {best[1][0]} 支 + 核 K={best[1][1]} | '
          f'val {best[0]*100:.2f} | test {best[1][2]*100:.2f}', flush=True)
    rec['selected_by_val'] = {'deep_n': best[1][0], 'K': best[1][1],
                              'val': best[0], 'test': best[1][2], 'w': best[1][3]}
    (OUT / 'bci_crosssession_grid.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/bci_crosssession_grid.json', flush=True)


if __name__ == '__main__':
    main()
