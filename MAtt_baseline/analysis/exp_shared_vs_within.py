# -*- coding: utf-8 -*-
"""共享 vs 被试内:差距在「前端」还是「头」? 2×2 交叉分解(跨 session 主协议)。

设计(每格都用闭式 LW-LDA 当读出,零额外训练):
  A 共享前端 + 共享头   (= 文献里的共享基线)
  B 共享前端 + 逐人头
  C 逐人前端 + 共享头
  D 逐人前端 + 逐人头   (= 被试内上界)
若 B≈D 且 C 低  => 差距在「头」:共享头不会为每人调决策边界
若 C≈D 且 B 低  => 差距在「前端」:共享前端学不出每人的表示
两者都低         => 前端与头各占一半

共享前端用已训的 net_s42.pt;逐人前端每人训一个(288 试次,约 10 秒/人)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from mAtt.optimizer import MixOptimizer                            # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'shared_vs_within'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN, STEP = 192, 96
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)


def load_split():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        H = apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X))
        per.append((H[:288], ys[0], H[288:], ys[1]))
    return per


def grp(per):
    return (float(np.mean([per[SUBJ[i]] for i in LOW])),
            float(np.mean([per[SUBJ[i]] for i in HIGH])))


def sc(ok, sb):
    per = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    return float(ok.mean()), lo, hi, per


def feats(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            out.append(net._features(torch.tensor(X[i:i + 128], dtype=torch.float32).to(DEV).unsqueeze(1)).cpu().numpy())
    return np.concatenate(out, 0)


def train_one(Xtr, ytr, seed=42, epochs=100):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Xtr))
    nva = max(1, int(0.15 * len(Xtr)))
    iva, itr = idx[:nva], idx[nva:]
    torch.manual_seed(seed)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    base = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=5e-4, weight_decay=0.02)
    opt = MixOptimizer(base)
    Xtr = np.asarray(Xtr, np.float32)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 31, 32):
            j = pm[i:i + 32]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 9 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    return net


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    # 共享前端
    netS = build_matt_model(WIN, STEP, ARGS).to(DEV)
    netS.load_state_dict(torch.load(ROOT / 'MAtt_baseline' / 'results' / 'crosssession_hybrid' / 'net_s42.pt',
                                    map_location=DEV))
    netS.eval()
    Ftr_S, Fte_S = feats(netS, Xtr), feats(netS, Xte)
    print('共享前端特征', Ftr_S.shape, flush=True)

    # 逐人前端
    t0 = time.time()
    Ftr_P, Fte_P = np.zeros_like(Ftr_S), np.zeros_like(Fte_S)
    for i in range(9):
        net = train_one(np.asarray(per[i][0], np.float32), per[i][1])
        m_tr = sb == i
        m_te = sb == i
        Ftr_P[m_tr] = feats(net, per[i][0])
        Fte_P[m_te] = feats(net, per[i][2])
        print('  per-subject %s 训练完成 (%.0fs)' % (SUBJ[i], time.time() - t0), flush=True)
    print('逐人前端特征完成 %.0fs' % (time.time() - t0), flush=True)

    def shared_head(A, ya, B, yb):
        mu, sg = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
        clf = LDA(solver='lsqr', shrinkage='auto').fit((A - mu) / sg, ya)
        return clf.predict((B - mu) / sg) == yb

    def per_head(A, ya, B, yb, sb):
        pred = np.zeros(len(yb), dtype=np.int64)
        for i in range(9):
            m = sb == i
            mu, sg = A[m].mean(0, keepdims=True), A[m].std(0, keepdims=True) + 1e-6
            clf = LDA(solver='lsqr', shrinkage='auto').fit((A[m] - mu) / sg, ya[m])
            pred[m] = clf.predict((B[m] - mu) / sg)
        return pred == yb

    rec = {}
    for tag, A, B, head in (
            ('A 共享前端+共享头', Ftr_S, Fte_S, 'shared'),
            ('B 共享前端+逐人头', Ftr_S, Fte_S, 'per'),
            ('C 逐人前端+共享头', Ftr_P, Fte_P, 'shared'),
            ('D 逐人前端+逐人头', Ftr_P, Fte_P, 'per')):
        ok = shared_head(A, ytr, B, yte) if head == 'shared' else per_head(A, ytr, B, yte, sb)
        ov, lo, hi, _ = sc(ok, sb)
        rec[tag] = {'pooled': ov, 'low4': lo, 'high5': hi}
        print('  %-18s %6.2f%%  (低四 %.1f / 高五 %.1f)' % (tag, 100 * ov, 100 * lo, 100 * hi), flush=True)

    (OUT / 'shared_vs_within.json').write_text(
        json.dumps({'note': '跨 session;闭式 LW-LDA;特征=后端 684 维', **rec}, indent=2, ensure_ascii=False),
        encoding='utf-8')
    print('[OK] %s/shared_vs_within.json' % OUT)


if __name__ == '__main__':
    main()
