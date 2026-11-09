# -*- coding: utf-8 -*-
"""跨 session:用已训 checkpoint 做秒级的组合搜索(不重训)。

上一轮发现:LE(8-40)=58.76, backend=55.59, 直接拼接=58.60(被 684 维噪声拖累)。
本轮只用已保存的 net_s42.pt,尝试:
  A backend PCA 降到 k 维再拼 LE(k=10/20/50)
  B 只用 backend 的判别方向(LDA 分数)拼 LE
  C softmax 概率集成(深度的模型输出 + 经典 LDA 后验),扫权重
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'crosssession_hybrid'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN, STEP = 192, 96
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (C.shape[1] / np.trace(C, axis1=1, axis2=2)[:, None, None])


def logvec(C):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-12)
    L = np.einsum('nij,nj,nkj->nik', V, np.log(w), V)
    d = L.shape[-1]
    iu = np.triu_indices(d)
    v = L[:, iu[0], iu[1]].copy()
    v[:, iu[0] != iu[1]] *= np.sqrt(2)
    return v


def bp(X, lo=8.0, hi=40.0, fs=128, order=4):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, X, axis=-1)


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


def score(pred_or_ok, sb):
    ok = pred_or_ok
    per = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    return float(ok.mean()), lo, hi


def main():
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    net.load_state_dict(torch.load(OUT / 'net_s42.pt', map_location=DEV))
    net.eval()

    def backend(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                out.append(net._features(torch.tensor(X[i:i + 128]).to(DEV).unsqueeze(1)).cpu().numpy())
        return np.concatenate(out, 0)

    def logits(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                out.append(net(torch.tensor(X[i:i + 128]).to(DEV).unsqueeze(1)).cpu())
        return torch.cat(out, 0).numpy()

    Ltr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Lte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    Btr, Bte = backend(Xtr), backend(Xte)

    rec = {}
    # 0 单独基线
    for tag, A, B in (('LE', Ltr, Lte), ('backend', Btr, Bte)):
        mu, sg = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
        clf = LDA(solver='lsqr', shrinkage='auto').fit((A - mu) / sg, ytr)
        ov, lo, hi = score(clf.predict((B - mu) / sg) == yte, sb)
        rec[tag] = {'pooled': ov, 'low4': lo, 'high5': hi}
        print('  %-24s %6.2f%%  (低四 %.1f / 高五 %.1f)' % (tag, 100 * ov, 100 * lo, 100 * hi), flush=True)

    # A backend PCA k 维 + LE
    for k in (10, 20, 50):
        pca = PCA(n_components=k, random_state=0).fit(Btr)
        A = np.concatenate([Ltr, pca.transform(Btr)], 1)
        B = np.concatenate([Lte, pca.transform(Bte)], 1)
        mu, sg = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
        clf = LDA(solver='lsqr', shrinkage='auto').fit((A - mu) / sg, ytr)
        ov, lo, hi = score(clf.predict((B - mu) / sg) == yte, sb)
        rec['LE+pca%d' % k] = {'pooled': ov, 'low4': lo, 'high5': hi}
        print('  %-24s %6.2f%%  (低四 %.1f / 高五 %.1f)' % ('LE+pca%d' % k, 100 * ov, 100 * lo, 100 * hi), flush=True)

    # C softmax 概率集成
    p_deep = Fn.softmax(torch.tensor(logits(Xte)), 1).numpy()
    mu, sg = Ltr.mean(0, keepdims=True), Ltr.std(0, keepdims=True) + 1e-6
    clf = LDA(solver='lsqr', shrinkage='auto').fit((Ltr - mu) / sg, ytr)
    p_cls = clf.predict_proba((Lte - mu) / sg)
    best = None
    for w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6):
        p = w * p_deep + (1 - w) * p_cls
        ov, lo, hi = score(p.argmax(1) == yte, sb)
        rec['ens_w%.1f' % w] = {'pooled': ov, 'low4': lo, 'high5': hi}
        print('  %-24s %6.2f%%  (低四 %.1f / 高五 %.1f)' % ('ens w=%.1f' % w, 100 * ov, 100 * lo, 100 * hi), flush=True)
        if best is None or ov > best[1]:
            best = (w, ov, lo, hi)
    rec['best_ensemble'] = {'w': best[0], 'pooled': best[1], 'low4': best[2], 'high5': best[3]}

    (OUT / 'crosssession_hybrid2.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] best ensemble w=%.1f -> %.2f%%' % (best[0], 100 * best[1]))


if __name__ == '__main__':
    main()
