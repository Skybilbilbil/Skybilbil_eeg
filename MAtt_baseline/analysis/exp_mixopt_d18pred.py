# -*- coding: utf-8 -*-
"""d_attn_out=18 版（原配置），只跑 seed 42 并落盘预测，供 d=12 ⊕ d=18 ⊕ 经典 三路组合。

MixOptimizer + 经典集成:3 seed 定稿读数(跨 session 主协议)。"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402
from mAtt.optimizer import MixOptimizer                            # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'mixopt_d18pred'
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


def sc(ok, sb):
    per = {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}
    lo, hi = grp(per)
    return float(ok.mean()), lo, hi, per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    Ltr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Lte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    mu, sg = Ltr.mean(0, keepdims=True), Ltr.std(0, keepdims=True) + 1e-6
    clf = LDA(solver='lsqr', shrinkage='auto').fit((Ltr - mu) / sg, ytr)
    p_cls_te = clf.predict_proba((Lte - mu) / sg)
    ov0, lo0, hi0, _ = sc(p_cls_te.argmax(1) == yte, sb)
    print('经典单独 %.2f%% (低四 %.1f / 高五 %.1f)' % (100 * ov0, 100 * lo0, 100 * hi0), flush=True)

    rec = {'classic': {'pooled': ov0, 'low4': lo0, 'high5': hi0}, 'seeds': {}}
    accs, ens = [], []
    for seed in (2024, 7):
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xtr))
        nva = int(0.15 * len(Xtr))
        iva, itr = idx[:nva], idx[nva:]
        torch.manual_seed(seed)
        net = build_matt_model(WIN, STEP, ARGS).to(DEV)
        base = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=5e-4, weight_decay=0.02)
        opt = MixOptimizer(base)
        Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
        Xv = torch.tensor(Xtr[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
        best = (1e10, None)
        t0 = time.time()
        for ep in range(100):
            net.train()
            pm = rng.permutation(itr)
            for i in range(0, len(pm) - 127, 128):
                j = pm[i:i + 128]
                loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
                opt.zero_grad(); loss.backward(); opt.step()
            if ep % 5 == 4 or ep == 99:
                net.eval()
                with torch.no_grad():
                    vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
                if vl < best[0]:
                    best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
        net.load_state_dict(best[1]); net.eval()

        def soft(X):
            with torch.no_grad():
                lg = torch.cat([net(torch.tensor(X[i:i + 128]).to(DEV).unsqueeze(1)).cpu()
                                for i in range(0, len(X), 128)], 0)
            return Fn.softmax(lg, 1).numpy()

        p_d_te = soft(Xte)
        ov, lo, hi, _ = sc(p_d_te.argmax(1) == yte, sb)
        accs.append(ov)
        # 集成:w 在 T 内验证集上选
        p_d_va = soft(Xtr[iva])
        cc = LDA(solver='lsqr', shrinkage='auto').fit((Ltr[itr] - mu) / sg, ytr[itr])
        p_c_va = cc.predict_proba((Ltr[iva] - mu) / sg)
        bestw = None
        for w in np.arange(0.0, 0.95, 0.1):
            pv = w * p_d_va + (1 - w) * p_c_va
            a = float((pv.argmax(1) == ytr[iva]).mean())
            if bestw is None or a > bestw[1]:
                bestw = (float(w), a)
        p = bestw[0] * p_d_te + (1 - bestw[0]) * p_cls_te
        ov2, lo2, hi2, _ = sc(p.argmax(1) == yte, sb)
        ens.append(ov2)
        np.savez(OUT / ('pred_%d.npz' % seed), p_deep=p_d_te, p_cls=p_cls_te,
                 yte=yte, sb=sb, p_d_va=p_d_va, p_c_va=p_c_va, yva=ytr[iva])
        rec['seeds'][str(seed)] = {'alone': ov, 'ens': ov2, 'w': bestw[0],
                                   'low4': lo2, 'high5': hi2, 'sec': round(time.time() - t0)}
        print('seed %-5d 单独 %.2f%% | 集成 w=%.1f -> %.2f%% (低四 %.1f / 高五 %.1f)  %.0fs'
              % (seed, 100 * ov, bestw[0], 100 * ov2, 100 * lo2, 100 * hi2, time.time() - t0),
              flush=True)
    rec['alone_mean'] = float(np.mean(accs)); rec['alone_sd'] = float(np.std(accs))
    rec['ens_mean'] = float(np.mean(ens)); rec['ens_sd'] = float(np.std(ens))
    print('\n=== MixOpt 3 seed ===')
    print('单独 %.2f ± %.2f | 集成 %.2f ± %.2f | 经典 %.2f'
          % (100 * rec['alone_mean'], 100 * rec['alone_sd'],
             100 * rec['ens_mean'], 100 * rec['ens_sd'], 100 * ov0), flush=True)
    (OUT / 'mixopt_d18pred.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
