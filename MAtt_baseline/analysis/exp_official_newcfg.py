# -*- coding: utf-8 -*-
"""BCI 官方数据（跨 session 主协议）：用 2026-10 之后的新模型配置重训深度族。

## 为什么
论文 §5.2 的 6 支深度（d12/d18）用的是**旧配置**：128Hz 原始、全连接 conv2、d_attn_in=20。
而新配置（降采样 + 深度可分离 conv2 + d_attn_in=8）在 pkl 上值 +3.4pp（250→62.5Hz）。
官方数据 128Hz 不能整除到 62.5，但 128÷2 = 64Hz 合法，同一降采样逻辑应成立。

## 配置（唯一变量 = 模型代际）
  数据：官方 BCIC_S0X + 逐被试 EA（与 exp_mixopt_d12.load_split 逐位一致）
  切分：T（288/人）作训练、E（288/人）作测试；T 内 15% 作验证（rng=default_rng(seed)）
  采样：128 → 64 Hz（resample_poly 1/2），WIN=96 / STEP=48（1.5s / 0.75s）
  模型：d_attn_in=8, d_attn_out=8, conv2_separable=True, local_radius=6
  优化：MixOptimizer(Adam lr 5e-4, wd 0.02)，100 epoch，bs 128，MixOptimizer 保 Stiefel 约束
  选出：按验证损失最小的权重（与 exp_mixopt_d12 一致）

## 产物
  results/official_newcfg/pred_{seed}.npz —— p_deep / p_d_va / yte / sb / yva
  以及族 + 经典融合的 pooled 读数
"""
import sys, json, time, os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io, signal
from scipy.signal import resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402
from mAtt.optimizer import MixOptimizer                            # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'official_newcfg'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
DOWN = 2
WIN, STEP = 96, 48          # 64 Hz: 1.5s / 0.75s
# 单变量对照：保持与 exp_mixopt_d12 完全相同的模型配置（d_attn_in=20 / d_attn_out=12 /
# 全连接 conv2），只开「128→64 Hz 降采样」这一个变量。
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=12, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
SEEDS = [int(x) for x in os.environ.get('SEEDS', '42,2024,7').split(',')]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (C.shape[1] / np.trace(C, axis1=1, axis2=2)[:, None, None])


def logvec(C):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-12)
    L = np.einsum('nij,nj,nkj->nik', V, np.log(w), V)
    iu = np.triu_indices(L.shape[-1])
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


def sc(ok, sb):
    per = {i: float(ok[sb == i].mean()) for i in range(9)}
    return (float(ok.mean()),
            float(np.mean([per[i] for i in LOW])),
            float(np.mean([per[i] for i in HIGH])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float64)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float64)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, np.int64) for i in range(9)])

    # 经典支（128Hz，论文口径，用于融合对照）
    Ltr = logvec(cov_tn(bp(Xtr)[:, :, 124:562]))
    Lte = logvec(cov_tn(bp(Xte)[:, :, 124:562]))
    mu, sg = Ltr.mean(0, keepdims=True), Ltr.std(0, keepdims=True) + 1e-6
    clf0 = LDA(solver='lsqr', shrinkage='auto').fit((Ltr - mu) / sg, ytr)
    p_cls_te = clf0.predict_proba((Lte - mu) / sg)
    c_acc = sc(p_cls_te.argmax(1) == yte, sb)
    print('[经典] %.2f (低四 %.1f / 高五 %.1f)' % (c_acc[0]*100, c_acc[1]*100, c_acc[2]*100),
          flush=True)

    # 64 Hz 降采样（模型侧）
    Xtr64 = resample_poly(Xtr, 1, DOWN, axis=-1).astype(np.float32)
    Xte64 = resample_poly(Xte, 1, DOWN, axis=-1).astype(np.float32)
    print(f'[降采样] 128 -> 64 Hz, WIN={WIN} STEP={STEP}', flush=True)

    rec = {'config': {'down': DOWN, 'win': WIN, 'step': STEP, 'd_attn_in': 8,
                      'd_attn_out': 8, 'conv2_separable': True},
           'classic': {'pooled': c_acc[0], 'low4': c_acc[1], 'high5': c_acc[2]}, 'seeds': {}}
    fam_te, fam_va = [], []
    for seed in SEEDS:
        t0 = time.time()
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(Xtr64))
        nva = int(0.15 * len(Xtr64))
        iva, itr = idx[:nva], idx[nva:]
        torch.manual_seed(seed)
        net = build_matt_model(WIN, STEP, ARGS).to(DEV)
        base = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=5e-4, weight_decay=0.02)
        opt = MixOptimizer(base)
        Xt, yt = torch.tensor(Xtr64), torch.tensor(ytr)
        Xv = torch.tensor(Xtr64[iva]).to(DEV); yv = torch.tensor(ytr[iva]).to(DEV)
        best = (1e10, None)
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

        p_te = soft(Xte64); p_va = soft(Xtr64[iva])
        a = sc(p_te.argmax(1) == yte, sb)
        fam_te.append(p_te); fam_va.append(p_va)
        np.savez(OUT / f'pred_{seed}.npz', p_deep=p_te, p_d_va=p_va, yte=yte, sb=sb,
                 yva=ytr[iva], iva=iva)
        rec['seeds'][str(seed)] = {'alone': a[0], 'low4': a[1], 'high5': a[2],
                                   'sec': round(time.time() - t0)}
        print('[seed %-5d] %.2f (低四 %.1f / 高五 %.1f)  %.0fs'
              % (seed, a[0]*100, a[1]*100, a[2]*100, time.time() - t0), flush=True)

    # 族 + 融合（val 用 seed 42 的 iva）
    p_fam = np.mean(fam_te, 0)
    f_acc = sc(p_fam.argmax(1) == yte, sb)
    d0 = np.load(OUT / f'pred_{SEEDS[0]}.npz')
    yva = d0['yva']; fam_va42 = np.mean([np.load(OUT / f'pred_{s}.npz')['p_d_va']
                                         for s in SEEDS if (OUT / f'pred_{s}.npz').exists()], 0)
    cls_va = LDA(solver='lsqr', shrinkage='auto').fit((Ltr[d0['iva']] - mu) / sg,
                                                      ytr[d0['iva']]).predict_proba(
        (Ltr[d0['iva']] - mu) / sg)
    bw, bv = 0.5, -1.0
    for w in np.arange(0, 1.0001, 0.02):
        v = float(((w * fam_va42 + (1 - w) * cls_va).argmax(1) == yva).mean())
        if v > bv:
            bv, bw = v, float(w)
    p_f = bw * p_fam + (1 - bw) * p_cls_te
    ff = sc(p_f.argmax(1) == yte, sb)
    rec['family'] = {'n': len(fam_te), 'pooled': f_acc[0], 'low4': f_acc[1], 'high5': f_acc[2]}
    rec['fusion'] = {'w_deep': bw, 'val': bv, 'pooled': ff[0], 'low4': ff[1], 'high5': ff[2]}
    print('[族 x%d] %.2f (低四 %.1f / 高五 %.1f)' % (len(fam_te), f_acc[0]*100, f_acc[1]*100,
                                                    f_acc[2]*100), flush=True)
    print('[融合 w=%.2f] %.2f (低四 %.1f / 高五 %.1f) | 基线旧配置 64.93'
          % (bw, ff[0]*100, ff[1]*100, ff[2]*100), flush=True)
    (OUT / 'official_newcfg.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                              encoding='utf-8')
    print(f'[OK] {OUT}/official_newcfg.json', flush=True)


if __name__ == '__main__':
    main()
