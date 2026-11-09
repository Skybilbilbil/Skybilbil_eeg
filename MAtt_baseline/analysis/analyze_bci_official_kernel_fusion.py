# -*- coding: utf-8 -*-
"""BCI 官方数据口径（跨 session 主协议）：把多切图核接进论文 §5.2 的族融合。

## 已有资产（论文 64.93 的全部输入）
  results/mixopt_d12/pred_{42,2024,7}.npz     —— d_attn_out=12 的 3 支
  results/mixopt_d18new/pred_{123,456,789}.npz —— d_attn_out=18 的 3 支（稳健性 seed）
  每个 npz: p_deep(2592,4) p_cls(2592,4) yte sb p_d_va(388,4) p_c_va(388,4) yva

## 本脚本
  1. 复现官方数据加载与 EA（与 exp_mixopt_d12.load_split 逐位一致）；
  2. 复现 seed 42 的验证折索引（rng=default_rng(42), 前 15% ⇒ 388 样本）；
  3. 在同一批特征上算**多切图核**（K=1/4/8，trace 归一 + logm，参考点取训练折随机试次）；
  4. 基线 = 6 支深度族 + 经典（论文 §5.2 的融合算子，w=0.70）；
  5. 加核：val 上搜「基线 vs 核」的一维权，test 只读一次。

产物：results/bci_official_kernel_fusion/bci_official_kernel_fusion.json
"""
import sys, json
from pathlib import Path
import numpy as np
from scipy import io, signal
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
RES = ROOT / 'MAtt_baseline' / 'results'
OUT = RES / 'bci_official_kernel_fusion'
# 论文 §5.2 的 6 支 = d_attn_out ∈ {12,18} × seed ∈ {42,2024,7}
NPZ = [(RES / 'mixopt_d12' / f'pred_{s}.npz') for s in (42, 2024, 7)] + \
      [(RES / 'mixopt_d18pred' / f'pred_{s}.npz') for s in (42, 2024, 7)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (C.shape[1] / np.trace(C, axis1=1, axis2=2)[:, None, None])


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
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    sb = np.concatenate([np.full(288, i, np.int64) for i in range(9)])
    print(f'[数据] T {len(ytr)} / E {len(yte)}', flush=True)

    # seed 42 的验证折（与 exp_mixopt_d12 逐位一致）
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]

    # 6 支预测
    ds, cs, ds_va, cs_va = [], [], [], []
    for p in NPZ:
        if not p.exists():
            print(f'  [跳过] 缺 {p.name}', flush=True); continue
        d = np.load(p)
        ds.append(d['p_deep']); cs.append(d['p_cls'])
        ds_va.append(d['p_d_va']); cs_va.append(d['p_c_va'])
    print(f'[预测] 载入 {len(ds)} 支', flush=True)
    p_deep = np.mean(ds, 0); p_cls = np.mean(cs, 0)

    def acc(p):
        return sc(p.argmax(1) == yte, sb)

    a_d = acc(p_deep); a_c = acc(p_cls)
    print('[单支族] 深度族 %.2f (低四 %.1f / 高五 %.1f)' % (a_d[0]*100, a_d[1]*100, a_d[2]*100), flush=True)
    print('[单支族] 经典   %.2f (低四 %.1f / 高五 %.1f)' % (a_c[0]*100, a_c[1]*100, a_c[2]*100), flush=True)

    # 基线：论文 §5.2 的融合（w=0.70）
    p_base = 0.70 * p_deep + 0.30 * p_cls
    b = acc(p_base)
    print('[基线 0.7/0.3] %.2f (低四 %.1f / 高五 %.1f)' % (b[0]*100, b[1]*100, b[2]*100), flush=True)

    # 多切图核
    Ctr_all = cov_tn(bp(Xtr)[:, :, 124:562])
    Cte_all = cov_tn(bp(Xte)[:, :, 124:562])
    Cva, Cte = Ctr_all[iva], Cte_all
    Ctr = Ctr_all[itr]
    rng2 = np.random.default_rng(0)
    rec = {'n_test': len(yte), 'n_va': int(nva), 'n_arms': len(ds),
           'deep_family': {'pooled': a_d[0], 'low4': a_d[1], 'high5': a_d[2]},
           'classic': {'pooled': a_c[0], 'low4': a_c[1], 'high5': a_c[2]},
           'base_w070': {'pooled': b[0], 'low4': b[1], 'high5': b[2]}, 'kernels': {}}
    # ⚠ 各 seed 的验证折索引不同（rng=default_rng(seed) 各自抽），故 val 只能用 seed 42 那支
    # seed 42 的验证折（d12 与 d18pred 的 pred_42 共用同一 iva）
    d0 = np.load(NPZ[0])
    d18_42 = np.load(NPZ[3])
    yva_ref = d0['yva']
    fam_va = np.mean([d0['p_d_va'], d18_42['p_d_va']], 0)   # 同 seed 的两支构成该 seed 的族
    base_va = 0.70 * fam_va + 0.30 * d0['p_c_va']
    base_va_acc = float((base_va.argmax(1) == yva_ref).mean())
    print(f'[基线] val {base_va_acc*100:.2f}', flush=True)

    for K in (1, 4, 8):
        if K > 1:
            refs = [Ctr[i] for i in rng2.choice(len(Ctr), K, replace=False)]
            Ftr, Fva, Fte = multichart(Ctr, refs), multichart(Cva, refs), multichart(Cte, refs)
        else:
            refs = [Ctr_all[itr].mean(0)]
            Ftr, Fva, Fte = multichart(Ctr, refs), multichart(Cva, refs), multichart(Cte, refs)
        r2 = np.random.default_rng(1)
        ridx = r2.choice(len(Ftr), min(800, len(Ftr)), replace=False)
        d2 = ((Ftr[ridx][:, None, :] - Ftr[ridx][None, :, :]) ** 2).sum(-1)
        m2 = float(np.median(d2[np.triu_indices(len(ridx), 1)]))
        bv, bg, bc = -1, None, None
        for gm in (1.0, 2.0, 3.0, 4.0, 6.0):
            clf = SVC(kernel='rbf', C=100., gamma=gm / (2 * m2)).fit(Ftr, ytr[itr])
            v = float((clf.predict(Fva) == yva_ref).mean())
            if v > bv:
                bv, bg, bc = v, gm, clf
        ok = bc.predict(Fte) == yte
        k = sc(ok, sb)
        # 融合：基线 vs 核，val 上搜 w
        def softmax(z):
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)
        pv_k = softmax(bc.decision_function(Fva))   # 与概率同尺度
        pt_k = softmax(bc.decision_function(Fte))
        bw, bvv = 1.0, -1.0
        for w in np.arange(0, 1.0001, 0.02):
            pv = w * base_va + (1 - w) * pv_k
            v = float((pv.argmax(1) == yva_ref).mean())
            if v > bvv:
                bvv, bw = v, float(w)
        pt = bw * p_base + (1 - bw) * pt_k
        f = sc(pt.argmax(1) == yte, sb)
        rec['kernels'][f'K{K}'] = {'gm': bg, 'dim': int(Ftr.shape[1]),
                                   'val': bv, 'test': {'pooled': k[0]},
                                   'fuse': {'w_base': bw, 'val': bvv,
                                            'pooled': f[0], 'low4': f[1], 'high5': f[2]}}
        print(f'[核 K={K}] dim {Ftr.shape[1]:5d} gm={bg} | val {bv*100:5.2f} | '
              f'test {k[0]*100:5.2f} | 融合 w={bw:.2f} -> {f[0]*100:5.2f} '
              f'(低四 {f[1]*100:.1f} / 高五 {f[2]*100:.1f})', flush=True)

    (OUT / 'bci_official_kernel_fusion.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/bci_official_kernel_fusion.json', flush=True)


if __name__ == '__main__':
    main()
