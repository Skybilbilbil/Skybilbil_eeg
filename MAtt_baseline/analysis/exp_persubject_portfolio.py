# -*- coding: utf-8 -*-
"""逐被试组合权重（Portfolio）：把「按人适配」做成不需要测试集标签的合法形式。

动机（文献）：
  - "Subject-Level Heterogeneity in EEG Motor Imagery Decoding: A Large-Scale Benchmark
    and Portfolio-Based Reduction of the Search Space"（2026）：不存在普遍最优的解码器，
    强个体差异要求按人组合（portfolio），而不是选一个全局最优。
  - "Average Rankings Mask Per-Subject Optimality"（2026）：1066 种配置、34 万次拟合，
    平均排名掩盖逐人最优性。
  两篇与项目既有结论（逐人 33-72%、均值掩盖一切）完全一致，但给出的行动方案我们没试过。

做法（全部只用 T 的标签，E 只在前向时出现）：
  1) 每个被试的 T(288 试次) 内部分层切 85% 训练 / 15% 验证；
  2) 共享深度模型在 85% 上训练（MixOptimizer，60 epoch，seed 42）；
  3) 经典 LE+LDA（8-40Hz 带通、任务窗、trace 归一、253 维 LE 展开）在同样 85% 上训练；
  4) 在 15% 验证集上搜组合权重 w（p = w·p_deep + (1-w)·p_lda）：
       arm A 全局 w（全组 389 个验证样本搜一个）
       arm B 逐人 w_s（每人约 43 个验证样本各搜一个）
  5) 用选出的 w 在 E 上评估。

报告：深度单独 / 经典单独 / 全局 w / 逐人 w_s，以及逐人 w_s 的分布与验证集上的选择噪声。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from scipy import io, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                        # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402
from mAtt.optimizer import MixOptimizer                              # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'persubject_portfolio'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)
WIN, STEP, T_OUT = 192, 96, 562
WIN_OFF = (124, 562)        # 经典读出的任务窗（与 analyze_official_classic 一致）
WGRID = np.round(np.arange(0.0, 1.001, 0.1), 2)


# ---------- 数据 ----------

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


def bandpass(X, lo=8.0, hi=40.0, fs=128, order=4):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, X, axis=-1)


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


def grp(ps):
    return (float(np.mean([ps[SUBJ[i]] for i in LOW])),
            float(np.mean([ps[SUBJ[i]] for i in HIGH])))


# ---------- 深度模型 ----------

def train_deep(Xtr, ytr, Xva, yva, epochs=60):
    torch.manual_seed(42)
    net = build_matt_model(WIN, STEP, ARGS)
    net.train()
    with torch.no_grad():
        net(torch.zeros(2, 1, 22, T_OUT, device=DEV))
    base = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                            lr=5e-4, weight_decay=0.02)
    opt = MixOptimizer(base)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xva).to(DEV); yv = torch.tensor(yva).to(DEV)
    rng = np.random.default_rng(42)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(epochs):
        net.train()
        pm = rng.permutation(len(Xt))
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    print('  [深度] 训练完成 %.0fs  最佳验证损失 %.4f' % (time.time() - t0, best[0]), flush=True)
    return net


def predict_deep(net, X):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), 128):
            logit = net(torch.tensor(X[i:i + 128]).to(DEV).unsqueeze(1))
            out.append(torch.softmax(logit, 1).cpu().numpy())
    return np.concatenate(out, 0)


# ---------- 主流程 ----------

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_split()

    # 每被试 T 内分层 85/15
    itr_all, iva_all = [], []
    off = 0
    for s in range(9):
        yT = per[s][1]
        i_tr, i_va = train_test_split(np.arange(288), test_size=0.15,
                                      stratify=yT, random_state=42)
        itr_all.append(i_tr + off); iva_all.append(i_va + off)
        off += 288
    itr = np.concatenate(itr_all); iva = np.concatenate(iva_all)
    print('[切分] 训练 %d / 验证 %d（逐人分层）' % (len(itr), len(iva)), flush=True)

    XT = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    yT = np.concatenate([p[1] for p in per], 0)
    XE = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yE = np.concatenate([p[3] for p in per], 0)
    subE = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])

    # 深度
    net = train_deep(XT[itr], yT[itr], XT[iva], yT[iva])
    pv_deep = predict_deep(net, XT[iva])
    pe_deep = predict_deep(net, XE)

    # 经典 LE+LDA（8-40Hz）
    XTb, XEb = bandpass(XT), bandpass(XE)
    Ftr = logvec(cov_tn(XTb[itr][:, :, WIN_OFF[0]:WIN_OFF[1]]))
    Fva = logvec(cov_tn(XTb[iva][:, :, WIN_OFF[0]:WIN_OFF[1]]))
    Fte = logvec(cov_tn(XEb[:, :, WIN_OFF[0]:WIN_OFF[1]]))
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, yT[itr])
    pv_lda = clf.predict_proba(Fva)
    pe_lda = clf.predict_proba(Fte)
    print('  [经典] 验证 %.2f%%  测试 %.2f%%'
          % (100 * (pv_lda.argmax(1) == yT[iva]).mean(),
             100 * (pe_lda.argmax(1) == yE).mean()), flush=True)

    # 验证集上深度单独
    yva = yT[iva]
    sub_va = np.repeat(np.arange(9), 0)  # 占位，下面按 itr/iva 重算
    sub_all = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    sub_va = sub_all[iva]

    def blend(w, p1, p2):
        return w * p1 + (1 - w) * p2

    def acc(p, y, sub=None):
        ok = p.argmax(1) == y
        if sub is None:
            return float(ok.mean())
        ps = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
        lo, hi = grp(ps)
        return float(ok.mean()), lo, hi, ps

    # arm A：全局 w
    va_scores = np.array([acc(blend(w, pv_deep, pv_lda), yva) for w in WGRID])
    w_glob = float(WGRID[int(va_scores.argmax())])
    # arm B：逐人 w_s
    w_per, va_per = {}, {}
    for i in range(9):
        m = sub_va == i
        sc = np.array([float((blend(w, pv_deep[m], pv_lda[m]).argmax(1) == yva[m]).mean())
                       for w in WGRID])
        w_per[SUBJ[i]] = float(WGRID[int(sc.argmax())])
        va_per[SUBJ[i]] = float(sc.max())

    res = {}
    res['深度单独'] = dict(zip(('pooled', 'low4', 'high5', 'per_subject'),
                              acc(pe_deep, yE, subE)))
    res['经典单独'] = dict(zip(('pooled', 'low4', 'high5', 'per_subject'),
                              acc(pe_lda, yE, subE)))
    res['全局 w'] = dict(zip(('pooled', 'low4', 'high5', 'per_subject'),
                             acc(blend(w_glob, pe_deep, pe_lda), yE, subE)))
    res['全局 w']['w'] = w_glob
    res['全局 w']['val_acc'] = float(acc(blend(w_glob, pv_deep, pv_lda), yva))

    pe_per = np.zeros_like(pe_deep)
    for i in range(9):
        m = subE == i
        pe_per[m] = blend(w_per[SUBJ[i]], pe_deep[m], pe_lda[m])
    res['逐人 w_s'] = dict(zip(('pooled', 'low4', 'high5', 'per_subject'),
                               acc(pe_per, yE, subE)))
    res['逐人 w_s']['w_per_subject'] = w_per
    res['逐人 w_s']['val_acc_per_subject'] = va_per

    # 记录：验证集上全局 w 曲线的形状（判断选择是否稳定）
    res['_val_curve'] = {('%.1f' % w): float(s) for w, s in zip(WGRID, va_scores)}

    print('', flush=True)
    for k in ('深度单独', '经典单独', '全局 w', '逐人 w_s'):
        v = res[k]
        print('%-8s %6.2f%%  (低四 %5.1f / 高五 %5.1f)' %
              (k, 100 * v['pooled'], 100 * v['low4'], 100 * v['high5']), flush=True)
    print('\n全局 w = %.1f；逐人 w_s = %s'
          % (w_glob, {k: v for k, v in w_per.items()}), flush=True)
    print('逐人 w 取值分布: %s' % dict(zip(*np.unique(list(w_per.values()), return_counts=True))), flush=True)

    (OUT / 'persubject_portfolio.json').write_text(json.dumps(
        {'note': '跨 session 主协议;官方数据+逐被试EA;T 内逐人分层 85/15;'
                 '深度 MixOptimizer 60 epoch seed42;经典 8-40Hz LE+LDA;'
                 '组合权重只在 T 的验证集上选，E 仅前向',
         'split': {'n_train': int(len(itr)), 'n_val': int(len(iva))},
         'w_grid': WGRID.tolist(),
         **res}, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK]', flush=True)


if __name__ == '__main__':
    main()
