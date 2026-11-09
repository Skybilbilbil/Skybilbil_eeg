# -*- coding: utf-8 -*-
"""手二类 × 共享读出的逐级探针:低四的手类信息死在管线哪一级。

## 依据(本日 error_decomposition + mirror_commonality)
低四的手二类:深度模型 44.1% <「log μ 功率 + 共享 LDA」54.2% <「ERD+反对称 +
共享 LDA」61.7%(同一测试折);高五相反(模型 77.3 > LDA 74.4)。⇒ 共享深度前端
对低四的手类信息有损。既往逐级追踪(pipeline_attribution)的探针是**被试内
重拟合**口径 ⇒ 共享口径下哪一级丢的从未测过。

## 设计
四级特征,同一批试次(主协议切分,手类试次):
  L0 入口   任务窗 (375,1126) 原始信号协方差 22×22(迹归一) log 向量 253 维
  L1 conv1  conv1 出口整段协方差 22×22(同上) 253 维
  L2 conv2  conv2 出口逐窗协方差 8×8(迹归一) 按窗算术平均后 log 向量 36 维
  L3 后端   net._features 180 维(模型实际喂给分类头的特征)
读出 = 共享 LDA(全被试训练折手类拟合)/ 被试内 LDA(各自训练折拟合),
在各自测试折手类上评;逐人 + 低四/高五均值。基线参照 = 模型自身手二类(多数票)。
只写 results/hand_shared_trace/。纯推理。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis        # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'hand_shared_trace'
RUNS = {42: ('verify_samepool_70_15_15', 'verify70')}   # 单 seed(诊断探针,用户指示)
if len(sys.argv) == 3:                                   # 可选:指定别的 run(如 conv1id)
    RUNS = {42: (sys.argv[1], sys.argv[2])}
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)


def cov_tn(x):
    """x (B,C,T) -> 迹归一协方差 (B,C,C)"""
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (C.shape[1] / np.trace(C, axis1=1, axis2=2)[:, None, None])


def spdvec(M):
    d = M.shape[-1]
    iu = np.triu_indices(d)
    v = M[:, iu[0], iu[1]].copy()
    v[:, iu[0] != iu[1]] *= np.sqrt(2)
    return v


def logvec(C):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-12)
    L = np.einsum('nij,nj,nkj->nik', V, np.log(w), V)
    return spdvec(L)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    # 三个二分类任务:手内 L vs R / 手 vs 非手 / 脚 vs 舌(把 4 类的 6 条边界拆开)
    TASKS = {'LvsR': (np.isin(y_tr, (0, 1)), np.isin(y_te, (0, 1)), y_tr, y_te),
             'handvsrest': (np.ones(len(y_tr), bool), np.ones(len(y_te), bool),
                            (y_tr < 2).astype(int), (y_te < 2).astype(int)),
             'feetvsrest': (np.isin(y_tr, (2, 3)), np.isin(y_te, (2, 3)),
                            (y_tr == 2).astype(int), (y_te == 2).astype(int))}
    rec = {'per_seed': {}, 'mean_3seed': {}}
    acc = {}
    print('三个二分类边界 × 四级共享读出,3 seed(模型自身手二类(多数票) 低四 44.1 / 高五 77.3)')
    for seed, (dirname, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), 375, 188, ARGS)
        net.eval()
        dev = next(net.parameters()).device
        store = {}

        def grab(name):
            def h(m, i, o):
                store[name] = o.detach()
            return h

        hs = [net.conv1.register_forward_hook(grab('conv1')),
              net.conv2.register_forward_hook(grab('conv2'))]

        def levels(X):
            """返回 {级名: 特征}。L0=任务窗输入协方差;L0w=整段输入协方差(含静息窗,
            用于隔离「窗口稀释」);L1=conv1 后整段;L2=conv2 后逐窗(模型同款 5 窗)均值;
            L2t=只取任务 4 窗(去掉含静息的窗 0);L3=后端 _features。"""
            starts = (0, 188, 376, 564, 752)
            out = {k: [] for k in ('L0', 'L0w', 'L1', 'L2', 'L2t', 'L3')}

            def wincov(seg, sts):
                cs = [cov_tn(seg[:, :, s:s + 375]) for s in sts
                      if s + 375 <= seg.shape[2]]
                return np.mean(cs, 0)
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    xb = torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)
                    out['L3'].append(net._features(xb).cpu().numpy())
                    _ = net(xb)                   # 触发 conv 钩子
                    x2 = xb.squeeze(1).cpu().numpy()
                    out['L0'].append(logvec(cov_tn(x2[:, :, 375:1126])))
                    out['L0w'].append(logvec(cov_tn(x2)))
                    c1 = store['conv1'].squeeze(2).cpu().numpy()
                    out['L1'].append(logvec(cov_tn(c1)))
                    c2 = store['conv2'].squeeze(2).cpu().numpy()   # (b,8,T')
                    out['L2'].append(logvec(wincov(c2, starts)))
                    out['L2t'].append(logvec(wincov(c2, starts[1:])))
            return {k: np.concatenate(v, 0) for k, v in out.items()}

        Ftr = levels(X_tr)
        Fte = levels(X_te)
        for h in hs:
            h.remove()
        rec['per_seed'][str(seed)] = {}
        for tname, (mtr, mte, ytr2, yte2) in TASKS.items():
            for k in ('L0', 'L0w', 'L1', 'L2', 'L2t', 'L3'):
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                clf.fit(Ftr[k][mtr], ytr2[mtr])
                pr = clf.predict(Fte[k][mte])
                ok = pr == yte2[mte]
                po = {SUBJ[i]: float(ok[sub_te[mte] == i].mean())
                      for i in range(9) if (sub_te[mte] == i).sum()}
                lo = float(np.mean([po[SUBJ[i]] for i in LOW]))
                hi = float(np.mean([po[SUBJ[i]] for i in HIGH]))
                rec['per_seed'][str(seed)].setdefault(tname, {})[k] = {
                    'low4': lo, 'high5': hi, 'pooled': po}
                acc.setdefault((tname, k), []).append((lo, hi))
        print('  seed%d 完成' % seed, flush=True)

    for tname in TASKS:
        print('\n== %s == (3-seed 均值;括号 = 逐 seed 低四)' % tname)
        print('%-4s %7s %7s | 逐人(均值)' % ('级', '共享低四', '共享高五'))
        for k in ('L0', 'L0w', 'L1', 'L2', 'L2t', 'L3'):
            los = [v[0] for v in acc[(tname, k)]]
            his = [v[1] for v in acc[(tname, k)]]
            ps = {s: float(np.mean([rec['per_seed'][str(sd)][tname][k]['pooled'][s]
                                    for sd in RUNS if s in rec['per_seed'][str(sd)][tname][k]['pooled']]))
                  for s in SUBJ}
            rec['mean_3seed'].setdefault(tname, {})[k] = {
                'low4': float(np.mean(los)), 'high5': float(np.mean(his)),
                'low4_per_seed': los, 'pooled': ps}
            print('%-4s %7.1f %7.1f | %s   (%s)'
                  % (k, np.mean(los) * 100, np.mean(his) * 100,
                     ' '.join('%s%.0f' % (s, ps[s] * 100) for s in SUBJ),
                     '/'.join('%.1f' % (v * 100) for v in los)))
    (OUT / 'hand_shared_trace.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/hand_shared_trace.json' % OUT)


if __name__ == '__main__':
    main()
