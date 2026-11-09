# -*- coding: utf-8 -*-
"""天花板分解:修复后模型贴上限了吗?融合全部表示后共享读出最高能到哪(零训练)。

## 三问(对应「效果一般→分析原因 or 换角度」)
① 摄入修复(nbncw=bn+通道加权恒等)之后,模型(56.4)是否已贴其表示的共享读出上限?
   逐人重拟合差距(基线上 +10.8/+8.2)在修复后的表示上还在不在?
② 把**全部已知表示**融合(后端切向 180 ⊕ 任务窗原始协方差 253 ⊕ μ/β 带功率 44),
   共享**线性**读出能到多少——表示层还有没有可挖的共享余量;
③ 共享**非线性**读出(MLP)在融合表示上能到多少——瓶颈是「线性」还是「共享」。

数据:主协议切分(同测试折);模型 seed42(baseline=verify70 / nbncw)。
只写 results/ceiling_decomposition/。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, TBL, SUBJ, LOW, HIGH, pow_band  # noqa: E402
import analyze_erd_decomposition as ED                                      # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

OUT = BASE / 'results' / 'ceiling_decomposition'
BETA = (15.0, 25.0)
TASK = ED.TASK
ARGS_IN = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
               local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
MODELS = {'baseline': ('verify_samepool_70_15_15', 'verify70', 'in', False),
          'nbncw': ('nbncw_s42', 'nbncw42', 'bn', True)}


def cov_tn(x):
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


def shared_lda(Ftr, ytr, Fte, yte, sub_te):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(Ftr, ytr)
    ok = clf.predict(Fte) == yte
    return float(ok.mean()), {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}


def persub_lda(Ftr, ytr, sub_tr, Fte, yte, sub_te):
    accs = {}
    for i, s in enumerate(SUBJ):
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Ftr[sub_tr == i], ytr[sub_tr == i])
        ok = clf.predict(Fte[sub_te == i]) == yte[sub_te == i]
        accs[s] = float(ok.mean())
    return accs


def g(p):
    lo = np.mean([p[SUBJ[i]] for i in LOW])
    hi = np.mean([p[SUBJ[i]] for i in HIGH])
    return lo, hi


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]

    # 带功率特征(μ IAF±1.5 + β 15-25,log):直接在训练/测试数组上按被试重算
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}

    def bp(Xb, subb):
        out = []
        for i in range(9):
            m = subb == i
            iaf = ref[SUBJ[i]]['iaf_hz']
            out.append(np.log(np.concatenate(
                [pow_band(Xb[m], iaf - 1.5, iaf + 1.5, TASK[0], TASK[1], 125),
                 pow_band(Xb[m], *BETA, TASK[0], TASK[1], 125)], 1)))
        o = np.zeros((len(Xb), 44))
        for i in range(9):
            o[subb == i] = out[i]
        return o
    Fbp_tr, Fbp_te = bp(X_tr, sub_tr), bp(X_te, sub_te)
    L0_tr, L0_te = logvec(cov_tn(X_tr[:, :, TASK[0]:TASK[1]])), \
        logvec(cov_tn(X_te[:, :, TASK[0]:TASK[1]]))

    rec = {}
    for name, (dirname, tag, norm, cwid) in MODELS.items():
        ck = BASE / 'results' / dirname / f'{tag}_cross_mix_best_acc.pt'
        args = dict(ARGS_IN, norm_type=norm)
        net = load_matt_model(str(ck), 375, 188, args)
        net.eval()
        dev = next(net.parameters()).device

        def backend(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    out.append(net._features(
                        torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy())
            return np.concatenate(out, 0)
        L3_tr, L3_te = backend(X_tr), backend(X_te)
        rec[name] = {}
        # ① 单表示:共享 vs 逐人
        for fname, Ftr, Fte in (('L3', L3_tr, L3_te), ('L0', L0_tr, L0_te),
                                ('BP44', Fbp_tr, Fbp_te)):
            ov, po = shared_lda(Ftr, y_tr, Fte, y_te, sub_te)
            ws = persub_lda(Ftr, y_tr, sub_tr, Fte, y_te, sub_te)
            lo, hi = g(po)
            wlo, whi = g(ws)
            rec[name][fname] = {'shared': {'overall': ov, 'per': po},
                                'within': ws,
                                'gap_low': wlo - lo, 'gap_high': whi - hi}
            print('%-8s %-5s 共享 %.1f (低%.1f 高%.1f) 逐人 (低%.1f 高%.1f)  gap %+.1f/%+.1f'
                  % (name, fname, ov * 100, lo * 100, hi * 100,
                     wlo * 100, whi * 100, (wlo - lo) * 100, (whi - hi) * 100))
        # ②③ 融合表示:共享线性 + 共享 MLP
        FU_tr = np.concatenate([L3_tr, L0_tr, Fbp_tr], 1)
        FU_te = np.concatenate([L3_te, L0_te, Fbp_te], 1)
        ov, po = shared_lda(FU_tr, y_tr, FU_te, y_te, sub_te)
        ws = persub_lda(FU_tr, y_tr, sub_tr, FU_te, y_te, sub_te)
        lo, hi = g(po)
        wlo, whi = g(ws)
        rec[name]['FUSED477_lda'] = {'shared': {'overall': ov, 'per': po}, 'within': ws}
        print('%-8s FUSED 共享LDA %.1f (低%.1f 高%.1f) 逐人 (低%.1f 高%.1f)'
              % (name, ov * 100, lo * 100, hi * 100, wlo * 100, whi * 100))
        mlp = MLPClassifier(hidden_layer_sizes=(128,), max_iter=300,
                            early_stopping=True, random_state=0)
        mlp.fit(FU_tr, y_tr)
        ok = mlp.predict(FU_te) == y_te
        po = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        lo, hi = g(po)
        rec[name]['FUSED477_mlp'] = {'overall': float(ok.mean()), 'per': po}
        print('%-8s FUSED 共享MLP %.1f (低%.1f 高%.1f)' % (name, ok.mean() * 100, lo * 100, hi * 100))
        # 模型自身
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 128):
                preds.append(net(torch.Tensor(X_te[i:i + 128]).unsqueeze(1).to(
                    dev)).argmax(1).cpu().numpy())
        pr = np.concatenate(preds)
        ok = pr == y_te
        po = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        lo, hi = g(po)
        rec[name]['model_itself'] = {'overall': float(ok.mean()), 'per': po}
        print('%-8s 模型自身 %.1f (低%.1f 高%.1f)' % (name, ok.mean() * 100, lo * 100, hi * 100))
        print()

    (OUT / 'ceiling_decomposition.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/ceiling_decomposition.json' % OUT)


if __name__ == '__main__':
    main()
