# -*- coding: utf-8 -*-
"""PhysionetMI 前 50% cohort 上的改进诊断（零训练，CPU）。

cohort = results/physionet_gap/cohort_by_trainacc.json（训练集深度 acc 排序前 52 人）。
本脚本回答两件事：
  A 逐类混淆：深度 / 共享CSP / 共享LE 各自把哪一类读错（4 类 = left/right/feet/hands）。
  B 三路融合：深度 ⊕ 共享CSP ⊕ 共享LE，在 **cohort 的 val 折**上搜权重、test 只读一次。
     BCI-2a 上这是唯一同时满足「低四抬起 + 高五不拉低」的方案（w=0.70, 64.93）；
     Physionet 全 103 上 CBS 融合只 +0.5pp（噪声内），此处检验它在精选 cohort 上是否成立。

口径：CSP/LE 用**全 103 训练折**拟合（共享预训练可用全量 103），只在 cohort 的 val/test 上评估。
产物：results/physionet_cohort/physionet_cohort.json
"""
import sys, json
from pathlib import Path
import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(HERE))
from analyze_physionet_gap import cov, fit_csp, csp_feats
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
CKPT = BASE / 'results' / 'physionet_mix80_s42' / 'pm80_cross_mix_best_acc.pt'
COHORT = BASE / 'results' / 'physionet_gap' / 'cohort_by_trainacc.json'
OUT = BASE / 'results' / 'physionet_cohort'
ARGS = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=True)
FS, WIN, STEP = 80, 120, 60
T0, T1 = 80, 361          # 任务期 1.0s 起（80Hz）
CLS = ['left', 'right', 'feet', 'hands']


def le_feats(C):
    C = C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 1e-12, None)
    L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
    iu = np.triu_indices(C.shape[1])
    return L[:, iu[0], iu[1]]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    from scipy.signal import resample_poly
    sd = [(resample_poly(np.asarray(X, np.float32), 1, 2, axis=-1).astype(np.float32), y)
          for X, y in sd]
    X = np.concatenate([s[0] for s in sd], 0); y = np.concatenate([s[1] for s in sd], 0)
    sub = np.concatenate([np.full(len(s[1]), i, np.int64) for i, s in enumerate(sd)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Xtr, ytr, str_, Xva, yva, sva, Xte, yte, ste = (
        ret[0], ret[1], ret[6], ret[2], ret[3], ret[7], ret[4], ret[5], ret[8])
    keep = sorted(int(s) - 1 for s in json.load(open(COHORT))['cohort_1based'])
    mva, mte = np.isin(sva, keep), np.isin(ste, keep)
    print(f'[cohort] n={len(keep)} | val {mva.sum()} | test {mte.sum()}', flush=True)

    # ---- 深度 ----
    net = load_matt_model(str(CKPT), WIN, STEP, ARGS).cpu().eval()

    def infer(S):
        out = []
        with torch.no_grad():
            for i in range(0, len(S), 256):
                out.append(net(torch.Tensor(S[i:i + 256]).unsqueeze(1)).numpy())
        return np.concatenate(out)

    def softmax(z):
        z = z - z.max(1, keepdims=True); e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    pv_d, pt_d = softmax(infer(Xva)), softmax(infer(Xte))

    # ---- 经典：CSP 与 LE（全 103 训练折拟合）----
    Ctr = cov(Xtr[:, :, T0:T1].astype(np.float64))
    W = fit_csp(Ctr, ytr)
    lda_c = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(
        csp_feats(Ctr, W), ytr)
    pv_c = lda_c.predict_proba(csp_feats(cov(Xva[:, :, T0:T1].astype(np.float64)), W))
    pt_c = lda_c.predict_proba(csp_feats(cov(Xte[:, :, T0:T1].astype(np.float64)), W))
    lda_l = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(
        le_feats(Ctr), ytr)
    pv_l = lda_l.predict_proba(le_feats(cov(Xva[:, :, T0:T1].astype(np.float64))))
    pt_l = lda_l.predict_proba(le_feats(cov(Xte[:, :, T0:T1].astype(np.float64))))
    for p in ('pv_c', 'pt_c', 'pv_l', 'pt_l'):
        pass
    pv_c, pt_c = pv_c[:, np.argsort(lda_c.classes_)], pt_c[:, np.argsort(lda_c.classes_)]
    pv_l, pt_l = pv_l[:, np.argsort(lda_l.classes_)], pt_l[:, np.argsort(lda_l.classes_)]

    def acc(p, y, m):
        return float((p[m].argmax(1) == y[m]).mean())

    rec = {'cohort_n': len(keep), 'test_n': int(mte.sum()), 'single': {}, 'fused': {}}
    for name, pv, pt in (('deep', pv_d, pt_d), ('csp', pv_c, pt_c), ('le', pv_l, pt_l)):
        rec['single'][name] = {'val': acc(pv, yva, mva), 'test': acc(pt, yte, mte)}
        print(f'[单支 {name:5s}] val {rec["single"][name]["val"]*100:5.2f} | '
              f'test {rec["single"][name]["test"]*100:5.2f}', flush=True)

    # ---- 两路融合：val 搜 w ----
    def fuse2(tag, pv_c2, pt_c2):
        best = (-1, 0.5)
        for w in np.arange(0, 1.0001, 0.02):
            v = acc(w * pv_d + (1 - w) * pv_c2, yva, mva)
            if v > best[0]:
                best = (v, float(w))
        _, w = best
        t = acc(w * pt_d + (1 - w) * pt_c2, yte, mte)
        rec['fused'][tag] = {'w_deep': w, 'val': best[0], 'test': t}
        print(f'[融合 {tag:10s}] w_deep={w:.2f} | val {best[0]*100:5.2f} | test {t*100:5.2f}',
              flush=True)

    fuse2('deep+csp', pv_c, pt_c)
    fuse2('deep+le', pv_l, pt_l)
    # 三路：先把两个经典读出等权平均，再与深度加权（避免小 val 上搜二维单纯形退化）
    fuse2('deep+classics', 0.5 * pv_c + 0.5 * pv_l, 0.5 * pt_c + 0.5 * pt_l)

    # ---- 混淆矩阵（cohort test）----
    cm = {}
    for name, pt in (('deep', pt_d), ('csp', pt_c), ('le', pt_l)):
        pred = pt[mte].argmax(1); yy = yte[mte]
        M = np.zeros((4, 4), int)
        for a_, b_ in zip(yy, pred):
            M[a_, b_] += 1
        cm[name] = M.tolist()
        rec_ = [f'{CLS[c]} {M[c, c]/max(M[c].sum(),1)*100:.0f}%' for c in range(4)]
        print(f'[混淆 {name:4s}] 逐类召回 ' + ' | '.join(rec_), flush=True)
    rec['confusion'] = cm
    rec['confusion_classes'] = CLS

    (OUT / 'physionet_cohort.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/physionet_cohort.json', flush=True)


if __name__ == '__main__':
    main()
