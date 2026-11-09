# -*- coding: utf-8 -*-
"""conv1 丢失机制分解:被删的方向是不是中线/对称结构(零训练,seed42)。

## 依据(2026-10-24 推进五)
低四的「手 vs 非手」共享可读性在 conv1 从 74.5 掉到 63.6(−10.9pp),高五只掉
6.1pp;镜像分解显示脚/舌信息在对称+中线子空间。conv1 谱:cond≈1023,尾部
奇异值 ~5e-4 ⇒ 软秩截断(协方差层衰减 ~σ²≈2.5e-7,等于删除)。

## 三问
① W 尾部(被删)奇异向量在电极空间里的构成:镜像对称 vs 反对称分解、中线 vs
   外侧通道质量 —— 被删的是不是正是「手-非手」的载体几何;
② 输入端几何预言:μ/β 带功率 {anti8, symmid14} × {LvsR, handvsrest} 共享探针
   —— handvsrest 应住 symmid、LvsR 应住 anti;
③ 恢复测试:L1 的 handvsrest 换 RBF-SVM(核读出)能否把 63.6 拉回 ~74 ——
   能 ⇒ 信息还在只是共享线性读出表达不出(B);不能 ⇒ 真删(A)。
只写 results/conv1_mechanism/。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import (PKL, TBL, SUBJ, LOW, HIGH, IL, IR,   # noqa: E402
                                        IMID, pow_band, project)
import analyze_erd_decomposition as ED                                      # noqa: E402
from mAtt.mAtt import BCI2A_CHANNELS                                        # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

OUT = BASE / 'results' / 'conv1_mechanism'
BETA = (15.0, 25.0)
REST, TASK = ED.REST, ED.TASK
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
MIDX = IMID


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


def mirror_perm():
    pairs = [('FC3', 'FC4'), ('FC1', 'FC2'), ('C5', 'C6'), ('C3', 'C4'),
             ('C1', 'C2'), ('CP3', 'CP4'), ('CP1', 'CP2'), ('P1', 'P2')]
    ch = {n: i for i, n in enumerate(BCI2A_CHANNELS)}
    P = np.eye(22)
    for a, b in pairs:
        P[ch[a], ch[a]] = P[ch[b], ch[b]] = 0
        P[ch[a], ch[b]] = P[ch[b], ch[a]] = 1
    return P


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {}
    net = load_matt_model(str(BASE / 'results' / 'verify_samepool_70_15_15' /
                              'verify70_cross_mix_best_acc.pt'), 375, 188, ARGS)
    net.eval()
    W = net.conv1.weight.detach().cpu().numpy().reshape(22, 22)   # (out, in)
    Pm = mirror_perm()

    # ① 谱 + 尾部奇异向量的几何
    U, s, Vt = np.linalg.svd(W)
    v_sym = (Vt + Vt @ Pm.T) / 2      # 每个输入方向 v 的镜像对称分量
    v_anti = (Vt - Vt @ Pm.T) / 2
    sym_norm = np.linalg.norm(v_sym, axis=1)
    anti_norm = np.linalg.norm(v_anti, axis=1)
    mid_mass = (np.abs(Vt[:, MIDX]).sum(1) / np.abs(Vt).sum(1))
    rec['spectrum'] = {'s': [round(float(x), 5) for x in s],
                       'cond': round(float(s[0] / s[-1]), 1)}
    print('① conv1 奇异谱: max %.4f min %.5g cond %.0f;尾部 6 个 σ/σmax = %s'
          % (s[0], s[-1], s[0] / s[-1],
             np.round((s[-6:] / s[0]).tolist(), 4)))
    print('   逐方向: σ/σmax | 对称分量范数 | 反对称范数 | 中线质量')
    for i in list(range(6)) + list(range(16, 22)):
        print('   v%02d  %8.4f | %6.3f | %6.3f | %.3f'
              % (i, s[i] / s[0], sym_norm[i], anti_norm[i], mid_mass[i]))
    tail = np.arange(16, 22)
    head = np.arange(6)
    rec['sv_geometry'] = {
        'tail6_sym_mean': round(float(sym_norm[tail].mean()), 4),
        'tail6_anti_mean': round(float(anti_norm[tail].mean()), 4),
        'tail6_mid_mass': round(float(mid_mass[tail].mean()), 4),
        'head6_sym_mean': round(float(sym_norm[head].mean()), 4),
        'head6_anti_mean': round(float(anti_norm[head].mean()), 4),
        'head6_mid_mass': round(float(mid_mass[head].mean()), 4)}
    print('   汇总: 尾6(被删) 对称 %.3f / 反对称 %.3f / 中线 %.3f ;头6(保留) %.3f / %.3f / %.3f'
          % (rec['sv_geometry']['tail6_sym_mean'], rec['sv_geometry']['tail6_anti_mean'],
             rec['sv_geometry']['tail6_mid_mass'], rec['sv_geometry']['head6_sym_mean'],
             rec['sv_geometry']['head6_anti_mean'], rec['sv_geometry']['head6_mid_mass']))

    # 数据与切分(与主线全同)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([x[0] for x in sd], 0).astype(np.float64)
    y_all = np.concatenate([x[1] for x in sd], 0)
    sub_all = np.concatenate([np.full(len(x[1]), i, dtype=np.int64) for i, x in enumerate(sd)])
    ret = split_train_valid_test(np.zeros((len(y_all), 1)), y_all, 0.70, 0.15,
                                 random_state=42,
                                 others=[sub_all, np.arange(len(y_all))])
    y_tr, y_te = ret[1], ret[5]
    sub_tr, sub_te, itr, ite = ret[6], ret[8], ret[9], ret[11]

    # ② 输入子空间几何:带功率 {anti,symmid,full} × {LvsR, handvsrest} 共享 LDA
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}
    print('\n② 输入端子空间几何(共享 LDA,低四/高五):')
    rec['subspace_probes'] = {}
    for bname, bandfn in (('mu', lambda m: None), ('beta', lambda m: BETA)):
        A = []
        for i, s in enumerate(SUBJ):
            iaf = ref[s]['iaf_hz']
            lo, hi = (iaf - 1.5, iaf + 1.5) if bandfn(None) is None else bandfn(None)
            m = sub_all == i
            A.append(pow_band(X_all[m], lo, hi, TASK[0], TASK[1], 125))
        A = np.concatenate(A, 0)
        F = np.log(A)
        TASKS = {'LvsR': (np.isin(y_tr, (0, 1)), np.isin(y_te, (0, 1)),
                          y_tr[np.isin(y_tr, (0, 1))], y_te[np.isin(y_te, (0, 1))]),
                 'handvsrest': (np.ones(len(y_tr), bool), np.ones(len(y_te), bool),
                                (y_tr < 2).astype(int), (y_te < 2).astype(int))}
        for pname in ('anti', 'symmid', 'full'):
            Fp = project(F, pname)
            for tname, (mtr, mte, y2tr, y2te) in TASKS.items():
                Ftr, Fte = Fp[itr][mtr], Fp[ite][mte]
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                clf.fit(Ftr, y2tr)
                ok = clf.predict(Fte) == y2te
                po = {SUBJ[i]: float(ok[sub_te[mte] == i].mean())
                      for i in range(9) if (sub_te[mte] == i).sum()}
                lo4 = float(np.mean([po[SUBJ[i]] for i in LOW]))
                hi5 = float(np.mean([po[SUBJ[i]] for i in HIGH]))
                rec['subspace_probes'][f'{bname}_{pname}_{tname}'] = {
                    'low4': lo4, 'high5': hi5, 'pooled': po}
                print('  %-4s %-8s %-10s 低四 %5.1f 高五 %5.1f'
                      % (bname, pname, tname, lo4 * 100, hi5 * 100))

    # ③ L0/L1 的 handvsrest:线性 vs RBF 核读出
    print('\n③ 恢复测试(handvsrest,低四/高五):')
    dev = next(net.parameters()).device
    store = {}
    net.conv1.register_forward_hook(
        lambda m, i, o: store.__setitem__('c1', o.detach()))

    def feats01(X):
        A0, A1 = [], []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)
                _ = net(xb)
                x2 = xb.squeeze(1).cpu().numpy()
                A0.append(logvec(cov_tn(x2[:, :, 375:1126])))
                A1.append(logvec(cov_tn(store['c1'].squeeze(2).cpu().numpy())))
        return np.concatenate(A0, 0), np.concatenate(A1, 0)

    F0tr, F1tr = feats01(X_all[itr])
    F0te, F1te = feats01(X_all[ite])
    y2tr, y2te = (y_tr < 2).astype(int), (y_te < 2).astype(int)
    rec['recovery'] = {}
    for lname, Ftr, Fte in (('L0', F0tr, F0te), ('L1', F1tr, F1te)):
        for rname, mk in (('LDA', 'lda'), ('RBF-SVM', 'svm')):
            if mk == 'lda':
                clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
            else:
                clf = SVC(kernel='rbf', C=10, gamma='scale')
            clf.fit(Ftr, y2tr)
            ok = clf.predict(Fte) == y2te
            po = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
            lo4 = float(np.mean([po[SUBJ[i]] for i in LOW]))
            hi5 = float(np.mean([po[SUBJ[i]] for i in HIGH]))
            rec['recovery'][f'{lname}_{rname}'] = {'low4': lo4, 'high5': hi5,
                                                   'pooled': po}
            print('  %s %-8s 低四 %5.1f 高五 %5.1f' % (lname, rname, lo4 * 100, hi5 * 100))

    (OUT / 'conv1_mechanism.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/conv1_mechanism.json' % OUT)


if __name__ == '__main__':
    main()
