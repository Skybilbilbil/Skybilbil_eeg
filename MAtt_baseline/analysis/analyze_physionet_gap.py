# -*- coding: utf-8 -*-
"""PhysionetMI 差距定位（零训练，CPU）：①逐人 train/val/test 精度 ②类差分方向余弦矩阵。

动机（2026-11-08）：同一环节对照下，BCI-2a(9 人) 共享模型 train 64.6% / test 59.4%，
Physionet(103 人) train 48.3% / test 42.6% —— 两个数据集的 train-test 差都只有 5~6pp，
所以差距不在"泛化"，而在"共享算子能拟合多少训练集"。本脚本量化这一点，不训练任何模型。

A 逐人精度：用已有 ckpt 在 train/val/test 三段分别推理，按被试统计。
  · 若相当一部分人 train acc 贴 25%（4 类随机），说明共享读法只服务了部分坐标系。

B 类差分方向余弦：
  · 每人 4 类，类协方差均值 M_c 的 log 欧氏切向量 v_c = vec(logm(M_c))；
  · 类差分方向 d_c = v_c - mean_c v_c（去掉每人常数偏移，Σ_c d_c = 0）；
  · 跨 103 人两两余弦 cos(d_s^c, d_t^c) 的分布 = 「不同人的类差分方向能否被同一个线性读出共享」；
  · 地板：每人试次按类劈半，两个半的 d 之间余弦（= 测量噪声能到多高）；
  · 只有 cos_cross << cos_half 时，"方向异质"才是真结论而不是噪声。

C 共享 CSP+LDA 的 train/val/test（闭式，零训练）：一个 CSP 矩阵 + 一个 LDA 服务 103 人，
  与深度模型的 train 可拟合度直接对照。

用法:
  CUDA_VISIBLE_DEVICES= python analysis/analyze_physionet_gap.py --resample 2 \
      --ckpt results/physionet_mix80_s42/pm80_cross_mix_best_acc.pt --tag gap80
产物: results/physionet_gap/physionet_gap_<tag>.json
"""
import sys, json, argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.linalg import eigh as sp_eigh   # 广义特征值 Ca w = λ Cb w（numpy.linalg.eigh 无 b 参数）

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model

PKL = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
OUT_DIR = BASE / 'results' / 'physionet_gap'
ARGS = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=True)   # 2026-10-30 起深度侧默认=深度可分离 conv2
N_CLS = 4
# 任务期窗口，与 analyze_physionet_adapt.TASK=(FS,10*FS) 同口径：
# Physionet epoch = [-1, 3.5]s ⇒ 任务期从 epoch 内第 1.0 秒起，到末尾（T 截断）。
TASK_T0_SEC, TASK_T1_SEC = 1.0, 10.0


# ---------- 数学原语 ----------
def cov(seg):
    """单试次协方差: (n, ch, T) -> (n, ch, ch)。去均值后 x^T x / T。"""
    s = seg - seg.mean(axis=2, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / seg.shape[-1]


def logm_spd(C):
    """SPD 矩阵对数（特征值分解）: log(C) = V log(L) V^T。"""
    w, V = np.linalg.eigh(C)
    return (V * np.log(np.maximum(w, 1e-12))) @ V.T


def vec_sym(L):
    """对称矩阵上三角展开（含对角），22x22 -> 253 维。"""
    iu = np.triu_indices(L.shape[0])
    return L[iu]


def class_dir(X, y, n_cls=N_CLS):
    """类差分方向 (n_cls, 253): d_c = v_c - mean_c(v_c)，v_c = mean(logm(C)) over class c。"""
    V = np.stack([vec_sym(logm_spd(c)) for c in cov(X)])
    v_c = np.stack([V[y == c].mean(0) for c in range(n_cls)])
    return v_c - v_c.mean(0, keepdims=True)


def cos_rows(d, e):
    """两批方向的余弦（按行归一化）: d,e 同形 (K,) -> 标量。"""
    return float(d @ e / (np.linalg.norm(d) * np.linalg.norm(e) + 1e-30))


def fit_csp(C, y, k=2, reg=0.05):
    """一对多 CSP：对每个类解广义特征值问题 Ca w = λ (Cb + reg·tr·I) w，取两端各 k 个。"""
    Ws = []
    for c in np.unique(y):
        Ca, Cb = C[y == c].mean(0), C[y != c].mean(0)
        tr = np.trace(Ca + Cb) / C.shape[-1]
        _, V = sp_eigh(Ca, Cb + reg * tr * np.eye(C.shape[-1]))
        Ws += [V[:, -k:], V[:, :k]]
    return np.concatenate(Ws, axis=1)


def csp_feats(C, W):
    return np.log(np.maximum(np.einsum('ia,nij,ja->na', W, C, W), 1e-12))


def csp_acc(C, y, tr, te):
    """CSP+LDA 单折精度（供逐人 5 折 CV 用）。"""
    W = fit_csp(C[tr], y[tr])
    F = csp_feats(C, W)
    return float(LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                 .fit(F[tr], y[tr]).score(F[te], y[te]))


def split_half_dirs(X, y, seed=0):
    """按类随机对半，返回两半各自的类差分方向；奇数量类的最后一个试次丢弃。"""
    rng = np.random.RandomState(seed)
    ia, ib = [], []
    for c in range(N_CLS):
        idx = rng.permutation(np.where(y == c)[0])
        h = len(idx) // 2
        ia += idx[:h].tolist()
        ib += idx[h:2 * h].tolist()
    ia, ib = np.array(sorted(ia)), np.array(sorted(ib))
    return class_dir(X[ia], y[ia]), class_dir(X[ib], y[ib])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resample', type=int, default=2, help='160/resample = 目标采样率')
    ap.add_argument('--ckpt', type=str,
                    default='results/physionet_mix80_s42/pm80_cross_mix_best_acc.pt')
    ap.add_argument('--tag', type=str, default='gap80')
    ap.add_argument('--skip_infer', action='store_true', help='只跑 B/C，跳过深度推理')
    a = ap.parse_args()

    fs = 160 // a.resample
    WIN, STEP = int(1.5 * fs), int(0.75 * fs)
    out = {'tag': a.tag, 'fs': fs, 'win': WIN, 'step': STEP, 'n_classes': N_CLS}

    subject_data, _ = load_preprocessed_dataset(str(PKL))
    n_sub = len(subject_data)
    print(f'[数据] {n_sub} 被试, 每人 {subject_data[0][0].shape[0]} 试次, '
          f'原 T={subject_data[0][0].shape[-1]}')

    if a.resample > 1:
        from scipy.signal import resample_poly
        subject_data = [(resample_poly(np.asarray(X, np.float32), 1, a.resample, axis=-1)
                         .astype(np.float32), y) for X, y in subject_data]
        print(f'[重采样] 160 -> {fs} Hz（1/{a.resample} 抽取）')

    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_tr, y_tr, X_va, y_va, X_te, y_te = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
    sub_tr, sub_va, sub_te = ret[6], ret[7], ret[8]
    print(f'[切分] train {len(y_tr)} / val {len(y_va)} / test {len(y_te)}  '
          f'(覆盖 {len(np.unique(sub_all))} 人)')

    # ---------- A 逐人精度 ----------
    if not a.skip_infer:
        net = load_matt_model(str(BASE / a.ckpt), WIN, STEP, ARGS).cpu().eval()

        def infer(X):
            outs = []
            with torch.no_grad():
                for i in range(0, len(X), 256):
                    outs.append(net(torch.Tensor(X[i:i + 256]).unsqueeze(1)).numpy())
            return np.concatenate(outs)

        parts, logits_store = {}, {}
        for name, X, y, s in (('train', X_tr, y_tr, sub_tr),
                              ('valid', X_va, y_va, sub_va),
                              ('test', X_te, y_te, sub_te)):
            logits = infer(X)
            logits_store[name] = logits
            pred = logits.argmax(1)
            per = []
            for k in np.unique(s):
                m = s == k
                per.append({'subject': int(k), 'n': int(m.sum()),
                            'acc': float((pred[m] == y[m]).mean())})
            acc = float((pred == y).mean())
            arr = np.array([p['acc'] for p in per]) * 100
            parts[name] = {'overall_acc': acc, 'per_subject': per,
                           'per_subject_pct': {'min': float(arr.min()),
                                               'q1': float(np.percentile(arr, 25)),
                                               'median': float(np.median(arr)),
                                               'q3': float(np.percentile(arr, 75)),
                                               'max': float(arr.max())}}
            print(f'[A {name}] 总体 {acc*100:.2f}% | 逐人 min={arr.min():.1f} '
                  f'Q1={np.percentile(arr,25):.1f} 中位={np.median(arr):.1f} '
                  f'Q3={np.percentile(arr,75):.1f} max={arr.max():.1f}  '
                  f'| <30% 的人数 {(arr<30).sum()}/{len(arr)}  <35% {(arr<35).sum()}')
        out['depth'] = parts

        # 逐人 train vs test 的关系（同一个人，训练集能不能拟合 ⇒ 测试还能不能保住）
        tr_by = {p['subject']: p['acc'] for p in parts['train']['per_subject']}
        te_by = {p['subject']: p['acc'] for p in parts['test']['per_subject']}
        common = sorted(set(tr_by) & set(te_by))
        out['depth']['train_vs_test_across_subjects'] = {
            'n_subjects': len(common),
            'mean_train': float(np.mean([tr_by[k] for k in common])),
            'mean_test': float(np.mean([te_by[k] for k in common])),
        }
        print(f'[A 关系] 逐人均值: train {out["depth"]["train_vs_test_across_subjects"]["mean_train"]*100:.2f}% '
              f'→ test {out["depth"]["train_vs_test_across_subjects"]["mean_test"]*100:.2f}%')

    # ---------- B 类差分方向 ----------
    t0, t1 = int(TASK_T0_SEC * fs), min(int(TASK_T1_SEC * fs), X_all.shape[-1])
    dirs, half_cos = [], []
    for X, y in subject_data:
        Xt = X[:, :, t0:t1]
        dirs.append(class_dir(Xt, y))
        da, db = split_half_dirs(Xt, y)
        half_cos.append([cos_rows(da[c], db[c]) for c in range(N_CLS)])
    dirs = np.stack(dirs)          # (n_sub, n_cls, 253)
    half_cos = np.array(half_cos)  # (n_sub, n_cls)

    B = {'task_t0_sec': TASK_T0_SEC, 'task_t1_sec': TASK_T1_SEC,
         'task_samples': t1 - t0,
         'split_half_cos': {'per_class_mean': np.mean(half_cos, axis=0).tolist(),
                            'overall_mean': float(half_cos.mean()),
                            'overall_q25_q75': [float(np.percentile(half_cos, 25)),
                                                float(np.percentile(half_cos, 75))]}}
    print(f'[B 地板] 劈半余弦 逐类 ' +
          ' '.join(f'{v:+.3f}' for v in B['split_half_cos']['per_class_mean']) +
          f' | 全体均值 {half_cos.mean():+.3f}')

    cross = np.zeros((N_CLS, n_sub, n_sub))
    for c in range(N_CLS):
        D = dirs[:, c, :]
        Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-30)
        cross[c] = Dn @ Dn.T
    iu = np.triu_indices(n_sub, 1)
    B['cross_subject_cos'] = {
        'per_class_mean': [float(cross[c][iu].mean()) for c in range(N_CLS)],
        'per_class_q25_q75': [[float(np.percentile(cross[c][iu], 25)),
                               float(np.percentile(cross[c][iu], 75))] for c in range(N_CLS)],
        'overall_mean': float(cross[:, iu[0], iu[1]].mean()),
        'overall_abs_mean': float(np.abs(cross[:, iu[0], iu[1]]).mean()),
        'offdiag_q05_q95': [float(np.percentile(cross[:, iu[0], iu[1]], 5)),
                            float(np.percentile(cross[:, iu[0], iu[1]], 95))],
    }
    print(f'[B 跨人] 逐类余弦均值 ' +
          ' '.join(f'{v:+.3f}' for v in B['cross_subject_cos']['per_class_mean']) +
          f' | 全体 {B["cross_subject_cos"]["overall_mean"]:+.3f}'
          f'（|cos| 均值 {B["cross_subject_cos"]["overall_abs_mean"]:.3f}）')
    B['contrast'] = {
        'interpretation': 'cos_cross 显著低于 cos_half ⇒ 跨人方向异质是真实结构而非噪声',
        'ratio': float(abs(B['cross_subject_cos']['overall_mean']) /
                       (abs(B['split_half_cos']['overall_mean']) + 1e-30)),
    }
    out['class_direction'] = B

    # ---------- C 共享 CSP+LDA 的 train/val/test（闭式） ----------
    C_tr = cov(X_tr[:, :, t0:t1].astype(np.float64))
    W = fit_csp(C_tr, y_tr)
    F_tr, F_va, F_te = (csp_feats(cov(S[:, :, t0:t1].astype(np.float64)), W)
                        for S in (X_tr, X_va, X_te))
    lda = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F_tr, y_tr)
    csp = {}
    for name, F, y, s in (('train', F_tr, y_tr, sub_tr),
                          ('valid', F_va, y_va, sub_va),
                          ('test', F_te, y_te, sub_te)):
        pred = lda.predict(F)
        per = [{'subject': int(k), 'n': int((s == k).sum()),
                'acc': float((pred[s == k] == y[s == k]).mean())} for k in np.unique(s)]
        arr = np.array([p['acc'] for p in per]) * 100
        csp[name] = {'overall_acc': float((pred == y).mean()), 'per_subject': per,
                     'per_subject_pct': {'min': float(arr.min()),
                                         'median': float(np.median(arr)),
                                         'max': float(arr.max())}}
        print(f'[C 共享CSP+LDA {name}] 总体 {csp[name]["overall_acc"]*100:.2f}% | '
              f'逐人 min={arr.min():.1f} 中位={np.median(arr):.1f} max={arr.max():.1f}')
    out['shared_csp_lda'] = csp


    # ---------- C2 共享 LE+LDA（trace 归一协方差 → log 切空间 → LDA） ----------
    # 这是 BCI-2a 上 62.9% 那条零训练经典读出的同构物（analyze_ceiling_decomposition::cov_tn）。
    # CSP 是类监督的（16 个滤波器）；LE 是无监督的（253 维全切空间），BCI 上后者高 9.6pp。
    def le_feats(S):
        C = cov(S[:, :, t0:t1].astype(np.float64))
        C = C / (np.trace(C, axis1=1, axis2=2)[:, None, None] + 1e-30)   # trace 归一
        return np.stack([vec_sym(logm_spd(c)) for c in C])

    F_le_tr, F_le_va, F_le_te = le_feats(X_tr), le_feats(X_va), le_feats(X_te)
    lda_le = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(F_le_tr, y_tr)
    le = {}
    for name, F, y, s in (('train', F_le_tr, y_tr, sub_tr),
                          ('valid', F_le_va, y_va, sub_va),
                          ('test', F_le_te, y_te, sub_te)):
        pred = lda_le.predict(F)
        per = [{'subject': int(k), 'n': int((s == k).sum()),
                'acc': float((pred[s == k] == y[s == k]).mean())} for k in np.unique(s)]
        arr = np.array([p['acc'] for p in per]) * 100
        le[name] = {'overall_acc': float((pred == y).mean()), 'per_subject': per,
                    'per_subject_pct': {'min': float(arr.min()),
                                        'median': float(np.median(arr)),
                                        'max': float(arr.max())}}
        print(f'[C2 共享LE+LDA {name}] 总体 {le[name]["overall_acc"]*100:.2f}% | '
              f'逐人 min={arr.min():.1f} 中位={np.median(arr):.1f} max={arr.max():.1f}')
    out['shared_le_lda'] = le

    # ---------- D 族融合：深度 ⊕ 共享经典，val 搜 w、test 只读一次 ----------
    # BCI-2a 上唯一同时满足「低四抬起 + 高五不拉低」的方案就是这一族（w=0.70）。
    # 两路各自共享、只在一维权重上做选择；LE+LDA 为主臂（BCI 同构物），CSP+LDA 作对照。
    if not a.skip_infer:
        def softmax(z):
            z = z - z.max(1, keepdims=True)
            e = np.exp(z)
            return e / e.sum(1, keepdims=True)

        pv_d, pt_d = softmax(logits_store['valid']), softmax(logits_store['test'])

        def acc_of(p, y):
            return float((p.argmax(1) == y).mean())

        def fuse(tag, pv_c, pt_c):
            best_w, best_va = 0.5, -1.0
            for w in np.arange(0.0, 1.0001, 0.02):
                v = acc_of(w * pv_d + (1 - w) * pv_c, y_va)
                if v > best_va:
                    best_va, best_w = v, float(w)
            p_te = best_w * pt_d + (1 - best_w) * pt_c
            per = [{'subject': int(k), 'n': int((sub_te == k).sum()),
                    'acc': float((p_te[sub_te == k].argmax(1) == y_te[sub_te == k]).mean())}
                   for k in np.unique(sub_te)]
            arr = np.array([p['acc'] for p in per]) * 100
            print(f'[D 融合/{tag}] w_deep={best_w:.2f} | val 深{acc_of(pv_d, y_va)*100:.2f} '
                  f'经{acc_of(pv_c, y_va)*100:.2f} →融合{best_va*100:.2f} | '
                  f'test 深{acc_of(pt_d, y_te)*100:.2f} 经{acc_of(pt_c, y_te)*100:.2f} '
                  f'→融合{acc_of(p_te, y_te)*100:.2f}')
            return {'w_deep': best_w, 'w_classic': 1 - best_w,
                    'valid': {'deep_only': acc_of(pv_d, y_va),
                              'classic_only': acc_of(pv_c, y_va), 'fused': best_va},
                    'test': {'deep_only': acc_of(pt_d, y_te),
                             'classic_only': acc_of(pt_c, y_te),
                             'fused': acc_of(p_te, y_te)},
                    'test_per_subject': per,
                    'test_per_subject_pct': {'min': float(arr.min()),
                                             'median': float(np.median(arr)),
                                             'max': float(arr.max())}}

        pv_le, pt_le = lda_le.predict_proba(F_le_va), lda_le.predict_proba(F_le_te)
        o = np.argsort(lda_le.classes_)
        pv_le, pt_le = pv_le[:, o], pt_le[:, o]
        pv_csp, pt_csp = lda.predict_proba(F_va), lda.predict_proba(F_te)
        o2 = np.argsort(lda.classes_)
        pv_csp, pt_csp = pv_csp[:, o2], pt_csp[:, o2]
        out['fusion'] = {'le_lda': fuse('LE+LDA', pv_le, pt_le),
                         'csp_lda': fuse('CSP+LDA', pv_csp, pt_csp)}

    # ---------- E 身份可分性（103 类）vs 类别（4 类）:同特征、同分类器 ----------
    # BCI-2a 九人口径:subject_id 在 EA 后任务期 = 0.9996（analyze_ea_identity，历史绝对 ridge）。
    # 这里用仓库正式口径（相对 ridge RIDGE*tr(C)/n），并把两种都报出来以保可比性。
    from sklearn.model_selection import StratifiedKFold as SKF
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import silhouette_score
    from scipy.spatial.distance import cdist

    def tangent_rel(C):
        """正式口径切空间: log(C + RIDGE*tr(C)/n * I) 上三角。"""
        C = C + 1e-3 * np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1] \
            * np.eye(C.shape[1])
        w, V = np.linalg.eigh(C)
        w = np.clip(w, 1e-12, None)
        L = (V * np.log(w)[:, None, :]) @ np.transpose(V, (0, 2, 1))
        iu = np.triu_indices(C.shape[1])
        return L[:, iu[0], iu[1]]

    def cv_acc_generic(F, y, k=5):
        return float(np.mean([LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
                              .fit(F[tr], y[tr]).score(F[te], y[te])
                              for tr, te in SKF(k, shuffle=True, random_state=0).split(F, y)]))

    F_id_all = tangent_rel(cov(X_all[:, :, t0:t1].astype(np.float64)))
    id_acc = cv_acc_generic(F_id_all, sub_all)
    cls_acc = cv_acc_generic(F_id_all, y_all)
    F_id_tr = tangent_rel(cov(X_tr[:, :, t0:t1].astype(np.float64)))
    F_id_te = tangent_rel(cov(X_te[:, :, t0:t1].astype(np.float64)))
    D_te_tr = cdist(F_id_te, F_id_tr)
    order = np.argsort(D_te_tr, axis=1)
    nn_same = {f'k{k}': float((sub_tr[order[:, :k]] == sub_te[:, None]).mean())
               for k in (1, 15, 30)}
    sil_sub = float(silhouette_score(F_id_te, sub_te)) if len(np.unique(sub_te)) > 1 else float('nan')
    sil_cls = float(silhouette_score(F_id_te, y_te))
    knn9 = KNeighborsClassifier(n_neighbors=9).fit(F_id_tr, sub_tr)
    knn_id = float((knn9.predict(F_id_te) == sub_te).mean())
    out['identity'] = {
        'tangent': '相对 ridge RIDGE*tr(C)/n（仓库正式口径）',
        'task_window': [TASK_T0_SEC, TASK_T1_SEC],
        'identity_acc_lda5fold_103cls': id_acc, 'class_acc_lda5fold_same_feats': cls_acc,
        'chance_identity': 1.0 / n_sub, 'chance_class': 0.25,
        'knn9_identity_acc': knn_id,
        'nn_same_subject_frac': nn_same,
        'silhouette_test': {'subject': sil_sub, 'class': sil_cls},
    }
    print(f'[E 身份] 103 类 LDA5 折 = {id_acc:.4f}（随机 {1/n_sub:.4f}）| 4 类 = {cls_acc:.4f}'
          f'（随机 0.25）| kNN9 身份 = {knn_id:.4f} | 最近邻同人率 k1/15/30 = '
          f'{nn_same["k1"]:.3f}/{nn_same["k15"]:.3f}/{nn_same["k30"]:.3f}'
          f' | silhouette 身份 {sil_sub:+.3f} vs 类别 {sil_cls:+.3f}')

    # ---------- F 前 50% cohort（训练折 CSP 5 折排序，cohort 盲选） ----------
    # 用户 2026-11-08:「不要那几个特别差的被试，只用前 50% 左右」。
    # 纪律:排序只用训练折、测试折不参与;筛选读数不报作主协议（与 §1.7/§1.8 同）。
    if not a.skip_infer:
        scores = np.full(n_sub, np.nan)
        for k in np.unique(sub_tr):
            m = sub_tr == k
            Ck = cov(X_tr[m][:, :, t0:t1].astype(np.float64))
            yk = y_tr[m]
            cfg = [(tri, tei) for tri, tei in SKF(5, shuffle=True, random_state=0).split(Ck, yk)]
            scores[int(k)] = float(np.mean([csp_acc(Ck, yk, tri, tei) for tri, tei in cfg]))
        keep = int(np.ceil(n_sub * 0.5))
        cohort = np.argsort(-scores)[:keep]
        mask_te, mask_va = np.isin(sub_te, cohort), np.isin(sub_va, cohort)

        pt_d = softmax(logits_store['test'])
        pt_c = lda.predict_proba(F_te)
        pt_c = pt_c[:, np.argsort(lda.classes_)]
        pt_l = lda_le.predict_proba(F_le_te)
        pt_l = pt_l[:, np.argsort(lda_le.classes_)]

        def pooled(p, mask):
            return float((p[mask].argmax(1) == y_te[mask]).mean())

        out['cohort_top50'] = {
            'n_subjects': keep, 'threshold_csp_trainfold': float(scores[cohort].min()),
            'subjects': [int(s) + 1 for s in np.sort(cohort)],
            'test_n': int(mask_te.sum()),
            'test_acc': {'deep': pooled(pt_d, mask_te), 'csp_lda': pooled(pt_c, mask_te),
                         'le_lda': pooled(pt_l, mask_te)},
            'all103_test_acc': {'deep': pooled(pt_d, np.ones_like(mask_te)),
                                'csp_lda': pooled(pt_c, np.ones_like(mask_te)),
                                'le_lda': pooled(pt_l, np.ones_like(mask_te))},
        }
        print(f'[F 前50%] n={keep} 人, test {int(mask_te.sum())} 试次 | '
              f'深 {pooled(pt_d, mask_te)*100:.2f} / CSP {pooled(pt_c, mask_te)*100:.2f} / '
              f'LE {pooled(pt_l, mask_te)*100:.2f} | 全103对照: '
              f'{pooled(pt_d, np.ones_like(mask_te))*100:.2f} / '
              f'{pooled(pt_c, np.ones_like(mask_te))*100:.2f} / '
              f'{pooled(pt_l, np.ones_like(mask_te))*100:.2f}')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    p = OUT_DIR / f'physionet_gap_{a.tag}.json'
    with open(p, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'[OK] {p}')


if __name__ == '__main__':
    main()
