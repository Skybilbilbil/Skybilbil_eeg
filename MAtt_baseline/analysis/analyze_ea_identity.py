# -*- coding: utf-8 -*-
"""EA 与被试身份可分性 —— 只差一个 P 的因果消融 + 估计器几何归因。

要推翻的 caveat
--------------
`analyze_individual_diff.py` 给出 subj_id_nonEA_{rest,task}=0.9417/0.9828 与
subj_id_EA_{rest,task}=0.9971/0.9996,材料文档记为"两条通路不只差一个 EA
⇒ 2–5pp 不作因果归因"。但 `preprocess/main_bci2a.py:40` 生成 pkl 时调的就是
`load_subject_mi_trials(subj_id)`,与该脚本非 EA 臂调的是**同一个函数** ⇒ 两臂
试次集合/滤波链/通道集/epoch 边界/排列顺序全同,唯一差别是 X -> P X。
残留风险只有"pkl 是 09-01 快照、moabb_bci2a.py 改到 09-12" ⇒ 第 1 步验它。

关键前提(数学,不是测量)
----------------------
`apply_ea_to_numpy` = einsum('ij,bjt->bit', P, data),即 X_EA = P X_raw,P 存在
pkl 的 subject_metadata[i]['ea_matrix'] ⇒ X_raw = P^-1 X_EA **可精确还原单电极
时间序列**(仍受 8–30 Hz 带限约束)。材料文档 §1 "一阶时域量不可逆、单电极时间
序列拿不回来" 的说法在此被检验并推翻。

进一步:C -> P^-1 C P^-T 是**双射** ⇒ 电极空间与 EA 空间携带的身份信息
**完全等量**。因此任何身份可分性差异都**不可能**是"EA 制造/放大了身份信息",
只能是**估计器几何效应**(同一个解码器在两种几何下响应不同)。本脚本的全部
对照都是为了把这个效应归因到具体机制:

  EA          C_EA                                   历史口径基线
  invEA       P^-1 C_EA P^-T                         精确电极空间(无需重放)
  globalEA    G (P^-1 C_EA P^-T) G,G 来自**全体被试共用**的迹归一化平均协方差
              ⇒ 分离"白化本身"与"逐人白化"
  EA_trnorm   迹归一化后的 C_EA                       分离"尺度"与"几何"
  invEA_trnorm 迹归一化后的 P^-1 C_EA P^-T
  ctrlSharedDir  Q^-1 C_EA Q^-T,Q = W diag(eig(P_i)) W^T,**W 对 9 人共用**
              ⇒ 谱逐人特异、方向不携带身份。分离"谱/条件数"与"EA 的特定方向"
  (上一版的 ctrlSPD 给每人抽独立随机 W,等于亲手注入被试标签,恒为 1.0,已废)

解码器不变性检验
----------------
主口径 = tangent(矩阵 log,无迹归一化,上三角 253 维)+ LDA(lsqr, shrinkage=auto)
+ StratifiedKFold(5, shuffle, random_state=0),与历史完全一致。
另加 **MDM**(最小距离到类均值,log-Euclidean 度量,切向量按 √2 加权使欧氏距离
= Frobenius 范数)。MDM 是黎曼几何的自然分类器、对特征缩放不敏感:
若 MDM 在 EA 与 invEA 两臂都给 ~1.0 ⇒ 0.94 纯属 tangent+LDA 的几何伪影;
若 MDM 也复现出落差 ⇒ 落差有几何以外的成分。

产物: results/ea_identity/{ea_identity.json, ea_identity.log}
运行: cd MAtt_baseline && python analysis/analyze_ea_identity.py 2>&1 | tee results/ea_identity/ea_identity.log
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))

from analyze_individual_diff import KEYS, PKL, REST, TASK, cov, cv_acc, tangent  # noqa: E402

OUT = BASE / 'results' / 'ea_identity'
N_SUB = 9
CHANCE = 1.0 / N_SUB
R_PASS = 1 - 1e-6        # 逐通道 Pearson r 判据:pkl 为 float32 且 cond(P)~1e4,
                         # 舍入噪声经 P^-1 放大 1e2~1e4 倍,故不能用相对误差做判据
SEED = 42
ARMS = ('EA', 'invEA', 'globalEA', 'EA_trnorm', 'invEA_trnorm', 'ctrlSharedDir')


def load_ea():
    """pkl -> 每被试 (X_ea, P)。X 按 KEYS 顺序拼接,与 analyze_individual_diff 一致。"""
    import pickle
    with open(PKL, 'rb') as fh:
        d = pickle.load(fh)
    Xs, Ps = [], []
    for s in range(len(d['subject_metadata'])):
        Xs.append(np.concatenate([np.asarray(d[k][s]) for k in KEYS], 0).astype(np.float64))
        Ps.append(np.asarray(d['subject_metadata'][s]['ea_matrix'], np.float64))
    return Xs, Ps, d


def inv_sym(A):
    """对称阵求逆与谱: A = V w V^T -> A^-1 = V (1/w) V^T（比通用 inv 更保对称性）"""
    w, V = np.linalg.eigh(A)
    return (V * (1.0 / w)) @ V.T, w, V


def whiten(Cbar):
    """迹归一化平均协方差 -> 白化算子 W = U Λ^-1/2 U^T（与 compute_ea_matrix_from_numpy 同口径）"""
    Cbar = Cbar / np.trace(Cbar)
    w, V = np.linalg.eigh(Cbar)
    w = np.maximum(w, 1e-10)
    return (V * (1.0 / np.sqrt(w))) @ V.T


def congruent(C, A):
    """C -> A C A^T, C 形状 (n,22,22)"""
    return np.einsum('ij,njk,lk->nil', A, C, A)


def trace_norm(C):
    return C / np.trace(C, axis1=-2, axis2=-1)[..., None, None]


def tangent_w(C):
    """切向量,off-diagonal 乘 √2 ⇒ 欧氏距离 == ||log C||_F（MDM 用）"""
    out = np.empty((len(C), 253))
    r, c = np.triu_indices(22)
    scale = np.where(r == c, 1.0, np.sqrt(2.0))
    for i, M in enumerate(C):
        w, V = np.linalg.eigh(M + 1e-6 * np.eye(22))
        L = (V * np.log(np.maximum(w, 1e-12))) @ V.T
        out[i] = L[r, c] * scale
    return out


def tangent_absridge(C):
    """【历史口径，故意保留】绝对 ridge 1e-6*I 的切空间映射。

    这是 `analyze_individual_diff.py` 2026-09-23 之前的实现。pkl 是 Volt 单位、
    协方差量级 1e-10 ⇒ 地板比每个特征值大 3–4 个数量级，log 退化为一阶线性
    "常数 -13.8155*I + 1e6*C"。本脚本的六臂数字（`results/ea_identity/` 已落盘）
    全部在此口径下产出，故保留原样以保复现；仓库正式口径已统一为
    `analyze_spd_separability.tangent`（相对 ridge RIDGE*tr(C)/n*I，尺度无关）。
    """
    out = np.empty((len(C), 253))
    for i, M in enumerate(C):
        w, V = np.linalg.eigh(M + 1e-6 * np.eye(22))
        L = V @ np.diag(np.log(np.maximum(w, 1e-12))) @ V.T
        out[i] = L[np.triu_indices(22)]
    return out


def mdm_oof(V, y, k=5):
    """log-Euclidean MDM 的折外预测:类均值 = 切向量算术平均(= Log-Euclidean 均值)"""
    pred = np.full(len(y), -1)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(V, y):
        cls = np.unique(y[tr])
        M = np.stack([V[tr][y[tr] == c].mean(0) for c in cls])
        d = ((V[te][:, None, :] - M[None]) ** 2).sum(-1)
        pred[te] = cls[d.argmin(1)]
    return pred


def oof_predict(F, y, k=5):
    """与 cv_acc 完全同折,但返回折外标签,以便算逐被试 recall
    （历史表把组级值复制进 9 行,不是逐被试读数,这是要修掉的缺陷）"""
    pred = np.full(len(y), -1)
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=0).split(F, y):
        pred[te] = LDA(solver='lsqr', shrinkage='auto').fit(F[tr], y[tr]).predict(F[te])
    return pred


def perm_null(F, y, n=20, seed=SEED):
    """标签置换零分布 ⇒ 估计器的乐观偏差地板"""
    rng = np.random.default_rng(seed)
    return [cv_acc(F, rng.permutation(y)) for _ in range(n)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {'n_subjects': N_SUB, 'chance': CHANCE, 'r_pass': R_PASS, 'arms': list(ARMS)}

    Xs, Ps, raw_pkl = load_ea()
    res['pkl_dtype'] = str(np.asarray(raw_pkl[KEYS[0]][0]).dtype)
    res['trial_shape'] = list(Xs[0].shape)
    print('[数据] pkl dtype=%s, 每被试 X=%s' % (res['pkl_dtype'], Xs[0].shape), flush=True)

    # ---------- 1. 自校验: P^-1 X_EA 是否等于重放值 ----------
    # 不依赖任何度量定义,只验"pkl 快照与当前预处理代码是否还一致"。
    from moabb_bci2a import VALID_SUBJECTS, load_subject_mi_trials
    chk = []
    for s, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        L, R, Fe, To = load_subject_mi_trials(sid)
        X_rp = np.concatenate([L, R, Fe, To], 0).astype(np.float64)
        Pinv = inv_sym(Ps[s])[0]
        X_rec = np.einsum('ij,bjt->bit', Pinv, Xs[s])
        same_shape = X_rp.shape == X_rec.shape
        rel = float(np.abs(X_rec - X_rp).max() / max(np.abs(X_rp).max(), 1e-12)) if same_shape else float('nan')
        rs = ([float(np.corrcoef(X_rec[:, c].ravel(), X_rp[:, c].ravel())[0, 1])
               for c in range(X_rp.shape[1])] if same_shape else [])
        chk.append({'subject': 'S%02d' % sid, 'same_shape': same_shape,
                    'shape_replay': list(X_rp.shape), 'shape_pkl': list(Xs[s].shape),
                    'max_rel_err': rel, 'min_channel_r': min(rs) if rs else None,
                    'mean_channel_r': float(np.mean(rs)) if rs else None,
                    'replay_seconds': round(time.time() - t0, 1)})
        print('  [自校验] S%02d 形状%s 最大相对误差=%.3e 逐通道最小r=%.8f (%.0fs)'
              % (sid, '一致' if same_shape else '不一致', rel,
                 min(rs) if rs else float('nan'), time.time() - t0), flush=True)
    res['selfcheck_inverse_ea'] = chk
    res['selfcheck_pass'] = bool(all(c['same_shape'] and c['min_channel_r'] >= R_PASS for c in chk))
    res['selfcheck_max_rel_err'] = max(c['max_rel_err'] for c in chk)
    res['selfcheck_min_channel_r'] = min(c['min_channel_r'] for c in chk)
    print('[自校验] P^-1 X_EA ≈ 重放值 ? %s (逐通道最小 r=%.8f, 判据 r>=%.7f; '
          '最大相对误差 %.3e 属 float32 落盘 × cond(P)~1e4 的放大,非真实不一致)'
          % ('通过' if res['selfcheck_pass'] else '不通过', res['selfcheck_min_channel_r'],
             R_PASS, res['selfcheck_max_rel_err']), flush=True)
    print('         ⇒ 历史 caveat "两条通路不只差一个 EA" %s'
          % ('不成立,0.9417→0.9971 可作因果归因' if res['selfcheck_pass'] else '仍成立'), flush=True)

    # ---------- 2. 逐被试协方差与各臂变换 ----------
    W_shared = np.linalg.qr(np.random.default_rng(SEED).standard_normal((22, 22)))[0]
    C_raw, C_ea, condP = {}, {}, []
    for s in range(N_SUB):
        Pinv, wP, VP = inv_sym(Ps[s])
        condP.append(float(wP.max() / wP.min()))
        for tag, (a, b) in (('rest', REST), ('task', TASK)):
            Ce = cov(Xs[s][:, :, a:b])
            C_ea[(s, tag)] = Ce
            C_raw[(s, tag)] = congruent(Ce, Pinv)          # 精确电极空间
        print('  [协方差] S%02d cond(P)=%.3e' % (s + 1, condP[-1]), flush=True)
    res['cond_P'] = [round(c, 1) for c in condP]

    # globalEA 的共用白化算子:全体被试全部试次(两窗合并)的迹归一化平均协方差
    Gall = np.mean(np.concatenate([trace_norm(C_raw[(s, t)]) for s in range(N_SUB)
                                   for t in ('rest', 'task')], 0), 0)
    G = whiten(Gall)
    res['cond_G'] = round(float(np.linalg.cond(G)), 1)

    # ctrlSharedDir: 逐人谱 + 9 人共用方向 W ⇒ 方向不携带身份
    Qinv = {}
    for s in range(N_SUB):
        wP = np.linalg.eigvalsh(Ps[s])
        Qinv[s] = (W_shared * (1.0 / wP)) @ W_shared.T

    def arm_feats(name, s, tag):
        if name == 'EA':
            return C_ea[(s, tag)]
        if name == 'invEA':
            return C_raw[(s, tag)]
        if name == 'globalEA':
            return congruent(C_raw[(s, tag)], G)
        if name == 'EA_trnorm':
            return trace_norm(C_ea[(s, tag)])
        if name == 'invEA_trnorm':
            return trace_norm(C_raw[(s, tag)])
        if name == 'ctrlSharedDir':
            return congruent(C_ea[(s, tag)], Qinv[s])
        raise KeyError(name)

    # ---------- 3. 逐臂评估 ----------
    out = {}
    for name in ARMS:
        out[name] = {}
        for tag in ('rest', 'task'):
            Fs = [tangent_absridge(arm_feats(name, s, tag)) for s in range(N_SUB)]
            F = np.concatenate(Fs)
            y = np.concatenate([np.full(f.shape[0], s) for s, f in enumerate(Fs)])
            acc = cv_acc(F, y)
            pred = oof_predict(F, y)
            recall = [float(np.mean(pred[y == s] == s)) for s in range(N_SUB)]
            out[name][tag] = {'acc': round(acc, 4),
                              'recall_per_subject': [round(r, 4) for r in recall]}
            print('[%-13s] %s 9 类身份 = %.4f (随机 %.4f)' % (name, tag, acc, CHANCE), flush=True)
        # MDM 解码器不变性检验:只对两主臂做
        if name in ('EA', 'invEA'):
            for tag in ('rest', 'task'):
                Vs = [tangent_w(arm_feats(name, s, tag)) for s in range(N_SUB)]
                V = np.concatenate(Vs)
                y = np.concatenate([np.full(v.shape[0], s) for s, v in enumerate(Vs)])
                pred = mdm_oof(V, y)
                acc = float(np.mean(pred == y))
                out[name]['mdm_' + tag] = {'acc': round(acc, 4),
                                           'recall_per_subject': [round(float(np.mean(pred[y == s] == s)), 4)
                                                                  for s in range(N_SUB)]}
                print('[%-13s] MDM %s = %.4f' % (name, tag, acc), flush=True)
    Fs = [tangent_absridge(arm_feats('EA', s, 'task')) for s in range(N_SUB)]
    F = np.concatenate(Fs)
    y = np.concatenate([np.full(f.shape[0], s) for s, f in enumerate(Fs)])
    nl = perm_null(F, y)
    out['EA']['task_permnull'] = {'mean': round(float(np.mean(nl)), 4),
                                  'std': round(float(np.std(nl)), 4), 'n': len(nl)}
    print('[置换零分布] EA task: %.4f ± %.4f' % (np.mean(nl), np.std(nl)), flush=True)
    res['results'] = out

    # ---------- 4. 与历史值对照 ----------
    res['history'] = {'subj_id_nonEA_rest': 0.9417, 'subj_id_nonEA_task': 0.9828,
                      'subj_id_EA_rest': 0.9971, 'subj_id_EA_task': 0.9996}
    res['reproduce'] = {'EA_rest': out['EA']['rest']['acc'], 'EA_task': out['EA']['task']['acc'],
                        'invEA_rest': out['invEA']['rest']['acc'],
                        'invEA_task': out['invEA']['task']['acc']}
    print('[复现] EA  %.4f/%.4f vs 历史 0.9971/0.9996' % (out['EA']['rest']['acc'],
                                                          out['EA']['task']['acc']), flush=True)
    print('[复现] invEA %.4f/%.4f vs 历史重放 0.9417/0.9828' % (out['invEA']['rest']['acc'],
                                                                out['invEA']['task']['acc']), flush=True)

    # ---------- 5. 机制: cond(P) 与逐人增益 ----------
    for tag in ('rest', 'task'):
        g = np.array(out['EA'][tag]['recall_per_subject']) - np.array(out['invEA'][tag]['recall_per_subject'])
        rho, p = spearmanr(condP, g)
        res['mech_' + tag] = {'gain_per_subject': [round(float(x), 4) for x in g],
                              'spearman_condP_gain_rho': round(float(rho), 4),
                              'spearman_condP_gain_p': round(float(p), 5)}
        print('[机制] %s: cond(P) × (EA-invEA) 逐人 recall 增益 ρ=%+.4f p=%.4f' % (tag, rho, p), flush=True)

    # ---------- 6. 判决: 换成尺度无关的相对 ridge,落差是否消失 ----------
    # 这一节是全部机制讨论的终点:若两臂都升到 ~1.0,则历史落差 100% 是绝对 ridge 的伪影。
    # 此处的 `tangent` 来自 analyze_individual_diff,已统一为 analyze_spd_separability 的相对 ridge。
    rel = {}
    for tag in ('rest', 'task'):
        for nm, Cdict in (('EA', C_ea), ('invEA', C_raw)):
            Fs = [tangent(Cdict[(s, tag)]) for s in range(N_SUB)]
            F = np.concatenate(Fs)
            y = np.concatenate([np.full(f.shape[0], s) for s, f in enumerate(Fs)])
            rel['%s_%s' % (nm, tag)] = round(cv_acc(F, y), 4)
    res['relative_ridge'] = rel
    print('[判决] 相对 ridge(尺度无关)下: EA rest/task = %.4f/%.4f, 电极空间 = %.4f/%.4f'
          % (rel['EA_rest'], rel['EA_task'], rel['invEA_rest'], rel['invEA_task']), flush=True)
    print('       vs 绝对 ridge: EA %.4f/%.4f, 电极空间 %.4f/%.4f ⇒ 落差%s'
          % (out['EA']['rest']['acc'], out['EA']['task']['acc'],
             out['invEA']['rest']['acc'], out['invEA']['task']['acc'],
             '消失(伪影)' if abs(rel['EA_rest'] - rel['invEA_rest']) < 0.01 else '仍在'), flush=True)

    with open(OUT / 'ea_identity.json', 'w', encoding='utf-8') as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    print('[OK] %s' % (OUT / 'ea_identity.json'), flush=True)


if __name__ == '__main__':
    main()
