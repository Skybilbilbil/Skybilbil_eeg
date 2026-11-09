# -*- coding: utf-8 -*-
"""官方数据上的同源经典基线(零训练)。

问题:深度端到端 64.40 是官方数据口径,而 62.9 是本项目 pkl 口径 —— 两者跨数据源,
不能直接比。本脚本在同一份官方数据、同一批切分上跑经典零训练读出。

口径 = results/ceiling_decomposition 的 L0:
  任务窗协方差 -> 迹归一(trace/d) -> Log-Euclidean 展开(253 维) -> 共享 LW-LDA
窗口 = 官方模型同款输入窗 x[:,:,:,124:562](论文 §8.10 / 材料 §1.1)。

两个臂:
  raw    直接用作者数据(4-40Hz)      —— 与深度模型"同输入"的对照
  bp8-40 额外带通 8-40Hz             —— 经典读出自身最优的对照

自检:先在 pkl(raw_ica) 上复现 62.9,确认口径没走样。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy import io, signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy         # noqa: E402

PKL = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'official_classic'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
TASK_PKL = (375, 1126)      # pkl 250Hz: cue+1.5s -> 末尾(与 ceiling_decomposition 一致)
WIN_OFF = (124, 562)        # 官方模型同款输入窗 x[:,:,:,124:562]


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


def shared_lda(Ftr, ytr, Fte, yte, sub_te):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
    ok = clf.predict(Fte) == yte
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    return float(ok.mean()), per


def group(per):
    return (float(np.mean([per[SUBJ[i]] for i in LOW])),
            float(np.mean([per[SUBJ[i]] for i in HIGH])))


def selfcheck_pkl():
    subject_data, _ = load_preprocessed_dataset(str(PKL))
    X = np.concatenate([s[0] for s in subject_data], 0).astype(np.float64)
    y = np.concatenate([s[1] for s in subject_data], 0)
    sub = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                          for i, s in enumerate(subject_data)])
    ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
    Ftr = logvec(cov_tn(ret[0][:, :, TASK_PKL[0]:TASK_PKL[1]]))
    Fte = logvec(cov_tn(ret[4][:, :, TASK_PKL[0]:TASK_PKL[1]]))
    ov, per = shared_lda(Ftr, ret[1], Fte, ret[5], ret[8])
    lo, hi = group(per)
    print('[自检 pkl] 共享 LDA = %.2f%%  (低四 %.1f / 高五 %.1f)  期望 62.9'
          % (100 * ov, 100 * lo, 100 * hi), flush=True)
    return {'pooled': ov, 'per_subject': per, 'low4': lo, 'high5': hi, 'expect': 0.629}


def load_official():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        per.append((apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X)),
                    np.concatenate(ys, 0)))
    return per


def bandpass(X, lo=8.0, hi=40.0, fs=128, order=4):
    b, a = signal.butter(order, [lo / (fs / 2), hi / (fs / 2)], btype='band')
    return signal.filtfilt(b, a, X, axis=-1)


def arm(X, y, sub, tag):
    F = logvec(cov_tn(X[:, :, WIN_OFF[0]:WIN_OFF[1]]))
    i_all = np.arange(len(y))
    seeds, pers, accs = {}, {}, []
    for seed in (42, 2024, 7):
        tr, tmp = train_test_split(i_all, test_size=0.30, stratify=y, random_state=seed)
        va, te = train_test_split(tmp, test_size=0.5, stratify=y[tmp], random_state=seed)
        ov, p = shared_lda(F[tr], y[tr], F[te], y[te], sub[te])
        seeds[str(seed)] = ov
        pers[str(seed)] = p
        accs.append(ov)
        lo, hi = group(p)
        print('  %-8s seed %-5d %.2f%%  (低四 %.1f / 高五 %.1f)'
              % (tag, seed, 100 * ov, 100 * lo, 100 * hi), flush=True)
    mean_per = {k: float(np.mean([pers[s][k] for s in pers])) for k in pers['42']}
    lo, hi = group(mean_per)
    res = {'seeds': seeds, 'mean': float(np.mean(accs)), 'sd': float(np.std(accs)),
           'per_subject_mean': mean_per, 'low4': lo, 'high5': hi}
    print('  [%s] %.2f ± %.2f  (低四 %.1f / 高五 %.1f)\n'
          % (tag, 100 * res['mean'], 100 * res['sd'], 100 * lo, 100 * hi), flush=True)
    return res


def official_crosssession():
    """真跨 session(T 训练 -> E 测试),与 §8.10 的 58.29 同口径。"""
    tr, te = [], []
    for s in range(1, 10):
        mT = io.loadmat(str(DATA / ('BCIC_S%02d_T.mat' % s)))
        mE = io.loadmat(str(DATA / ('BCIC_S%02d_E.mat' % s)))
        XT = np.asarray(mT['x_train'], dtype=np.float64)
        yT = np.asarray(mT['y_train']).reshape(-1).astype(np.int64)
        XE = np.asarray(mE['x_test'], dtype=np.float64)
        yE = np.asarray(mE['y_test']).reshape(-1).astype(np.int64)
        X = np.concatenate([XT, XE], 0)
        H = apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X))
        tr.append((H[:288], yT))
        te.append((H[288:], yE))
    Xtr = np.concatenate([p[0] for p in tr], 0)
    ytr = np.concatenate([p[1] for p in tr], 0)
    Xte = np.concatenate([p[0] for p in te], 0)
    yte = np.concatenate([p[1] for p in te], 0)
    sub_te = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    out = {}
    for tag, XA, XB in (('raw_4_40', Xtr, Xte),
                        ('bp_8_40', bandpass(Xtr, 8, 40), bandpass(Xte, 8, 40))):
        Fa = logvec(cov_tn(XA[:, :, WIN_OFF[0]:WIN_OFF[1]]))
        Fb = logvec(cov_tn(XB[:, :, WIN_OFF[0]:WIN_OFF[1]]))
        ov, p = shared_lda(Fa, ytr, Fb, yte, sub_te)
        lo, hi = group(p)
        out[tag] = {'pooled': ov, 'per_subject': p, 'low4': lo, 'high5': hi}
        print('  [跨session %s] %.2f%%  (低四 %.1f / 高五 %.1f)'
              % (tag, 100 * ov, 100 * lo, 100 * hi), flush=True)
    return out


def official_3seed():
    per = load_official()
    X = np.concatenate([p[0] for p in per], 0)
    y = np.concatenate([p[1] for p in per], 0)
    sub = np.concatenate([np.full(len(p[1]), i, dtype=np.int64) for i, p in enumerate(per)])
    return {'window': 'official 124:562 (438 pts @128Hz)',
            'raw_4_40': arm(X, y, sub, 'raw'),
            'bp_8_40': arm(bandpass(X, 8, 40), y, sub, 'bp8-40')}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'title': '官方数据同源经典基线(零训练,任务窗协方差+LE+共享LW-LDA)',
           'date': '2026-11-02',
           'protocol': 'shared LW-LDA; per-seed split identical to exp_official_3seed',
           'selfcheck_pkl': selfcheck_pkl(),
           'official_ea_128hz': official_3seed(),
           'official_crosssession_T_to_E': official_crosssession()}
    (OUT / 'official_classic.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/official_classic.json' % OUT)


if __name__ == '__main__':
    main()
