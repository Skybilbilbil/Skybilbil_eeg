# -*- coding: utf-8 -*-
"""个体化差异材料: 逐被试的【生理指标】+【判别指标】+【身份可分性】

⚠ 关键前提: saved_data/preprocessed_data_bci2a_raw_ica.pkl 已做逐被试 EA
  (alignment.py:91 的 P = V D^-1/2 V^T 是 22x22 全通道白化) —— 白化后
  【通道编号不再对应电极】。所以:
    A/B 生理指标  -> 走 moabb_bci2a.load_subject_mi_trials() 的【非 EA】数据
                     (电极空间, 才可解释 C3/C4/灶通道)
    C  判别指标  -> 走 EA 后的 pkl(与模型输入一致)
    D  身份可分  -> 两边都算, 差值 = EA 白化本身注入的'身份签名'

产出 results/individual_diff/: subject_table.csv/.json + features.npz
纯 CPU; 不改模型、不训练。约 10 分钟(非 EA 需从 GDF 缓存重放 9 个被试)。
"""
import sys, json, csv, time
from pathlib import Path
from typing import Any
import numpy as np
from scipy.signal import welch, butter, sosfiltfilt
from scipy.linalg import eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))   # moabb_bci2a / preprocessing(2026-09-17 起在 preprocess/)
from mAtt.mAtt import BCI2A_CHANNELS
# 全仓库唯一的 tangent():尺度无关的相对 ridge(RIDGE*tr(C)/n*I)。
# 本文件原先自带一份绝对 ridge(1e-6*I)的实现,而 pkl 是 Volt 单位、协方差量级 1e-10
# ⇒ 地板压过信号、log 退化为一阶线性(见 实验记录.md ⑰)。2026-09-23 起统一用这一份。
from analyze_spd_separability import tangent  # noqa: E402

FS = 250
REST, TASK = (0, 250), (375, 1126)
PKL = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'individual_diff'
KEYS = ['aligned_left_epochs_list', 'aligned_right_epochs_list',
        'aligned_feet_epochs_list', 'aligned_hands_epochs_list']
LABELS = ['left', 'right', 'feet', 'tongue']
CH = {n: i for i, n in enumerate(BCI2A_CHANNELS)}
ALPHA_BAND, BETA_BAND = (8, 13), (15, 25)


def bp(P, f, lo, hi):
    m = (f >= lo) & (f <= hi)
    return float(np.trapezoid(P[m], f[m])) if m.sum() > 1 else float('nan')


def psd(seg):
    f, P = welch(seg, fs=FS, nperseg=min(256, seg.shape[-1]), noverlap=0,
                 nfft=2048, axis=-1)
    return f, P.mean(axis=(0, 1))


def ch_psd(seg):
    f, P = welch(seg, fs=FS, nperseg=min(256, seg.shape[-1]), noverlap=0,
                 nfft=2048, axis=-1)
    return f, P.mean(axis=0)


def cov(seg):
    s = seg - seg.mean(axis=2, keepdims=True)
    return np.einsum('nct,ndt->ncd', s, s) / seg.shape[-1]


def eff_rank(C):
    """参与比 (sum L)^2 / sum L^2 —— 对近零特征值稳健, = 有效空间维度数"""
    ev = np.maximum(np.linalg.eigvalsh(C), 0.0)
    s1, s2 = ev.sum(), (ev ** 2).sum()
    return float(s1 * s1 / (s2 + 1e-30))


def cv_acc(F, y, k=5):
    return float(np.mean([LDA(solver='lsqr', shrinkage='auto').fit(F[tr], y[tr])
                          .score(F[te], y[te])
                          for tr, te in StratifiedKFold(k, shuffle=True,
                                                        random_state=0).split(F, y)]))


def fit_csp(C, y, k=2, reg=0.05):
    Ws = []
    for c in np.unique(y):
        Ca, Cb = C[y == c].mean(0), C[y != c].mean(0)
        tr = np.trace(Ca + Cb) / C.shape[-1]
        _, V = eigh(Ca, Cb + reg * tr * np.eye(C.shape[-1]))
        Ws += [V[:, -k:], V[:, :k]]
    return np.concatenate(Ws, axis=1)


def csp_acc(C, y, tr, te):
    W = fit_csp(C[tr], y[tr])
    F = np.log(np.maximum(np.einsum('ia,nij,ja->na', W, C, W), 1e-12))
    return LDA(solver='lsqr', shrinkage='auto').fit(F[tr], y[tr]).score(F[te], y[te])


def physiology(X, y):
    """非 EA 电极空间数据 -> 生理列"""
    r = {}
    rest, task = X[:, :, REST[0]:REST[1]], X[:, :, TASK[0]:TASK[1]]
    fr, Pr = psd(rest)
    m_a = (fr >= ALPHA_BAND[0]) & (fr <= ALPHA_BAND[1])
    gi = int(np.where(m_a)[0][np.argmax(Pr[m_a])])
    iaf = float(fr[gi])
    if 0 < gi < len(Pr) - 1:
        a, b, c = (np.log(Pr[gi - 1] + 1e-30), np.log(Pr[gi] + 1e-30),
                   np.log(Pr[gi + 1] + 1e-30))
        den = a - 2 * b + c
        if abs(den) > 1e-12:
            iaf += float(0.5 * (a - c) / den * (fr[1] - fr[0]))
    r['iaf_hz'] = round(iaf, 3)
    lo, hi = iaf - 1.5, iaf + 1.5
    r['alpha_pow_db'] = round(10 * np.log10(bp(Pr, fr, lo, hi) + 1e-30), 3)
    r['alpha_snr_db'] = round(10 * np.log10(
        (Pr[m_a].max() + 1e-30) / (np.median(Pr[(fr >= 5) & (fr <= 30)]) + 1e-30)), 3)
    f2, Pc = ch_psd(rest)
    m2 = (f2 >= lo) & (f2 <= hi)
    bpc = Pc[:, m2].mean(1)
    r['mu_focus_ch'] = BCI2A_CHANNELS[int(np.argmax(bpc))]
    r['mu_focus_rel'] = round(float(bpc.max() / (bpc.mean() + 1e-30)), 3)
    r['c3c4_asym_db'] = round(10 * np.log10(
        (bpc[CH['C4']] + 1e-30) / (bpc[CH['C3']] + 1e-30)), 3)
    r['rest_eff_rank'] = round(eff_rank(cov(rest).mean(0)), 3)
    ft, Pt = psd(task)
    fr2, Ptc = ch_psd(task)
    m3 = (fr2 >= lo) & (fr2 <= hi)
    for nm, (l2, h2) in (('mu', (lo, hi)), ('beta', BETA_BAND)):
        r['erd_%s_db' % nm] = round(10 * np.log10(
            (bp(Pt, ft, l2, h2) + 1e-30) / (bp(Pr, fr, l2, h2) + 1e-30)), 3)
    w, st, ts, vals = 125, 25, [], []
    for a0 in range(0, task.shape[-1] - w, st):
        _, Ps = psd(task[:, :, a0:a0 + w])
        ts.append((a0 + w / 2.0) / FS)
        vals.append(Ps[m3].mean())
    r['erd_peak_lat_s'] = round(float(ts[int(np.argmin(vals))]), 3)
    for ci, nm in ((CH['C3'], 'c3'), (CH['C4'], 'c4')):
        r['erd_%s_db' % nm] = round(10 * np.log10(
            (Ptc[ci, m3].mean() + 1e-30) / (Pc[ci, m2].mean() + 1e-30)), 3)
    r['erd_contra_db'] = round(min(r['erd_c3_db'], r['erd_c4_db']), 3)
    for c, nm in enumerate(LABELS):
        _, Pcc = ch_psd(task[y == c])
        r['erd_cls_%s_db' % nm] = round(10 * np.log10(
            (bp(Pcc.mean(0), fr2, lo, hi) + 1e-30) / (bp(Pr, fr, lo, hi) + 1e-30)), 3)
    return r


def curves(X, y, iaf, step=62, win=125):
    """原始曲线(全部非 EA 电极空间): 静息 PSD / 逐类 mu 功率时间过程 / 逐窗 log 功率"""
    f, P = welch(X[:, :, REST[0]:REST[1]], fs=FS, nperseg=250, noverlap=0,
                 nfft=2048, axis=-1)
    P = P.mean(axis=(0, 1))
    band = (f >= 2) & (f <= 45)
    lo, hi = iaf - 1.5, iaf + 1.5
    sos = butter(4, [lo / (FS / 2), hi / (FS / 2)], btype='band', output='sos')
    pw = np.abs(sosfiltfilt(sos, X, axis=-1)) ** 2
    pw = np.apply_along_axis(lambda v: np.convolve(v, np.ones(win) / win, 'same'),
                             -1, pw)
    edges = np.arange(0, X.shape[-1] - win + 1, step)
    lp = np.log(pw[:, :, edges + win // 2].mean(axis=1) + 1e-20)
    tc = np.stack([lp[y == c].mean(0) for c in range(4)])
    return {'freq': f[band].tolist(),
            'psd_db': (10 * np.log10(P[band] + 1e-30)).tolist(),
            't_s': ((edges + win // 2) / FS).tolist(),
            'mu_logpow': tc.tolist(), 'lp_all': lp.tolist()}


def main():
    import pickle
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = [{'subject': 'S%02d' % (i + 1)} for i in range(9)]
    curves_all = []

    # ===== A/B 生理指标: 非 EA(电极空间) =====
    from moabb_bci2a import load_subject_mi_trials, VALID_SUBJECTS
    Fn_task, Fn_rest, yn, sn = [], [], [], []
    for i, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        L, R, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, R, Fe, To], 0).astype(np.float64)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, R, Fe, To])])
        rows[i].update(physiology(X, y))
        curves_all.append(curves(X, y, rows[i]['iaf_hz']))
        Fn_task.append(tangent(cov(X[:, :, TASK[0]:TASK[1]])))
        Fn_rest.append(tangent(cov(X[:, :, REST[0]:REST[1]])))
        yn.append(y); sn.append(np.full(len(y), i))
        print('  [非EA] S%02d 完成 (%.0fs)' % (sid, time.time() - t0), flush=True)
    Fn_task = np.concatenate(Fn_task); Fn_rest = np.concatenate(Fn_rest)
    yn = np.concatenate(yn); sn = np.concatenate(sn)
    from scipy.stats import f_oneway
    allv = np.concatenate([np.array(c['lp_all']) for c in curves_all])
    Fstat = [float(f_oneway(*[allv[yn == k][:, j] for k in range(4)]).statistic)
             for j in range(allv.shape[1])]
    with open(OUT / 'raw_curves.json', 'w', encoding='utf-8') as fh:
        json.dump({'subjects': [r['subject'] for r in rows],
                   'freq': curves_all[0]['freq'],
                   'psd_db': [c['psd_db'] for c in curves_all],
                   't_s': curves_all[0]['t_s'],
                   'mu_logpow': [c['mu_logpow'] for c in curves_all],
                   'F_stat': Fstat}, fh)
    print('[OK] %s/raw_curves.json' % OUT, flush=True)
    for tag, F in (('rest', Fn_rest), ('task', Fn_task)):
        a = cv_acc(F, sn)
        print('[身份可分 非EA] %s -> 9 类 = %.1f%% (随机 11.1%%)' % (tag, a * 100), flush=True)
        for i in range(9):
            rows[i]['subj_id_nonEA_%s' % tag] = round(a, 4)

    # ===== C 判别指标: EA 后的 pkl(与模型输入一致) =====
    with open(PKL, 'rb') as fh:
        d = pickle.load(fh)
    n = len(d['subject_metadata'])
    Fe_task, Fe_rest, ye, se, Ct_list = [], [], [], [], []
    for s in range(n):
        X = np.concatenate([np.asarray(d[k][s]) for k in KEYS], 0).astype(np.float64)
        y = np.concatenate([np.full(len(d[k][s]), c) for c, k in enumerate(KEYS)])
        sess = np.concatenate([np.r_[np.zeros(len(d[k][s]) // 2, int),
                                     np.ones(len(d[k][s]) - len(d[k][s]) // 2, int)]
                               for k in KEYS])
        Ct = cov(X[:, :, TASK[0]:TASK[1]])
        mT, mE = np.where(sess == 0)[0], np.where(sess == 1)[0]
        wi = float(np.mean([csp_acc(Ct, y, tr, te) for tr, te in
                            StratifiedKFold(5, shuffle=True,
                                            random_state=0).split(Ct, y)]))
        cr = float(np.mean([csp_acc(Ct, y, mT, mE), csp_acc(Ct, y, mE, mT)]))
        Ft = tangent(Ct)
        rows[s]['csp_within'] = round(wi, 4)
        rows[s]['csp_cross_sess'] = round(cr, 4)
        rows[s]['session_gap'] = round(wi - cr, 4)
        rows[s]['tangent_lda_within'] = round(cv_acc(Ft, y), 4)
        Fe_task.append(Ft); Fe_rest.append(tangent(cov(X[:, :, REST[0]:REST[1]])))
        ye.append(y); se.append(np.full(len(y), s)); Ct_list.append(Ct)
    Fe_task = np.concatenate(Fe_task); Fe_rest = np.concatenate(Fe_rest)
    ye = np.concatenate(ye); se = np.concatenate(se)
    for s in range(n):
        tr = (se != s) & (np.tile(np.concatenate([np.r_[np.zeros(72, int),
                                                  np.ones(72, int)] for _ in KEYS]), n) == 0)
        te = (se == s) & (np.tile(np.concatenate([np.r_[np.zeros(72, int),
                                                  np.ones(72, int)] for _ in KEYS]), n) == 0)
        rows[s]['receivability'] = round(float(LDA(solver='lsqr', shrinkage='auto')
                                                .fit(Fe_task[tr], ye[tr])
                                                .score(Fe_task[te], ye[te])), 4)
    for tag, F in (('rest', Fe_rest), ('task', Fe_task)):
        a = cv_acc(F, se)
        print('[身份可分 EA]   %s -> 9 类 = %.1f%%' % (tag, a * 100), flush=True)
        for i in range(9):
            rows[i]['subj_id_EA_%s' % tag] = round(a, 4)

    # E' 主协议 70/15/15 逐被试深度模型精度(3-seed 均值): 来自 eval70_persubject.py
    # 的 verify70_persubject.json;不存在时跳过该列,表格其余部分自洽。
    _ps = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
    if _ps.exists():
        _d = json.loads(_ps.read_text(encoding='utf-8'))
        _seeds = [s for s in _d if 'best_acc' in _d[s]]
        for i in range(9):
            _v = [_d[s]['best_acc']['per_subject']['S%02d' % (i + 1)][1]
                  for s in _seeds]
            rows[i]['deep70_acc'] = round(sum(_v) / len(_v), 4)
            rows[i]['deep70_std'] = round(float(np.std(_v, ddof=1)), 4)

    cols = list(rows[0].keys())
    with open(OUT / 'subject_table.csv', 'w', newline='', encoding='utf-8-sig') as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader(); wr.writerows(rows)
    with open(OUT / 'subject_table.json', 'w', encoding='utf-8') as fh:
        json.dump(rows, fh, indent=2, ensure_ascii=False)
    np.savez_compressed(OUT / 'features.npz',
                        nonEA_task=Fn_task, nonEA_rest=Fn_rest, y_nonEA=yn, subj_nonEA=sn,
                        EA_task=Fe_task, EA_rest=Fe_rest, y_EA=ye, subj_EA=se)
    print('\n[OK] %s/  subject_table.csv | .json | features.npz  (%d 行 x %d 列)'
          % (OUT, len(rows), len(cols)))
    print('\n' + ' | '.join(cols))
    for r in rows:
        print(' | '.join(str(r.get(c, '')) for c in cols))


if __name__ == '__main__':
    main()
