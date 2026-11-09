# -*- coding: utf-8 -*-
"""从已确定的量【Q = 手类任务期 μ ERD 深度】往下追一层:把它拆成
「类共同分量」与「类差分分量」,看哪一支才是与精度同序的那一支。

## 为什么这是"向下"而不是"旁边"
Q 的定义式 = 10log10( mean_{i,c}(A) / mean_{i,c}(R) ),i = 手类试次, c = 22 通道。
它**把左右手两类合在一起平均** ⇒ Q 在构造上就是【类共同】的:两只手一起平均,
偏侧化的部分互相抵消。而任何协方差/线性解码器只能读【类差分】的东西。
于是有一个可直接检验的预言(由已确定的干预判决推出):
    动任务窗 μ 的"量"对精度几乎无效(a 扫描 ±2 dB 全档极差 0.31pp;β_W 不显著),
    而"改了哪一块通道"却差 5–9 倍
  ⇒ **Q 不可能是解码器读到的量;它只能是"类差分调制幅度"的一个间接标尺。**
若真如此,则:ρ(Q, 精度 | D) 应当塌掉,而 ρ(D, 精度) 应当很强。
D := 类差分分量的幅度,在**同一层**(逐通道 μ 带功率的对数比)上定义,不经模型、不经切空间。

## 三个读数(全是 μ 带功率层,非 EA,各人 IAF±1.5,窗与 erd_decomposition 相同)
  E_L[c], E_R[c] : 逐通道的 Pfurtscheller ERD = 10log10( mean_i A_ic / mean_i R_ic )
  Q              : 已确定的量(22 通道先平均功率再取比,两类合并)—— 检查点复现
  E_diff         : E_L − E_R  (22 维,"两手的空间 μ 反应差在哪")
  D              : ||E_diff||  —— 类差分幅度
  D 的噪声地板   : 每类试次随机劈半各算一次 E_diff,得 ||d_half||² 的期望
                   ⇒ D_true² = D_obs² − E||d_half||²/4
  align          : 留一法模板余弦 cos(E_diff_i, mean_{j≠i} E_diff_j)
                   —— 该被试的类差分"空间形式"与其余八人是否一致(解码器要能用就得一致)
  acc_mu_topog   : 只用逐试次 22 维 μ 带功率做 LDA 的 5 折被试内精度
                   —— "μ 功率层到底有没有可解码的类信息"的零训练体检

只写 results/erd_class_diff/。约 12 分钟(GDF 重放是瓶颈)。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, rankdata, spearmanr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / 'preprocess'))
from mAtt.mAtt import BCI2A_CHANNELS                       # noqa: E402
import analyze_erd_decomposition as ED                     # noqa: E402

OUT = BASE / 'results' / 'erd_class_diff'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'
REF = BASE / 'results' / 'erd_decomposition' / 'erd_decomposition.json'
FS, REST, TASK = ED.FS, ED.REST, ED.TASK
CH = ED.CH
ROI = ED.ROI_L + ED.ROI_R
PERF = ('deep70_acc', 'csp_within')


def ch_erd(A, R):
    """逐通道 Pfurtscheller ERD(dB):10log10(mean_i A_ic / mean_i R_ic) -> (22,)"""
    return 10 * np.log10(A.mean(0) / R.mean(0))


def split_tb(A, R):
    """已确定的量 Q = 22 通道先平均功率再取比(两类合并)。"""
    T, B = 10 * np.log10(A.mean()), 10 * np.log10(R.mean())
    return float(T), float(B), float(T - B)


def pcorr(x, y, z):
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    Z = np.column_stack([np.ones_like(rz), rz])
    ex = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
    ey = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    r, p = pearsonr(ex, ey)
    return round(float(r), 4), round(float(p), 5)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}
    pub = json.loads(REF.read_text(encoding='utf-8'))['per_subject']
    from moabb_bci2a import load_subject_mi_trials, VALID_SUBJECTS

    per, edges = {}, {}
    for i, sid in enumerate(VALID_SUBJECTS):
        t0 = time.time()
        name = 'S%02d' % (i + 1)
        iaf = ref[name]['iaf_hz']
        L, Rr, Fe, To = load_subject_mi_trials(sid)
        X = np.concatenate([L, Rr, Fe, To], 0).astype(np.float64)
        n = len(X)
        y = np.concatenate([np.full(len(a), c) for c, a in enumerate([L, Rr, Fe, To])])
        ci = np.arange(n) % (n // 4)                # 类内序号
        hL, hR = (y == 0), (y == 1)
        s0 = ci < n // 8                            # 每类前一半 = 0train

        A = ED.band_pow(X, iaf, TASK[0], TASK[1], 125)     # (n,22) 任务窗
        Rb = ED.band_pow(X, iaf, REST[0], REST[1], 0)      # (n,22) 静息窗
        row = {'iaf_hz': round(float(iaf), 3)}
        # 检查点
        row['Q_check_vs_published'] = round(
            split_tb(A[hL | hR], Rb[hL | hR])[2] - pub[name]['brain_hand'], 6)

        arms = {}
        for tag, m in (('pool', hL | hR), ('L', hL), ('R', hR)):
            arms[tag] = (A[m], Rb[m])
        # 已确定的量 + 两条分量
        T_, B_, Q = split_tb(*arms['pool'])
        E_L, E_R = ch_erd(*arms['L']), ch_erd(*arms['R'])
        E_diff = E_L - E_R
        D_obs = float(np.linalg.norm(E_diff))
        D_roi = float(np.linalg.norm(E_diff[[CH[c] for c in ROI]]))
        # 类共同分量的逐通道版(供对照)
        E_com = 0.5 * (E_L + E_R)
        row.update({'T_db': round(T_, 4), 'B_db': round(B_, 4), 'Q_db': round(Q, 4),
                    'E_com_montage_db': round(float(E_com.mean()), 4),
                    'D_obs': round(D_obs, 4), 'D_roi_obs': round(D_roi, 4)})

        # D 的噪声地板:每类随机劈半
        rng = np.random.default_rng(i)
        nh = []
        for _ in range(200):
            d = []
            for m in (hL, hR):
                idx = np.where(m)[0]
                k = rng.permutation(len(idx))
                a, b = idx[k[:len(idx) // 2]], idx[k[len(idx) // 2:]]
                d.append(ch_erd(A[a], Rb[a]) - ch_erd(A[b], Rb[b]))
            nh.append(sum(float((x ** 2).sum()) for x in d))
        d_null = float(np.mean(nh)) / 4.0        # = 22*sigma2_full
        row['D_noise_var_total'] = round(d_null, 4)
        row['D_true'] = round(float(np.sqrt(max(D_obs ** 2 - d_null, 0))), 4)
        row['D_true_roi'] = round(float(np.sqrt(max(D_roi ** 2 - d_null * len(ROI) / 22, 0))), 4)

        # μ 功率层有没有可解码的类信息(零训练体检)
        Fmu = np.log(np.concatenate([A[hL], A[hR]], 0))
        yy = np.r_[np.zeros(hL.sum()), np.ones(hR.sum())]
        skf = StratifiedKFold(5, shuffle=True, random_state=0)
        accs = []
        for tr, te in skf.split(Fmu, yy):
            clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Fmu[tr], yy[tr])
            accs.append(float((clf.predict(Fmu[te]) == yy[te]).mean()))
        row['acc_mu_topog'] = round(float(np.mean(accs)), 4)
        for tag, mm in (('s0', s0), ('s1', ~s0)):
            Fh = np.log(np.concatenate([A[hL & mm], A[hR & mm]], 0))
            yh = np.r_[np.zeros((hL & mm).sum()), np.ones((hR & mm).sum())]
            aa = []
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(Fh, yh):
                c2 = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Fh[tr], yh[tr])
                aa.append(float((c2.predict(Fh[te]) == yh[te]).mean()))
            row.setdefault('acc_mu_topog_by_session', {})[tag] = round(float(np.mean(aa)), 4)

        # 逐 session
        for tag, m in (('s0', (hL & s0) | (hR & s0)), ('s1', (hL & ~s0) | (hR & ~s0))):
            mL, mR = m & (y == 0), m & (y == 1)
            ed = ch_erd(A[mL], Rb[mL]) - ch_erd(A[mR], Rb[mR])
            row.setdefault('by_session', {})[tag] = {
                'Q_db': round(split_tb(A[m], Rb[m])[2], 4),
                'D_obs': round(float(np.linalg.norm(ed)), 4)}
        edges[name] = E_diff
        row['E_diff'] = [round(float(v), 4) for v in E_diff]
        row['E_L'] = [round(float(v), 4) for v in E_L]
        row['E_R'] = [round(float(v), 4) for v in E_R]
        per[name] = row
        print('  %s iaf=%.2f  检查点 %+.1e  Q=%+.3f  D_obs=%.3f (真值 %.3f)  '
              'μ功率LDA=%.1f%%  (%.0fs)'
              % (name, iaf, row['Q_check_vs_published'], Q, D_obs, row['D_true'],
                 row['acc_mu_topog'] * 100, time.time() - t0), flush=True)

    S = list(per)
    Q = np.array([per[s]['Q_db'] for s in S])
    D = np.array([per[s]['D_true'] for s in S])
    Droi = np.array([per[s]['D_true_roi'] for s in S])
    Dobs = np.array([per[s]['D_obs'] for s in S])
    mt = np.array([per[s]['acc_mu_topog'] for s in S])
    acc = np.array([ref[s]['deep70_acc'] for s in S])
    csp = np.array([ref[s]['csp_within'] for s in S])
    E = np.stack([edges[s] for s in S])
    # 留一法模板余弦:该被试的类差分空间形式与其余八人是否一致
    align = np.array([float(np.dot(E[i], E.sum(0) - E[i]) /
                            (np.linalg.norm(E[i]) * np.linalg.norm(E.sum(0) - E[i])))
                      for i in range(len(S))])

    N = np.linalg.norm(E, axis=1)
    COS = (E @ E.T) / np.outer(N, N)
    off = COS[~np.eye(len(S), dtype=bool)]
    print('\n=== 类差分「空间形式」的被试间一致性(22 维 E_diff 的两两余弦) ===')
    print('  36 对余弦: %.3f ~ %.3f  均值 %.3f  (自身=1)'
          % (off.min(), off.max(), off.mean()))
    print('  模板余弦(留一法): ' + ' '.join('%s %+.2f' % (s, align[j]) for j, s in enumerate(S)))

    print('\n=== 检查点 ===')
    _d = max(abs(per[s]['Q_check_vs_published']) for s in S)
    print('Q 与已发布 brain_hand 的最大偏差: %.1e dB  %s'
          % (_d, 'PASS' if _d < 1e-4 else 'FAIL')
          + '  (容差 1e-4:已发布值在 json 里是 4 位小数取整)')
    print('类差分幅度 D: 观测 %.3f~%.3f (%.2fx) → 扣噪声 %.3f~%.3f (%.2fx)'
          % (Dobs.min(), Dobs.max(), Dobs.max() / Dobs.min(),
             D.min(), D.max(), D.max() / D.min()))
    print('类共同分量(逐通道均值): %.3f~%.3f dB' % (
        min(per[s]['E_com_montage_db'] for s in S), max(per[s]['E_com_montage_db'] for s in S)))
    print('\n%-6s %9s %9s %8s %9s %9s' % ('被试', 'Q(类共同)', 'D(类差分)', 'D_roi', 'μ功率LDA', '深度精度'))
    for j, s in enumerate(S):
        print('%-6s %9.3f %9.3f %8.3f %8.1f%% %8.1f%%'
              % (s, Q[j], D[j], Droi[j], mt[j] * 100, acc[j] * 100))

    print('\n=== 谁与精度同序(Spearman, n=9) ===')
    tab = {}
    for j, s in enumerate(S):
        per[s]['align_loo'] = round(float(align[j]), 4)
    for nm, v in (('Q 类共同 ERD 深度', Q), ('D 类差分幅度(扣噪声)', D),
                  ('D_roi 中央区类差分', Droi), ('D_obs 类差分(未扣噪声)', Dobs),
                  ('align 类差分空间形式一致性', align), ('μ功率层 LDA 精度', mt)):
        r1, p1 = spearmanr(v, acc)
        r2, p2 = spearmanr(v, csp)
        tab[nm] = {'vs_deep70_acc': [round(float(r1), 4), round(float(p1), 6)],
                   'vs_csp_within': [round(float(r2), 4), round(float(p2), 6)]}
        print('%-26s ρ(深度模型)=%+.3f (p=%.4f)   ρ(CSP)=%+.3f (p=%.4f)'
              % (nm, r1, p1, r2, p2))
    print('\nρ(Q, D) = %+.3f  ⇒ Q 与类差分幅度是不是同一件事' % spearmanr(Q, D)[0])
    print('偏相关(判决量):')
    for nm, z in (('控制 D', D), ('控制 Q', Q)):
        a, b = pcorr(Q, acc, z) if nm == '控制 D' else pcorr(D, acc, z)
        print('  ρ(%s, 深度精度 | %s) = %+.3f (p=%.4f)'
              % ('Q' if nm == '控制 D' else 'D', nm[3:], a, b))

    print('\n=== 跨 session 重测(类共同 vs 类差分) ===')
    q0 = np.array([per[s]['by_session']['s0']['Q_db'] for s in S])
    q1 = np.array([per[s]['by_session']['s1']['Q_db'] for s in S])
    d0 = np.array([per[s]['by_session']['s0']['D_obs'] for s in S])
    d1 = np.array([per[s]['by_session']['s1']['D_obs'] for s in S])
    for nm, a, b in (('Q 类共同', q0, q1), ('D 类差分', d0, d1)):
        print('  %-8s 两日 Pearson r=%+.3f  Spearman=%+.3f  (s0 与精度 ρ=%+.3f / s1 ρ=%+.3f)'
              % (nm, pearsonr(a, b)[0], spearmanr(a, b)[0],
                 spearmanr(a, acc)[0], spearmanr(b, acc)[0]))

    rec = {'estimator': 'per-channel Pfurtscheller ERD, mu band = per-subject IAF±1.5, '
                        'Welch(Hann) band integral; non-EA; hand classes; TASK=(375,1126), '
                        'REST=(0,250)',
           'D_definition': 'E_diff = E_left - E_right (22-dim per-channel dB); '
                           'D_true^2 = ||E_diff||^2 - E||E_diff_half1-E_diff_half2||^2/4 '
                           '(200 random half-splits per class)',
           'perf_corr_spearman': tab, 'per_subject': per,
           'spatial_form': {'pairwise_cosine_36pairs': {
               'min': round(float(off.min()), 4), 'max': round(float(off.max()), 4),
               'mean': round(float(off.mean()), 4), 'sd': round(float(off.std(ddof=1)), 4)},
               'loo_template_cosine': {s: round(float(align[j]), 4) for j, s in enumerate(S)}},
           'session_retest': {'Q': {'pearson': round(float(pearsonr(q0, q1)[0]), 4),
                                    'spearman': round(float(spearmanr(q0, q1)[0]), 4)},
                              'D': {'pearson': round(float(pearsonr(d0, d1)[0]), 4),
                                    'spearman': round(float(spearmanr(d0, d1)[0]), 4)}}}
    (OUT / 'erd_class_diff.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                             encoding='utf-8')
    print('\n[OK] %s/erd_class_diff.json' % OUT)


if __name__ == '__main__':
    main()
