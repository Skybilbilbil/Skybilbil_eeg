# -*- coding: utf-8 -*-
"""镜像对称/反对称分解:类差分的「共同点」到底在哪个子空间。

## 依据(全部已确立)
X-3:解码效应由**空间偏侧化**驱动(对侧/同侧反号,9/9 同向);
erd_class_diff:类差分方向 E_diff 在 22 维全拓扑上高度异质(36 对余弦 −0.758~+0.811,
均值 +0.017)——这正是共享读法读不出的原因。两条合起来 ⇒ 一个没试过的分解:
把特征分到「左右同源对的反对称分量」(编码偏侧化,共同的解剖组织)与
「对称分量 + 中线」(编码双侧水平,个体特质层)。若类信息集中在反对称子空间
且该子空间跨被试更对齐,就找到了一个「更通用的特征坐标」。

## 三问
① 反对称分量(8 维)携带多少类信息(被试内 LDA vs 全 22 维);
② E_diff 的反对称部分跨被试是否比对齐(两两余弦 vs 全拓扑基线);
③ 哪种特征让**共享(pooled)读出**逐人精度最高——逐试次基线校正(log A−log R,
   被试相对化)× 投影子空间(全/反对称/对称+中线)哪个组合最接近被试内上限,
   尤其低四人。这是「共同点特征」能否喂给模型的判据。

臂 = {μ(IAF±1.5), β(15-25)} × {A 任务窗, ERD=A−R} × {full22, anti8, symmid14};
读出 = 被试内 5 折 CV(与 acc_mu_topog 同口径) + 主协议切分下的被试内/共享配对;
类别 = 手二类(主) + 四类(次)。数据 = 非 EA pkl(电极空间)。只写 results/mirror_commonality/。
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from mAtt.mAtt import BCI2A_CHANNELS                       # noqa: E402
import analyze_erd_decomposition as ED                     # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from scipy.signal import welch


def pow_band(X, lo, hi, a, b, noverlap):
    """X (n,ch,T) 的 [a,b) 窗 -> 每 trial 每通道 [lo,hi] Hz 带功率 (n,ch)。
    Welch 口径与 ED.band_pow 逐字相同(Hann、nperseg=min(NPERSEG,窗长));
    band_pow 本体只支持 IAF±1.5,任意频带在此展开。"""
    f, P = welch(X[:, :, a:b], fs=ED.FS, nperseg=min(ED.NPERSEG, b - a),
                 noverlap=noverlap, axis=-1)
    m = (f >= lo) & (f <= hi)
    return np.trapezoid(P[..., m], f[m], axis=-1)

PKL = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
TBL = BASE / 'results' / 'individual_diff' / 'subject_table.json'
OUT = BASE / 'results' / 'mirror_commonality'
BETA = (15.0, 25.0)
REST, TASK = ED.REST, ED.TASK
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
_CH = {n: i for i, n in enumerate(BCI2A_CHANNELS)}
PAIRS = [('FC3', 'FC4'), ('FC1', 'FC2'), ('C5', 'C6'), ('C3', 'C4'),
         ('C1', 'C2'), ('CP3', 'CP4'), ('CP1', 'CP2'), ('P1', 'P2')]
IL = np.array([_CH[a] for a, _ in PAIRS])
IR = np.array([_CH[b] for _, b in PAIRS])
IMID = np.array([_CH[n] for n in ('Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz')])


def project(F, mode):
    if mode == 'full':
        return F
    if mode == 'anti':
        return (F[:, IL] - F[:, IR]) / np.sqrt(2)
    return np.concatenate([(F[:, IL] + F[:, IR]) / np.sqrt(2), F[:, IMID]], 1)


def lda_fit_pred(trF, try_, teF):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(trF, try_)
    return clf.predict(teF)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ref = {r['subject']: r for r in json.loads(TBL.read_text(encoding='utf-8'))}
    sd, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in sd], 0).astype(np.float64)
    y_all = np.concatenate([s[1] for s in sd], 0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(sd)])
    ret = split_train_valid_test(np.zeros((len(y_all), 1)), y_all, 0.70, 0.15,
                                 random_state=42, others=[sub_all, np.arange(len(y_all))])
    y_tr, y_te = ret[1], ret[5]
    sub_tr, sub_te, itr, ite = ret[6], ret[8], ret[9], ret[11]
    print('镜像分解: %d 对同源对 + %d 中线;训练 %d / 测试 %d 试次'
          % (len(PAIRS), len(IMID), len(y_tr), len(y_te)), flush=True)

    t0 = time.time()
    P = {}   # P[band][win] = 逐试次带功率 (n,22)
    for bname, band in (('mu', None), ('beta', BETA)):
        per = []
        for i, s in enumerate(SUBJ):
            iaf = ref[s]['iaf_hz']
            lo, hi = (iaf - 1.5, iaf + 1.5) if band is None else band
            m = sub_all == i
            A = pow_band(X_all[m], lo, hi, TASK[0], TASK[1], 125)
            R = pow_band(X_all[m], lo, hi, REST[0], REST[1], 0)
            per.append((A, R))
        P[bname] = {'A': np.concatenate([p[0] for p in per], 0),
                    'R': np.concatenate([p[1] for p in per], 0)}
        print('  %s 带功率完成 (%.0fs)' % (bname, time.time() - t0), flush=True)

    arms = [(b, f, p) for b in ('mu', 'beta') for f in ('A', 'ERD')
            for p in ('full', 'anti', 'symmid')]
    hand_tr, hand_te = y_tr < 2, y_te < 2
    rec = {'config': {'pairs': [list(p) for p in PAIRS],
                      'mid': ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'POz'],
                      'n_test': int(len(y_te))}, 'arms': {}}

    # ② E_diff 反对称/对称部分的跨被试对齐(μ 带,任务窗)
    E = []
    for i, s in enumerate(SUBJ):
        m = (sub_all == i)
        hL, hR = m & (y_all == 0), m & (y_all == 1)
        eL = np.log(P['mu']['A'][hL]).mean(0)
        eR = np.log(P['mu']['A'][hR]).mean(0)
        E.append(eL - eR)
    E = np.stack(E)
    out = {}
    for tag, V in (('full22', E), ('anti8', (E[:, IL] - E[:, IR]) / np.sqrt(2)),
                   ('symmid14', np.concatenate(
                       [(E[:, IL] + E[:, IR]) / np.sqrt(2), E[:, IMID]], 1))):
        N = np.linalg.norm(V, axis=1)
        C = (V @ V.T) / np.outer(N, N)
        off = C[~np.eye(9, dtype=bool)]
        vT = V / N[:, None]
        pc1 = np.linalg.svd(vT - vT.mean(0), compute_uv=False)[0] ** 2 / np.sum(
            np.linalg.svd(vT - vT.mean(0), compute_uv=False) ** 2)
        out[tag] = {'cos_mean': round(float(off.mean()), 4),
                    'cos_min': round(float(off.min()), 4),
                    'cos_max': round(float(off.max()), 4),
                    'norm_mean': round(float(N.mean()), 4),
                    'pc1_share': round(float(pc1), 4)}
        print('E_diff 对齐 %s: 两两余弦均值 %+.3f (%+.3f~%+.3f)  范数 %.3f  PC1 %.3f'
              % (tag, off.mean(), off.min(), off.max(), N.mean(), pc1), flush=True)
    rec['e_diff_alignment'] = out
    anti_share = np.array([np.linalg.norm((E[i, IL] - E[i, IR]) / np.sqrt(2))
                           / np.linalg.norm(E[i]) for i in range(9)])
    rec['e_diff_alignment']['anti_norm_share_per_subject'] = {
        SUBJ[i]: round(float(anti_share[i]), 4) for i in range(9)}

    # ①③ 分类:12 臂 × {手二类, 四类} × {被试内CV, 被试内切分, 共享切分}
    print('\n%-16s %6s | %8s %8s %8s | %8s %8s %8s  (手二类: CV内/切分内/共享;四类同)'
          % ('臂(带/特征/投影)', 'dim', '低四共享', '高五共享', '共享极差',
             '低四内', '高五内', 'deep70'))
    for bname, fname, pname in arms:
        if fname == 'A':
            F = P[bname]['A']
        else:
            F = P[bname]['A'] - P[bname]['R']
        key = '%s_%s_%s' % (bname, fname, pname)
        Fp = project(F, pname)
        Ftr, Fte = Fp[itr], Fp[ite]
        rec['arms'][key] = {}
        for cls in ('hand2', 'all4'):
            mtr = hand_tr if cls == 'hand2' else np.ones(len(y_tr), bool)
            mte = hand_te if cls == 'hand2' else np.ones(len(y_te), bool)
            row = {'within_cv': {}, 'within_split': {}, 'pooled': {}}
            accs_cv, accs_ws, accs_po = [], [], []
            for i, s in enumerate(SUBJ):
                m = sub_all == i
                mi = m if cls == 'all4' else (m & (y_all < 2))
                Fi, yi = Fp[mi], y_all[mi]
                accs = []
                for trn, ten in StratifiedKFold(5, shuffle=True,
                                                random_state=0).split(Fi, yi):
                    accs.append(float((lda_fit_pred(Fi[trn], yi[trn], Fi[ten])
                                       == yi[ten]).mean()))
                accs_cv.append(float(np.mean(accs)))
                trm, tem = (sub_tr == i) & mtr, (sub_te == i) & mte
                accs_ws.append(float((lda_fit_pred(Ftr[trm], y_tr[trm], Fte[tem])
                                      == y_te[tem]).mean()))
                accs_po.append(None)   # 共享在下面一次拟合后统一填
            row['within_cv'] = {SUBJ[i]: accs_cv[i] for i in range(9)}
            row['within_split'] = {SUBJ[i]: accs_ws[i] for i in range(9)}
            # 共享:全被试训练折一次拟合
            pr = lda_fit_pred(Ftr[mtr], y_tr[mtr], Fte[mte])
            ok = pr == y_te[mte]
            for i, s in enumerate(SUBJ):
                accs_po[i] = float(ok[sub_te[mte] == i].mean())
            row['pooled'] = {SUBJ[i]: accs_po[i] for i in range(9)}
            for tag, arr in (('cv', accs_cv), ('ws', accs_ws), ('po', accs_po)):
                row['%s_low' % tag] = round(float(np.mean([arr[i] for i in LOW])), 4)
                row['%s_high' % tag] = round(float(np.mean([arr[i] for i in HIGH])), 4)
                row['%s_range' % tag] = round(float(np.ptp(arr)), 4)
            row['gap_low'] = round(row['ws_low'] - row['po_low'], 4)
            row['gap_high'] = round(row['ws_high'] - row['po_high'], 4)
            rec['arms'][key][cls] = row
            if cls == 'hand2':
                print('%-16s %6d | %8.1f %8.1f %8.1f | %8.1f %8.1f  gap低 %+.1f'
                      % (key, Fp.shape[1],
                         row['po_low'] * 100, row['po_high'] * 100,
                         row['po_range'] * 100, row['ws_low'] * 100,
                         row['ws_high'] * 100, row['gap_low'] * 100), flush=True)
    deep = [ref[s]['deep70_acc'] for s in SUBJ]
    rec['deep70_acc'] = {SUBJ[i]: deep[i] for i in range(9)}
    print('\ndeep70: 低四 %.1f 高五 %.1f' % (
        np.mean([deep[i] for i in LOW]) * 100, np.mean([deep[i] for i in HIGH]) * 100))
    (OUT / 'mirror_commonality.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/mirror_commonality.json' % OUT)


if __name__ == '__main__':
    main()
