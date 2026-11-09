# -*- coding: utf-8 -*-
"""PhysionetMI 适配 · 零训练通道挑选:用 62 通道信息选一套 ~20 通道(2026-10-25,适配八)。

用户协议(2026-10-25 二次/三次澄清):
  ① 模型输入 ~20 通道即够,62/64 通道信息只用于不消耗模型性能的环节;
  ② 共享预训练可用全量数据;
  ③ 逐人实验只做可分性强的被试(top-30),但**管线不得利用"测试被试是谁"的信息**
     ⇒ 通道排名/子集取舍规则只允许用**全量 103 人训练折**(与共享训练同可见度),
     top-30 只作为报告 cohort;test 折全程不参与。

做法(全零训练,CPU):
  1) 排名信号:全量 103 人训练折,逐被试逐通道 μ/β(8-30Hz) 任务期 log 功率的
     4 类 one-vs-rest F 分,被试内 z 化后平均 ⇒ 共享通道排名(不含 cohort 信息);
  2) 候选子集(全部由全量排名或解剖先验导出): bci22 参照 / univariate top20 /
     top20 镜像对称化 / 中央区解剖集 / 全 62;
  3) 报告:每个子集在 full103 与 top30 两个 cohort 的训练折内 5 折 CSP+LDA CV。
     部署取哪个子集由全量口径决定,top30 列只作确认读数。

产物: results/physionet_chsel/physionet_chsel.json(stdout 重定向由调用方带)。
"""
import sys, json
from pathlib import Path

import numpy as np
from scipy.stats import f_oneway

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from analyze_individual_diff import fit_csp, cov

P62 = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica_62ch.pkl'
P22 = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
FS, T0, T1 = 160, 160, 721          # 任务窗 [1, 4.5)s
BAND = (8.0, 30.0)
TOP30 = json.load(open(BASE / 'results' / 'physionet_top30' / 'subjects.json'))['top30']


def band_logpower(X):
    """X (n,ch,T) → (n,ch) 8-30Hz log 功率。"""
    f = np.fft.rfftfreq(X.shape[-1], d=1.0 / FS)
    m = (f >= BAND[0]) & (f <= BAND[1])
    P = (np.abs(np.fft.rfft(X, axis=-1))[:, :, m] ** 2).sum(-1)
    return np.log(P + 1e-12)


def mirror(name):
    """10-10 系统 L↔R 镜像:C3↔C4, FC5↔FC6, Cz↔Cz。"""
    i = len(name) - 1
    while i >= 0 and name[i].isdigit():
        i -= 1
    pre, num = name[:i + 1], name[i + 1:]
    if not num or num == 'z':
        return name
    n = int(num)
    return f"{pre}{n + 1 if n % 2 else n - 1}"


def main():
    data, meta = load_preprocessed_dataset(str(P62))
    names62 = meta[0]['channel_names']
    # bci22 参照名单直接读 22ch pkl 的元数据(不加载其数据数组)
    import pickle
    with open(P22, 'rb') as fh:
        names22 = pickle.load(fh)['subject_metadata'][0]['channel_names']
    X = np.concatenate([sd[0] for sd in data], axis=0)
    y = np.concatenate([sd[1] for sd in data], axis=0)
    sub = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64) for i, sd in enumerate(data)])
    X_tr, y_tr, _, _, _, _, sub_tr, _, _ = split_train_valid_test(
        X, y, train_ratio=0.70, valid_ratio=0.15, random_state=42, others=[sub])
    print(f'训练折全量 {len(y_tr)} 试次 / {len(np.unique(sub_tr))} 人; 通道数 {len(names62)}')

    # ---- 1) 共享通道排名:只用全量 103 训练折(与共享训练同可见度,无 cohort 信息) ----
    F = np.zeros(len(names62))
    for s in np.unique(sub_tr):
        m = sub_tr == s
        P = band_logpower(X_tr[m][:, :, T0:T1])
        # 4 类 one-vs-rest: 每类 F 分取均值(f_oneway 沿 axis=0 逐通道)
        fs = np.mean([f_oneway(P[y_tr[m] == c], P[y_tr[m] != c]).statistic
                      for c in range(4)], axis=0)
        F += (fs - fs.mean()) / (fs.std() + 1e-12)
    order = np.argsort(-F)
    print('通道排名(全量训练折, 共享 z-F, 前 25):')
    for j in order[:25]:
        print(f'  {names62[j]:>4s} {F[j]:+.2f}')

    top20 = [names62[j] for j in order[:20]]
    sym = list(dict.fromkeys([x for n in top20 for x in (n, mirror(n))]))[:20]
    # 通道名归一化(62ch 名带 BrainVision 点占位: 'C3..'/'Cp3.'; 22ch 名格式不同)
    norm = lambda s: s.lower().rstrip('.')
    nmap = {norm(n): i for i, n in enumerate(names62)}
    n22 = [norm(x) for x in names22 if norm(x) in nmap]
    print(f'bci22 参照: 22ch 名单 {len(names22)} 个,在 62ch 中命中 {len(n22)} 个')
    central = [n for n in nmap if n.startswith('fc') or n.startswith('c')]
    subsets_n = {
        f'bci{len(n22)}': sorted(n22),
        'top20': [norm(c) for c in top20],
        'top20_mirror': [norm(c) for c in sym],
        f'central{len(central)}': sorted(central),
        'full62': sorted(nmap),
    }
    # 通道集合相同的子集只评估一次(镜像集常与 top20 重合)
    aliases, uniq = {}, {}
    for k, chs in subsets_n.items():
        key = tuple(chs)
        if key in uniq:
            aliases[k] = uniq[key]
            print(f'子集 {k}: 与 {aliases[k]} 相同,跳过重复评估')
        else:
            uniq[key] = k
            aliases[k] = k
    for k, v in subsets_n.items():
        if aliases[k] == k:
            print(f'子集 {k}: n={len(v)} | {",".join(v)}')

    # ---- 2) 报告:full103 与 top30 两个 cohort 的训练折内 5 折 CSP+LDA CV ----
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    from sklearn.model_selection import StratifiedKFold
    cohorts = {'full103': (X_tr, y_tr, sub_tr)}
    keep = np.isin(sub_tr + 1, TOP30)
    cohorts['top30'] = (X_tr[keep], y_tr[keep], sub_tr[keep])
    res = {}
    for k, chs in subsets_n.items():
        src = aliases[k]
        if src in res and src != k:
            res[k] = res[src]
            for cname in cohorts:
                v = res[src][cname]
                print(f"[{k}] 同 {src} @{cname}: mean {v['cv_mean']:.2f}")
            continue
        idx = [nmap[c] for c in chs]
        res[k] = {'n_ch': len(chs), 'channels': chs}
        for cname, (Xc, yc, sc) in cohorts.items():
            accs = []
            for s in np.unique(sc):
                m = sc == s
                Xs, ys = Xc[m][:, idx][:, :, T0:T1].astype(np.float64), yc[m]
                if len(np.unique(ys)) < 4:
                    continue
                a = []
                for tr_i, te_i in StratifiedKFold(5, shuffle=True, random_state=42).split(Xs, ys):
                    W = fit_csp(cov(Xs[tr_i]), ys[tr_i])
                    f_tr = np.log(np.maximum(np.einsum('ia,nij,ja->na', W, cov(Xs[tr_i]), W), 1e-12))
                    f_te = np.log(np.maximum(np.einsum('ia,nij,ja->na', W, cov(Xs[te_i]), W), 1e-12))
                    a.append(LDA(solver='lsqr', shrinkage='auto').fit(f_tr, ys[tr_i]).score(f_te, ys[te_i]))
                accs.append(np.mean(a))
            accs = np.array(accs) * 100
            res[k][cname] = {'cv_mean': float(accs.mean()), 'cv_median': float(np.median(accs)),
                             'cv_min': float(accs.min()), 'cv_max': float(accs.max()),
                             'per_subject': [float(a) for a in accs]}
            print(f"[{k}] n={len(chs):2d} @{cname}: mean {accs.mean():.2f} | median {np.median(accs):.2f}"
                  f" | range {accs.min():.1f}-{accs.max():.1f}")

    out = {'protocol': 'ranking/selection on full-103 train folds only (same visibility as shared '
                       'training); top30 is reporting cohort; test folds untouched; zero training',
           'band': BAND, 'ranking': [(names62[j], float(F[j])) for j in order],
           'subsets': res}
    od = BASE / 'results' / 'physionet_chsel'
    od.mkdir(parents=True, exist_ok=True)
    with open(od / 'physionet_chsel.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'[OK] {od}/physionet_chsel.json')


if __name__ == '__main__':
    main()
