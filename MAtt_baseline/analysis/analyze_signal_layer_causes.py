# -*- coding: utf-8 -*-
"""被试间精度差异的原因,必须去算法之外找:直接在**原始信号**上量。

## 为什么换口径
"类间分离度 d_b" 是在「22×22 协方差 → log-Euclidean 切空间」里算的,那是
**算法自己的中间表示**。用它解释精度是同义反复(说"模型内部这个数决定了模型输出
那个数")。真正要找的原因必须在输入信号本身。

## 本脚本只做最小加工:带通 + 希尔伯特包络,不算协方差、不进切空间、不用模型
  · erd      : 10log10(任务窗 μ 功率 / 基线窗 μ 功率)      —— 经典 ERD
  · cv_cls   : 四个类别的「对侧 ROI μ 功率」均值之间的相对离散度
               = sd_c(mean_c) / mean —— **这就是"类别依赖的 μ 调制幅度"**
  · cv_elec  : 同上但只用单个对侧电极(不用 ROI 平均)
  · spatial  : 逐通道、逐类的任务窗功率均值矩阵的离散度 —— 空间层面的类别分化
  · m_contra / m_diff : 对侧绝对功率 /（对侧−同侧）偏侧化
  · rest_mu  : 基线窗 μ 绝对功率

## ⚠ 两个已踩过的坑(都写进注释)
  ① 循环变量不要用 `c` —— 本文件里 `ci` 是通道索引,曾把 `ci` 覆盖成类号 ⇒ NaN。
  ② 单电极分支不要用 `mean(axis=1)` —— (144,1,T) 会被压成 (144,),再按时间切片
     就得到空数组,均值是 NaN。统一先取成 (n, T) 再切时间。
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from mAtt.mAtt import BCI2A_CHANNELS                                   # noqa: E402
from analyze_causal_deep_reader import ROI_R, ROI_L, _CH               # noqa: E402
from utils.dataset import load_preprocessed_dataset                    # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'signal_layer_causes'
FS, TASK, BASE_WIN = 250, (375, 1126), (0, 250)
SUBJ = [f'S{i+1:02d}' for i in range(9)]
SOS = butter(4, [8 / (FS / 2), 13 / (FS / 2)], btype='band', output='sos')
LAT = {0: ('C4', 'C3'), 1: ('C3', 'C4')}          # 类号 → (对侧单电极, 同侧单电极)
ROI_L_I = np.array([_CH[n] for n in ROI_L])
ROI_R_I = np.array([_CH[n] for n in ROI_R])


def mu_env(X):
    """(N,C,T) → μ 带包络平方 (N,C,T)。"""
    return np.abs(hilbert(sosfiltfilt(SOS, X, axis=-1), axis=-1)) ** 2


def win_mean(trace_nt):
    """(n, T) → (n,) 任务窗均值。"""
    return trace_nt[:, TASK[0]:TASK[1]].mean(axis=1)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    tbl = json.load(open(BASE / 'results' / 'individual_diff' / 'subject_table.json'))
    acc = np.array([r['deep70_acc'] for r in tbl])
    csp = np.array([r['csp_within'] for r in tbl])

    rows = []
    print(f"{'被试':>5s}{'erd':>8s}{'cv_cls':>9s}{'cv_elec':>9s}{'spatial':>9s}"
          f"{'m_contra':>11s}{'m_diff':>11s}{'rest_mu':>11s}{'精度':>8s}")
    for s in range(9):
        X = np.asarray(sd[s][0], dtype=np.float64)
        y = np.asarray(sd[s][1])
        E = mu_env(X)
        # --- 对侧/同侧 μ 功率(逐类) ---
        # 手类用对侧单电极;脚/舌用镜像 ROI 的左右(它们本无"对侧")
        contra, ipsi = [], []
        for cls in (0, 1):
            cc, ii = LAT[cls]
            contra.append(E[y == cls][:, _CH[cc], :])          # (144, T)
            ipsi.append(E[y == cls][:, _CH[ii], :])
        for cls in (2, 3):
            contra.append(E[y == cls][:, ROI_L_I, :].mean(axis=1))
            ipsi.append(E[y == cls][:, ROI_R_I, :].mean(axis=1))
        m_c = np.array([win_mean(t).mean() for t in contra])     # 每类一个数
        m_i = np.array([win_mean(t).mean() for t in ipsi])
        cv_cls = float(np.std(m_c) / max(m_c.mean(), 1e-30))
        # --- 单电极版(手类对侧单电极;脚/舌用 Cz)---
        e_contra = []
        for cls in (0, 1):
            e_contra.append(E[y == cls][:, _CH[LAT[cls][0]], :])
        for cls in (2, 3):
            e_contra.append(E[y == cls][:, _CH['Cz'], :])
        me = np.array([win_mean(t).mean() for t in e_contra])
        cv_elec = float(np.std(me) / max(me.mean(), 1e-30))
        # --- 经典 ERD(全 22 通道 μ 功率比)---
        rest = float(E[:, :, BASE_WIN[0]:BASE_WIN[1]].mean())
        task = float(E[:, :, TASK[0]:TASK[1]].mean())
        erd = float(10 * np.log10(max(task, 1e-30) / max(rest, 1e-30)))
        # --- 空间层面的类别分化:逐通道类别均值矩阵的跨通道离散度 ---
        seg = E[:, :, TASK[0]:TASK[1]].mean(axis=-1)             # (n, 22)
        cen = np.stack([seg[y == cls].mean(0) for cls in range(4)])   # (4, 22)
        spatial = float(np.std(cen, axis=0).mean() / max(cen.mean(), 1e-30))
        rows.append({'subject': SUBJ[s], 'erd': erd, 'cv_cls': cv_cls,
                     'cv_elec': cv_elec, 'spatial': spatial,
                     'm_contra': float(m_c.mean()),
                     'm_diff': float((m_c - m_i).mean()), 'rest_mu': rest,
                     'deep_acc': float(acc[s]), 'csp_within': float(csp[s])})
        print(f'{SUBJ[s]:>5s}{erd:>+8.3f}{cv_cls:>9.4f}{cv_elec:>9.4f}{spatial:>9.4f}'
              f'{m_c.mean():>11.3e}{(m_c-m_i).mean():>+11.3e}{rest:>11.3e}'
              f'{acc[s]*100:>7.1f}%')

    print('\n' + '=' * 80)
    print('★ 纯信号层的量与精度的关系(n=9,Spearman)')
    print('=' * 80)
    print(f"{'量':>10s}{'范围':>28s}{'极差':>8s}{'ρ(深度模型)':>12s}{'p':>8s}{'ρ(CSP)':>10s}")
    for nm, k in [('erd', 'erd'), ('cv_cls', 'cv_cls'), ('cv_elec', 'cv_elec'),
                  ('spatial', 'spatial'), ('m_contra', 'm_contra'),
                  ('m_diff', 'm_diff'), ('rest_mu', 'rest_mu')]:
        v = np.array([r[k] for r in rows])
        r1, p1 = spearmanr(v, acc)
        r2, _ = spearmanr(v, csp)
        print(f'{nm:>10s}{v.min():>13.4g} ~ {v.max():12.4g}'
              f'{v.max()/max(abs(v.min()), 1e-30):>8.2f}x{r1:>+12.3f}{p1:>8.3f}'
              f'{r2:>+10.3f}')
    (OUT / 'signal_layer_causes.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/signal_layer_causes.json')


if __name__ == '__main__':
    main()
