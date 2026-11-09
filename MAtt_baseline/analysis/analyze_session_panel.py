# -*- coding: utf-8 -*-
"""逐 session 面板:把每个被试的 session0 / session1 各自算一套量,供被试内配对检验。

问题:同一个被试两次采集之间**精度会变**(S02 27%(deep)/54%(CSP) → 50%/55%),
那这个变化跟着哪个量一起变?——这才是「找到真正因果」的正确问法:
  在**被试内**比较两个 session,任何**不随 session 变**的东西自动被消掉,
  剩下的差异只可能来自「随 session 变的东西」。

量(全部沿用项目既有的非 EA 口径):
  · mu ERD(手类,brain_hand = 全脑 22 通道 task/rest μ 功率比)—— 与报告同口径
  · IAF / α 功率(ROI 静息)
  · 峰显著度 peak_ratio / 峰锐度 CI(节律干不干净)
  · 高频残差/μ(肌电与宽带噪声代理)
  · 有效秩 rest_eff_rank(协方差谱的平坦度)
  · 切空间 oracle D²(类间/类内,确定性)
  · CSP+LDA 精度(直接读 results/session_gap/session_gap.json,不重算)
  · 深度模型多数投票精度(逐 session)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_spd_separability import tangent                          # noqa: E402
from analyze_causal_deep_reader import ROI_R, ROI_L, ROI_SYM, _CH     # noqa: E402
from analyze_causal_x2 import predict_ok, RUNS, ARGS, WIN, STEP       # noqa: E402
from analyze_session_within_causal import session_index               # noqa: E402
from analyze_why_correlated import periodogram, band_mean             # noqa: E402
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                           # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'session_within_causal'
FS, CUE = 250, 250
BASE_WIN = (0, 250)
TASK = (375, 1126)
SUBJ = [f'S{i+1:02d}' for i in range(9)]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = [np.asarray(d[0], dtype=np.float64) for d in sd]
    y_all = [np.asarray(d[1]) for d in sd]
    sub_orig = np.concatenate([np.full(len(y_all[i]), i) for i in range(9)])
    Xc, yc = np.concatenate(X_all, 0), np.concatenate(y_all, 0)
    ret = split_train_valid_test(Xc, yc, 0.70, 0.15, random_state=42,
                                 others=[sub_orig])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    # 行对齐:用 (被试, 特征和) 唯一键(已在上一脚本验证 576/576 唯一)
    key = {}
    for i in range(len(sub_orig)):
        key.setdefault((int(sub_orig[i]), float(Xc[i].sum())), i)
    rows = np.array([key[(int(sub_te[i]), float(X_te[i].sum()))]
                     for i in range(len(sub_te))])
    assert (yc[rows] == y_te).all(), '行对齐失败'
    print(f'  行对齐校验通过(测试折 {len(y_te)} 试次)')

    nets = [load_matt_model(str(BASE / 'results' / d / f'{t}_cross_mix_best_acc.pt'),
                            WIN, STEP, ARGS) for (d, t) in RUNS.values()]
    ok_te, _ = predict_ok(nets, X_te, y_te)
    gap = json.load(open(BASE / 'results' / 'session_gap' / 'session_gap.json'))

    recs = []
    for s in range(9):
        X, y = X_all[s], y_all[s]
        sess = session_index(y)
        ci_all = np.arange(22)
        ci_roi = np.array([_CH[n] for n in ROI_SYM])
        # 逐 session 的类别归属:left/right 手类(0/1);brain_hand 只用这两类
        m_te = sub_te == s
        off = 0 if s == 0 else sum(len(y_all[j]) for j in range(s))
        loc_all = rows[m_te] - off                      # 该被试测试折试次在 s 内的行号
        y_te_s = y_te[m_te]
        ok_s = ok_te[m_te]
        for k in (0, 1):
            in_sess = sess == k
            mk = in_sess[loc_all] if len(loc_all) else np.zeros(0, bool)
            # --- ERD(手类,全脑):task/rest μ 功率比,逐 session 内按 96 个手类试次 ---
            hand = (y == 0) | (y == 1)
            sel = in_sess & hand
            Xr = X[:, ci_roi, :]
            Xr = Xr - Xr.mean(axis=-1, keepdims=True)
            Ph, f = periodogram(Xr[sel][:, :, BASE_WIN[0]:BASE_WIN[1]])
            Pt, _ = periodogram(Xr[sel][:, :, TASK[0]:TASK[1]])
            iaf_s = None
            # IAF:静息 ROI 谱的 7–13 Hz 峰
            bandsel = (f >= 7) & (f <= 13)
            iaf_s = float(f[bandsel][np.argmax(Ph[:, bandsel].mean(0))])
            mu_r = band_mean(Ph, f, iaf_s - 1.5, iaf_s + 1.5)
            mu_t = band_mean(Pt, f, iaf_s - 1.5, iaf_s + 1.5)
            lo = band_mean(Ph, f, iaf_s - 4.5, iaf_s - 1.5)
            hi = band_mean(Ph, f, iaf_s + 1.5, iaf_s + 4.5)
            side = 0.5 * (lo + hi)
            Ph_all, _ = periodogram(X[sel][:, :, BASE_WIN[0]:BASE_WIN[1]])
            hf = band_mean(Ph_all, f, 20, 45)
            wmu = band_mean(Ph_all, f, iaf_s - 1.5, iaf_s + 1.5)
            # 有效秩(静息 ROI 协方差)
            Xres = Xr[in_sess][:, :, BASE_WIN[0]:BASE_WIN[1]]
            C = np.einsum('nct,ndt->ncd', Xres - Xres.mean(-1, keepdims=True),
                          Xres - Xres.mean(-1, keepdims=True)) / Xres.shape[-1]
            Cm = C.mean(0)
            ev = np.linalg.eigvalsh(Cm)
            ev = ev / ev.sum()
            eff_rank = float(np.exp(-(ev * np.log(np.maximum(ev, 1e-30))).sum()))
            # 切空间 oracle D²(该 session 的任务窗、全 22 通道)
            seg = X[in_sess][:, :, TASK[0]:TASK[1]]
            seg = seg - seg.mean(-1, keepdims=True)
            Cs = np.einsum('nct,ndt->ncd', seg, seg) / seg.shape[-1]
            F = tangent(Cs)
            F = F - F.mean(0, keepdims=True)
            yy = y[in_sess]
            cen = np.stack([F[yy == c].mean(0) for c in range(4)])
            r2 = float(((cen - F.mean(0)) ** 2).sum(1).mean())
            D2 = r2 / float((F ** 2).sum(1).mean())
            rec = {'subject': SUBJ[s], 'session': k,
                   'n_test': int(mk.sum()),
                   'erd_hand_dB': float(10 * np.log10(mu_t / max(mu_r, 1e-30))),
                   'iaf_hz': iaf_s, 'alpha_pow': float(mu_r),
                   'peak_ratio': float(mu_r / max(side, 1e-30)),
                   'ci': float((mu_r - side) / max(mu_r, 1e-30)),
                   'hf_over_mu': float(hf / max(wmu, 1e-30)),
                   'eff_rank': eff_rank, 'oracle_D2': float(D2),
                   'deep_acc': float(ok_s[mk].mean()) if mk.sum() else float('nan'),
                   'csp_acc': float(gap['B_session0_to_1' if k == 0 else
                                         'C_session1_to_0'][s])}
            recs.append(rec)
    print(f"{'被试':>5s}{'s':>3s}{'ERD dB':>9s}{'IAF':>7s}{'峰比':>7s}{'CI':>7s}"
          f"{'有效秩':>8s}{'D²':>8s}{'deep':>8s}{'CSP':>8s}")
    for r in recs:
        print(f"{r['subject']:>5s}{r['session']:>3d}{r['erd_hand_dB']:>+9.3f}"
              f"{r['iaf_hz']:>7.2f}{r['peak_ratio']:>7.2f}{r['ci']:>+7.3f}"
              f"{r['eff_rank']:>8.3f}{r['oracle_D2']:>8.4f}"
              f"{r['deep_acc']*100:>7.1f}%{r['csp_acc']*100:>7.1f}%")
    (OUT / 'session_panel.json').write_text(
        json.dumps(recs, ensure_ascii=False, indent=1))
    print(f'\n总耗时 {time.time()-t0:.1f}s  json -> {OUT}/session_panel.json')


if __name__ == '__main__':
    main()
