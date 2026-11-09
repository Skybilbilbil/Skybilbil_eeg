# -*- coding: utf-8 -*-
"""PhysionetMI 适配(零训练部分):数据普查 + 逐人读数噪声地板 + 逐人 CSP+LDA 劈半信度
+ 现有 probe_n103(wd=0) checkpoint 的离线逐人复评。

问题定位(①线个体差异,2026-10-24):把 BCI-2a(9 人,每人每类 144 试次)上那套
「主协议 + 逐人精度分布」读法推广到 PhysionetMI(103 人,每人每类 21-24 试次)时,
适配层有哪些问题、哪些读数还测得准:
  A 普查       人数/试次/类平衡/T/窗数 M(160Hz 自动换算 win=240/step=120)
  B 噪声地板   主协议 samepool 70/15/15(random_state=42, 与训练入口逐位一致)下
               每人 test 仅 ~13 试次 ⇒ 逐人精度的 Wilson 95% CI 半宽;
               与 BCI-2a 的逐人极差(38.5pp)/组内差(6-9pp)对比, 判断单人读数是否还有测量力
  C CSP+LDA    逐人 5 折 CV(与 analyze_individual_diff::csp_acc 同一实现 ⇒ 与 BCI
               csp_within 45-60% 同尺); 同被试试次劈半 A/B 各自 CV 的跨被试相关
               = 21-24 试次/类下「逐人线性可分性」本身的信度(Spearman-Brown 校正)
  D 离线复评   复现确定性切分, 加载 results/probe_n103/pn103_cross_mix_best_acc.pt
               逐人 test 精度 → 103 人分布; 与 C 的 CSP 排序做 Spearman(BCI 上
               deep70×csp_within ρ=0.917, 此处看稀疏+多坐标系下还剩多少)

产物: results/physionet_adapt/physionet_adapt.json(本脚本 stdout 由调用方重定向)。
纯 CPU(推理亦不上 GPU, 避免与并行训练抢卡); 不训练、不改任何共享文件。
"""
import sys, json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, norm
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent          # analysis/
BASE = HERE.parent                               # MAtt_baseline/
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
# fit_csp/csp_acc/cov 与 BCI 的 csp_within 同一实现(复制自 analyze_individual_diff,
# 其余部分带 BCI 专属常量, 整体 import 会拖入无关模块级路径设置, 只借这三个纯函数)
from analyze_individual_diff import fit_csp, csp_acc, cov

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
FS = 160
WIN, STEP = 240, 120          # 1.5s/0.75s @160Hz, 与 train_pair 自动换算一致
TASK = (FS, 10 * FS)          # epoch [-1,3.5]s 中任务期 [0,3.5) → 样本 160:1600→ 实际由 T 截断
CLASS_KEYS = ['aligned_left_epochs_list', 'aligned_right_epochs_list',
              'aligned_feet_epochs_list', 'aligned_hands_epochs_list']
CLASS_NAMES = ['left', 'right', 'feet', 'hands']   # physionet 第 4 类 = 双手(hands), 非 bci 的 tongue
OUT_DIR = BASE / 'results' / 'physionet_adapt'
CKPT_N103 = BASE / 'results' / 'probe_n103' / 'pn103_cross_mix_best_acc.pt'
Z95 = float(norm.ppf(0.975))


def wilson_ci(k, n):
    """Wilson 95% CI (center, half_width) —— 小 n 下比正态近似不塌零。"""
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + Z95 ** 2 / n
    c = (p + Z95 ** 2 / (2 * n)) / d
    h = Z95 * np.sqrt(p * (1 - p) / n + Z95 ** 2 / (4 * n * n)) / d
    return (float(c), float(h))


def census(subject_data, metadata):
    n_sub = len(subject_data)
    T = subject_data[0][0].shape[-1]
    per_sub = []
    for X, y in subject_data:
        cnt = [int((y == c).sum()) for c in range(4)]
        per_sub.append({'n_total': int(len(y)), 'per_class': cnt})
    M = (T - WIN) // STEP + 1
    tot = np.array([p['n_total'] for p in per_sub])
    print(f'[A 普查] n_subjects={n_sub}, T={T}({T/FS:.2f}s@{FS}Hz), M={M} 窗')
    print(f'  每人试次: min={tot.min()} 中位={int(np.median(tot))} max={tot.max()}'
          f' | 全池 {tot.sum()} | 每类每人 '
          f'{min(min(p["per_class"]) for p in per_sub)}-'
          f'{max(max(p["per_class"]) for p in per_sub)}')
    print(f'  subject_metadata[0] 键: {list(metadata[0].keys()) if metadata else "无"}')
    return {'n_subjects': n_sub, 'T': int(T), 'M': int(M),
            'per_subject': per_sub,
            'trial_total': int(tot.sum()),
            'n_per_class_range': [int(min(min(p['per_class']) for p in per_sub)),
                                  int(max(max(p['per_class']) for p in per_sub))]}


def split_floor(subject_data):
    """复现训练入口的混合切分 → 每人 test 样本量与 Wilson 半宽(逐人读数的噪声地板)。"""
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    # 无模型参与: 半宽在预期精度 p=0.35(历史 wd0/wd0.02 读数区间)处评估,
    # 回答「若某人真实精度 35%, 单人一次读数的不确定度有多大」
    rows = []
    for s in np.unique(sub_te):
        m = sub_te == s
        n = int(m.sum())
        _, hw = wilson_ci(int(round(0.35 * n)), n)
        rows.append({'subject': int(s), 'n_test': n, 'wilson_half_width': hw,
                     'per_class_test': [int(((sub_te == s) & (y_te == c)).sum())
                                        for c in range(4)]})
    hw = np.array([r['wilson_half_width'] for r in rows]) * 100
    nt = np.array([r['n_test'] for r in rows])
    print(f'[B 地板] test 总 {len(y_te)} 试次; 每人 n_test min={nt.min()} 中位={int(np.median(nt))}'
          f' max={nt.max()}; Wilson95% 半宽 中位 ±{np.median(hw):.1f}pp (范围 ±{hw.min():.1f}~±{hw.max():.1f}pp)')
    print(f'  对照: BCI-2a 每人 n_test≈86(576×0.15) → 半宽约 ±9~10pp;'
          f' 而 BCI 逐人实测全距 38.5pp、高/低组内仅 6-9pp。')
    return {'n_test_total': int(len(y_te)), 'per_subject': rows,
            'wilson_half_width_median_pp': float(np.median(hw)),
            'wilson_half_width_min_pp': float(hw.min()),
            'wilson_half_width_max_pp': float(hw.max()),
            'bci_reference': {'n_test_per_subject': 86,
                              'note': 'BCI-2a 576 试次×0.15≈86, 解析值非落盘'}}


def csp_within(subject_data):
    """逐人 CSP+LDA 5 折 CV + 试次劈半信度。协方差取任务期 [0,3.5)s。"""
    accs, accA, accB = [], [], []
    rng = np.random.RandomState(0)
    for X, y in subject_data:
        t0, t1 = TASK[0], min(TASK[1], X.shape[-1])
        C = cov(X[:, :, t0:t1].astype(np.float64))
        cv = [csp_acc(C, y, tr, te) for tr, te in
              StratifiedKFold(5, shuffle=True, random_state=0).split(C, y)]
        accs.append(float(np.mean(cv)))
        # 劈半: 逐类随机对半(奇数类丢 1 个到较少用的 B 半), 每半 4 折 CV
        ia, ib = [], []
        for c in np.unique(y):
            idx = rng.permutation(np.where(y == c)[0])
            h = len(idx) // 2
            ia += idx[:h].tolist()
            ib += idx[h:2 * h].tolist()
        for half, out in ((ia, accA), (ib, accB)):
            ia_ = np.array(sorted(half))
            cvh = [csp_acc(C[ia_], y[ia_], tr, te) for tr, te in
                   StratifiedKFold(4, shuffle=True, random_state=0).split(C[ia_], y[ia_])]
            out.append(float(np.mean(cvh)))
    a = np.array(accs) * 100
    rho, pval = spearmanr(accA, accB)
    rho_sb = 2 * rho / (1 + rho) if rho > -1 else float('nan')
    print(f'[C CSP] 逐人 5 折 CV: min={a.min():.1f}% 中位={np.median(a):.1f}% '
          f'max={a.max():.1f}% (4 类随机 25%)  ← 对照 BCI csp_within 45-60%')
    print(f'  劈半信度: Spearman(A,B) ρ={rho:.3f} (p={pval:.1e}), Spearman-Brown 校正 ρ_SB={rho_sb:.3f}')
    return {'acc_by_subject': accs,
            'summary': {'min': float(a.min()), 'median': float(np.median(a)),
                        'max': float(a.max()), 'mean': float(a.mean())},
            'split_half': {'accA': accA, 'accB': accB, 'rho': float(rho),
                           'p': float(pval), 'rho_spearman_brown': float(rho_sb)}}


def eval_n103_persubject(subject_data, csp_accs):
    """确定性切分复现 + probe_n103 best_acc 逐人 test 复评(CPU)。"""
    import torch
    from utils.functions import load_matt_model
    if not CKPT_N103.exists():
        print(f'[D 复评] 跳过: {CKPT_N103} 不存在')
        return None
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    args = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
                local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
    net = load_matt_model(str(CKPT_N103), WIN, STEP, args).cpu().eval()
    x_t = torch.Tensor(X_te).unsqueeze(1)
    with torch.no_grad():
        pred = []
        for i in range(0, len(x_t), 256):
            pred += net(x_t[i:i + 256]).argmax(1).numpy().tolist()
    pred = np.array(pred)
    per = []
    for s in np.unique(sub_te):
        m = sub_te == s
        k, n = int((pred[m] == y_te[m]).sum()), int(m.sum())
        _, hw = wilson_ci(k, n)
        per.append({'subject': int(s), 'n_test': n, 'acc': k / n,
                    'wilson_half_width': hw})
    a = np.array([p['acc'] for p in per]) * 100
    overall = float((pred == y_te).mean())
    rho, pval = spearmanr([p['acc'] for p in per], csp_accs)
    print(f'[D 复评] probe_n103(wd=0,150it) best_acc: 总体 test {overall*100:.2f}% | '
          f'逐人 min={a.min():.1f}% 中位={np.median(a):.1f}% max={a.max():.1f}%')
    hw_med = float(np.median([p['wilson_half_width'] for p in per])) * 100
    print(f'  逐人 × CSP Spearman ρ={rho:.3f} (p={pval:.1e}); 逐人 CI 半宽中位 ±{hw_med:.1f}pp'
          f' ⇒ 单人排序读数被二项噪声大幅衰减(BCI 上同款相关 ρ=0.917)')
    return {'ckpt': str(CKPT_N103), 'overall_test_acc': overall,
            'per_subject': per,
            'spearman_vs_csp': {'rho': float(rho), 'p': float(pval)},
            'wilson_half_width_median_pp': hw_med}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_data, metadata = load_preprocessed_dataset(str(DATA_PATH))
    out = {}
    out['census'] = census(subject_data, metadata)
    out['split_floor'] = split_floor(subject_data)
    out['csp_within'] = csp_within(subject_data)
    out['probe_n103_persubject'] = eval_n103_persubject(
        subject_data, out['csp_within']['acc_by_subject'])
    with open(OUT_DIR / 'physionet_adapt.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'[OK] {OUT_DIR}/physionet_adapt.json')


if __name__ == '__main__':
    main()
