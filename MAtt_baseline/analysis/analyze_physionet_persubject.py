# -*- coding: utf-8 -*-
"""PhysionetMI 适配 · 主协议 seed42 逐人复评 + 逐人方差三分解(零训练, CPU)。

输入: results/physionet_mix_s42/pm42_cross_mix_best_{acc,loss}.pt(主协议 wd=0.02 训练)
     + results/physionet_adapt/physionet_adapt.json(CSP 逐人/地板/probe_n103 复评)
切分: 与训练入口逐位一致(split_train_valid_test, random_state=42) ⇒ 测试集共享。

读数(①线个体差异, 把 BCI 上的三分解判据推广到 103 人):
  1 逐人 test 精度分布(min/四分位/max)——只作总体分布读, 单人读数不可用(地板见 adapt)
  2 逐人方差三分解: Var_between(实测) vs mean[p̂(1-p̂)/n](二项) ⇒ 超出部分 = 真个体差(+run 噪声)
  3 逐人 × CSP Spearman(对照 BCI 0.917 / probe_n103 复评 0.537)
  4 两个独立训练 run(pm42 wd=0.02 vs probe_n103 wd=0)的逐人相关 = 「训练 run 间可复现份额」
产物: results/physionet_persubject/physionet_persubject.json
"""
import sys, json
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model
from analyze_physionet_adapt import wilson_ci

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'
WIN, STEP = 240, 120
OUT_DIR = BASE / 'results' / 'physionet_persubject'
RUN = BASE / 'results' / 'physionet_mix_s42'
ADAPT = BASE / 'results' / 'physionet_adapt' / 'physionet_adapt.json'
ARGS = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print(f'test {len(y_te)} trials, {sub_te.max()+1} subjects')

    x_t = torch.Tensor(X_te).unsqueeze(1)
    y_np = np.asarray(y_te)
    per_seed = {}
    for kind in ('best_acc', 'best_loss'):
        ck = RUN / f'pm42_cross_mix_{kind}.pt'
        if not ck.exists():
            sys.exit(f'[错误] 缺 {ck} —— 先完成 physionet_mix_s42 训练')
        net = load_matt_model(str(ck), WIN, STEP, ARGS).cpu().eval()
        pred = []
        with torch.no_grad():
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
        per_seed[kind] = {
            'ckpt': str(ck),
            'overall_test_acc': float((pred == y_np).mean()),
            'per_subject': per,
            'per_class_recall': [float((pred[y_np == c] == c).mean()) for c in range(4)],
        }
        a = np.array([p['acc'] for p in per]) * 100
        print(f'[{kind}] 总体 {per_seed[kind]["overall_test_acc"]*100:.2f}% | '
              f'逐人 min={a.min():.1f} Q1={np.percentile(a,25):.1f} 中位={np.median(a):.1f} '
              f'Q3={np.percentile(a,75):.1f} max={a.max():.1f} | 逐类召回 '
              + ' '.join(f'{v*100:.0f}' for v in per_seed[kind]['per_class_recall']))

    # ---- 逐人方差三分解(best_acc 读数) ----
    accs = np.array([p['acc'] for p in per_seed['best_acc']['per_subject']])
    ns = np.array([p['n_test'] for p in per_seed['best_acc']['per_subject']])
    var_between = float(np.var(accs))
    var_binom = float(np.mean(accs * (1 - accs) / ns))
    excess = var_between - var_binom
    print(f'[三分解] Var_between={var_between*1e4:.1f}pp² | 二项期望={var_binom*1e4:.1f}pp² '
          f'| 超出(真个体差+run噪声)={excess*1e4:.1f}pp² '
          f'⇒ 逐人观测方差的 {excess/var_between*100:.0f}% 不是抽样噪声')

    # ---- 相关读数 ----
    adapt = json.load(open(ADAPT))
    csp = np.array(adapt['csp_within']['acc_by_subject'])
    rho_csp, p_csp = spearmanr(accs, csp)
    print(f'[×CSP] Spearman ρ={rho_csp:.3f} (p={p_csp:.1e}) | 对照: BCI 0.917, probe_n103 复评 0.537')

    n103 = None
    if adapt.get('probe_n103_persubject'):
        n103 = np.array([p['acc'] for p in adapt['probe_n103_persubject']['per_subject']])
        rho_run, p_run = spearmanr(accs, n103)
        print(f'[run间] pm42(wd.02) × pn103(wd0) 逐人 Spearman ρ={rho_run:.3f} (p={p_run:.1e}) '
              f'⇒ 两个独立训练 run 共享的逐人份额')
    else:
        rho_run = None

    out = {
        'run': 'physionet_mix_s42 (主协议 wd=0.02, 100it, samepool 70/15/15, seed42)',
        'test_n': int(len(y_te)),
        **per_seed,
        'variance_decomposition': {
            'var_between_pp2': var_between * 1e4,
            'binomial_expected_pp2': var_binom * 1e4,
            'excess_pp2': excess * 1e4,
            'excess_share': excess / var_between,
            'note': '超出二项的部分 = 真个体差 + 训练run特异; 与劈半地板/三分解判据同框架',
        },
        'spearman_vs_csp': {'rho': float(rho_csp), 'p': float(p_csp)},
        'spearman_run_vs_probe_n103': ({'rho': float(rho_run)} if rho_run is not None else None),
    }
    with open(OUT_DIR / 'physionet_persubject.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'[OK] {OUT_DIR}/physionet_persubject.json')


if __name__ == '__main__':
    main()
