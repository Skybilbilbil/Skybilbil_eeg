# -*- coding: utf-8 -*-
"""I-4′(整改方案 §4.1b):读 conv2 学到的核,看它的幅频响应是宽带塑形还是选频。

## 背景(论文初稿 §8.1 表 12 与 §11 第 8 条)
L1 算术:conv2 核长 12 点 @250 Hz ⇒ Rayleigh 极限 20.8 Hz,而输入带宽 8–30 Hz 只有
22 Hz 宽 ⇒ 时间-带宽积 1.06,原理上放不下第二个瓣。由此推断(现为 L4)「conv2 学到的
是宽带整体塑形,不是频带选择」。固定带通梯子实验(12/24/96 抽头)从外侧支持它,
但**从未直接读过学到的核**。本脚本把这个 L4 升成 L2。

## 数据与做法
主协议 3 seed 的 cross_mix best_acc ckpt(verify_samepool_70_15_15 / verify70_seed2024 /
verify70_seed7),conv2.weight 形状 (8, 22, 1, 12):176 个 12 抽头滤波器/seed。
对每个核 h 做 256 点 DFT,只看 8–30 Hz(输入已带限,带外无能量):
  - 带内响应起伏系数:cv = std_f |H(f)|² / mean_f |H(f)|², f∈[8,30]
  - μ vs β 选择性:sel = |log10( mean|H|²[8,13] / mean|H|²[13,30] )|
对照 = 同长度 iid 高斯随机核(1000 次抽样)。理想 3 Hz μ 选择器(不可实现,只作上限参照)。
判读:学到的核的 cv / sel 分布若与随机核无本质差别(且远低于理想选择器),
则「宽带塑形而非选频」由直接测量确立。

只写 results/conv2_spectrum/。秒级,纯 CPU。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / 'results' / 'conv2_spectrum'
CKPTS = [
    ('seed42', BASE / 'results' / 'verify_samepool_70_15_15' / 'verify70_cross_mix_best_acc.pt'),
    ('seed2024', BASE / 'results' / 'verify70_seed2024' / 'v70s2024_cross_mix_best_acc.pt'),
    ('seed7', BASE / 'results' / 'verify70_seed7' / 'v70s7_cross_mix_best_acc.pt'),
]
FS = 250
F = np.fft.rfftfreq(256, 1 / FS)                    # 0..125 Hz, 1 Hz 步
IN_BAND = (F >= 8) & (F <= 30)
MU = (F >= 8) & (F <= 13)
BETA = (F >= 13) & (F <= 30)


def kernel_metrics(h):
    H2 = np.abs(np.fft.rfft(h, 256)) ** 2           # |H(f)|²
    w = H2[IN_BAND]
    mu, beta = H2[MU].mean(), H2[BETA].mean()
    return (float(w.std() / w.mean()),
            float(abs(np.log10(max(mu, 1e-12) / max(beta, 1e-12)))))


def ideal_mu_selector_metrics():
    """理想 3 Hz μ 通带(矩形频响 → 时域 sinc,截断到 12 抽头)——不可实现的上限参照。"""
    t = (np.arange(12) - 5.5) / FS
    h = np.sinc(2 * 1.5 * t) * np.cos(2 * np.pi * 10.5 * t)
    return kernel_metrics(h)


def main():
    import torch
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {'purpose': 'I-4prime: read learned conv2 kernels, test broadband-shaping vs '
                      'band-selection (paper §8.1 L4 -> L2)',
           'kernels_per_seed': 176, 'fs': FS, 'band': '8-30 Hz', 'mu': '8-13', 'beta': '13-30'}
    learned = {}
    for tag, path in CKPTS:
        sd = torch.load(path, map_location='cpu')
        W = sd['conv2.weight'].float().numpy()      # (8,22,1,12)
        ks = W.reshape(-1, W.shape[-1])
        m = np.array([kernel_metrics(k) for k in ks])
        learned[tag] = m
        rec['per_seed_%s' % tag] = {
            'cv_inband_mean': round(float(m[:, 0].mean()), 4),
            'cv_inband_p95': round(float(np.percentile(m[:, 0], 95)), 4),
            'sel_mu_beta_mean': round(float(m[:, 1].mean()), 4),
            'sel_mu_beta_p95': round(float(np.percentile(m[:, 1], 95)), 4)}
        print('%s: 带内起伏 cv 均值 %.3f (p95 %.3f)  μ/β 选择性均值 %.3f (p95 %.3f)'
              % (tag, m[:, 0].mean(), np.percentile(m[:, 0], 95),
                 m[:, 1].mean(), np.percentile(m[:, 1], 95)))

    rng = np.random.default_rng(42)
    rnd = np.array([kernel_metrics(rng.normal(size=12)) for _ in range(2000)])
    ide = ideal_mu_selector_metrics()
    allr = np.concatenate(list(learned.values()))
    rec['random_kernels'] = {
        'cv_inband_mean': round(float(rnd[:, 0].mean()), 4),
        'cv_inband_p95': round(float(np.percentile(rnd[:, 0], 95)), 4),
        'sel_mu_beta_mean': round(float(rnd[:, 1].mean()), 4),
        'sel_mu_beta_p95': round(float(np.percentile(rnd[:, 1], 95)), 4)}
    rec['ideal_3hz_selector_truncated12'] = {'cv_inband': round(ide[0], 4),
                                             'sel_mu_beta': round(ide[1], 4)}
    rec['learned_all_seeds'] = {
        'cv_inband_mean': round(float(allr[:, 0].mean()), 4),
        'sel_mu_beta_mean': round(float(allr[:, 1].mean()), 4),
        'frac_kernels_sel_gt_random_p95': round(
            float((allr[:, 1] > np.percentile(rnd[:, 1], 95)).mean()), 4)}

    print('\n随机 12 抽头核(2000 次): cv 均值 %.3f (p95 %.3f)  sel 均值 %.3f (p95 %.3f)'
          % (rnd[:, 0].mean(), np.percentile(rnd[:, 0], 95),
             rnd[:, 1].mean(), np.percentile(rnd[:, 1], 95)))
    print('理想 3Hz 选择器(截断到 12 抽头): cv %.3f  sel %.3f' % ide)
    print('学到核 sel > 随机 p95 的比例: %.1f%%'
          % (100 * rec['learned_all_seeds']['frac_kernels_sel_gt_random_p95']))
    print('\n判读:学到的核若与随机核同分布(cv/sel 均接近)且远低于理想选择器,')
    print('     则 conv2 = 宽带塑形而非选频,由 L4 升 L2。')

    (OUT / 'conv2_spectrum.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/conv2_spectrum.json' % OUT)


if __name__ == '__main__':
    main()
