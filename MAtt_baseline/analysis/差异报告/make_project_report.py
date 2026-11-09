# -*- coding: utf-8 -*-
"""生成《项目成果汇报材料：EEG 个体差异与模型改进》Word 文档。

复用 `make_individual_diff_report.py` 的渲染器（中文首行缩进 / 三线表 / 图题 / 零宽断行），
只换输入 markdown 与输出路径。新画 2 张图（逐人精度、逐级归因），其余复用 results 下已有图。
纯 CPU，秒级。产物 `results/汇报材料/`。
"""
import json, shutil
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from docx import Document

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]                      # = MAtt_baseline/
R = BASE / 'results'
OUT = R / '汇报材料'
FIG = OUT / 'figs'
MD = HERE / '项目成果汇报材料.md'
DOCX = OUT / '项目成果汇报材料.docx'

for _f in ('/mnt/c/Windows/Fonts/simhei.ttf', r'C:\Windows\Fonts\simhei.ttf'):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

LOW4 = ['S02', 'S04', 'S05', 'S06']
HIGH5 = ['S01', 'S03', 'S07', 'S08', 'S09']
SUBS = ['S01', 'S02', 'S03', 'S04', 'S05', 'S06', 'S07', 'S08', 'S09']


def load():
    st = json.load(open(R / 'individual_diff' / 'subject_table.json', encoding='utf-8'))
    erd = json.load(open(R / 'erd_decomposition' / 'erd_decomposition.json', encoding='utf-8'))
    # 2026-10-30: 新口径(62.5 Hz)逐级探针优先,缺省回退旧口径
    pa_path = R / 'pipeline_attribution_62.5hz' / 'pipeline_attribution.json'
    if not pa_path.exists():
        pa_path = R / 'pipeline_attribution' / 'pipeline_attribution.json'
    pa = json.load(open(pa_path, encoding='utf-8'))
    vs = json.load(open(R / 'val_selection_audit' / 'val_selection_audit.json', encoding='utf-8'))
    return st, erd, pa, vs


def new_figs(st, erd, pa):
    FIG.mkdir(parents=True, exist_ok=True)
    # ---- 图 9：逐人精度（按值排序）----
    # 2026-10-30: 新口径逐人精度(sep62_eval)优先,缺省回退旧 deep70
    ev_path = R / 'sep62_eval' / 'sep62_eval.json'
    if ev_path.exists():
        ev = json.loads(ev_path.read_text(encoding='utf-8'))['per_subject']
        acc = {s: v['mean'] * 100 for s, v in ev.items()}
        sd = {s: (v['max'] - v['min']) / 2 * 100 for s, v in ev.items()}
        mean_line = 58.8
    else:
        acc = {r['subject']: r['deep70_acc'] * 100 for r in st}
        sd = {r['subject']: r['deep70_std'] * 100 for r in st}
        mean_line = 55.4
    order = sorted(SUBS, key=lambda s: acc[s])
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    cols = ['#c0392b' if s in LOW4 else '#2e6fb7' for s in order]
    ax.bar(order, [acc[s] for s in order], yerr=[sd[s] for s in order],
           color=cols, capsize=3, error_kw=dict(lw=0.8))
    ax.axhline(mean_line, ls='--', lw=1, color='#555555')
    ax.text(0.1, mean_line + 1.0, '9 人混合均值 %.1f%%' % mean_line, fontsize=8, color='#555555')
    ax.set_ylabel('测试精度 (%)', fontsize=9); ax.set_ylim(0, 82)
    ax.tick_params(labelsize=9); ax.grid(alpha=0.3, axis='y')
    for i, s in enumerate(order):
        ax.text(i, acc[s] + sd[s] + 0.8, '%.1f' % acc[s], ha='center', fontsize=7.5)
    ax.set_title('主协议 3-seed 逐被试测试精度（红=低四人，蓝=高五人）', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'fig9_persubject.png'); plt.close(fig)
    # ---- 图 10：逐级归因（逐人探针 vs 共享读出 两个口径并列）----
    levels = ['01_入口_22通道协方差', 'conv1_协方差', 'conv2_协方差',
              'E2R出口_log均值', 'SPDNorm后_log均值', '后端切空间出口']
    names = ['入口协方差\n(22通道)', 'conv1后', 'conv2后\n(22→8)',
             'E2R出口', 'SPDNorm后', '后端切空间\n出口']
    runs = list(pa['runs'].keys())

    def _avg(key):
        return [sum(pa['runs'][s]['trace'][l][key] for s in runs) / len(runs) * 100 for l in levels]
    lo, hi = _avg('low4_full'), _avg('high5_full')
    slo, shi = _avg('shared_low4'), _avg('shared_high5')
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(names, lo, 'o-', color='#c0392b', lw=1.9, label='逐人探针·低四人')
    ax.plot(names, hi, 's-', color='#2e6fb7', lw=1.9, label='逐人探针·高五人')
    ax.plot(names, slo, 'o--', mfc='white', color='#c0392b', lw=1.2, label='共享读出·低四人')
    ax.plot(names, shi, 's--', mfc='white', color='#2e6fb7', lw=1.2, label='共享读出·高五人')
    for x, (a, b) in enumerate(zip(lo, hi)):
        ax.text(x, a - 3.2, '%.1f' % a, ha='center', fontsize=7.2, color='#c0392b')
        ax.text(x, b + 1.4, '%.1f' % b, ha='center', fontsize=7.2, color='#2e6fb7')
    for x, (a, b) in enumerate(zip(slo, shi)):
        ax.text(x, a + 1.4, '%.1f' % a, ha='center', fontsize=6.6, color='#c0392b', alpha=.85)
        ax.text(x, b - 3.0, '%.1f' % b, ha='center', fontsize=6.6, color='#2e6fb7', alpha=.85)
    ax.annotate('共享读出最低点', xy=(2, slo[2]), xytext=(2.1, 28),
                fontsize=7.5, color='#a33', arrowprops=dict(arrowstyle='->', color='#a33', lw=0.9))
    ax.text(3.5, 82, 'E2R 与 SPDNorm 两级逐位相同\n（log 域仿射，对本读出恒等）',
            ha='center', fontsize=7, color='#555')
    ax.set_ylabel('被试内 4 类可分性 (%)', fontsize=9)
    ax.set_ylim(24, 88); ax.tick_params(labelsize=8); ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=7.5, loc='lower right', ncol=2, framealpha=0.9)
    ax.set_title('逐级串联追踪：两级读法（逐人 5 折探针 / 共享读出）', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'fig10_layers.png'); plt.close(fig)
    # ---- 图 11：共享基对比（随机基线按低四/高五各给一条）----
    keys = ['all22', 'conv1_22dir', 'conv1_top8_to_lda', 'subject_pca8']
    labs = ['全 22 维\n(入口)', 'conv1 全部\n22 方向', 'conv1 与判别\n最对齐 8 维', '被试自身\n最优 8 维']
    lo2 = [sum(pa['runs'][s]['basis_comparison'][k]['low4_high5'][0] for s in runs) / len(runs) * 100 for k in keys]
    hi2 = [sum(pa['runs'][s]['basis_comparison'][k]['low4_high5'][1] for s in runs) / len(runs) * 100 for k in keys]
    r8l = sum(pa['runs'][s]['basis_comparison']['random8_low4_mean'] for s in runs) / len(runs) * 100
    r8h = sum(pa['runs'][s]['basis_comparison']['random8_high5_mean'] for s in runs) / len(runs) * 100
    import numpy as np
    x = np.arange(len(keys)); w = 0.36
    fig, ax = plt.subplots(figsize=(6.8, 3.3))
    ax.bar(x - w / 2, lo2, w, color='#c0392b', label='低四人')
    ax.bar(x + w / 2, hi2, w, color='#2e6fb7', label='高五人')
    ax.axhline(r8l, ls='--', lw=1.2, color='#8e2b1f', alpha=.65)
    ax.axhline(r8h, ls='--', lw=1.2, color='#1f4e86', alpha=.65)
    ax.text(len(keys) - 0.55, r8l - 3.4, '同维随机·低四人 %.1f%%' % r8l,
            fontsize=7.5, color='#8e2b1f', ha='right')
    ax.text(len(keys) - 0.55, r8h + 1.2, '同维随机·高五人 %.1f%%' % r8h,
            fontsize=7.5, color='#1f4e86', ha='right')
    ax.set_xticks(x); ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel('被试内可分性 (%)', fontsize=9); ax.set_ylim(0, 82)
    ax.grid(alpha=0.3, axis='y'); ax.legend(fontsize=8, loc='upper right')
    ax.set_title('共享空间基 vs 全维：8 维投影略高于随机、明显低于全 22 维', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'fig11_basis.png'); plt.close(fig)
    return [FIG / n for n in ('fig9_persubject.png', 'fig10_layers.png', 'fig11_basis.png')]


def trial_figs():
    """图 16:试次 vs 个体 —— 数据层三个尺度 + 模型层抽样噪声收缩。"""
    tv = json.load(open(R / 'trial_vs_subject' / 'trial_vs_subject.json', encoding='utf-8'))
    ms = json.load(open(R / 'model_sensitivity' / 'model_sensitivity.json', encoding='utf-8'))
    A = tv['A']
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.2))
    labs = ['个体间距\nbetween', '试次散布\nwithin', '类间信号\n(要读的东西)']
    vals = [A['between_rms'], A['within_rms'], A['class_dist_mean']]
    ax[0].bar(labs, vals, color=['#2e6fb7', '#e08b2e', '#4a934a'], width=0.6)
    for i, v in enumerate(vals):
        ax[0].text(i, v * 1.08, '%.2f' % v, ha='center', fontsize=8)
    ax[0].set_yscale('log'); ax[0].set_ylim(1, 40)
    ax[0].set_ylabel('SPD 仿射不变距离', fontsize=9)
    ax[0].tick_params(labelsize=8); ax[0].grid(alpha=0.3, axis='y')
    ax[0].set_title('① 数据层：个体 >> 试次 >> 类间信号', fontsize=10)
    c, sb = ms['within_curve'], ms['sigma_between']
    ax[1].plot([x['m'] for x in c], [x['sigma_within_mean'] for x in c], 'o-',
               color='#e08b2e', lw=1.8, label='换一批试次：σ_trial(m)')
    ax[1].axhline(sb, ls='--', color='#2e6fb7', lw=1.5,
                  label='换人：σ_between = %.1fpp' % sb)
    ax[1].axhline(sb / 2, ls=':', color='#777777', lw=1)
    ax[1].text(10, sb / 2 + 0.5, 'σ_between/2', fontsize=7.5, color='#666666')
    ax[1].set_xlabel('每人测试试次数 m', fontsize=9)
    ax[1].set_ylabel('逐人精度标准差 (pp)', fontsize=9)
    ax[1].set_ylim(0, 19); ax[1].tick_params(labelsize=8); ax[1].grid(alpha=0.3)
    ax[1].legend(fontsize=7.5)
    ax[1].set_title('② 模型层：试次噪声可压，个体差异不可压', fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / 'fig16_trial_vs_subject.png'); plt.close(fig)
    return FIG / 'fig16_trial_vs_subject.png'


def copy_existing():
    FIG.mkdir(parents=True, exist_ok=True)
    src = R / 'individual_diff' / 'figs'
    pairs = [('fig1_identity.png', 'fig1.png'), ('fig2_psd.png', 'fig2.png'),
             ('fig3_spatial.png', 'fig3.png'), ('fig4_erd.png', 'fig4.png'),
             ('fig5_erd_decomp.png', 'fig5.png'), ('fig6_scatter.png', 'fig6.png'),
             ('fig7_corrheat.png', 'fig7.png'), ('fig8_session.png', 'fig8.png'),
             ('../verify_samepool_70_15_15/pair_training_curves.png', 'fig12_curves.png'),
             ('fig7_corrheat.png', 'fig13_corr.png'),
             ('fig3_spatial.png', 'fig14_spatial.png'),
             ('fig8_session.png', 'fig15_session.png')]
    for s, d in pairs:
        p = (src / s) if not s.startswith('..') else (R / 'verify_samepool_70_15_15' / 'pair_training_curves.png')
        shutil.copyfile(p, FIG / d)
    return [FIG / d for _, d in pairs]


if __name__ == '__main__':
    OUT.mkdir(parents=True, exist_ok=True)
    st, erd, pa, vs = load()
    new_figs(st, erd, pa)
    trial_figs()
    copy_existing()
    print('figs ok:', len(list(FIG.glob('*.png'))))