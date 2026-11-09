# -*- coding: utf-8 -*-
"""降采样 / 降参数对训练过程的影响（纯分析，零训练）。

数据全部来自已保存的 pair_training.json（正式口径 MixOptimizer，100 it，11 个快照）：
  旧 = 250 Hz + 全连接 conv2 : results/verify_samepool_70_15_15 · verify70_seed2024 · verify70_seed7
  新 = 62.5 Hz + 可分离 conv2 : results/sep62_full_s{42,2024,7}
两套同切分、同 seed、同迭代预算，唯一差别就是这两处改动。

分析维度（都是过程量，不是只看最终精度）：
  1) 学习曲线速度：val/test 曲线、达阈值的最早迭代、峰值迭代
  2) 参数范数演化：Conv1核 / Conv2核 / 通道加权 / 分类器W（11 点）
  3) 表示与注意力：silhouette、注意力对角线集中度
  4) 梯度：各模块梯度范数的首末变化

产物 results/conv2_cost/process_analysis/。
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

BASE = Path(__file__).resolve().parents[1]
OUT = BASE / 'results' / 'conv2_cost' / 'process_analysis'
FIG = OUT / 'figs'
for _f in ('/mnt/c/Windows/Fonts/simhei.ttf',):
    if Path(_f).exists():
        font_manager.fontManager.addfont(_f)
        plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 140

OLD = {'42': 'results/verify_samepool_70_15_15/pair_training.json',
       '2024': 'results/verify70_seed2024/pair_training.json',
       '7': 'results/verify70_seed7/pair_training.json'}
NEW = {'42': 'results/sep62_full_s42/pair_training.json',
       '2024': 'results/sep62_full_s2024/pair_training.json',
       '7': 'results/sep62_full_s7/pair_training.json'}
GROUPS = [('Conv1核', 'Conv1 空间核'), ('Conv2核', 'Conv2 时序核'),
          ('通道加权scale', '通道加权'), ('分类器W', '分类器 W')]
LOW4 = ['S02', 'S04', 'S05', 'S06']


def load(src):
    out = {}
    for sd, f in src.items():
        m = json.load(open(BASE / f))['models']['cross_mix']
        out[sd] = m
    return out


def curve(runs, key):
    xs = [np.asarray(r[key], dtype=float) for r in runs.values()]
    n = min(len(x) for x in xs)
    A = np.vstack([x[:n] for x in xs])
    return np.arange(n), A.mean(0), A.std(0)


def first_reach(runs, key, thr):
    out = []
    for r in runs.values():
        a = np.asarray(r[key], dtype=float)
        idx = np.where(a >= thr)[0]
        out.append(int(idx[0]) if len(idx) else len(a))
    return float(np.mean(out))


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    old, new = load(OLD), load(NEW)
    rec = {'note': '旧=250Hz+全连接, 新=62.5Hz+可分离; 3 seed 均值'}

    # ---------- 1) 学习曲线 ----------
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.1))
    for ax, key, lab in zip(axs, ['train_loss', 'val_acc', 'test_acc'],
                            ['训练损失', '验证准确率 (%)', '测试准确率 (%)']):
        for runs, name, c in ((old, '旧 250Hz+全连接', '#2e6fb7'), (new, '新 62.5Hz+可分离', '#c0392b')):
            it, mu, sd = curve(runs, key)
            if key == 'test_acc':        # test 是每 5 it 评估一次, x 轴换算成真实迭代
                it = np.asarray(list(runs.values())[0]['test_iter'])
            y = mu * 100 if key != 'train_loss' else mu
            e = sd * 100 if key != 'train_loss' else sd
            ax.plot(it, y, color=c, lw=1.6, label=name)
            ax.fill_between(it, y - e, y + e, color=c, alpha=0.18)
        ax.set_xlabel('迭代'); ax.set_title(lab, fontsize=10); ax.grid(alpha=0.3)
        if key == 'train_loss':
            ax.legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / 'fig_curves.png'); plt.close(fig)

    # ---------- 2) 参数范数演化 ----------
    fig, axs = plt.subplots(1, 4, figsize=(13, 2.9))
    for ax, (k, lab) in zip(axs, GROUPS):
        for runs, name, c in ((old, '旧', '#2e6fb7'), (new, '新', '#c0392b')):
            vals = np.array([[p.get(k, np.nan) for p in r['params']] for r in runs.values()], dtype=float)
            si = np.arange(vals.shape[1])
            ax.plot(si, vals.mean(0), color=c, lw=1.6, marker='o', ms=2.5, label=name)
        ax.set_title(lab, fontsize=9.5); ax.set_xlabel('快照 (每 10 it)'); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / 'fig_params.png'); plt.close(fig)

    # ---------- 3) silhouette / 注意力 ----------
    fig, axs = plt.subplots(1, 2, figsize=(8.4, 2.9))
    for ax, key, lab in zip(axs, ['silhouette', 'attn_conc'], ['表示 silhouette', '注意力对角线集中度']):
        for runs, name, c in ((old, '旧', '#2e6fb7'), (new, '新', '#c0392b')):
            vals = np.array([r[key] for r in runs.values()], dtype=float)
            ax.plot(np.arange(vals.shape[1]), vals.mean(0), color=c, lw=1.6, marker='o', ms=2.5, label=name)
        ax.set_title(lab, fontsize=9.5); ax.set_xlabel('快照 (每 10 it)'); ax.grid(alpha=0.3)
    axs[0].legend(fontsize=7.5)
    fig.tight_layout(); fig.savefig(FIG / 'fig_repr.png'); plt.close(fig)

    # ---------- 汇总表 ----------
    def summ(runs):
        out = {}
        for key, thr in (('val_acc', 0.55), ('val_acc', 0.60), ('test_acc', 0.55)):
            out[f'first_{key}>={int(thr*100)}'] = first_reach(runs, key, thr)
        fin = lambda k: float(np.mean([r[k][-1] for r in runs.values()])) * 100
        pk = lambda k: float(np.mean([max(r[k]) for r in runs.values()])) * 100
        out['val_final'] = fin('val_acc'); out['val_peak'] = pk('val_acc')
        out['test_final'] = fin('test_acc'); out['test_peak'] = pk('test_acc')
        out['train_loss_final'] = float(np.mean([r['train_loss'][-1] for r in runs.values()]))
        out['silhouette_final'] = float(np.mean([r['silhouette'][-1] for r in runs.values()]))
        out['attn_final'] = float(np.mean([r['attn_conc'][-1] for r in runs.values()]))
        out['params_final'] = {k: float(np.mean([p.get(k, np.nan) for p in [r['params'][-1] for r in runs.values()]])) for k, _ in GROUPS}
        out['params_first'] = {k: float(np.mean([p.get(k, np.nan) for p in [r['params'][0] for r in runs.values()]])) for k, _ in GROUPS}
        out['grad_first'] = {k: float(np.mean([r['grad'][0].get(k, np.nan) for r in runs.values()])) for k, _ in GROUPS}
        out['grad_last'] = {k: float(np.mean([r['grad'][-1].get(k, np.nan) for r in runs.values()])) for k, _ in GROUPS}
        return out

    rec['old'] = summ(old); rec['new'] = summ(new)
    json.dump(rec, open(OUT / 'process_analysis.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)

    print('%-26s %12s %12s' % ('过程量', '旧 250Hz+全连', '新 62.5+可分'))
    rows = [('val 首次≥55% (it)', 'first_val_acc>=55'), ('val 首次≥60% (it)', 'first_val_acc>=60'),
            ('test 首次≥55% (it)', 'first_test_acc>=55'), ('val 最终 (%)', 'val_final'),
            ('val 峰值 (%)', 'val_peak'), ('test 最终 (%)', 'test_final'), ('test 峰值 (%)', 'test_peak'),
            ('train_loss 最终', 'train_loss_final'), ('silhouette 最终', 'silhouette_final'),
            ('attn 集中度 最终', 'attn_final')]
    for lab, k in rows:
        print('%-26s %12.2f %12.2f' % (lab, rec['old'][k], rec['new'][k]))
    print('\n参数范数 首→末')
    for k, lab in GROUPS:
        print('  %-10s 旧 %.3f→%.3f | 新 %.3f→%.3f'
              % (lab, rec['old']['params_first'][k], rec['old']['params_final'][k],
                 rec['new']['params_first'][k], rec['new']['params_final'][k]))
    print('\n梯度范数 首→末')
    for k, lab in GROUPS:
        print('  %-10s 旧 %.4f→%.4f | 新 %.4f→%.4f'
              % (lab, rec['old']['grad_first'][k], rec['old']['grad_last'][k],
                 rec['new']['grad_first'][k], rec['new']['grad_last'][k]))
    print(f'\n[落盘] {OUT}/process_analysis.json + figs/')


if __name__ == '__main__':
    main()
