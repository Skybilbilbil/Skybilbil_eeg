# -*- coding: utf-8 -*-
"""PhysionetMI 筛选部署协议(用户 2026-10-25 指定):好个体 + 好片段上的分类任务。

协议(全部 label-blind 于测试折):
  1 个体筛选: 训练折 CSP 5 折 CV 上四分位 n≈25(与全量版名单对照报告重叠)
  2 片段门控: 标签盲 ERD 门(去同步更强的 keep_frac 试次)+ MAD>3 离群剔除,
    train/test 折同规则分别过滤(测试试次过滤合法,因门不使用标签)
  3 分类: 最佳部署配方 = 全 103 池预训练 ckpt + 筛后训练折逐人微调(150it);
    参照臂 = 筛后池从零共享重训 + 逐人微调 / 同切分同门控 CSP+LDA
  4 ①线锚点: 逐人平均 ERD 分数 × 筛选分数 Spearman(BCI 上 μ ERD 深度×精度 ρ=−0.950
    的 Physionet 对应物,个体层)
读数口径: **筛选部署协议,不报作 103 人主协议**。产物 results/physionet_screened/。
"""
import sys, json, os, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model
from mAtt.mAtt import _SparseAttentionBase
from analyze_individual_diff import fit_csp
from physionet_screen_lib import (subject_train_csp_cv, gate_mask, trial_erd_scores,
                                  _cov, TASK)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

WIN, STEP = 240, 120
OUT = BASE / 'results' / 'physionet_screened'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ARGS = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
CH = os.environ.get('CH', '62')
PKL = {'22': BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl',
       '62': BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica_62ch.pkl'}[CH]
NCH = int(CH)
POOL_CKPT = {'22': BASE / 'results' / 'physionet_mix_s42' / 'pm42_cross_mix_best_acc.pt',
             '62': BASE / 'results' / 'physionet_62ch_s42' / 'p62_cross_mix_best_acc.pt'}[CH]
LR, ITERS, WD = 1e-4, 150, 0.02
KEEP_FRACS = (0.5, 0.7)
TAG = os.environ.get('TAG', '')


def build_net():
    if NCH == 22:
        return load_matt_model(str(POOL_CKPT), WIN, STEP, ARGS).to(DEVICE)
    return _SparseAttentionBase(
        WIN, STEP, n_classes=4, d_attn_in=8, d_attn_out=8,
        local_radius=6, anchor_stride=None, n_heads=1, n_windows=None,
        use_hann=True, use_channel_weight=False, n_spatial_filters=22,
        conv1_kernel=NCH, dropout=0.3, norm_type='in').to(DEVICE)


def finetune_eval(net, state, tr, te):
    torch.manual_seed(42)
    net.load_state_dict(state)
    for p in net.parameters():
        p.requires_grad = True
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss()
    Xtr = torch.Tensor(tr[0]).unsqueeze(1).to(DEVICE)
    ytr = torch.Tensor(tr[1]).long().to(DEVICE)

    def acc(X, y):
        net.eval()
        with torch.no_grad():
            pred = net(torch.Tensor(X).unsqueeze(1).to(DEVICE)).argmax(1).cpu().numpy()
        net.train()
        return float((pred == y).mean()), pred
    net.train()
    for _ in range(ITERS):
        opt.zero_grad()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        loss = ce(net(Xtr), ytr)
        loss.backward()
        opt.step()
    return acc(*te)[0]


def csp_eval(tr, te):
    C_tr = _cov(tr[0][:, :, TASK[0]:TASK[1]].astype(np.float64))
    C_te = _cov(te[0][:, :, TASK[0]:TASK[1]].astype(np.float64))
    W = fit_csp(C_tr, tr[1])
    F = lambda Cs: np.log(np.maximum(np.einsum('ia,nij,ja->na', W, Cs, W), 1e-12))
    return float(LDA(solver='lsqr', shrinkage='auto')
                .fit(F(C_tr), tr[1]).score(F(C_te), te[1]))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sd, _ = load_preprocessed_dataset(str(PKL))
    X_all = np.concatenate([s[0] for s in sd], axis=0)
    y_all = np.concatenate([s[1] for s in sd], axis=0)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(sd)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, np.arange(len(y_all))])
    X_tr, y_tr, X_va, y_va, X_te, y_te = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
    sub_tr, sub_va, sub_te = ret[6], ret[7], ret[8]
    idx_tr = ret[9]

    # 1) 个体筛选(训练折 CSP CV)
    t0 = time.time()
    screen = subject_train_csp_cv(sd, sub_tr, y_tr, idx_tr)
    order = np.argsort(screen)[::-1]
    n_strong = 103 // 4
    strong_ids = sorted(int(i) + 1 for i in order[:n_strong])
    old = set(json.load(open(BASE / 'results' / 'physionet_strong' / 'subjects.json'))
              ['strong_top_quartile'])
    overlap = len(old & set(strong_ids))
    print(f'[筛选] 训练侧 CSP CV 上四分位 n={n_strong}, 与全量版名单重叠 {overlap}/{n_strong}'
          f' ({time.time()-t0:.0f}s)')

    # ①线锚点: 逐人平均 ERD 分数 × 筛选分数
    erd_mean = [float(np.mean(trial_erd_scores(s[0]))) for s in sd]
    rho_erd, p_erd = spearmanr(erd_mean, screen)
    print(f'[①锚点] 逐人平均 logdet-ERD × 训练侧CSP Spearman ρ={rho_erd:.3f} (p={p_erd:.1e})'
          f'  ← BCI 对应物: μ ERD 深度×精度 ρ=−0.950')

    net = build_net()
    if NCH == 62:   # 22ch 的 load_matt_model 已载权重; 62ch 需显式载入(同时物化 LazyLinear)
        net.load_state_dict(torch.load(str(POOL_CKPT), map_location='cpu'))
    state = {k: v.clone() for k, v in net.state_dict().items()}
    strong = set(strong_ids)
    res = {'ch': NCH, 'strong_ids': strong_ids, 'screen_scores': screen,
           'screen_overlap_with_alltrial_list': overlap,
           'anchor': {'rho_erd_vs_screen': float(rho_erd), 'p': float(p_erd)},
           'keep_fracs': {}}

    # 2)+3) 逐 keep_frac 门控 → 各臂(每人对自己的分布定阈值, train/test 各自过滤)
    for kf in KEEP_FRACS:
        rows = {'ft': [], 'csp': [], 'n_tr': [], 'n_te': []}
        pooled_correct, pooled_n = 0.0, 0
        for s in sorted(strong):
            mtr, mte = (sub_tr == s), (sub_te == s)
            Xs_tr, ys_tr = X_tr[mtr], y_tr[mtr]
            Xs_te, ys_te = X_te[mte], y_te[mte]
            ktr, _ = gate_mask(Xs_tr, kf)
            kte, _ = gate_mask(Xs_te, kf)   # 测试折同规则(标签盲)
            tr = (Xs_tr[ktr], ys_tr[ktr])
            te = (Xs_te[kte], ys_te[kte])
            rows['n_tr'].append(int(ktr.sum()))
            rows['n_te'].append(int(kte.sum()))
            pooled_n += len(te[1])
            a = finetune_eval(net, state, tr, te)
            pooled_correct += a * len(te[1])
            rows['ft'].append(a)
            rows['csp'].append(csp_eval(tr, te))
        pooled = pooled_correct / max(pooled_n, 1)
        macro_ft, macro_csp = float(np.mean(rows['ft'])), float(np.mean(rows['csp']))
        print(f'[keep={kf}] 筛选部署(池ckpt+微调{ITERS}it): pooled {pooled*100:.2f}% | '
              f'macro {macro_ft*100:.1f}% | CSP 同门控 {macro_csp*100:.1f}% | '
              f'人均 train {np.mean(rows["n_tr"]):.0f} / test {np.mean(rows["n_te"]):.0f} 试次')
        res['keep_fracs'][str(kf)] = {'pooled_ft': pooled, 'macro_ft': macro_ft,
                                      'macro_csp': macro_csp, **rows}
    json.dump(res, open(OUT / f'screened_{CH}ch{TAG}.json', 'w'), indent=2,
              ensure_ascii=False)
    print(f'[OK] {OUT}/screened_{CH}ch{TAG}.json')


if __name__ == '__main__':
    main()
