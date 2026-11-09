# -*- coding: utf-8 -*-
"""PhysionetMI 适配 · 共享预训练 + 逐人适配阶梯(主协议内, seed42, GPU)。

动机: 41.0% 是把 BCI 超参原样搬到 103 人的共享模型读数(「移植」), 不是「适配」。
人数多×单人试次少的设定的标准适配 = 共享预训练 + 逐人微调(Köllød 2023 同款迁移式,
64ch 文献读数 46.5%被试内 / 53.8%迁移)。本脚本在同一切分上量化四级阶梯:

  A 共享零适配     —— 已有读数 40.98%(physionet_persubject.json, 本脚本不重跑, 只引用)
  B 逐人只调读出头 —— 冻结除 linear 外全部参数, 每人用自己的训练折(约63试次)重拟合分类头
  C 逐人全参微调   —— 从 pm42 best_acc 出发, lr=1e-4, 50it, 每人独立
  D 逐人 CSP+LDA   —— 同切分线性参照(train 折拟合, test 折评估; 22ch 上的「10 秒基线」)

口径: samepool 70/15/15 确定性切分(random_state=42)与训练 run 完全共享测试集;
      逐人 seed=42; 选择规则双报(final 与 best-val), headline=best-val(与主协议一致)。
      ⚠ 逐人 val 仅 ~14 试次 ⇒ best-val 选择本身 ±噪声大, 双报以示诚实。
产物: results/physionet_finetune/physionet_finetune.json
不改任何共享文件; 不触碰已判死方向(本人训练折标签微调, 非 LOSO/域适配)。
"""
import sys, json, time, os
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
from mAtt.optimizer import MixOptimizer
from analyze_individual_diff import fit_csp, cov

DATA_PATH = Path(os.environ.get(
    'FT_PKL', BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica.pkl'))
CKPT = Path(os.environ.get(
    'FT_CKPT', BASE / 'results' / 'physionet_mix_s42' / 'pm42_cross_mix_best_acc.pt'))
ARCH = os.environ.get('FT_ARCH', '22ch')   # '22ch'=mAtt_bci 主协议 | '62ch'=全导联变体
OUT_DIR = BASE / 'results' / os.environ.get(
    'FT_OUT', 'physionet_finetune' if ARCH == '22ch' else 'physionet_finetune_62ch')
# 强组等子集协议: 逗号分隔 1-based 被试号(与训练侧同款, 切分后掩码过滤)
SUBJECTS = ([int(x) for x in os.environ['FT_SUBJECTS'].split(',') if x.strip()]
            if os.environ.get('FT_SUBJECTS') else [])
# 2026-11-08 新增 env 门控（默认值 = 历史行为逐位不变）：
#   FT_RESAMPLE : 160/FT_RESAMPLE = 目标采样率（2→80Hz，与 pm80 ckpt 对齐）
#   FT_SEPARABLE: 1 = conv2 深度可分离（2026-10-30 起的模型默认；旧 ckpt 必须为 0）
#   FT_MIXOPT   : 1 = 用 MixOptimizer 包住 Adam（保住 StiefelParameter 正交性）
RESAMPLE = int(os.environ.get('FT_RESAMPLE', '1'))
SEPARABLE = os.environ.get('FT_SEPARABLE', '0') == '1'
MIXOPT = os.environ.get('FT_MIXOPT', '0') == '1'
ARGS = dict(dataset='physionetmi', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=SEPARABLE)
WIN, STEP, FS = int(1.5 * 160 / RESAMPLE), int(0.75 * 160 / RESAMPLE), 160 // RESAMPLE
N_CH = 62   # FT_ARCH='62ch' 时的通道数(64 EEG − Fp1/Fp2)


def build_net():
    """22ch → 主协议 mAtt_bci; 62ch → _SparseAttentionBase(conv1_kernel=62, 无通道加权)。"""
    if ARCH != '62ch':
        return load_matt_model(str(CKPT), WIN, STEP, ARGS).to(DEVICE)
    from mAtt.mAtt import _SparseAttentionBase
    return _SparseAttentionBase(
        WIN, STEP, n_classes=4, d_attn_in=int(os.environ.get('DATTN', '8')), d_attn_out=8,
        local_radius=6, anchor_stride=None, n_heads=1, n_windows=None,
        use_hann=True, use_channel_weight=False,
        n_spatial_filters=int(os.environ.get('FT_NSPAT', 22)),
        conv1_kernel=N_CH, dropout=0.3, norm_type='in').to(DEVICE)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 预算扫描用 env 覆盖(默认 = 主读数配置);FT_HEAD=0 跳过调头臂, FT_TAG 区分输出名
LR_FULL = float(os.environ.get('FT_LR', 1e-4))
LR_HEAD, ITERS, WD = 5e-4, int(os.environ.get('FT_ITERS', 50)), 0.02
RUN_HEAD = os.environ.get('FT_HEAD', '1') != '0'
TAG = os.environ.get('FT_TAG', '')
TASK = (FS, 10 * FS)   # 任务期 [0,3.5)s → 样本 160:721


def make_sub_data(X_tr, y_tr, X_va, y_va, X_te, y_te, sub_tr, sub_va, sub_te):
    keep = set(SUBJECTS)
    for s in np.unique(sub_te):
        if keep and (s + 1) not in keep:
            continue
        m_tr, m_va, m_te = sub_tr == s, sub_va == s, sub_te == s
        yield s, (X_tr[m_tr], y_tr[m_tr]), (X_va[m_va], y_va[m_va]), (X_te[m_te], y_te[m_te])


def eval_acc(net, X, y):
    net.eval()
    with torch.no_grad():
        xb = torch.Tensor(X).unsqueeze(1).to(DEVICE)
        pred = net(xb).argmax(1).cpu().numpy()
    net.train()
    return float((pred == y).mean()), pred


def finetune(net, shared_state, tr, va, te, head_only):
    """从共享权重出发逐人微调; 返回 (final, best_val 两口径的 test 精度)。"""
    torch.manual_seed(42)
    net.load_state_dict(shared_state)
    for p in net.parameters():
        p.requires_grad = True
    if head_only:
        for n, p in net.named_parameters():
            if not n.startswith('linear.'):
                p.requires_grad = False
    trainable = [p for p in net.parameters() if p.requires_grad]
    base_opt = torch.optim.Adam(trainable, lr=LR_HEAD if head_only else LR_FULL)
    opt = MixOptimizer(base_opt) if MIXOPT else base_opt
    ce = nn.CrossEntropyLoss()
    Xtr = torch.Tensor(tr[0]).unsqueeze(1).to(DEVICE)
    ytr = torch.Tensor(tr[1]).long().to(DEVICE)
    net.train()
    best_val, best_test, final_test = -1.0, float('nan'), float('nan')
    for ite in range(ITERS):
        opt.zero_grad()
        out = net(Xtr)
        loss = ce(out, ytr)
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
        opt.step()
        va_acc, _ = eval_acc(net, *va)
        te_acc, _ = eval_acc(net, *te)
        final_test = te_acc
        if va_acc > best_val:
            best_val, best_test = va_acc, te_acc
    return final_test, best_test


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    if RESAMPLE > 1:
        from scipy.signal import resample_poly
        subject_data = [(resample_poly(np.asarray(X, np.float32), 1, RESAMPLE, axis=-1)
                         .astype(np.float32), y) for X, y in subject_data]
        print(f'[重采样] 160 -> {FS} Hz（1/{RESAMPLE}），WIN={WIN} STEP={STEP}', flush=True)
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_tr, y_tr, X_va, y_va, X_te, y_te = ret[0], ret[1], ret[2], ret[3], ret[4], ret[5]
    sub_tr, sub_va, sub_te = ret[6], ret[7], ret[8]

    net = build_net()
    net.load_state_dict(torch.load(str(CKPT), map_location='cpu'))
    shared_state = {k: v.clone() for k, v in net.state_dict().items()}

    rows = []
    t0 = time.time()
    for s, tr, va, te in make_sub_data(X_tr, y_tr, X_va, y_va, X_te, y_te, sub_tr, sub_va, sub_te):
        row = {'subject': int(s), 'n_train': int(len(tr[1])), 'n_test': int(len(te[1]))}
        # B / C
        arms = (('head_final', True), ('full_final', False)) if RUN_HEAD \
            else (('full_final', False),)
        for arm, head_only in arms:
            fin, best = finetune(net, shared_state, tr, va, te, head_only)
            row[arm] = fin
            row[arm.replace('_final', '_bestval')] = best
        # D 同切分 CSP+LDA
        Xs, ys = tr[0], tr[1]
        t0i, t1i = TASK[0], min(TASK[1], Xs.shape[-1])
        C_tr = cov(Xs[:, :, t0i:t1i].astype(np.float64))
        C_te = cov(te[0][:, :, t0i:t1i].astype(np.float64))
        W = fit_csp(C_tr, ys)
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        F_tr = np.log(np.maximum(np.einsum('ia,nij,ja->na', W, C_tr, W), 1e-12))
        F_te = np.log(np.maximum(np.einsum('ia,nij,ja->na', W, C_te, W), 1e-12))
        row['csp'] = float(LDA(solver='lsqr', shrinkage='auto')
                           .fit(F_tr, ys).score(F_te, te[1]))
        rows.append(row)
        if len(rows) % 20 == 0:
            print(f'  {len(rows)} subjects done, {time.time()-t0:.0f}s', flush=True)

    def pooled(key):
        k = np.array([r['n_test'] for r in rows])
        a = np.array([r[key] for r in rows])
        return float((a * k).sum() / k.sum())

    def summ(key):
        a = np.array([r[key] for r in rows]) * 100
        return {'pooled': pooled(key) * 100, 'macro_mean': float(a.mean()),
                'macro_sd': float(a.std()), 'min': float(a.min()), 'max': float(a.max())}

    out = {'ckpt': str(CKPT), 'lr_full': LR_FULL, 'lr_head': LR_HEAD,
           'iterations': ITERS, 'wd': WD, 'subjects_subset': SUBJECTS, 'rows': rows,
           'summary': {k: summ(k) for k in
                       ('head_final', 'head_bestval', 'full_final', 'full_bestval', 'csp')
                       if any(k in r for r in rows)},
           'armA_shared_reference': float(os.environ.get('FT_ARMA', 40.98))}
    for k, v in out['summary'].items():
        print(f"[{k}] pooled {v['pooled']:.2f}% | macro {v['macro_mean']:.1f}±{v['macro_sd']:.1f}%"
              f" | range {v['min']:.1f}-{v['max']:.1f}%")
    rho, p = spearmanr([r['full_bestval'] for r in rows], [r['csp'] for r in rows])
    print(f'[full_bestval × csp] Spearman ρ={rho:.3f} (p={p:.1e})')
    out['spearman_full_vs_csp'] = {'rho': float(rho), 'p': float(p)}
    fname = f'physionet_finetune{TAG}.json'
    with open(OUT_DIR / fname, 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'[OK] {OUT_DIR}/{fname}  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
