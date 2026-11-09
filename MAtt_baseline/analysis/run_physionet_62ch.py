# -*- coding: utf-8 -*-
"""PhysionetMI 适配 · 62 通道全导联版:共享模型训练 + 逐人复评(seed42, GPU)。

动机: 22ch(BCI-2a 导联子集)共享 41.0% / 微调 44.4%;文献 EEGMMIDB 读数均用全导联
(64ch)。62ch = 全部 64 EEG − Fp1/Fp2(转 eog 作 ICA 参考、ICA 后剔除)。
数据: saved_data/preprocessed_data_moabb_raw_ica_62ch.pkl(PHYSIONET_ALL_EEG=1 重生成,
同一切分口径/同清洗规范/同 EA;由用户 2026-10-24「适配本体未完成」指令授权新增,
22ch 冻结版保留不动)。
模型: _SparseAttentionBase(conv1_kernel=62, use_channel_weight=False) —— 与主协议
mAtt_bci 逐参一致(d_attn 8/8, local_radius 6, dropout 0.3, conv1_out=22 ⇒ SPD 路径
维度不变),仅两处差异:conv1 核跨 62 通道;去掉 A9 解剖通道加权(EA 白化后逐通道
增益≈均匀, 且 BCI2A 名单不适用 62ch)。⇒ 零共享文件改动。
训练: samepool 70/15/15(random_state=42 固定)、Adam lr 5e-4 wd 0.02、100it、bs 128、
grad clip 1.0、test 每 5 it —— 与 train_pair 主协议同构(该入口无法喂 62ch:
build_matt_model 硬接 mAtt_bci 的 conv1_kernel=22)。
产物: results/physionet_62ch_s42/(ckpt + physionet_62ch.json + 本脚本 stdout 由调用方重定向)
"""
import sys, json, time, os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(HERE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from mAtt.mAtt import _SparseAttentionBase
from analyze_physionet_adapt import wilson_ci
from mAtt.optimizer import MixOptimizer

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_moabb_raw_ica_62ch.pkl'
# 2026-11-08 env 门控（默认值 = 历史行为）：
#   RESAMPLE : 160/RESAMPLE = 目标采样率（2 → 80Hz，与 pm80 ckpt 同口径）
#   MIXOPT   : 1 = MixOptimizer 包住 Adam（保住 StiefelParameter 正交性）
RESAMPLE = int(os.environ.get('RESAMPLE', '1'))
MIXOPT = os.environ.get('MIXOPT', '0') == '1'
WIN, STEP = int(1.5 * 160 / RESAMPLE), int(0.75 * 160 / RESAMPLE)
OUT_DIR = BASE / 'results' / os.environ.get('OUT_NAME', 'physionet_62ch_s42')
# 强组等子集协议: 逗号分隔 1-based 被试号; 切分仍全局 random_state=42 后按掩码过滤
# (与 train_pair --subjects 同款, 逐人 test 成员与全量 run 完全一致)
SUBJECTS = ([int(x) for x in os.environ['SUBJECTS'].split(',') if x.strip()]
            if os.environ.get('SUBJECTS') else [])
N_CH = 62
NSPAT = int(os.environ.get('NSPAT', 22))   # conv1_out(空间分量宽度); 默认 22=主协议
DATTN = int(os.environ.get('DATTN', '8'))  # d_attn_in(注意力嵌入维); 默认 8=主协议
SEED, ITERS, LR, WD, BS = 42, 100, 5e-4, 0.02, 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_net():
    return _SparseAttentionBase(
        WIN, STEP, n_classes=4, d_attn_in=DATTN, d_attn_out=8,
        local_radius=6, anchor_stride=None, n_heads=1, n_windows=None,
        use_hann=True, use_channel_weight=False, n_spatial_filters=NSPAT,
        conv1_kernel=N_CH, dropout=0.3, norm_type='in').to(DEVICE)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(SEED)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    if RESAMPLE > 1:
        from scipy.signal import resample_poly
        subject_data = [(resample_poly(np.asarray(X, np.float32), 1, RESAMPLE, axis=-1)
                         .astype(np.float32), y) for X, y in subject_data]
        print(f'[重采样] 160 -> {160 // RESAMPLE} Hz（1/{RESAMPLE}），WIN={WIN} STEP={STEP}',
              flush=True)
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_tr, y_tr, X_va, y_va, X_te, y_te, sub_tr, sub_va, sub_te = ret
    if SUBJECTS:
        m_tr = np.isin(sub_tr + 1, SUBJECTS)
        m_va = np.isin(sub_va + 1, SUBJECTS)
        m_te = np.isin(sub_te + 1, SUBJECTS)
        X_tr, y_tr, sub_tr = X_tr[m_tr], y_tr[m_tr], sub_tr[m_tr]
        X_va, y_va, sub_va = X_va[m_va], y_va[m_va], sub_va[m_va]
        X_te, y_te, sub_te = X_te[m_te], y_te[m_te], sub_te[m_te]
        print(f'[子集协议] 强组 {len(SUBJECTS)} 人: train/val/test={len(y_tr)}/{len(y_va)}/{len(y_te)}')
    print(f'n={len(y_all)} train/val/test={len(y_tr)}/{len(y_va)}/{len(y_te)} ch={X_all.shape[1]}')

    def tl(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1),
                                            torch.Tensor(y).long())
        return torch.utils.data.DataLoader(ds, batch_size=BS, shuffle=shuffle)
    tr_ld, va_ld, te_ld = tl(X_tr, y_tr, True), tl(X_va, y_va, False), tl(X_te, y_te, False)

    net = make_net()
    base_opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WD)
    opt = MixOptimizer(base_opt) if MIXOPT else base_opt
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=ITERS)
    ce = nn.CrossEntropyLoss()
    rec = {'train_loss': [], 'train_acc': [], 'val_acc': [], 'test_acc': [], 'test_iter': []}
    best_acc = -1.0
    t0 = time.time()
    for ite in range(ITERS):
        net.train()
        ls = ns = nb = 0.0
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = net(xb)
            loss = ce(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            ls += loss.item(); ns += (out.argmax(1) == yb).sum().item(); nb += len(yb)
        sched.step()
        net.eval()
        with torch.no_grad():
            va = np.mean([ (net(xb.to(DEVICE)).argmax(1).cpu().numpy() == yb.numpy()).mean()
                           for xb, yb in va_ld])
        rec['train_loss'].append(ls / max(len(tr_ld), 1))
        rec['train_acc'].append(ns / max(len(y_tr), 1))
        rec['val_acc'].append(float(va))
        if ite % 5 == 0 or ite == ITERS - 1:
            net.eval()
            with torch.no_grad():
                te = np.mean([(net(xb.to(DEVICE)).argmax(1).cpu().numpy() == yb.numpy()).mean()
                              for xb, yb in te_ld])
            rec['test_acc'].append(float(te)); rec['test_iter'].append(ite)
            print(f'  ite {ite:02d}: trA={rec["train_acc"][-1]*100:.1f}% '
                  f'vaA={va*100:.1f}% teA={te*100:.2f}%', flush=True)
            if va > best_acc:
                best_acc = float(va)
                torch.save(net.state_dict(), OUT_DIR / 'p62_cross_mix_best_acc.pt')

    # best_loss 简化: 训练损失最低的快照即末段, 只存末值模型
    torch.save(net.state_dict(), OUT_DIR / 'p62_cross_mix_final.pt')

    # ---- 逐人复评(best_acc ckpt, 与训练共享测试集) ----
    net = make_net()
    net.load_state_dict(torch.load(OUT_DIR / 'p62_cross_mix_best_acc.pt', map_location='cpu'))
    net.eval()
    x_t = torch.Tensor(X_te).unsqueeze(1)
    pred = []
    with torch.no_grad():
        for i in range(0, len(x_t), 256):
            pred += net(x_t[i:i + 256].to(DEVICE)).argmax(1).cpu().numpy().tolist()
    pred = np.array(pred)
    per = []
    for s in np.unique(sub_te):
        m = sub_te == s
        k, n = int((pred[m] == y_te[m]).sum()), int(m.sum())
        _, hw = wilson_ci(k, n)
        per.append({'subject': int(s), 'n_test': n, 'acc': k / n, 'wilson_half_width': hw})
    a = np.array([p['acc'] for p in per]) * 100
    overall = float((pred == y_te).mean())
    print(f'[best_acc] 总体 {overall*100:.2f}% | 逐人 min={a.min():.1f} Q1={np.percentile(a,25):.1f} '
          f'中位={np.median(a):.1f} Q3={np.percentile(a,75):.1f} max={a.max():.1f} | 逐类召回 '
          + ' '.join(f'{(pred[y_te==c]==c).mean()*100:.0f}' for c in range(4)))

    out = {'config': {'n_ch': N_CH, 'seed': SEED, 'iterations': ITERS, 'lr': LR, 'wd': WD,
                      'conv1_kernel': N_CH, 'channel_weight': False, 'n_spatial_filters': NSPAT,
                      'subjects_subset': SUBJECTS},
           'trajectories': rec,
           'best_acc': {'overall_test_acc': overall, 'per_subject': per,
                        'per_class_recall': [float((pred[y_te == c] == c).mean()) for c in range(4)]}}
    with open(OUT_DIR / 'physionet_62ch.json', 'w', encoding='utf-8') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f'[OK] {OUT_DIR}/physionet_62ch.json  ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
