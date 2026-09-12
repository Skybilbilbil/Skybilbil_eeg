# -*- coding: utf-8 -*-
"""
统一对比脚本：三个实验（数据管线 / 想象段 / 窗口数量）
=========================================================
实验1 (数据管线)：official(官方mat) | official_band (官方+8-30Hz) |
                   official_band_ea (官方+8-30Hz+独立EA) | self (自建管线)
实验2 (只保留想象段)：self 全段 vs self 只想象段(0.5s 起, 去静息窗)
实验3 (窗口数量)：同一数据 / 步长 0.5s / 窗长 1.0s, 2.5s, 3.5s → 8/5/3 窗

评估协议：S03/S09 被试内 + 9人混合跨被试（与 train_pair 一致）
输出：test_acc + 注意力对角线占比 + 想象窗权重差
"""
import argparse, os, sys, time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from scipy.signal import butter, sosfiltfilt

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'MAtt_baseline'))

from mAtt.mAtt import mAtt_bci, E2R
from mAtt.optimizer import MixOptimizer
from utils.GetPhysionetMI import load_preprocessed_dataset, split_train_valid_test
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS = 250
N_CLASSES = 4
LR, BS = 5e-4, 64
WD = 0.02
GRAD_CLIP = 1.0
ITERATIONS = 50
N_SPATIAL_FILTERS = 22

SELF_DATA = BASE / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OFFICIAL_DATA = BASE / 'saved_data' / 'official_bciciv2a_matt_raw.pkl'


def bandpass_8_30(x, fs=FS):
    """8-30Hz 零相位带通（我方预处理管线的 filter 步骤）。"""
    sos = butter(4, [8, 30], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, x, axis=-1)


def load_self_data():
    subject_data, _ = load_preprocessed_dataset(str(SELF_DATA))
    return subject_data


def load_official_data(add_band=False, add_ea=True):
    """官方 mat → (subject_data, metadata)。注意 EA 保持 matlab 内逐被试；
    band 对 raw 加 8-30Hz。"""
    subject_data, metadata = load_preprocessed_dataset(str(OFFICIAL_DATA))
    if add_band:
        subject_data = [(bandpass_8_30(X).astype(np.float32), y) for X, y in subject_data]
    if add_ea:
        for i, (X, y) in enumerate(subject_data):
            P = compute_ea_matrix_from_numpy(X)
            subject_data[i] = (apply_ea_to_numpy(X, P).astype(np.float32), y)
    return subject_data


def cut_imagonly(subject_data, start_sec=0.5, end_sec=3.5):
    """只保留想象段 [start_sec, end_sec]（去掉静息窗）。"""
    s0 = int(round(start_sec * FS))
    e0 = int(round(end_sec * FS))
    return [(X[:, :, s0:e0], y) for X, y in subject_data]


def split_data(subject_data, sample_ratio=0.5, subjects=(3, 9), n_all=9):
    """S03/S09 被试内 + 9人混合跨被试划分（与 train_pair 一致）。"""
    from sklearn.model_selection import train_test_split
    splits = {}
    for sid in subjects:
        X, y = subject_data[sid - 1]
        _, X, _, y = train_test_split(X, y, test_size=sample_ratio, stratify=y, random_state=42)
        splits[sid] = split_train_valid_test(X, y)
    X_all = np.concatenate([subject_data[s - 1][0] for s in range(1, n_all + 1)], axis=0)
    y_all = np.concatenate([subject_data[s - 1][1] for s in range(1, n_all + 1)], axis=0)
    X9_tr, y9_tr, X9_va, y9_va, X9_te, y9_te = split_train_valid_test(X_all, y_all)
    return splits, (X9_tr, y9_tr, X9_va, y9_va, X9_te, y9_te)


def make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te, bs=BS):
    def tl(X, y, shuffle):
        ds = torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long())
        return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle,
                                           pin_memory=True, num_workers=0)
    return tl(X_tr, y_tr, True), tl(X_va, y_va, False), tl(X_te, y_te, False)


def train_eval(tag, net, tr_ld, va_ld, te_ld):
    """精简训练：50 iters，返回 best test acc & 注意力指标。"""
    CE = nn.CrossEntropyLoss()
    base_opt = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                                lr=LR, weight_decay=WD)
    opt = MixOptimizer(base_opt)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(base_opt, T_max=ITERATIONS)
    best_acc, best_att = 0.0, None
    net.train()
    for ite in range(ITERATIONS):
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            out = net(xb)
            loss = CE(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=GRAD_CLIP)
            opt.step()
        sched.step()
        net.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in te_ld:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                correct += (net(xb).argmax(1) == yb).sum().item()
                total += len(yb)
        acc = correct / max(total, 1)
        if acc > best_acc:
            best_acc = acc
    # 注意力指标：对角线占比 + 想象窗(后段)相对前段权重差
    att = attn_metrics(net, va_ld)
    net.train()
    return best_acc, att


def attn_metrics(net, loader):
    """加载验证集，统计注意力概率矩阵的对角线占比与"后半窗-前半窗"权重差。"""
    probs = []
    net.eval()
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(DEVICE)
            # 与 net.forward 相同主干部（截至 spdnorm）
            x = net.channel_weight(xb)
            x = net.In1(net.conv1(x))
            x = net.In2(net.conv2(x))
            spd = net.ract1(x)
            spd = net.spdnorm(spd)
            spd = net.manifold_change(spd)
            probs.append(net.att2.attention_prob(spd).cpu().numpy())
    P = np.concatenate(probs, axis=0).mean(0)      # (M,M)
    m = P.shape[0]
    diag = np.trace(P) / m if m else 0.0
    half = m // 2
    late = P[:, half:].mean() if m > 1 else 0.0
    early = P[:, :half].mean() if half > 0 else 0.0
    net.train()
    return {'diag': float(diag), 'late_early': float(late - early)}


def build_net(win_samples, step_samples, n_windows):
    net = mAtt_bci(win_samples, step_samples, n_classes=N_CLASSES,
                   local_radius=1, anchor_stride=None, n_heads=1,
                   n_windows=n_windows,
                   dropout=0.5)  # 显式保护本脚本历史基线(win5/win3 数字出自 0.5)
    net = net.to(DEVICE)
    return net


def run_config(name, subject_data, win_sec, step_sec):
    T = subject_data[0][0].shape[-1]
    win = int(round(win_sec * FS))
    step = int(round(step_sec * FS))
    M = E2R.n_windows(T, win, step)
    if M < 2:
        print(f'  [skip {name}] T={T} win={win} step={step} → M={M} < 2')
        return None

    splits, nine = split_data(subject_data)
    results = {}
    for sid in (3, 9):
        X_tr, y_tr, X_va, y_va, X_te, y_te = splits[sid]
        tr_ld, va_ld, te_ld = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)
        net = build_net(win, step, M)
        acc, att = train_eval(f'{name}_S{sid}', net, tr_ld, va_ld, te_ld)
        results[f'S{sid}'] = {'acc': acc, 'attn': att}
        print(f'  [{name}] S{sid}: test_acc={acc*100:.2f}% '
              f'diag={att["diag"]:.3f} late-early={att["late_early"]:+.3f}')

    X_tr, y_tr, X_va, y_va, X_te, y_te = nine
    tr_ld, va_ld, te_ld = make_loaders(X_tr, y_tr, X_va, y_va, X_te, y_te)
    net = build_net(win, step, M)
    acc, att = train_eval(f'{name}_9mix', net, tr_ld, va_ld, te_ld)
    results['9mix'] = {'acc': acc, 'attn': att}
    print(f'  [{name}] 9mix: test_acc={acc*100:.2f}% '
          f'diag={att["diag"]:.3f} late-early={att["late_early"]:+.3f}')
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', type=str, default='1',
                    help='1=数据管线 2=想象段 3=窗口数量（可用逗号组合 "1,3"）')
    args = ap.parse_args()
    exps = [e.strip() for e in args.exp.split(',')]

    t0 = time.time()
    summary = {}

    if '1' in exps:
        print('\n========== 实验1：数据管线对比 (self 数据 4.5s/8窗 为基准) ==========')
        self_sd = load_self_data()
        off_raw = load_official_data(add_band=False)
        off_band = load_official_data(add_band=True)
        off_band_ea = load_official_data(add_band=True, add_ea=True)
        exp1 = {}
        for name, sd in [('self', self_sd),
                         ('official_raw', off_raw),
                         ('official_band', off_band),
                         ('official_band_ea', off_band_ea)]:
            T = sd[0][0].shape[-1]
            win_sec = 1.0
            step_sec = 0.5
            win = int(round(win_sec * FS))
            step = int(round(step_sec * FS))
            M = E2R.n_windows(T, win, step)
            print(f'\n  --- {name}: T={T} ({T/FS:.2f}s) → M={M} 窗 ---')
            r = run_config(name, sd, win_sec, step_sec)
            exp1[name] = r
        summary['experiment1'] = exp1

    if '2' in exps:
        print('\n========== 实验2：只保留想象段 ==========')
        self_sd = load_self_data()
        exp2 = {}
        for name, sd in [('self_full', self_sd),
                         ('self_imagonly', cut_imagonly(self_sd, 0.5, 3.5))]:
            T = sd[0][0].shape[-1]
            win, step = int(round(1.0*FS)), int(round(0.5*FS))
            M = E2R.n_windows(T, win, step)
            print(f'\n  --- {name}: T={T} ({T/FS:.2f}s) → M={M} 窗 ---')
            exp2[name] = run_config(name, sd, 1.0, 0.5)
        summary['experiment2'] = exp2

    if '3' in exps:
        print('\n========== 实验3：窗口数量对比 (self 4.5s 全景, 步长0.5s) ==========')
        self_sd = load_self_data()
        exp3 = {}
        for nwin, win_sec in [(8, 1.0), (5, 2.5), (3, 3.5)]:
            name = f'win{nwin}'
            M = E2R.n_windows(1125, int(round(win_sec*FS)), int(round(0.5*FS)))
            print(f'\n  --- {name}: {win_sec}s 窗 / 0.5s 步 → M={M} 窗 ---')
            exp3[name] = run_config(name, self_sd, win_sec, 0.5)
        summary['experiment3'] = exp3

    print(f'\n==== 总耗时 {time.time()-t0:.0f}s ====')
    import json
    with open(BASE / '_exp_compare_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1, default=str)
    print('已保存: _exp_compare_summary.json')


if __name__ == '__main__':
    main()