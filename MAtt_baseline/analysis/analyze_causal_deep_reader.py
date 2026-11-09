# -*- coding: utf-8 -*-
"""X-1 读出器 2 + X-3 特异性对照:冻结主协议 ckpt,只看输入被干预后的精度。

配套: analysis/差异报告/因果性分析方案_ERD到精度.md §4 X-1(读出器 2) / X-3(四臂)、§6 边界 3。
读出器 1(切空间 + 收缩 LDA)在 analyze_causal_erd_injection.py 里;本脚本做「同一个干预、
换读出器」与「同一个读出器、换干预作用域」,所以**信号空间的实现必须与协方差空间那一版一致**:

    Z(a) = Z + (sqrt(a) - 1) * (Z_band - mu),   mu := 本被试任务窗 Z_band 的跨试次均值

为什么深度模型必须走信号空间:协方差那一版等价于直接给 M 加一个(可能不半正定的)增量,
而模型前端看到的是波形 —— 只有一个真实存在的波形才能被前向。

用法:
    python analysis/analyze_causal_deep_reader.py                # X-1 主结果
    python analysis/analyze_causal_deep_reader.py --x3           # X-3 特异性四臂
"""
import argparse
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
from mAtt.mAtt import BCI2A_CHANNELS                                     # noqa: E402
from analyze_causal_erd_injection import (MU, BROAD, TASK, BASE_WIN, FS,  # noqa: E402
                                          ALPHAS, CTRL_ALPHAS, narrow, band_pow)
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                  # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
WIN, STEP = 375, 188          # 与训练入口一致(1.5s/0.75s @250Hz)
RUNS = {'seed42':   ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024',        'v70s2024'),
        'seed7':    ('verify70_seed7',           'v70s7')}
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
SUBJ = [f'S{i+1:02d}' for i in range(9)]
BETA = (15.0, 25.0)           # β 带(与 analyze_trial_vs_subject 一致)

# ---- X-3 的通道集(沿用项目已核过的偏侧化定义,见 analyze_erd_decomposition) ----
ROI_R = ['FC4', 'FC2', 'C6', 'C4', 'C2', 'CP4', 'CP2']
ROI_L = ['FC3', 'FC1', 'C5', 'C3', 'C1', 'CP3', 'CP1']
ROI_SYM = sorted(set(ROI_R) | set(ROI_L))
_CH = {n: i for i, n in enumerate(BCI2A_CHANNELS)}


def chan_sets(y, side):
    """按 cue 侧给通道索引:left(0)→右半球对侧、right(1)→左半球;脚/舌→双侧合计。"""
    out = []
    for c in y:
        contra = ROI_R if c == 0 else ROI_L if c == 1 else ROI_SYM
        ipsi = ROI_L if c == 0 else ROI_R if c == 1 else ROI_SYM
        out.append(np.array([_CH[n] for n in (contra if side == 'contra' else ipsi)]))
    return out


def mix_epochs(X, band, a, chans=None):
    """按 (sqrt(a)-1)*(Z_band - mu) 改造任务窗。chans 给定时只在该试次的通道子集上做。"""
    Zb = narrow(X, band)
    seg = Zb[:, :, TASK[0]:TASK[1]]
    mu = seg.mean(axis=0, keepdims=True)
    d = X[:, :, TASK[0]:TASK[1]] + (np.sqrt(a) - 1.0) * (seg - mu)
    Y = X.copy()
    if chans is None:
        Y[:, :, TASK[0]:TASK[1]] = d
    else:
        for t, ci in enumerate(chans):
            Y[t, ci, TASK[0]:TASK[1]] = d[t, ci]
    return Y


def evaluate(net, x_np, y_np, sub_np):
    x = torch.Tensor(x_np.astype(np.float32)).unsqueeze(1).to(
        next(net.parameters()).device)
    net.eval()
    with torch.no_grad():
        pred = net(x).argmax(dim=1).cpu().numpy()
    ok = pred == y_np
    per = {SUBJ[s]: float(ok[sub_np == s].mean()) for s in range(9)
           if (sub_np == s).sum() > 0}
    return float(ok.mean()), per


def arm_defs(x3):
    """返回 {臂名: (band, alphas, scope)};scope 为 None 表示全通道。"""
    if not x3:
        return {'mu': (MU, ALPHAS, None),
                'broadband_control': (BROAD, CTRL_ALPHAS, None)}
    return {'mu_contra': (MU, ALPHAS, 'contra'),
            'mu_ipsi': (MU, ALPHAS, 'ipsi'),
            'beta_contra': (BETA, ALPHAS, 'contra'),
            'broadband': (BROAD, CTRL_ALPHAS, None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x3', action='store_true', help='跑 X-3 特异性四臂')
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print('X-3 特异性对照(冻结深度模型)' if a.x3 else 'X-1 读出器 2(冻结深度模型)')
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print(f'测试集 {len(y_te)} 试次 / {len(set(sub_te.tolist()))} 被试')

    arms = arm_defs(a.x3)
    mixed = {}
    for aname, (band, alphas, ch) in arms.items():
        cs = chan_sets(y_te, ch) if isinstance(ch, str) else None
        mixed[aname] = {al: mix_epochs(X_te, band, al, cs) for al in alphas}
        print(f'  臂「{aname}」(带 {band}, 作用域 {ch or "全通道"}, {len(alphas)} 档)已生成')

    out = {'config': {'mode': 'x3' if a.x3 else 'x1',
                      'arms': {k: {'band': list(v[0]), 'alphas': v[1], 'scope': v[2]}
                               for k, v in arms.items()},
                      'win': WIN, 'step': STEP, 'n_test': int(len(y_te))}}
    for name, (dirname, tag) in RUNS.items():
        ck = BASE / 'results' / dirname / f'{tag}_cross_mix_best_acc.pt'
        if not ck.exists():
            print(f'  跳过 {name}:{ck.name} 不存在')
            continue
        net = load_matt_model(str(ck), WIN, STEP, ARGS)
        net.eval()
        rec = {}
        for aname, (band, alphas, ch) in arms.items():
            accs, per = [], {}
            for al in alphas:
                acc, p = evaluate(net, mixed[aname][al], y_te, sub_te)
                accs.append(acc)
                per[str(al)] = p
            rec[aname] = {'alphas': alphas, 'overall': accs, 'per_subject': per}
            print(f'  {name} / {aname}: ' + ' '.join(f'{v:.4f}' for v in accs))
        out[name] = rec
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    fn = 'causal_erd_deep_x3.json' if a.x3 else 'causal_erd_deep.json'
    (OUT / fn).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\n总耗时 {time.time()-t0:.1f}s  json -> {OUT}/{fn}')


if __name__ == '__main__':
    main()
