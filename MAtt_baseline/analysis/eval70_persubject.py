# -*- coding: utf-8 -*-
"""离线评估:70/15/15 samepool 逐被试精度(多 seed 汇总)。

切分复现: split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42, others=[sub_all])
与 train_pair_with_recording.load_data() 逐位一致 → 测试集与训练 run 共享。
模型: load_matt_model(utils.functions), 架构参数须与 train_pair 实际值一致
(D_ATTN_IN/OUT=8/8, local_radius=6 — build_matt_model 的 fallback 默认 20/18/1 是错的)。
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent   # = MAtt_baseline/（本脚本在 analysis/ 下）
sys.path.insert(0, str(BASE))

from utils.dataset import (load_preprocessed_dataset,
                           split_train_valid_test)
from utils.functions import load_matt_model

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
WIN, STEP = 375, 188          # 与训练入口一致(1.5s/0.75s @250Hz)
OUT = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
RUNS = {                       # seed → (结果目录, ckpt 前缀)
    'seed42':   ('verify_samepool_70_15_15', 'verify70'),
    'seed2024': ('verify70_seed2024',        'v70s2024'),
    'seed7':    ('verify70_seed7',           'v70s7'),
}
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)


def main():
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70,
                                 valid_ratio=0.15, random_state=42,
                                 others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print(f'test {len(y_te)} trials, {len(set(sub_te.tolist()))} subjects')

    x_t = torch.Tensor(X_te).unsqueeze(1)
    y_np = np.asarray(y_te)

    out = {}
    for name, (dirname, tag) in RUNS.items():
        d = BASE / 'results' / dirname
        accs = {}
        for kind in ('best_acc', 'best_loss'):
            ck = d / f'{tag}_cross_mix_{kind}.pt'
            if not ck.exists():
                print(f'skip {name}/{kind}: {ck.name} 不存在')
                continue
            net = load_matt_model(str(ck), WIN, STEP, ARGS)
            net.eval()
            x_t = x_t.to(next(net.parameters()).device)
            with torch.no_grad():
                pred = net(x_t).argmax(dim=1).cpu().numpy()
            ok = (pred == y_np)
            per = {}
            for s in range(9):
                m = sub_te == s
                if m.sum() > 0:
                    per[f'S{s+1:02d}'] = [int(m.sum()), round(float(ok[m].mean()), 4)]
            accs[kind] = {'overall': round(float(ok.mean()), 4),
                          'per_subject': per}
        out[name] = accs
        if accs:
            k = 'best_acc' if 'best_acc' in accs else 'best_loss'
            a = accs[k]
            print(f"{name} {k} overall={a['overall']:.4f}  "
                  + '  '.join(f"{kk}:{vv[1]:.3f}" for kk, vv in a['per_subject'].items()))
    OUT.parent.mkdir(parents=True, exist_ok=True)   # 建目录放在写入点,不放模块级(被 import 时会凭空造空目录)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f'written {OUT}')


if __name__ == '__main__':
    main()
