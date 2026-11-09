# -*- coding: utf-8 -*-
"""表示里的时序信号清点:窗序信息到底存不存在(零训练,seed42)。

## 依据
① 主线的耦合轴含「μ ERD 深度 + 时间过程」;但 window_ablation 证明模型对窗序
不敏感(打乱窗序精度不掉,t=0.12),且 2026-10-23 的 10 开关大扫除把输入层差分
(le_diff1/2,塌)、静息相对化(le_win0,−3~−4.8)、打分层 ref/diff(−4~−7)、
窗先验(attn_prior,3-seed 回落)全部判阴。⇒ 唯一没关死的问题:**表示本身**
含不含线性可读的时序/动态信号——模型不敏感 ≠ 信息不存在。

## 设计
L3 = net._features(x) = 逐窗切向量展平 (180 = 5 窗 × 36 维)。五臂共享 LDA(all4):
  A full180   原序全特征(线性读出可用窗序)
  B mean36    逐窗平均(序盲;静态内容的上限)
  C shuf180   固定置换重排(=A 的重标记,数学恒等于 A,作一致性自检)
  D inc144    相邻窗切向量差 Δ_i = L_i − L_{i−1}(纯动态)
  E full+inc  A ⊕ D(静态 ⊕ 动态)
若 A≈B 且 D 低、E−A 小 ⇒ 表示内**无线性时序信号**,轨迹注意力在此表示上无可读目标。
只写 results/temporal_signal/。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'temporal_signal'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    net = load_matt_model(str(BASE / 'results' / 'verify_samepool_70_15_15' /
                              'verify70_cross_mix_best_acc.pt'), 375, 188, ARGS)
    net.eval()
    dev = next(net.parameters()).device

    def feats(X):
        out = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                xb = torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)
                out.append(net._features(xb).cpu().numpy())
        return np.concatenate(out, 0)

    Ftr, Fte = feats(X_tr).reshape(-1, 5, 36), feats(X_te).reshape(-1, 5, 36)
    rng = np.random.default_rng(0)
    perm = rng.permutation(5)

    def arms(F):
        return {'A_full180': F.reshape(len(F), -1),
                'B_mean36': F.mean(1),
                'C_shuf180': F[:, perm].reshape(len(F), -1),
                'D_inc144': (F[:, 1:] - F[:, :-1]).reshape(len(F), -1),
                'E_full+inc': np.concatenate(
                    [F.reshape(len(F), -1), (F[:, 1:] - F[:, :-1]).reshape(len(F), -1)], 1)}

    Atr, Ate = arms(Ftr), arms(Fte)
    rec = {}
    print('时序信号清点(L3 后端特征,all4 共享 LDA,低四/高五/全体):')
    for name in ('A_full180', 'B_mean36', 'C_shuf180', 'D_inc144', 'E_full+inc'):
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Atr[name], y_tr)
        ok = clf.predict(Ate[name]) == y_te
        po = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        lo = np.mean([po[SUBJ[i]] for i in LOW])
        hi = np.mean([po[SUBJ[i]] for i in HIGH])
        rec[name] = {'low4': float(lo), 'high5': float(hi),
                     'overall': float(ok.mean()), 'pooled': po}
        print('  %-10s 低四 %5.1f 高五 %5.1f 全体 %5.1f'
              % (name, lo * 100, hi * 100, ok.mean() * 100))
    (OUT / 'temporal_signal.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/temporal_signal.json' % OUT)


if __name__ == '__main__':
    main()
