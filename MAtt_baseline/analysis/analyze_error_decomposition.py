# -*- coding: utf-8 -*-
"""逐人 × 逐类误差分解:低四人(38%)与高五人(68%)的缺口落在哪个类别、哪种混淆。

## 依据(全部已确立)
mirror_commonality(`results/mirror_commonality/`):手二类类信息几乎全在镜像反对称
子空间,共享 LDA 在手二类上低四也能读到 54–62%(mu_A_full / mu_ERD_anti);
四类的带功率特征读不出脚/舌(共享 ≤34%)。⇒ 深度模型的逐人缺口在类别层
从未分解:若低四的手二类已被模型读出(≈共享 LDA 水平),缺口在脚/舌;
若低四连手二类都远低于共享 LDA,缺口在模型读手类的方式本身。

## 读数
3 seed 基线(verify70)在主协议测试折(778 试次)上:逐人×逐类精度、逐人手二类
精度、低四/高五分组的混淆矩阵(行归一化);并与 mirror_commonality 的
mu_A_full / mu_ERD_anti 共享 LDA 手二类逐人数并排。
只写 results/error_decomposition/。纯推理,分钟级。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'error_decomposition'
MIRROR = BASE / 'results' / 'mirror_commonality' / 'mirror_commonality.json'
RUNS = {42: ('verify_samepool_70_15_15', 'verify70'),
        2024: ('verify70_seed2024', 'v70s2024'),
        7: ('verify70_seed7', 'v70s7')}
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
CLS = ['左手', '右手', '脚', '舌']


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    x_t = torch.Tensor(X_te).unsqueeze(1)
    y_np = np.asarray(y_te)

    votes = np.zeros((len(y_np), 4))            # 3-seed 多数票(逐类分解用均值口径)
    preds = []
    for seed, (dirname, tag) in RUNS.items():
        ck = BASE / 'results' / dirname / f'{tag}_cross_mix_best_acc.pt'
        net = load_matt_model(str(ck), 375, 188,
                              dict(dataset='bci2a', n_classes=4, d_attn_in=8,
                                   d_attn_out=8, local_radius=6, anchor_stride=None,
                                   n_heads=1, dropout=0.3))
        net.eval()
        with torch.no_grad():
            p = net(x_t.to(next(net.parameters()).device)).argmax(1).cpu().numpy()
        preds.append(p)
        votes[np.arange(len(p)), p] += 1
    mv = votes.argmax(1)

    per, hand2, percls = {}, {}, {}
    for i, s in enumerate(SUBJ):
        m = sub_te == i
        per[s] = float((mv[m] == y_np[m]).mean())
        hm = m & (y_np < 2)
        hand2[s] = float((mv[hm] == y_np[hm]).mean())
        percls[s] = {CLS[c]: float((mv[m & (y_np == c)] == c).mean())
                     for c in range(4) if (m & (y_np == c)).sum()}

    mir = json.loads(MIRROR.read_text(encoding='utf-8'))['arms']
    print('3-seed 多数票;手二类对照 = mirror_commonality 的共享 LDA(同测试折、手二类)')
    print('%-5s %6s | %6s %6s %6s | 模型手二 vs 共享LDA手二(mu_A_full / mu_ERD_anti)'
          % ('被试', '总体', '左手', '脚', '舌'))
    for i, s in enumerate(SUBJ):
        a = mir['mu_A_full']['hand2']['pooled'][s]
        b = mir['mu_ERD_anti']['hand2']['pooled'][s]
        print('%-5s %6.1f | %6.1f %6.1f %6.1f | %6.1f  vs  %5.1f / %5.1f'
              % (s, per[s] * 100, percls[s]['左手'] * 100, percls[s]['脚'] * 100,
                 percls[s]['舌'] * 100, hand2[s] * 100, a * 100, b * 100))
    print('（右手列略:与左手同构;逐人完整表在 json）')

    rec = {'per_subject': per, 'hand2': hand2, 'per_class': percls,
           'group': {
               'low4_overall': float(np.mean([per[SUBJ[i]] for i in LOW])),
               'high5_overall': float(np.mean([per[SUBJ[i]] for i in HIGH])),
               'low4_hand2': float(np.mean([hand2[SUBJ[i]] for i in LOW])),
               'high5_hand2': float(np.mean([hand2[SUBJ[i]] for i in HIGH])),
               'low4_class': {c: float(np.mean([percls[SUBJ[i]][c] for i in LOW]))
                              for c in CLS if c in percls[SUBJ[0]]},
               'high5_class': {c: float(np.mean([percls[SUBJ[i]][c] for i in HIGH]))
                               for c in CLS if c in percls[SUBJ[0]]}}}

    print('\n分组均值: 低四 总体 %.1f 手二 %.1f | 高五 总体 %.1f 手二 %.1f'
          % (rec['group']['low4_overall'] * 100, rec['group']['low4_hand2'] * 100,
             rec['group']['high5_overall'] * 100, rec['group']['high5_hand2'] * 100))
    for g in ('low4_class', 'high5_class'):
        print('  %-11s' % g, {k: round(v * 100, 1) for k, v in rec['group'][g].items()})

    # 分组混淆矩阵(多数票,行归一化)
    for gname, idx in (('低四', LOW), ('高五', HIGH)):
        m = np.isin(sub_te, idx)
        C = np.zeros((4, 4))
        for t, pr in zip(y_np[m], mv[m]):
            C[t, pr] += 1
        C = C / C.sum(1, keepdims=True)
        rec['confusion_' + gname] = [[round(float(v), 4) for v in row] for row in C]
        print('\n%s 混淆(行=真,列=pred):' % gname)
        print('       ' + ''.join('%7s' % c for c in CLS))
        for r in range(4):
            print('%6s' % CLS[r] + ''.join('%7.1f' % (C[r, c] * 100) for c in range(4)))

    rec['per_seed_predictions'] = {str(s): p.tolist() for s, p in zip(RUNS, preds)}
    (OUT / 'error_decomposition.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n[OK] %s/error_decomposition.json' % OUT)


if __name__ == '__main__':
    main()
