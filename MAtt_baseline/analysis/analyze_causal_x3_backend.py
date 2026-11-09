# -*- coding: utf-8 -*-
"""判决性实验:同一条干预、同一个读出器,只看**表示层**。

## 要回答什么
X-3 显示深度模型的端到端精度随干预大幅移动(对侧 vs 同侧反号、β 带反向)。
问题是:这个移动是「**表示层的可分性被改变了**」还是「**表示层没动,只是末端那个固定
线性分类头被推过了决策边界**」?

## 怎么分开
用同一个冻结模型,取它自己的**后端出口特征**(后端切空间,180 维),然后
  ① 在该特征空间里做 oracle 类间分离度  D(a)/D(1)   —— 只看结构,不看分类头
  ② 在该特征空间里做**逐档重训**的 LDA(5 折)        —— 看「可读性」是否变化
  ③ 冻结点(a=1)训练的同一个 LDA 去测各档            —— 看固定决策规则被推多远
若 ①② 在 a≥0.25 几乎不动,而 ③ 大动 ⇒ **纯粹是决策边界效应**。

特征名:saved_data 里 "后端切空间出口" 对应模型 forward 的 tangent.vec 之前一级。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_causal_erd_injection import MU, TASK, ALPHAS, narrow      # noqa: E402
from analyze_causal_deep_reader import (mix_epochs, chan_sets, RUNS,   # noqa: E402
                                        ARGS, WIN, STEP, ROI_R, ROI_L, ROI_SYM)
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                  # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
PICK = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def backend_feats(net, x_np):
    """冻结模型的**后端出口**特征。

    ⚠ 必须调用模型自带的 `_features()` 而不是自己重排一遍模块链:第一版手写时漏了
    `manifold_change` 与 `flat(x.view(shape[0], shape[1], -1))`,形状变成 3890 维(应为 778×180),
    直接 IndexError。**能复用模型自己的方法就不要重写公式。**
    """
    x = torch.Tensor(x_np.astype(np.float32)).unsqueeze(1).to(
        next(net.parameters()).device)
    net.eval()
    with torch.no_grad():
        v = net._features(x)
    return v.cpu().numpy()


def main():
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    cs = chan_sets(y_te, 'contra')
    print(f'测试折 {len(y_te)} 试次;干预 = μ 带 / 对侧 ROI / 全 9 档', flush=True)

    out = {}
    for name, (dirname, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), WIN, STEP, ARGS)
        net.eval()
        feats = {}
        for al in PICK:
            feats[al] = backend_feats(net, mix_epochs(X_te, MU, al, cs))
        F1 = feats[1.0]
        g = F1.mean(0, keepdims=True)
        F1c = F1 - g
        cen1 = np.stack([F1c[y_te == c].mean(0) for c in range(4)])
        D1 = float(np.mean([np.linalg.norm(cen1[i] - cen1[j])
                            for i in range(4) for j in range(i + 1, 4)]))
        folds = list(StratifiedKFold(5, shuffle=True, random_state=42).split(F1, y_te))
        rec = {'D_over_D1': [], 'retrained': [], 'fixed': [], 'per_subject_fixed': {}}
        for al in PICK:
            Fc = feats[al] - feats[al].mean(0, keepdims=True)
            cen = np.stack([Fc[y_te == c].mean(0) for c in range(4)])
            rec['D_over_D1'].append(float(np.mean(
                [np.linalg.norm(cen[i] - cen[j])
                 for i in range(4) for j in range(i + 1, 4)])) / D1)
            rec['retrained'].append(float(np.mean(
                [LDA(solver='lsqr', shrinkage='auto').fit(Fc[tr], y_te[tr]).score(
                    Fc[te], y_te[te]) for tr, te in folds])))
            pred = np.empty(len(y_te), int)
            for tr, te in folds:
                m = LDA(solver='lsqr', shrinkage='auto').fit(F1c[tr], y_te[tr])
                pred[te] = m.predict(Fc[te])
            rec['fixed'].append(float((pred == y_te).mean()))
        out[name] = {'alphas': PICK, **rec}
        print(f'  {name}')
        print('    D(a)/D(1)  ' + ' '.join(f'{v:.4f}' for v in rec['D_over_D1']))
        print('    逐档重训   ' + ' '.join(f'{v:.4f}' for v in rec['retrained']))
        print('    冻结点     ' + ' '.join(f'{v:.4f}' for v in rec['fixed']), flush=True)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (OUT / 'causal_erd_x3_backend.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    seeds = list(out.keys())
    m = lambda k: np.mean([out[s][k] for s in seeds], 0)
    print('\n3-seed 平均')
    print('  D(a)/D(1)  ' + ' '.join(f'{v:.4f}' for v in m('D_over_D1')))
    print('  逐档重训   ' + ' '.join(f'{v:.4f}' for v in m('retrained')))
    print('  冻结点     ' + ' '.join(f'{v:.4f}' for v in m('fixed')))
    print(f'\nwritten {OUT}/causal_erd_x3_backend.json')


if __name__ == '__main__':
    main()
