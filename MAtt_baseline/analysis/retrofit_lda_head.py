# -*- coding: utf-8 -*-
"""LDA 头改装:把闭式共享 LDA 解直接写进 rcs 模型的线性头(零训练)。

## 依据
rcs 模型(并联原始协方差流)的 433 维特征上,共享 shrinkage LDA = 61.4,而其
SGD+CE 训练头只有 56.8(推进十)。LDA 的判别函数本来就是特征的仿射函数 ⇒
linear.weight=coef_、linear.bias=intercept_ 后,模型前向与 LDA 预测**逐位等价**。
本脚本对 3 个 seed 的 rcs 模型各做一次改装,保存 ckpt(可作为正式解码器交付)
并按主协议配对评估。产物 results/rcslda_eval/ + results/rcslda_s{seed}/*.pt。
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
OUT = BASE / 'results' / 'rcslda_eval'
BASELINE = BASE / 'results' / 'verify70_persubject' / 'verify70_persubject.json'
RUNS = {42: ('rcs_s42', 'rcs42'), 2024: ('rcs_s2024', 'rcs2024'), 7: ('rcs_s7', 'rcs7')}
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            raw_cov_stream=True)


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
    base = json.loads(BASELINE.read_text(encoding='utf-8'))
    rec = {'per_seed': {}}
    pooled, dlo, dhi = [], [], []
    ps_all = {s: [] for s in SUBJ}
    for seed, (dirname, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), 375, 188, ARGS)
        net.eval()
        dev = next(net.parameters()).device

        def feats(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    out.append(net._features(
                        torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy())
            return np.concatenate(out, 0)
        Ftr, Fte = feats(X_tr), feats(X_te)
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
        clf.fit(Ftr, y_tr)
        # 写进线性头(LazyLinear 已在 load 后由 forward 物化?保险:先跑一批)
        with torch.no_grad():
            _ = net(torch.Tensor(X_te[:2]).unsqueeze(1).to(dev))
        net.linear.weight.data = torch.Tensor(clf.coef_).to(dev)
        net.linear.bias.data = torch.Tensor(clf.intercept_).to(dev)
        # 端到端前向应与 LDA 预测逐位一致
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_te), 128):
                preds.append(net(torch.Tensor(X_te[i:i + 128]).unsqueeze(1).to(
                    dev)).argmax(1).cpu().numpy())
        pr = np.concatenate(preds)
        assert (pr == clf.predict(Fte)).all(), 'LDA 头与前向不一致'
        ok = pr == y_te
        per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
        bp = base['seed%d' % seed]['best_acc']
        d_ = [per[SUBJ[i]] - bp['per_subject'][SUBJ[i]][1] for i in range(9)]
        lo = float(np.mean([d_[i] for i in LOW]))
        hi = float(np.mean([d_[i] for i in HIGH]))
        pooled.append(float(ok.mean()))
        dlo.append(lo)
        dhi.append(hi)
        for s in SUBJ:
            ps_all[s].append(per[s])
        # 保存改装后 ckpt(可作正式解码器交付)
        d = BASE / 'results' / f'rcslda_s{seed}'
        d.mkdir(exist_ok=True)
        torch.save(net.state_dict(), d / f'rcslda{seed}_cross_mix_best_acc.pt')
        print('seed%d: pooled %.1f%%(基线 %.1f) Δ低四 %+.2f Δ高五 %+.2f | ckpt 已存'
              % (seed, ok.mean() * 100, bp['overall'] * 100, lo * 100, hi * 100))
        rec['per_seed'][str(seed)] = {'pooled': float(ok.mean()), 'per': per}
    ps = {s: float(np.mean(v)) for s, v in ps_all.items()}
    rec['summary'] = {'pooled_mean': float(np.mean(pooled)),
                      'd_low4': float(np.mean(dlo)), 'd_high5': float(np.mean(dhi)),
                      'd_low4_minus_high5': float(np.mean(dlo) - np.mean(dhi)),
                      'per_subject_mean': ps}
    (OUT / 'rcslda_eval.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False),
                                          encoding='utf-8')
    print('\n=== rcs + LDA 头(3 seed) ===')
    print('pooled %.1f%% | Δ低四 %+.2f Δ高五 %+.2f (Δ低−Δ高 %+.2f)'
          % (100 * rec['summary']['pooled_mean'], 100 * rec['summary']['d_low4'],
             100 * rec['summary']['d_high5'], 100 * rec['summary']['d_low4_minus_high5']))
    print('低四 %.1f 高五 %.1f | 逐人:' % (
        100 * np.mean([ps[SUBJ[i]] for i in LOW]),
        100 * np.mean([ps[SUBJ[i]] for i in HIGH])),
        ' '.join('%s%.1f' % (s, 100 * ps[s]) for s in SUBJ))
    print('[OK] %s/rcslda_eval.json' % OUT)


if __name__ == '__main__':
    main()
