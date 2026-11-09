# -*- coding: utf-8 -*-
"""sep62 新口径逐人复评（零训练）:给汇报稿 §3.1 与 图1 提供逐人数值。

复用 train_pair.load_data() 精确复现 70/15/15 切分（同 --resample_hz 62.5、同
random_state=42、同试次顺序），加载 results/sep62_s{42,2024,7} 的 best_acc 权重，
在同一批 778 试次 test 上复评：
  - 逐人 test 精度（3 seed 均值）→ 汇报稿 图1 与 §3.1 的"验证集选模"
  - 全体 test 精度（pooled）→ 与"最终迭代 58.33%"区分开（那是 best_acc 之外的读数）
只写 results/sep62_eval/。
"""
import sys, json
from pathlib import Path
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.argv = ['eval_sep62']            # train_pair 模块级会 parse_args，这里给空参（走新默认 62.5）
import torch
import train_pair_with_recording as TP
from utils.functions import build_matt_model

SEEDS = [42, 2024, 7]
CKPT = 'results/sep62_s%d/sep62s%d_cross_mix_best_acc.pt'
LOW4 = {2, 4, 5, 6}


def main():
    subject_data, T, M, splits, cross = TP.load_data()
    X_te, y_te, s_te = cross[4], cross[5], cross[8]
    print(f'[eval] T={T} M={M} test={len(y_te)} 被试={sorted(set(s_te.tolist()))}', flush=True)

    args = dict(n_classes=4, d_attn_in=TP.D_ATTN_IN, d_attn_out=TP.D_ATTN_OUT,
                local_radius=TP.LOCAL_RADIUS, anchor_stride=None, n_heads=1,
                n_windows=None, conv2_separable=True, conv2_multiscale=False,
                n_spatial_filters=TP.N_SPATIAL_FILTERS, channel_gaussian=False,
                gaussian_sigma=0.62, attn_change=None, attn_prior=None,
                manifold_change=None, dropout=TP.DROP_PROB, norm_type='in')

    out = {'fs': TP.FS, 'T': T, 'M': M, 'n_test': len(y_te), 'seeds': {}, 'per_subject': {}}
    runs = []
    for sd in SEEDS:
        torch.manual_seed(sd)
        net = build_matt_model(TP.WIN_SAMPLES, TP.STEP_SAMPLES, args)
        dev = next(net.parameters()).device
        net.eval()
        with torch.no_grad():
            net(torch.zeros(1, 1, 22, T, device=dev))
        ck = BASE / (CKPT % (sd, sd))
        net.load_state_dict(torch.load(ck, map_location='cpu'))
        net = net.to(dev)
        ps, ys = [], []
        with torch.no_grad():
            for i in range(0, len(y_te), 128):
                xb = torch.Tensor(X_te[i:i + 128]).unsqueeze(1).to(dev)
                ps.append(net(xb).argmax(1).cpu().numpy())
                ys.append(np.asarray(y_te[i:i + 128]))
        pred = np.concatenate(ps); yy = np.concatenate(ys)
        per = {}
        for s in np.unique(s_te):
            m = s_te == s
            per[f'S{int(s)+1:02d}'] = float((pred[m] == yy[m]).mean())
        pooled = float((pred == yy).mean())
        out['seeds'][str(sd)] = {'pooled': pooled, 'per_subject': per}
        runs.append(per)
        low = float(np.mean([v for k, v in per.items() if int(k[1:]) in LOW4]))
        high = float(np.mean([v for k, v in per.items() if int(k[1:]) not in LOW4]))
        print(f'  seed{sd}: pooled={pooled*100:.2f}% low4={low*100:.2f} high5={high*100:.2f}', flush=True)

    keys = sorted(runs[0])
    out['per_subject'] = {k: {'mean': float(np.mean([r[k] for r in runs])),
                              'min': float(np.min([r[k] for r in runs])),
                              'max': float(np.max([r[k] for r in runs]))} for k in keys}
    pooled_all = [out['seeds'][str(s)]['pooled'] for s in SEEDS]
    out['pooled_mean'] = float(np.mean(pooled_all))
    out['pooled_min'] = float(np.min(pooled_all))
    out['pooled_max'] = float(np.max(pooled_all))
    lowm = float(np.mean([out['per_subject'][k]['mean'] for k in keys if int(k[1:]) in LOW4]))
    highm = float(np.mean([out['per_subject'][k]['mean'] for k in keys if int(k[1:]) not in LOW4]))
    out['low4_mean'] = lowm
    out['high5_mean'] = highm
    print(f'\n[eval] pooled={out["pooled_mean"]*100:.2f}% ({out["pooled_min"]*100:.2f}-{out["pooled_max"]*100:.2f})')
    print('  逐人:' + ' '.join(f'{k}:{v["mean"]*100:.1f}' for k, v in sorted(out['per_subject'].items())))
    print(f'  低四={lowm*100:.2f} 高五={highm*100:.2f}  逐人极差={(max(v["mean"] for v in out["per_subject"].values())-min(v["mean"] for v in out["per_subject"].values()))*100:.1f}pp')
    od = BASE / 'results' / 'sep62_eval'
    od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od / 'sep62_eval.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'[落盘] {od}/sep62_eval.json')


if __name__ == '__main__':
    main()
