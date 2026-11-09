# -*- coding: utf-8 -*-
"""补测:22 通道空间的 oracle 类间分离度,针对**子集干预**(对侧/同侧 ROI)。

为什么必须另做:主 oracle(analyze_causal_erd_injection.step3c)是**全通道** μ 干预,
而 X-3 的四臂是**子集**干预。子集干预在协方差层不能写成 M+(a-1)Ñ(只有那一块通道变),
必须真的重建信号 → 重算协方差 → 再进切空间。

量:D(a)/D(1),分母固定为 a=1 的值(与 step3c 同口径),逐人算再平均。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from mAtt.mAtt import BCI2A_CHANNELS                                      # noqa: E402
from analyze_causal_erd_injection import (MU, TASK, ALPHAS, cov_task,  # noqa: E402
                                          tangent, centered)
from analyze_causal_deep_reader import mix_epochs, chan_sets, ROI_R, ROI_L, ROI_SYM  # noqa: E402
from utils.dataset import load_preprocessed_dataset                        # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
SUBJ = [f'S{i+1:02d}' for i in range(9)]
PICK = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
BETA_BAND = (15.0, 25.0)


def feats_of(X, band, a, chans):
    Zm = mix_epochs(X, band, a, chans)
    return centered(tangent(cov_task(Zm[:, :, TASK[0]:TASK[1]])))


def main():
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    out = {'alphas': PICK, 'per_subject': {}, 'mean': {}}
    arms = {'mu_contra': (MU, 'contra'), 'mu_ipsi': (MU, 'ipsi'),
            'beta_contra': (BETA_BAND, 'contra'), 'mu_all': (MU, None)}
    for aname, (band, scope) in arms.items():
        vals = []
        for s, (Xs, ys) in enumerate(sd):
            X = np.asarray(Xs, dtype=np.float64)
            y = np.asarray(ys)
            cs = chan_sets(y, scope) if scope else None
            F1 = feats_of(X, band, 1.0, cs)
            cen1 = np.stack([F1[y == c].mean(0) for c in range(4)])
            D1 = float(np.mean([np.linalg.norm(cen1[i] - cen1[j])
                                for i in range(4) for j in range(i + 1, 4)]))
            row = []
            for a in PICK:
                F = feats_of(X, band, a, cs)
                cen = np.stack([F[y == c].mean(0) for c in range(4)])
                row.append(float(np.mean([np.linalg.norm(cen[i] - cen[j])
                                          for i in range(4) for j in range(i + 1, 4)])) / D1)
            vals.append(row)
            out['per_subject'].setdefault(aname, {})[SUBJ[s]] = row
            print(f'  {aname:12s} {SUBJ[s]} ' + ' '.join(f'{v:.4f}' for v in row), flush=True)
        m = np.mean(vals, 0)
        out['mean'][aname] = m.tolist()
        print(f'  {aname:12s} 均值      ' + ' '.join(f'{v:.4f}' for v in m), flush=True)
    (OUT / 'causal_erd_oracle_subset.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    print('\n3 人以上平均(D(a)/D(1),分母固定 a=1):')
    print(f"{'臂':>13s} " + ''.join(f'{a:>9.2f}' for a in PICK))
    for aname in arms:
        print(f'{aname:>13s} ' + ''.join(f'{v:9.4f}' for v in out['mean'][aname]))
    print(f'\nwritten {OUT}/causal_erd_oracle_subset.json')


if __name__ == '__main__':
    main()
