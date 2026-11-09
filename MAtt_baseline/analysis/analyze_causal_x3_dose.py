# -*- coding: utf-8 -*-
"""X-3 的剂量归一化解释:子集干预的 D(a)/D(1) 大动,是因为「剂量更大」还是「方向不同」?

对每个臂每个 a 算 δ_s(a) = ‖居中切空间(干预后) − 居中切空间(a=1)‖,再与 D(a)/D(1) 并列。
δ 用与 step3c 相同的「全 22 通道切空间欧氏距离」口径,故可与 step3c 表直接比。
"""
import json
import sys
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from analyze_causal_erd_injection import (MU, TASK, cov_task, tangent,   # noqa: E402
                                          centered)
from analyze_causal_deep_reader import mix_epochs, chan_sets, ROI_R, ROI_L, ROI_SYM  # noqa: E402
from utils.dataset import load_preprocessed_dataset                        # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
PICK = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
BETA_BAND = (15.0, 25.0)


def main():
    sub = json.load(open(OUT / 'causal_erd_oracle_subset.json'))
    sd, _ = load_preprocessed_dataset(str(DATA_PATH))
    arms = {'mu_contra': (MU, 'contra'), 'mu_ipsi': (MU, 'ipsi'),
            'beta_contra': (BETA_BAND, 'contra'), 'mu_all': (MU, None)}
    print('臂            a      δ_s/σ_trial(9人中位)   D(a)/D(1)    两个量是否同向放大')
    rows = {}
    for aname, (band, scope) in arms.items():
        ds, Dl = [], sub['mean'][aname]
        for Xs, ys in sd:
            X = np.asarray(Xs, dtype=np.float64)
            y = np.asarray(ys)
            cs = chan_sets(y, scope) if scope else None
            def F(a):
                return centered(tangent(cov_task(
                    mix_epochs(X, band, a, cs)[:, :, TASK[0]:TASK[1]])))
            F1 = F(1.0)
            sig = float(np.sqrt((F1 ** 2).sum(1).mean()))
            ds.append([float(np.linalg.norm(F(a) - F1)) / sig for a in PICK])
        ds = np.array(ds)
        rows[aname] = {'delta_over_sigma': ds.tolist(), 'D_over_D1': Dl}
    print(f"{'a':>6s}" + ''.join(f'{k:>22s}' for k in arms))
    for i, a in enumerate(PICK):
        line = f'{a:6.2f}'
        for aname in arms:
            d = rows[aname]['delta_over_sigma']
            line += f'   δ {np.median([r[i] for r in d]):7.2f}  D {rows[aname]["D_over_D1"][i]:6.3f}'
        print(line)
    (OUT / 'causal_erd_x3_dose.json').write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/causal_erd_x3_dose.json')


if __name__ == '__main__':
    main()
