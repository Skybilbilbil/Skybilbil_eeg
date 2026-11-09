# -*- coding: utf-8 -*-
"""X-3 逐类正确率:哪一类被推动了?——判「侧向不对称」还是「任务窗功率」的判决性证据。

只重算关键 (臂, a) 组合:a ∈ {0, 0.25, 1, 2}(a=1 三臂共用,只算一次)。
逐类正确率 + 混淆矩阵,3 seed 平均。
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
ROOT = BASE.parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'analysis'))
from mAtt.mAtt import BCI2A_CHANNELS                                     # noqa: E402
from analyze_causal_erd_injection import MU, BROAD, TASK, ALPHAS          # noqa: E402
from analyze_causal_deep_reader import (mix_epochs, chan_sets, RUNS,      # noqa: E402
                                        ARGS, WIN, STEP, BETA)
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                  # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'causal_erd_injection'
CLS = ['左手', '右手', '脚', '舌头']
PICK = [0.0, 0.25, 1.0, 2.0]
ARMS = {'mu_contra': (MU, 'contra'), 'mu_ipsi': (MU, 'ipsi'),
        'beta_contra': (BETA, 'contra'), 'broadband': (BROAD, None)}


def main():
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0).astype(np.float64)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_te, y_te = ret[4], ret[5]
    out = {}
    for name, (dirname, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), WIN, STEP, ARGS)
        net.eval()
        rec = {}
        for aname, (band, scope) in ARMS.items():
            cs = chan_sets(y_te, scope) if scope else None
            rec[aname] = {}
            for al in PICK:
                Xm = mix_epochs(X_te, band, al, cs)
                x = torch.Tensor(Xm.astype(np.float32)).unsqueeze(1).to(
                    next(net.parameters()).device)
                with torch.no_grad():
                    pred = net(x).argmax(1).cpu().numpy()
                cm = np.zeros((4, 4), int)
                for t, p in zip(y_te, pred):
                    cm[t, p] += 1
                rec[aname][str(al)] = {'per_class': (cm.diagonal() /
                                                     np.maximum(cm.sum(1), 1)).tolist(),
                                       'confusion': cm.tolist(),
                                       'overall': float((pred == y_te).mean())}
        out[name] = rec
        print(f'{name} done', flush=True)
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    (OUT / 'causal_erd_x3_classes.json').write_text(
        json.dumps(out, ensure_ascii=False, indent=1))
    seeds = [k for k in out if k.startswith('seed')]
    print('\n3-seed 平均逐类正确率(%)')
    print(f"{'臂':>13s} {'a':>5s} " + ''.join(f'{c:>8s}' for c in CLS) + f'{"总体":>8s}')
    for aname in ARMS:
        for al in PICK:
            pc = np.mean([out[s][aname][str(al)]['per_class'] for s in seeds], 0) * 100
            ov = np.mean([out[s][aname][str(al)]['overall'] for s in seeds]) * 100
            print(f'{aname:>13s} {al:5.2f} ' + ''.join(f'{v:8.1f}' for v in pc)
                  + f'{ov:8.1f}')
    print(f'\nwritten {OUT}/causal_erd_x3_classes.json')


if __name__ == '__main__':
    main()
