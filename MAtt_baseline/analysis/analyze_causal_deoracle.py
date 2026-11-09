# -*- coding: utf-8 -*-
"""X-3 去 oracle(de-label)检验:唯一超 5pp 地板的正读数去掉标签条件后还剩多少。

## 背景(读数来自 results/causal_erd_injection/causal_erd_deep_x3.json)
X-3 四臂里 β 对侧 a=0(抹平对侧 ROI 的任务窗 β 试次级波动)是全项目唯一
「低四人被单独抬起」形状的读数:低四 +11.4pp / 高五 +3.9pp(低−高 +7.5,
逐 seed +10.6/+6.9/+4.9 全正)。但「对侧」按**试次类标签**选半球,是 oracle,
不能部署。μ 对侧 a=0.25 形状更极端(低四 +8.5 / 高五 −5.9)但拉低高五。

## 本脚本:两个去标签化改动,其余逐字复用 X-3 机制
  ① 作用域:对侧(按标签)→ 双侧 ROI 合集 ROI_SYM(14 通道,全部试次同一集合);
  ② 模板:X-3 的 mix_epochs 用测试折全体均值 → 改为**训练折**估计
     (pooled = 全体训练试次均值;per_subject = 逐人训练试次均值,更贴近
      「只删试次级波动、不碰被试均值」的原始定义)。
臂 = {μ(8-13), β(15-25)} × {pooled, per_subject} × a ∈ {0, 0.25};冻结 3 seed
主协议测试折(778 试次),零训练。判据 = 分组响应 + 逐人分布 + MDE(2×seed-sd)。
只写 results/causal_deoracle/。
"""
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
from analyze_causal_erd_injection import MU, TASK, narrow            # noqa: E402
from analyze_causal_deep_reader import (BETA, RUNS, ARGS, WIN, STEP,  # noqa: E402
                                        ROI_SYM, _CH, SUBJ)
from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                           # noqa: E402

DATA_PATH = ROOT / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
X3 = BASE / 'results' / 'causal_erd_injection' / 'causal_erd_deep_x3.json'
OUT = BASE / 'results' / 'causal_deoracle'
CH_BOTH = np.array([_CH[n] for n in ROI_SYM])
ALPHAS = (0.0, 0.25)
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


def mix_both(X, Zb, a, tpl, tpl_sub=None, sub=None):
    """双侧 ROI 合集上的窄带收缩(Y = X + (√a−1)(Z_band − 模板)),标签自由。

    tpl_sub 给定时用逐人模板(sub = 逐试次被试索引),否则全体共用 tpl。
    与 X-3 的 mix_epochs 唯一区别:通道集不随标签变、模板来自训练折。
    """
    t0, t1 = TASK
    Y = X.copy()
    k = np.sqrt(a) - 1.0
    if tpl_sub is None:
        Y[:, CH_BOTH, t0:t1] = (X[:, CH_BOTH, t0:t1]
                                + k * (Zb[:, CH_BOTH] - tpl[CH_BOTH][None]))
    else:
        for s in np.unique(sub):
            idx = np.where(sub == s)[0]
            Y[idx[:, None], CH_BOTH[None, :], t0:t1] = (
                X[idx[:, None], CH_BOTH[None, :], t0:t1]
                + k * (Zb[idx][:, CH_BOTH] - tpl_sub[s][CH_BOTH][None]))
    return Y


def eval_full(net, Y, y, sub):
    """单次前向同时给 pooled / 逐人 / 逐类三个读数。"""
    pred = net(torch.Tensor(Y.astype(np.float32)).unsqueeze(1).to(
        next(net.parameters()).device)).argmax(1).cpu().numpy()
    ok = pred == y
    return (float(ok.mean()),
            {SUBJ[s]: float(ok[sub == s].mean()) for s in range(9)},
            {int(c): float(ok[y == c].mean()) for c in range(4)})


def main():
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], 0).astype(np.float64)
    y_all = np.concatenate([sd[1] for sd in subject_data], 0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all])
    X_tr, y_tr, sub_tr = ret[0], ret[1], ret[6]
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print('去 oracle 检验:双侧 ROI(14ch,标签自由)× 训练折模板;测试折 %d 试次'
          % len(y_te), flush=True)

    nets = {}
    for name, (dirname, tag) in RUNS.items():
        ck = BASE / 'results' / dirname / f'{tag}_cross_mix_best_acc.pt'
        if not ck.exists():
            print('  缺 ckpt,跳过 %s' % name)
            continue
        net = load_matt_model(str(ck), WIN, STEP, ARGS)
        net.eval()
        nets[name] = net
    seeds = list(nets)
    base = {n: eval_full(nets[n], X_te, y_te, sub_te) for n in seeds}
    print('  干净基线: ' + ' '.join('%s %.1f%%' % (n, base[n][0] * 100) for n in seeds),
          flush=True)

    out = {'baseline': {n: {'overall': base[n][0], 'per_subject': base[n][1],
                            'per_class': base[n][2]} for n in seeds}, 'arms': {}}
    x3 = json.loads(X3.read_text(encoding='utf-8'))
    for bname, band in (('mu', MU), ('beta', BETA)):
        Z_tr = narrow(X_tr, band)[:, :, TASK[0]:TASK[1]]
        Z_te = narrow(X_te, band)[:, :, TASK[0]:TASK[1]]
        tpl_pool = Z_tr.mean(0)
        tpl_sub = {int(s): Z_tr[sub_tr == s].mean(0) for s in np.unique(sub_tr)}
        for tname in ('pooled', 'per_subject'):
            for a in ALPHAS:
                t0 = time.time()
                if tname == 'pooled':
                    Y = mix_both(X_te, Z_te, a, tpl_pool)
                else:
                    Y = mix_both(X_te, Z_te, a, None, tpl_sub, sub_te)
                rec = {'overall': {}, 'per_subject': {}, 'per_class': {}}
                for n in seeds:
                    acc, per, percls = eval_full(nets[n], Y, y_te, sub_te)
                    rec['overall'][n] = acc
                    rec['per_subject'][n] = per
                    rec['per_class'][n] = percls
                out['arms'][f'{bname}_{tname}_a{a}'] = rec
                dl = np.array([[rec['per_subject'][n][SUBJ[i]] -
                                base[n][1][SUBJ[i]] for i in range(9)] for n in seeds])
                dO = np.array([rec['overall'][n] - base[n][0] for n in seeds])
                lo, hi = dl[:, LOW].mean(1) * 100, dl[:, HIGH].mean(1) * 100
                print('%-22s 低四 %+5.2f 高五 %+5.2f (低−高 %+5.2f)  pooled %+5.2f  '
                      '逐人 %s  (%.0fs)'
                      % (f'{bname}/{tname}/a={a}', lo.mean(), hi.mean(),
                         lo.mean() - hi.mean(), dO.mean() * 100,
                         ' '.join('%s%+.1f' % (SUBJ[i], dl[:, i].mean() * 100)
                                  for i in range(9)), time.time() - t0), flush=True)
                del Y
        # 与 oracle 臂并排(X-3 原始数字,模板=测试折均值、作用域=按标签对侧)
        for a in ALPHAS:
            key = str(a)
            dl = []
            for sd in seeds:
                ps = x3[sd][f'{bname}_contra']['per_subject']
                dl.append(np.array([ps[key][SUBJ[i]] - ps['1.0'][SUBJ[i]]
                                    for i in range(9)]))
            m = np.mean(dl, 0)
            print('  [oracle 对照 %s_contra a=%s] 低四 %+5.2f 高五 %+5.2f (低−高 %+5.2f)'
                  % (bname, key, m[LOW].mean() * 100, m[HIGH].mean() * 100,
                     (m[LOW].mean() - m[HIGH].mean()) * 100), flush=True)

    (OUT / 'causal_deoracle.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/causal_deoracle.json' % OUT)


if __name__ == '__main__':
    main()
