# -*- coding: utf-8 -*-
"""模型层:试次间差异 vs 个体间差异 —— 冻结已有权重,零训练。

回答的问题:主协议深度模型上,
  (1) 「换人」造成多大落差(个体间信号)?
  (2) 「换一批试次」造成多大波动(试次抽样噪声)?
并把两者放在同一把尺子(百分点的标准差)上比。

做法:复现主协议切分(random_state=42)取测试集 → 3 个 seed 的 cross_mix 权重各跑一次
前向(只跑一次,不重训) → 逐人精度;然后在**同一批留出试次**里按人不重复抽样 m 个,
bootstrap 出逐人精度的抽样标准差 σ_trial(m) —— 不必重复推理,只是对已得预测重采样。

产物 results/model_sensitivity/{model_sensitivity.json, model_sensitivity.log}
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent      # = MAtt_baseline/
sys.path.insert(0, str(BASE))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test   # noqa: E402
from utils.functions import load_matt_model                                    # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'model_sensitivity'
WIN, STEP = 375, 188
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
RUNS = {'seed42': ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024', 'v70s2024'),
        'seed7': ('verify70_seed7', 'v70s7')}
MS = [8, 12, 16, 24, 32, 48, 64, 80]
REPS = 400


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], 0)
    y_all = np.concatenate([sd[1] for sd in subject_data], 0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, train_ratio=0.70, valid_ratio=0.15,
                                 random_state=42, others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print(f'测试集 {len(y_te)} 试次 / {len(set(sub_te.tolist()))} 人;'
          f'每人 {np.bincount(sub_te, minlength=9).tolist()}')

    x = torch.Tensor(X_te).unsqueeze(1)
    y = np.asarray(y_te)
    pred = {}
    for name, (d, tag) in RUNS.items():
        ck = BASE / 'results' / d / f'{tag}_cross_mix_best_acc.pt'
        net = load_matt_model(str(ck), WIN, STEP, ARGS)
        net.eval()
        xd = x.to(next(net.parameters()).device)
        with torch.no_grad():
            pred[name] = net(xd).argmax(1).cpu().numpy()
        print(f'  {name} 前向完成')

    ok = {k: (v == y) for k, v in pred.items()}
    subs = [f'S{i+1:02d}' for i in range(9)]

    # (1) 换人:同一权重下逐人精度的散布
    per = {k: np.array([ok[k][sub_te == s].mean() * 100 for s in range(9)]) for k in ok}
    mean_per = np.mean([per[k] for k in ok], 0)
    seed_sd = np.std([per[k] for k in ok], 0, ddof=1)          # 换训练随机性(同切分)
    # pooled = 全部 778 个测试试次一起算;mean_of_subject = 9 个逐人精度的算术平均。
    # 二者因每人试次数不等(80–92)而差 ~0.6pp,不可混用,故分开存。
    pooled = {k: float(ok[k].mean() * 100) for k in ok}
    print('\n① 换人(同一测试集,逐人精度 %,3 seed 均值):')
    print('   ' + '  '.join(f'{s}:{a:.1f}' for s, a in zip(subs, mean_per)))
    print(f'   pooled {np.mean(list(pooled.values())):.1f}%(778 试次总体),'
          f'逐人均值 {mean_per.mean():.1f}%,极差 {mean_per.max()-mean_per.min():.1f}pp,'
          f'σ_between(逐人精度标准差)={mean_per.std(ddof=1):.1f}pp')
    print(f'   换训练种子(同切分)逐人 σ 均值 {seed_sd.mean():.1f}pp,最大 {seed_sd.max():.1f}pp')

    # (2) 换一批试次:同一权重、同一被试,在其自己的留出试次里抽 m 个
    rng = np.random.default_rng(0)
    pools = [np.where(sub_te == s)[0] for s in range(9)]
    curve = []
    print('\n② 换一批试次(同一权重,每人从自己的留出池里抽 m 个,400 次重复):')
    print('     m     逐人精度抽样 σ(均值)   最差人 σ    全组 pooled σ')
    min_pool = min(len(p) for p in pools)
    for m in MS:
        if m > min_pool:
            print(f'   {m:4d}        跳过(最小留出池只有 {min_pool} 个试次)')
            continue
        sig, sig_pool = [], []
        for k in ok:
            accs = []
            for s in range(9):
                p = pools[s]
                if m > len(p):
                    accs = None
                    break
                idx = np.stack([rng.choice(p, m, replace=False) for _ in range(REPS)])
                accs.append(ok[k][idx].mean(1) * 100)
            if accs is None:
                continue
            A = np.array(accs)                        # (9, REPS) 逐人
            sig.append(A.std(1, ddof=1))
            sig_pool.append(A.mean(0).std(ddof=1))
        s_mean = float(np.mean(sig)); s_max = float(np.max(sig))
        sp = float(np.mean(sig_pool))
        curve.append({'m': m, 'sigma_within_mean': s_mean, 'sigma_within_max': s_max,
                      'sigma_pooled': sp})
        print(f'   {m:4d}        {s_mean:6.2f}            {s_max:6.2f}      {sp:6.2f}')

    sb = float(mean_per.std(ddof=1))
    for c in curve:
        c['ratio'] = sb / c['sigma_within_mean']
    tgt = [c for c in curve if c['sigma_within_mean'] <= sb / 2]
    ref = next((c for c in curve if c['m'] == 32), curve[len(curve) // 2])
    print(f'\n③ 两把尺子对比(单位 = 百分点的标准差):')
    print(f'   个体间 σ_between = {sb:.1f}pp(逐人精度真实散布,极差 '
          f'{mean_per.max()-mean_per.min():.1f}pp;换人不会让它变小)')
    print('   试次间 σ_trial 随每人测试试次数 m 收缩(∝1/√m):')
    for c in curve:
        print(f'      m={c["m"]:3d}   σ_trial={c["sigma_within_mean"]:5.2f}pp'
              f'   σ_between/σ_trial={c["ratio"]:5.2f}')
    print(f'   ⇒ 取 m={ref["m"]} 作参照:试次抽样噪声 ±{ref["sigma_within_mean"]:.1f}pp,'
          f'个体间距 ±{sb:.1f}pp ⇒ 后者是前者的 {ref["ratio"]:.1f} 倍(能量 {ref["ratio"]**2:.0f} 倍)')
    print(f'   ⇒ 只要每人 {tgt[0]["m"] if tgt else ">80"} 个试次,试次抽样噪声就降到个体间距的一半以下')
    print(f'   ⇒ 试次噪声 ∝ 1/√m 可继续压小;逐人排序与 '
          f'{mean_per.max()-mean_per.min():.1f}pp 极差**不随 m 变** —— 模型层的"个体差异主导"。')

    out = {'test_trials': len(y_te), 'per_subject_trials': np.bincount(sub_te, minlength=9).tolist(),
           'per_subject_acc': {k: per[k].tolist() for k in per},
           'acc_mean_per_subject': mean_per.tolist(), 'acc_seed_sd': seed_sd.tolist(),
           'pooled_acc': pooled,
           'mean_of_subject_acc': {k: float(np.mean(per[k])) for k in per},
           'sigma_between': sb, 'within_curve': curve,
           'm_for_half_between': (tgt[0]['m'] if tgt else None),
           'ratio_at_m32': ref['ratio']}
    (OUT / 'model_sensitivity.json').write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(f'\nwritten {OUT}/model_sensitivity.json')


if __name__ == '__main__':
    main()
