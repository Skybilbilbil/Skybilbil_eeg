# -*- coding: utf-8 -*-
"""E-1: 新口径(62.5 Hz + conv2 深度可分离)下重测闭式共享 LDA 头(零训练)。

## 依据
旧口径(250 Hz + 全连接):深度模型 55.4,其 433 维特征(后端 180 ⊕ 原始协方差 253)
上共享闭式 LW-LDA = 61.4(analysis/retrofit_lda_head.py → results/rcslda_eval/)。
新口径主协议 = 58.82,但闭式头上从未测过。判据:pooled 是否 > 62.9(真共享零训练基线)。

## 为什么零训练就够
闭式 LDA 头由训练折解析拟合;原始协方差流 _raw_cov_vec 是固定变换(无参数)。
故只需用已训 3 个 ckpt 的前向特征 + 数据本身构造的协方差流,无需重训。

## 臂(全部共享 shrinkage LW-LDA,主协议 rs=42 同一 train/test 折)
  A 433 = F62(180) ⊕ R62(253)      —— 新口径完整特征
  B 433 = F62(180) ⊕ R250(253)     —— 协方差用全带宽估计
  C 253 = R62                      —— 降采样对经典表示的影响
  D 180 = F62                      —— 深度后端在新口径下的线性可读性
  E 253 = R250                     —— 复现 62.9(锚点自检)

产物 results/lda_head_62/{lda_head_62.json, lda_head_62.log}
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly
from scipy.stats import binomtest
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import load_matt_model                                 # noqa: E402

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
OUT = BASE / 'results' / 'lda_head_62'
BASELINE = BASE / 'results' / 'sep62_full_summary' / 'sep62_full_summary.json'
RUNS = {42: ('sep62_full_s42', 'sep62f42'),
        2024: ('sep62_full_s2024', 'sep62f2024'),
        7: ('sep62_full_s7', 'sep62f7')}
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
WIN, DOWN = 94, 4                       # 62.5 Hz: win=94 点(1.5 s),原 fs 1/4 抽取
ARGS = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=True)


def raw_cov_vec(X, win):
    """原始任务窗协方差切向量(照抄 mAtt._raw_cov_vec 的构造)。

    X (n,22,T) → (n,253):窗 = win 起到试次末;trace 归一 + log-eig 上三角 √2 展开。
    """
    seg = np.asarray(X, dtype=np.float64)[:, :, win:]
    seg = seg - seg.mean(-1, keepdims=True)
    C = seg @ np.swapaxes(seg, 1, 2) / seg.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)
    C = C * (C.shape[1] / tr[:, None, None])
    s, u = np.linalg.eigh(C)
    s = np.clip(s, 1e-10, None)
    L = np.einsum('bij,bj,bkj->bik', u, np.log(s), u)
    iu = np.triu_indices(C.shape[1])
    v = L[:, iu[0], iu[1]].copy()
    v[:, iu[0] != iu[1]] *= np.sqrt(2)
    return v.astype(np.float32)


def _pack(ok, sub_te):
    per = {SUBJ[i]: float(ok[sub_te == i].mean()) for i in range(9)}
    return {'pooled': float(ok.mean()),
            'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([per[SUBJ[i]] for i in HIGH])),
            'per_subject': per}, ok


def evaluate(Ftr, Fte, y_tr, y_te, sub_te):
    """训练折拟合一套共享 LW-LDA → test 折读数 + 逐试次正确性。"""
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, y_tr)
    return _pack(clf.predict(Fte) == y_te, sub_te)


def per_subject_eval(Ftr, Fte, y_tr, y_te, sub_tr, sub_te):
    """共享性判据:拆成 9 个逐人 LDA(仅诊断,用到 subject_id)。"""
    ok = np.zeros(len(y_te), dtype=bool)
    for i in range(9):
        tr, te = sub_tr == i, sub_te == i
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'
                                         ).fit(Ftr[tr], y_tr[tr])
        ok[te] = clf.predict(Fte[te]) == y_te[te]
    return _pack(ok, sub_te)


def mcnemar(a, b):
    """两个读出的逐试次配对 McNemar(scipy 精确二项,双侧)。a,b 为 bool 正确性。"""
    n01, n10 = int(np.sum(~a & b)), int(np.sum(a & ~b))
    p = 1.0 if n01 + n10 == 0 else float(
        binomtest(min(n01, n10), n01 + n10, 0.5).pvalue)
    return {'n_a_only': n10, 'n_b_only': n01, 'p': p}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X250 = np.concatenate([np.asarray(sd[0], dtype=np.float32) for sd in subject_data], 0)
    y_all = np.concatenate([sd[1] for sd in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    X62 = np.concatenate([resample_poly(np.asarray(sd[0], dtype=np.float32), 1, DOWN,
                                        axis=-1) for sd in subject_data], 0)
    print('[数据] 250Hz %s → 62.5Hz %s' % (X250.shape, X62.shape), flush=True)

    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X250, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, ite = ret[9], ret[11]        # others=[sub_all, idx] ⇒ ret[6:]=sub, ret[9:]=idx
    y_tr, y_te, sub_te = ret[1], ret[5], ret[8]
    print('[切分] train %d / test %d' % (len(itr), len(ite)), flush=True)

    R62_tr, R62_te = raw_cov_vec(X62[itr], WIN), raw_cov_vec(X62[ite], WIN)
    R250_tr, R250_te = raw_cov_vec(X250[itr], WIN * DOWN), raw_cov_vec(X250[ite], WIN * DOWN)
    assert R62_tr.shape[1] == 253 == R250_tr.shape[1]

    sub_tr = ret[6]
    base = json.loads(BASELINE.read_text(encoding='utf-8'))
    base['cross_mix']['mean'] /= 100.0        # 汇总 json 存的是百分数
    base['low4_mean'] /= 100.0
    base['high5_mean'] /= 100.0
    base['per_subject'] = {k: v / 100.0 for k, v in base['per_subject'].items()}
    rec = {'note': '闭式共享 LW-LDA,零训练;主协议 rs=42 同折;'
                   'F=后端180(62.5Hz), R=原始协方差253(trace归一+log切空间)',
           'split': {'n_train': int(len(itr)), 'n_test': int(len(ite))},
           'baseline_sep62': {'pooled': base['cross_mix']['mean'],
                              'low4': base['low4_mean'], 'high5': base['high5_mean'],
                              'per_subject': base['per_subject']},
           'fixed_arms': {}, 'per_seed': {}, 'paired': {}}

    # 与 seed 无关的臂:纯数据表示
    fixed_pred = {}
    for name, (Ftr, Fte) in (('C_R62', (R62_tr, R62_te)),
                             ('E_R250', (R250_tr, R250_te))):
        sh, ok_sh = evaluate(Ftr, Fte, y_tr, y_te, sub_te)
        ps, _ = per_subject_eval(Ftr, Fte, y_tr, y_te, sub_tr, sub_te)
        rec['fixed_arms'][name] = {'shared': sh, 'per_subject_lda': ps}
        fixed_pred[name] = ok_sh
        print('[臂 %s] %d维 pooled %.1f | 低四 %.1f 高五 %.1f | 逐人拆开 %.1f'
              % (name, Ftr.shape[1], 100 * sh['pooled'], 100 * sh['low4'],
                 100 * sh['high5'], 100 * ps['pooled']), flush=True)
    rec['paired']['C_R62_vs_E_R250'] = mcnemar(fixed_pred['C_R62'],
                                               fixed_pred['E_R250'])

    # 与 seed 相关的臂:后端特征
    for seed, (dirname, tag) in RUNS.items():
        net = load_matt_model(str(BASE / 'results' / dirname /
                                  f'{tag}_cross_mix_best_acc.pt'), WIN, WIN // 2, ARGS)
        net.eval()
        dev = next(net.parameters()).device
        Xte = X62[ite]

        def feats(X):
            out = []
            with torch.no_grad():
                for i in range(0, len(X), 128):
                    out.append(net._features(
                        torch.Tensor(X[i:i + 128]).unsqueeze(1).to(dev)).cpu().numpy())
            return np.concatenate(out, 0)
        Ftr, Fte = feats(X62[itr]), feats(Xte)
        assert Ftr.shape[1] == 180, Ftr.shape
        # 训练头(SGD+CE,val 选模)在同一 test 折上的逐试次预测
        raw = []
        with torch.no_grad():
            for i in range(0, len(Xte), 128):
                raw.append(net(torch.Tensor(Xte[i:i + 128]).unsqueeze(1).to(
                    dev)).argmax(1).cpu().numpy())
        sh_sgd, ok_sgd = _pack(np.concatenate(raw) == y_te, sub_te)
        # 后端 180 = 5 窗 × 36 维(窗优先展平)⇒ [:, 36:] = 去掉 cue 前静息的窗 0(任务 4 窗)
        arms = {'A_F62_R62': (Ftr, Fte, R62_tr, R62_te),
                'B_F62_R250': (Ftr, Fte, R250_tr, R250_te),
                'D_F62': (Ftr, Fte, None, None),
                'D4_F62': (Ftr[:, 36:], Fte[:, 36:], None, None),
                'A4_F62_R250': (Ftr[:, 36:], Fte[:, 36:], R250_tr, R250_te)}
        rec['per_seed'][str(seed)] = {'sgd_head': sh_sgd}
        pred = {}
        for name, (a, b, c1, c2) in arms.items():
            A_tr = a if c1 is None else np.hstack([a, c1])
            A_te = b if c2 is None else np.hstack([b, c2])
            sh, ok_sh = evaluate(A_tr, A_te, y_tr, y_te, sub_te)
            ps, _ = per_subject_eval(A_tr, A_te, y_tr, y_te, sub_tr, sub_te)
            rec['per_seed'][str(seed)][name] = {
                'dim': int(A_tr.shape[1]), 'shared': sh, 'per_subject_lda': ps}
            pred[name] = ok_sh
            print('[seed%d 臂 %s] %d维 pooled %.1f | 低四 %.1f 高五 %.1f '
                  '| 逐人拆开 %.1f(vs 训练头 %.1f)'
                  % (seed, name, int(A_tr.shape[1]), 100 * sh['pooled'],
                     100 * sh['low4'], 100 * sh['high5'], 100 * ps['pooled'],
                     100 * sh_sgd['pooled']), flush=True)
        rec['paired']['seed%d_A_vs_D' % seed] = mcnemar(pred['A_F62_R62'],
                                                        pred['D_F62'])
        rec['paired']['seed%d_A_vs_E' % seed] = mcnemar(pred['A_F62_R62'],
                                                        fixed_pred['E_R250'])
        rec['paired']['seed%d_B_vs_E' % seed] = mcnemar(pred['B_F62_R250'],
                                                        fixed_pred['E_R250'])
        rec['paired']['seed%d_A_vs_sgd' % seed] = mcnemar(pred['A_F62_R62'], ok_sgd)
        rec['paired']['seed%d_D4_vs_D' % seed] = mcnemar(pred['D4_F62'], pred['D_F62'])
        rec['paired']['seed%d_A4_vs_A' % seed] = mcnemar(pred['A4_F62_R250'],
                                                         pred['A_F62_R62'])

    # 汇总(3 seed 平均;固定臂抄三份)
    DIMS = {'A_F62_R62': 433, 'B_F62_R250': 433, 'D_F62': 180, 'C_R62': 253,
            'E_R250': 253, 'D4_F62': 144, 'A4_F62_R250': 397}
    summary = {}
    for name in DIMS:
        rows = [rec['per_seed'][str(s)].get(name) or rec['fixed_arms'][name]
                for s in RUNS]
        out = {'dim': DIMS[name]}
        for kind in ('shared', 'per_subject_lda'):
            ps = {s: float(np.mean([r[kind]['per_subject'][s] for r in rows]))
                  for s in SUBJ}
            out[kind] = {'pooled': float(np.mean([r[kind]['pooled'] for r in rows])),
                         'low4': float(np.mean([ps[SUBJ[i]] for i in LOW])),
                         'high5': float(np.mean([ps[SUBJ[i]] for i in HIGH])),
                         'per_subject': ps}
        out['delta_vs_sep62_base'] = out['shared']['pooled'] - base['cross_mix']['mean']
        out['delta_low4'] = out['shared']['low4'] - base['low4_mean']
        out['delta_high5'] = out['shared']['high5'] - base['high5_mean']
        out['sharing_gain'] = out['per_subject_lda']['pooled'] - out['shared']['pooled']
        summary[name] = out
    # 训练头(SGD+CE,val 选模)同折读数,3 seed 平均
    sgd_ps = {s: float(np.mean([rec['per_seed'][str(x)]['sgd_head']['per_subject'][s]
                                for x in RUNS])) for s in SUBJ}
    summary['SGD_head'] = {
        'dim': 180,
        'pooled': float(np.mean([rec['per_seed'][str(x)]['sgd_head']['pooled']
                                 for x in RUNS])),
        'low4': float(np.mean([sgd_ps[SUBJ[i]] for i in LOW])),
        'high5': float(np.mean([sgd_ps[SUBJ[i]] for i in HIGH])),
        'per_subject': sgd_ps}
    rec['summary'] = summary
    (OUT / 'lda_head_62.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')

    print('\n=== E-1 新口径闭式 LDA 头(3 seed,同一 train/test 折) ===')
    print('%-12s %5s %7s %7s %7s %8s %9s'
          % ('臂', '维', 'pooled', '低四', '高五', 'Δpooled', '逐人拆开'))
    for name, r in summary.items():
        if 'shared' in r:
            sh, gain = r['shared'], '%+.1f' % (100 * r['sharing_gain'])
        else:
            sh, gain = r, '(诊断)'
        print('%-12s %5d %6.1f%% %6.1f%% %6.1f%% %+8.2f %9s'
              % (name, r['dim'], 100 * sh['pooled'], 100 * sh['low4'],
                 100 * sh['high5'],
                 100 * (sh['pooled'] - base['cross_mix']['mean']), gain))
    print('基线 sep62 深度模型(训练头): %.1f%%(低四 %.1f 高五 %.1f);'
          '真共享零训练基线 62.9'
          % (100 * base['cross_mix']['mean'], 100 * base['low4_mean'],
             100 * base['high5_mean']))
    print('配对 McNemar(精确二项,双侧;n=778):')
    for k, v in rec['paired'].items():
        print('  %-22s a独对 %3d / b独对 %3d  p=%.3g'
              % (k, v['n_a_only'], v['n_b_only'], v['p']))
    print('[OK] %s/lda_head_62.json' % OUT)


if __name__ == '__main__':
    main()
