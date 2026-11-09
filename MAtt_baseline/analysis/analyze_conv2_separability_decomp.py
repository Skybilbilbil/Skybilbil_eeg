# -*- coding: utf-8 -*-
"""分离 conv2 出口可分性下降的两个来源(零训练)。

问题: 新口径 sep62 下 conv2 出口的逐人探针低四从 46.5% 掉到 29.4%。
两个候选: ①62.5 Hz 每窗只有 94 点估 8x8 协方差(64 参数); ②深度可分离压掉 176 个
独立时间滤波器。本脚本用**旧 250 Hz 全连接 ckpt** 只改"估计点数",把两者分开:

  A 旧口径   : 250 Hz ckpt, conv2 输出按 5 窗各取 375 点 -> 协方差
  B 仅抽稀   : 同一网络/同一输出, 每窗 375 点做 4 倍平均池化到 94 点 -> 协方差
  C 新口径   : 62.5 Hz 可分离 ckpt, 每窗 94 点 -> 协方差(取自 pipeline_attribution_62.5hz)

B-A = 估计点数(降采样)的代价; C-B = 可分离表达力的代价。
只写 results/conv2_separability_decomp/。
"""
import sys, json
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'analysis'))
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model

torch.set_num_threads(4)
DEV = torch.device('cpu')
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
WIN, STEP = 375, 188
LOW4 = ['S02', 'S04', 'S05', 'S06']
HIGH5 = ['S01', 'S03', 'S07', 'S08', 'S09']
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3)
RUNS = {'seed42': ('verify_samepool_70_15_15', 'verify70'),
        'seed2024': ('verify70_seed2024', 'v70s2024'),
        'seed7': ('verify70_seed7', 'v70s7')}


def spd_feats(seg):
    """seg:(n,8,L) -> trace 归一化 + log 均值 -> (n,36)"""
    x = seg - seg.mean(axis=-1, keepdims=True)
    C = np.einsum('nct,ndt->ncd', x, x) / (x.shape[-1] - 1)
    C = C / np.trace(C, axis1=-2, axis2=-1)[..., None, None]
    C = C + 1e-5 * np.eye(C.shape[-1])[None]
    s, u = np.linalg.eigh(C)
    s = np.clip(np.abs(s), 1e-8, None)
    L = (u * np.log(s)[..., None, :]) @ np.swapaxes(u, -1, -2)
    r, c = np.triu_indices(L.shape[-1])
    return L[:, r, c]


def pool4(seg):
    """(n,8,L) -> (n,8,L//4) 4 点平均池化(等效 250->62.5)"""
    n, c, L = seg.shape
    L4 = L // 4
    return seg[:, :, :L4 * 4].reshape(n, c, L4, 4).mean(-1)


def readout(X, ys, seed=0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    nsp = max(2, min(5, int(np.bincount(ys).min())))
    cv = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    return float(cross_val_score(clf, X, ys, cv=cv).mean())


def main():
    data, _ = load_preprocessed_dataset(str(DATA))
    X_all = np.concatenate([d[0] for d in data], axis=0).astype(np.float32)
    y_all = np.concatenate([d[1] for d in data], axis=0)
    sub_all = np.concatenate([np.full(len(d[1]), i, dtype=np.int64)
                              for i, d in enumerate(data)])
    print(f'[data] {X_all.shape}', flush=True)

    res = {k: {} for k in ('A_375', 'B_94', 'C_94')}
    for tag, (dn, pre) in RUNS.items():
        ck = BASE / 'results' / dn / f'{pre}_cross_mix_best_acc.pt'
        net = load_matt_model(str(ck), WIN, STEP, ARGS).to(DEV).eval()
        bufs = {'A': [], 'B': []}
        with torch.no_grad():
            for b in range(0, len(X_all), 128):
                xb = torch.from_numpy(X_all[b:b + 128]).unsqueeze(1).to(DEV)
                h = net.channel_weight(xb)
                h = net.In1(net.conv1(h))
                h = net.conv2(h)                     # (B,8,1,T+1)
                h = h[:, :, 0, :].cpu().numpy()
                # 取 M=5 个窗
                for i in range(5):
                    seg = h[:, :, i * STEP:i * STEP + WIN]     # (B,8,375)
                    bufs['A'].append(seg)
                    bufs['B'].append(pool4(seg))               # (B,8,93)
        A = np.concatenate(bufs['A'], 0); B = np.concatenate(bufs['B'], 0)
        # 每试次 M 窗的 log 均值: 先按 (n*M,8,L) 算 -> (n,M,36) -> mean
        fa = spd_feats(A).reshape(len(X_all), 5, -1).mean(1)
        fb = spd_feats(B).reshape(len(X_all), 5, -1).mean(1)
        for k, F in (('A_375', fa), ('B_94', fb)):
            lo = [readout(F[sub_all == i], y_all[sub_all == i]) for i in range(9)]
            lo_w = np.mean([v for v, s in zip(lo, [f'S{i+1:02d}' for i in range(9)]) if s in LOW4])
            hi_w = np.mean([v for v, s in zip(lo, [f'S{i+1:02d}' for i in range(9)]) if s in HIGH5])
            res[k].setdefault(tag, {})['low4'] = float(lo_w * 100)
            res[k][tag]['high5'] = float(hi_w * 100)
        print(f'  {tag}: A(375 点) 低四 {res["A_375"][tag]["low4"]:.1f} 高五 {res["A_375"][tag]["high5"]:.1f}'
              f' | B(94 点) 低四 {res["B_94"][tag]["low4"]:.1f} 高五 {res["B_94"][tag]["high5"]:.1f}', flush=True)

    # C: 新口径直接读已有探针 json
    pa = json.load(open(BASE / 'results' / 'pipeline_attribution_62.5hz' / 'pipeline_attribution.json'))
    for tag in RUNS:
        t = pa['runs'][tag]['trace']['conv2_协方差']
        res['C_94'].setdefault(tag, {})['low4'] = t['low4_full'] * 100
        res['C_94'][tag]['high5'] = t['high5_full'] * 100

    summ = {}
    for k in res:
        summ[k] = {'low4': float(np.mean([res[k][t]['low4'] for t in res[k]])),
                   'high5': float(np.mean([res[k][t]['high5'] for t in res[k]]))}
    summ['_note'] = ('A=250Hz 全连接 conv2 + 375 点协方差; B=同一网络输出池化到 94 点; '
                     'C=62.5Hz 可分离 conv2 + 94 点。B-A 只含估计点数, C-B 只含表达力。')
    od = BASE / 'results' / 'conv2_separability_decomp'
    od.mkdir(parents=True, exist_ok=True)
    json.dump({'per_seed': res, 'summary': summ}, open(od / 'decomp.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=1)
    print('\n=== 分解(3 seed 均值) ===')
    for k in ('A_375', 'B_94', 'C_94'):
        print(f'  {k:6s} 低四 {summ[k]["low4"]:5.1f}  高五 {summ[k]["high5"]:5.1f}')
    print(f'  估计点数代价(B-A) 低四 {summ["B_94"]["low4"]-summ["A_375"]["low4"]:+.1f}'
          f' | 表达力代价(C-B) 低四 {summ["C_94"]["low4"]-summ["B_94"]["low4"]:+.1f}')
    print(f'[落盘] {od}/decomp.json')


if __name__ == '__main__':
    main()
