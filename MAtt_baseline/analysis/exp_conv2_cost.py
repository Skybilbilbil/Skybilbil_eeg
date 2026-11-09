# -*- coding: utf-8 -*-
"""conv2 算力对照: 降采样 125 Hz vs 深度可分离 vs 基线

## 背景(为什么做)
主协议 conv2 占 58.6% 参数 / 81.3% 乘加(d_attn_in=8);原文口径(d_attn_in=20)占 92.0%。
250 Hz 下 12 抽头 = 48 ms ⇒ 分辨率 20.8 Hz,而输入带宽 8–30 Hz 只有 22 Hz
⇒ 时间-带宽积 1.06,物理上放不下第二个瓣。原文是 128 Hz:12 抽头 = 94 ms、带宽 34 Hz
⇒ 时间-带宽积 3.19,够放 3 个频带。results/conv2_spectrum/ 已实测:学到核的
带内起伏 cv 0.495 vs 随机 12 抽头核 0.527,仅 4.5% 超随机 p95 ⇒ 250 Hz 下 conv2
没学到频带选择。本实验检验两件事:
  A) 把采样率对齐到原文设计(125 Hz,12 抽头 = 96 ms)后,算力减半是否掉点;
  B) 深度可分离(470 参数 / -79% 乘加)是否掉点 —— 8×22×12 张量不可分离
     (每输出 22×12 矩阵第一奇异值仅 25%)则 B 应有损。

## 臂(每臂只改一处,其余逐参一致)
  base   现状:250 Hz,conv2 = Conv2d(22→8,(1,12),pad=6)
  ds125  250→125 Hz 2 倍抽取(resample_poly, Kaiser 抗混叠),12 抽头 = 96 ms ≈ 原文 94 ms
  sep    250 Hz,conv2 = 深度可分离(dw 22×12 + pw 22→8)
  sep125 两处一起改:125 Hz + 深度可分离(2×2 表的第四格,检验两者是否可叠加)

## 配对
同 seed、同 70/15/15 切分(**在 250 Hz 上切完再按臂下采样**,故试次划分逐位一致)、
同 Adam(lr=5e-4, wd=0.02)、同 iters / bs=128 / grad_clip=1.0。

## 判读(事先写死)
  1) 配对差 d = arm − base,逐 seed 同切分同 seed 才是配对
  2) 单折 <5pp 不可测量(见 差异报告/差异测量的数据底座与噪声预算.md 的硬规则)
  3) 组差 d_low4 − d_high5:两者同步平移(<2pp)⇒ 只整体平移,不改组间结构
  4) "省算力且不掉点" = |d_pooled| < 5pp 且组差 <2pp

用法: python analysis/exp_conv2_cost.py --arms base,ds125,sep --seeds 42,2024,7 --iters 100
只写 results/conv2_cost/。
"""
import sys, json, time, argparse
from pathlib import Path
from math import gcd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
import torch, torch.nn as nn
from scipy.signal import resample_poly
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from mAtt.mAtt import mAtt_bci

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS0 = 250
LOW4 = {2, 4, 5, 6}          # S02,S04,S05,S06(1-based)
ARMS = {
    'base':   dict(fs=250,  sep=False),
    'ds125':  dict(fs=125,  sep=False),
    'sep':    dict(fs=250,  sep=True),
    'sep125': dict(fs=125,  sep=True),
    'ds62':   dict(fs=62.5, sep=False),
    'sep62':  dict(fs=62.5, sep=True),
}
log = lambda *a: print(*a, flush=True)


def load_splits():
    """在 250 Hz 上切分一次(所有臂共用同一组试次索引)。"""
    data, _ = load_preprocessed_dataset(
        str(BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'))
    Xs, ys, ss = [], [], []
    for s in range(len(data)):
        X = np.asarray(data[s][0], dtype=np.float32)
        y = np.asarray(data[s][1])
        Xs.append(X); ys.append(y); ss.append(np.full(len(y), s, dtype=np.int64))
    X = np.concatenate(Xs); y = np.concatenate(ys); sub = np.concatenate(ss)
    o = split_train_valid_test(X, y, train_ratio=.70, valid_ratio=.15,
                               random_state=42, others=[sub])
    return (o[0], o[1], o[6]), (o[2], o[3], o[7]), (o[4], o[5], o[8])


def to_fs(X, fs):
    """FS0/k 的整数倍抽取(250/2=125、250/4=62.5),Kaiser 抗混叠。"""
    if fs == FS0:
        return np.asarray(X, dtype=np.float32)
    down = int(round(FS0 / fs))
    return resample_poly(X, 1, down, axis=-1).astype(np.float32)


def make_net(fs, sep):
    win, step = int(round(1.5 * fs)), int(round(0.75 * fs))
    net = mAtt_bci(win, step, n_classes=4, d_attn_in=8, d_attn_out=8,
                   local_radius=6, anchor_stride=None, n_heads=1, n_windows=None,
                   use_hann=True, n_spatial_filters=22, dropout=0.3,
                   norm_type='in', conv2_separable=sep)
    return net


def mac_conv2(net, T):
    """conv2 的乘加(单样本);深度可分离按 dw+pw 计。"""
    if isinstance(net.conv2, nn.Conv2d):
        out_t = T + 2 * net.conv2.padding[1] - net.conv2.kernel_size[1] + 1
        return net.conv2.out_channels * out_t * (net.conv2.in_channels * net.conv2.kernel_size[1])
    m = net.conv2
    out_t = T + 2 * m.padding - m.kernel_size + 1
    dw = m.in_channels * T * m.kernel_size
    pw = m.out_channels * out_t * m.in_channels
    return dw + pw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arms', default='base,ds125,sep')
    ap.add_argument('--seeds', default='42,2024,7')
    ap.add_argument('--iters', type=int, default=100)
    a = ap.parse_args()
    arms = [s.strip() for s in a.arms.split(',') if s.strip()]
    seeds = [int(s) for s in a.seeds.split(',')]

    (Xtr, ytr, str_), (Xva, yva, sva), (Xte, yte, ste) = load_splits()
    log(f'[数据] tr={len(ytr)} va={len(yva)} te={len(yte)}  T={Xtr.shape[-1]} @ {FS0}Hz  dev={DEV}')
    out = {'iters': a.iters, 'seed_list': seeds, 'fs0': FS0, 'arms': {}}

    for arm in arms:
        cfg = ARMS[arm]
        Xa_tr, Xa_va, Xa_te = (to_fs(X, cfg['fs']) for X in (Xtr, Xva, Xte))
        T = Xa_tr.shape[-1]
        log(f'\n===== arm={arm} fs={cfg["fs"]} sep={cfg["sep"]} T={T} =====')
        rec_arm = {'cfg': cfg, 'T': T, 'runs': {}}

        for sd in seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            t0 = time.time()
            net = make_net(cfg['fs'], cfg['sep']).to(DEV)
            net.eval()                                  # 触发 LazyLinear 定维,不污染 running stats
            with torch.no_grad():
                net(torch.zeros(1, 1, 22, T, device=DEV))
            npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
            M = net.linear.weight.shape[1] // (8 * 9 // 2)
            mac = mac_conv2(net, T)
            log(f'  seed{sd} params={npar} M={M} conv2_MAC={mac/1e3:.0f}k')

            CE = nn.CrossEntropyLoss()
            opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad],
                                   lr=5e-4, weight_decay=0.02)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)

            def mk(X, y, s, sh):
                ds = torch.utils.data.TensorDataset(
                    torch.Tensor(X).unsqueeze(1), torch.Tensor(y).long(),
                    torch.Tensor(s).long())
                return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=sh)

            tr, va, te = mk(Xa_tr, ytr, str_, True), mk(Xa_va, yva, sva, False), mk(Xa_te, yte, ste, False)
            rec = {'iter': [], 'train_loss': [], 'val_acc': [], 'test_acc': []}
            for it in range(a.iters):
                net.train(); ls = n = 0
                for xb, yb, _ in tr:
                    xb, yb = xb.to(DEV), yb.to(DEV)
                    loss = CE(net(xb), yb)
                    opt.zero_grad(); loss.backward()
                    nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
                    opt.step()
                    ls += loss.item() * len(yb); n += len(yb)
                sch.step(); net.eval()

                def ev(ld):
                    ok = tot = 0
                    with torch.no_grad():
                        for xb, yb, _ in ld:
                            xb, yb = xb.to(DEV), yb.to(DEV)
                            ok += (net(xb).argmax(1) == yb).sum().item(); tot += len(yb)
                    return ok / tot

                rec['iter'].append(it)
                rec['train_loss'].append(ls / n)
                rec['val_acc'].append(ev(va))
                rec['test_acc'].append(ev(te))
                if it % 20 == 0 or it == a.iters - 1:
                    log(f'    it{it:3d} loss={ls/n:.4f} val={rec["val_acc"][-1]*100:.2f}% '
                        f'test={rec["test_acc"][-1]*100:.2f}%')

            net.eval(); ps, ys = [], []
            with torch.no_grad():
                for xb, yb, _ in te:
                    ps.append(net(xb.to(DEV)).argmax(1).cpu().numpy()); ys.append(yb.numpy())
            pred = np.concatenate(ps); yy = np.concatenate(ys)
            per = {}
            for s in np.unique(ste):
                m = ste == s
                per[f'S{int(s)+1:02d}'] = [int(m.sum()), float((pred[m] == yy[m]).mean())]
            rec.update(per_subject=per, secs=round(time.time() - t0, 1),
                       n_params=int(npar), M=int(M), conv2_mac=float(mac),
                       test_final=float(rec['test_acc'][-1]), test_peak=float(max(rec['test_acc'])))
            rec_arm['runs'][str(sd)] = rec
            log(f'  seed{sd} 完成 {rec["secs"]}s  final={rec["test_final"]*100:.2f}%  '
                f'peak={rec["test_peak"]*100:.2f}%')
        out['arms'][arm] = rec_arm

    # ---------- 汇总 ----------
    def agg(arm):
        runs = out['arms'][arm]['runs']
        pooled = [runs[str(s)]['test_final'] for s in seeds]
        low4, high5 = [], []
        for s in seeds:
            per = runs[str(s)]['per_subject']
            low4.append(float(np.mean([v[1] for k, v in per.items() if int(k[1:]) in LOW4])))
            high5.append(float(np.mean([v[1] for k, v in per.items() if int(k[1:]) not in LOW4])))
        return dict(pooled=float(np.mean(pooled)), pooled_sd=float(np.std(pooled, ddof=1)) if len(pooled) > 1 else 0.0,
                    low4=float(np.mean(low4)), high5=float(np.mean(high5)),
                    n_params=out['arms'][arm]['runs'][str(seeds[0])]['n_params'],
                    conv2_mac=out['arms'][arm]['runs'][str(seeds[0])]['conv2_mac'])

    summ = {arm: agg(arm) for arm in arms}
    if 'base' in summ:
        for arm in arms:
            if arm == 'base':
                continue
            summ[arm]['d_pooled'] = summ[arm]['pooled'] - summ['base']['pooled']
            summ[arm]['d_low4'] = summ[arm]['low4'] - summ['base']['low4']
            summ[arm]['d_high5'] = summ[arm]['high5'] - summ['base']['high5']
            summ[arm]['d_low4_minus_high5'] = summ[arm]['d_low4'] - summ[arm]['d_high5']
    out['summary'] = summ
    log('\n===== 汇总 =====')
    for arm, s in summ.items():
        log(f'{arm:6s} pooled={s["pooled"]*100:.2f}%±{s["pooled_sd"]*100:.2f} '
            f'low4={s["low4"]*100:.2f} high5={s["high5"]*100:.2f} '
            f'params={s["n_params"]} conv2MAC={s["conv2_mac"]/1e3:.0f}k'
            + (f'  d_pooled={s["d_pooled"]*100:+.2f} d_low4-h5={s["d_low4_minus_high5"]*100:+.2f}'
               if 'd_pooled' in s else ''))

    od = BASE / 'results' / 'conv2_cost'
    od.mkdir(parents=True, exist_ok=True)
    p = od / 'conv2_cost.json'
    json.dump(out, open(p, 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    log(f'[落盘] {p}')


if __name__ == '__main__':
    main()
