# -*- coding: utf-8 -*-
"""I-1'/I-2'/I-3' 逐级串联追踪（零训练整改第一档）。

回答：
  I-1' 信息在哪一级被丢掉：前向逐级出口，用同一套读出流程测被试内 4 类可分性
  I-2' 共享基够不够：conv1 学到的空间方向 vs 全 22 维（在协方差上直接做 W C W^T，等价于对信号做投影再算协方差）
  I-3' SPDNorm 的参考均值/gamma 在帮谁

数学要点：
  - 某级的协方差 C = cov(x)，把该级输出线性投影到新基 W 后的协方差严格等于 W C W^T
    （trace 归一化与中心化在两边一致）=> 不必重放时域信号即可比较「基」的作用。
  - SPDNorm: Y = expm(gamma*(logX - logM) + beta*I)。log 域平移是等距映射：
      d_LE(iso(X1), iso(X2)) = d_LE(X1, X2)
    => running_mean 不能改变被试间相对几何，唯一能改变相对关系的是 gamma 的乘性缩放。
  - E2R 对每窗 trace 归一化 => SPD 路线整体功率被消掉，只留形状（与项目已知结论一致）。

内存约束：本机 WSL 上限 6GB，故逐级只保留逐样本协方差（22x22 或 8x8），不缓存时域信号。
只读 ckpt + 原始信号，不训练。产物 results/pipeline_attribution/。
"""

import sys, json, time, os
from pathlib import Path
import numpy as np
import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import load_matt_model

torch.set_num_threads(4)   # 内存受限，别开满
DEV = torch.device('cpu')

DATA_PATH = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
# PA_FS 切换口径(2026-10-30): 默认 250 = 旧口径; PA_FS=62.5 量 train_pair 新默认
# (62.5 Hz + 深度可分离)的逐级特征。窗长按秒不变,故 WIN/STEP 随采样率缩放。
PA_FS = float(os.environ.get('PA_FS', '250'))
WIN, STEP = ((375, 188) if PA_FS == 250
             else (int(round(1.5 * PA_FS)), int(round(0.75 * PA_FS))))
LOW4 = [2, 4, 5, 6]
HIGH5 = [1, 3, 7, 8, 9]
ARGS = dict(dataset='bci2a', n_classes=4, d_attn_in=8, d_attn_out=8,
            local_radius=6, anchor_stride=None, n_heads=1, dropout=0.3,
            conv2_separable=(PA_FS != 250))
RUNS = ({'seed42':   ('verify_samepool_70_15_15', 'verify70'),
         'seed2024': ('verify70_seed2024',        'v70s2024'),
         'seed7':    ('verify70_seed7',           'v70s7')} if PA_FS == 250 else
        {'seed42':   ('sep62_s42',  'sep62s42'),
         'seed2024': ('sep62_s2024', 'sep62s2024'),
         'seed7':    ('sep62_s7',   'sep62s7')})
OUT = BASE / 'results' / ('pipeline_attribution' if PA_FS == 250
                              else f'pipeline_attribution_{PA_FS:g}hz')  # 不覆盖 250 Hz 旧产物
CH = ['Fz','FC3','FC1','FCz','FC2','FC4','C5','C3','C1','Cz','C2','C4','C6',
      'CP3','CP1','CPz','CP2','CP4','P1','Pz','P2','POz']


def _gmean(p, idxs):
    vals = [p.get(f'S{i:02d}', np.nan) for i in idxs]
    return float(np.nanmean(vals)) if any(v == v for v in vals) else float('nan')


def group_mean_low(p):
    return _gmean(p, LOW4)


def group_mean_high(p):
    return _gmean(p, HIGH5)


def readout(X, ys, seed=0):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    n = len(ys)
    if n < 20 or len(np.unique(ys)) < 2:
        return float('nan')
    nsp = max(2, min(5, int(np.bincount(ys).min())))
    cv = StratifiedKFold(n_splits=nsp, shuffle=True, random_state=seed)
    clf = make_pipeline(StandardScaler(),
                        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'))
    try:
        return float(cross_val_score(clf, X, ys, cv=cv).mean())
    except Exception:
        return float('nan')


def logm(M):
    s, u = np.linalg.eigh(M)
    s = np.clip(np.abs(s), 1e-8, None)
    return (u * np.log(s)[..., None, :]) @ np.swapaxes(u, -1, -2)


def spdvec(M):
    r, c = np.triu_indices(M.shape[-1])
    return M[..., r, c]


def cov_trace_norm(x):
    """x: (n,C,T) -> trace 归一化 + 1e-5*I（与 signal2spd 一致）"""
    x = x - x.mean(axis=-1, keepdims=True)
    C = np.einsum('nct,ndt->ncd', x, x) / (x.shape[-1] - 1)
    C = C / np.trace(C, axis1=-2, axis2=-1)[..., None, None]
    return C + 1e-5 * np.eye(C.shape[-1])


def _fingerprint(A):
    """把 (n,22,T) float32 压成逐行 128bit 指纹（int64 视图的简单混合和）。
    X_te 与 X 是同源同精度，逐位一致，故按内容匹配可靠。"""
    v = np.ascontiguousarray(A, dtype=np.float32).view(np.int64).reshape(len(A), -1)
    return v.sum(axis=1)


def match_test_rows(X_all, X_te, y_all, sub_all, Y, S):
    """把测试折的每一行映射回全量数组的行号（按内容指纹精确匹配）。
    split_train_valid_test 会打乱顺序，按（被试,类别）计数猜测不成立 —— 必须按内容匹配。"""
    fp_all, fp_te = _fingerprint(X_all), _fingerprint(X_te)
    out = np.full(len(fp_te), -1, dtype=int)
    pool = {}
    for i, f in enumerate(fp_all.tolist()):
        pool.setdefault((int(sub_all[i]), f), i)
    for k, f in enumerate(fp_te.tolist()):
        i = pool.get((int(S[k]), f), -1)
        if i < 0 or y_all[i] != Y[k]:
            raise RuntimeError(f'测试行 {k} 在全量数组中找不到精确匹配（被试 {S[k]} 标签 {Y[k]}）')
        out[k] = i
    if len(set(out.tolist())) != len(out):
        raise RuntimeError('指纹匹配出现重复行号')
    return out


def hook_out(o):
    if isinstance(o, tuple):
        for t in o:
            if isinstance(t, torch.Tensor):
                return t
        return None
    return o if isinstance(o, torch.Tensor) else None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    subject_data, _ = load_preprocessed_dataset(str(DATA_PATH))
    X_all = np.concatenate([sd[0] for sd in subject_data], axis=0).astype(np.float32)
    if PA_FS != 250:                       # 与新口径一致: 载入后整数抽取(pkl 不动)
        from scipy.signal import resample_poly
        X_all = resample_poly(X_all, 1, int(round(250 / PA_FS)), axis=-1).astype(np.float32)
        print(f'[PA_FS] 250 -> {PA_FS:g} Hz, X_all={X_all.shape}', flush=True)
    y_all = np.concatenate([sd[1] for sd in subject_data], axis=0)
    sub_all = np.concatenate([np.full(len(sd[1]), i, dtype=np.int64)
                              for i, sd in enumerate(subject_data)])
    ret = split_train_valid_test(X_all, y_all, 0.70, 0.15, random_state=42,
                                others=[sub_all])
    X_te, y_te, sub_te = ret[4], ret[5], ret[8]
    print(f'{time.time()-t0:.0f}s loaded all={X_all.shape} test={X_te.shape}', flush=True)
    S, Y = np.asarray(sub_te, dtype=int), np.asarray(y_te, dtype=int)
    # 测试折在全量数组里的行号：split_train_valid_test 返回的测试集顺序不保证与原序一致，
    # 故用 (被试, 标签, 该被试内出现次序) 做稳定匹配。
    idx_te = match_test_rows(X_all, X_te, y_all, sub_all, Y, S)
    res = {'meta': {'n_test': int(len(Y)), 'win': WIN, 'step': STEP,
                    'readout': 'StandardScaler+LDA(lsqr,shrinkage=auto) 分层5折CV',
                    'note': '逐级都用逐样本协方差的 log 向量做读出，保证唯一变量是特征来源'},
           'runs': {}}

    for seed_name, (dname, tag) in RUNS.items():
        ck = BASE / 'results' / dname / f'{tag}_cross_mix_best_acc.pt'
        if not ck.exists():
            print(f'skip {seed_name}', flush=True); continue
        net = load_matt_model(str(ck), WIN, STEP, ARGS).to(DEV).eval()

        store = {}

        def rec(key):
            def h(m, i, o):
                t = hook_out(o)
                if t is not None:
                    store[key] = t.detach()
            return h

        hs = [net.conv1.register_forward_hook(rec('conv1')),
              net.conv2.register_forward_hook(rec('conv2')),
              net.ract1.register_forward_hook(rec('ract1')),
              net.spdnorm.register_forward_hook(rec('spdnorm'))]

        covs = {k: [] for k in ['conv1', 'conv2', 'ract1', 'spdnorm', 'tangent']}
        scal = net.channel_weight.fixed_mask.clone()
        scal[net.channel_weight.keep_mask] = net.channel_weight.scale.detach()
        sc = scal.numpy().astype(np.float32)
        xin = torch.from_numpy(X_all).unsqueeze(1)
        with torch.no_grad():
            for b in range(0, len(X_all), 256):
                xb = xin[b:b + 256]
                spd_in = net._shared_backbone(xb)        # conv1..In2..E2R..SPDNorm
                a2, shape2 = net.att2(spd_in)
                a2 = net.ract2(a2)
                a2 = net.tangent(a2)
                _flat = net.flat(a2.view(shape2[0], shape2[1], -1))
                covs['conv1'].append(cov_trace_norm(
                    store['conv1'].squeeze(2).numpy().astype(np.float32)))
                covs['conv2'].append(cov_trace_norm(
                    store['conv2'].squeeze(2).numpy().astype(np.float32)))
                covs['ract1'].append(store['ract1'].numpy().astype(np.float32))
                covs['spdnorm'].append(store['spdnorm'].numpy().astype(np.float32))
                covs['tangent'].append(_flat.detach().numpy().astype(np.float32))
        for h in hs:
            h.remove()
        del store
        print(f'{seed_name} 前向完成 {time.time()-t0:.0f}s', flush=True)

        xw = X_all * sc[None, :, None]
        del xin

        # ---- I-1' 逐级协方差的可分性 ----
        lvl = {}
        lvl['01_入口_22通道协方差'] = cov_trace_norm(xw)
        for k in ['conv1', 'conv2']:
            lvl[f'{k}_协方差'] = np.concatenate(covs[k], axis=0)
        # conv2: padding=6 + 偶数核长 ⇒ 序列变长 1 点(1126 -> 1127)，故窗口数是 5 不是 4
        M_win = (X_all.shape[-1] + 2 * 6 - 12 + 1 - WIN) // STEP + 1
        lvl['后端切空间出口'] = np.concatenate(covs['tangent'], axis=0)
        for k, nm in [('ract1', 'E2R出口'), ('spdnorm', 'SPDNorm后')]:
            a = np.concatenate(covs[k], axis=0)
            if a.ndim == 3:                              # (n*M, C, C) => 还原成 (n, M, C, C)
                a = a.reshape(-1, M_win, a.shape[-2], a.shape[-1])
            n_, M_, Cc, _ = a.shape
            lm = logm(a.reshape(-1, Cc, Cc).astype(np.float64)).reshape(n_, M_, Cc, Cc)
            lvl[f'{nm}_log均值'] = lm.mean(axis=1).astype(np.float32)
            del a, lm
        # 这两级存的【已经是对数域均值】(逐窗 log 之后按窗平均)，绝不能再取一次 log：
        # 二次 log 会破坏 SPDNorm 的仿射不变性( log Y = γ·log X − γ·m + βI )，
        # 凭空造出一个"SPDNorm 丢失台阶"。两级在单次 log 下必然给出相同精度。
        LOG_DOMAIN = {'E2R出口_log均值', 'SPDNorm后_log均值'}
        test_mask = np.zeros(len(y_all), dtype=bool)
        test_mask[idx_te] = True          # 共享读出的训练集 = 全量去掉测试折
        trace = {}
        for nm, C in lvl.items():
            if nm in LOG_DOMAIN:
                F = spdvec(C.astype(np.float64)).astype(np.float32)
            elif C.ndim == 3:
                F = spdvec(logm(C.astype(np.float64))).astype(np.float32)
            else:
                F = C
            pf, pt = {}, {}
            for s in range(9):
                m = sub_all == s
                pf[f'S{s+1:02d}'] = readout(F[m], y_all[m])
                mte = idx_te[S == s]        # 全量行号
                if len(mte) > 8:
                    pt[f'S{s+1:02d}'] = readout(F[mte], Y[S == s])
            lo, hi = group_mean_low, group_mean_high
            trace[nm] = {'dim': int(F.shape[1]), 'per_subject_full': pf,
                         'per_subject_test': pt, 'low4_full': lo(pf), 'high5_full': hi(pf),
                         'low4_test': lo(pt), 'high5_test': hi(pt)}
            # ---- 共享读出（与模型同条件：不用被试身份，训练=非测试折 4406 试次，
            #      测试=同一 778 试次）。逐人 5 折探针与共享头差的是「重拟合权」，
            #      只有这一行才回答「共享读法在这一级读得出多少」。
            from sklearn.preprocessing import StandardScaler
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as _LDA
            tr_mask = ~test_mask
            sc_ = StandardScaler().fit(F[tr_mask])
            clf_ = _LDA(solver='lsqr', shrinkage='auto').fit(sc_.transform(F[tr_mask]),
                                                             y_all[tr_mask])
            ok_ = (clf_.predict(sc_.transform(F[idx_te])) == y_all[idx_te])
            sp_ = {f'S{i+1:02d}': float(ok_[sub_all[idx_te] == i].mean()) for i in range(9)}
            trace[nm].update({'shared_test': float(ok_.mean()),
                              'shared_low4': lo(sp_), 'shared_high5': hi(sp_),
                              'shared_per_subject': sp_})
            del sc_, clf_, ok_, sp_
            print(f"  {nm:20s} d={F.shape[1]:4d} 全576: 低{trace[nm]['low4_full']*100:5.1f} 高{trace[nm]['high5_full']*100:5.1f}"
                  f"  测试折: 低{trace[nm]['low4_test']*100:5.1f} 高{trace[nm]['high5_test']*100:5.1f}"
                  f"  | 共享读出 {trace[nm]['shared_test']*100:5.1f}"
                  f"(低{trace[nm]['shared_low4']*100:4.1f} 高{trace[nm]['shared_high5']*100:4.1f})", flush=True)
            del F

        # ---- I-2' 基的作用（全部在 22 通道协方差上做，等价线性投影） ----
        W = net.conv1.weight.detach().cpu().numpy()[:, 0, :, 0].astype(np.float64)  # (22,22)
        C22 = lvl['01_入口_22通道协方差']
        # 判别方向：入口切空间的 LDA 方向映射回 22x22 对称阵
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_FIT
        Fentry = spdvec(logm(C22.astype(np.float64)))
        # lsqr 求解器不暴露 scalings_；用判别系数 coef_（4 类 的判别方向）
        lda = LDA_FIT(solver='lsqr', shrinkage='auto').fit(Fentry, y_all)
        r22, c22 = np.triu_indices(22)
        D = np.zeros((22, 22))
        for _c in range(lda.coef_.shape[0]):        # 4 个类别的判别方向取上三角并重建对称阵
            Dk = np.zeros((22, 22)); Dk[r22, c22] = lda.coef_[_c]
            D = D + (Dk + Dk.T - np.diag(np.diag(Dk)))
        D = 0.5 * (D + D.T)
        # W 的第 j 行（22 维）= 第 j 个空间滤波器的通道载荷；D 是 22x22 的判别方向矩阵。
        # 两者同空间，Frobenius 内积 = 逐元素相乘求和，再除以各自 Frobenius 范数即余弦。
        dNorm = float(np.linalg.norm(D)) + 1e-12
        cosj = []
        for j in range(W.shape[0]):
            num = float(np.abs((W[j] * D).sum()))
            den = float(np.linalg.norm(W[j])) * dNorm + 1e-12
            cosj.append(num / den)
        order = list(np.argsort(cosj)[::-1])
        # 随机 8 维方向基线（30 次）。必须给到与其它臂【同一个统计量】——逐人被试精度，
        # 才能与 low4/high5 两列并列；早期版本只存了 9 人均值却被拿去和"低四/高五"比，
        # 属于不同统计量相比（判据方向因此反了）。
        rng = np.random.default_rng(0)
        rnd, rnd_low, rnd_high = [], [], []
        Cw_full = np.einsum('jc,ncd,ld->njl', W, C22, W)
        for _ in range(30):
            Wr = rng.standard_normal((8, 22))
            Wr /= np.linalg.norm(Wr, axis=1, keepdims=True)
            Cr = np.einsum('jc,ncd,ld->njl', Wr, C22, Wr)
            a = {}
            for s in range(9):
                m = sub_all == s
                a[f'S{s+1:02d}'] = readout(spdvec(logm(Cr[m].astype(np.float64))), y_all[m])
            rnd.append(float(np.nanmean(list(a.values()))))
            rnd_low.append(group_mean_low(a))
            rnd_high.append(group_mean_high(a))
        # 各被试自己的前 8 个主方向（该被试最优 8 维）
        basis = {'all22': {}, 'conv1_22dir': {}, 'conv1_top8_to_lda': {},
                 'subject_pca8': {}, 'random8_mean': float(np.mean(rnd)),
                 'random8_std': float(np.std(rnd)),
                 'random8_low4_mean': float(np.mean(rnd_low)),
                 'random8_low4_std': float(np.std(rnd_low)),
                 'random8_high5_mean': float(np.mean(rnd_high)),
                 'random8_high5_std': float(np.std(rnd_high))}
        for s in range(9):
            m = sub_all == s; ys = y_all[m]; k = f'S{s+1:02d}'
            basis['all22'][k] = readout(spdvec(logm(C22[m].astype(np.float64))), ys)

            basis['conv1_22dir'][k] = readout(spdvec(logm(Cw_full[m].astype(np.float64))), ys)
            Wt8 = W[order[:8]]
            C8 = np.einsum('jc,ncd,ld->njl', Wt8, C22, Wt8)
            basis['conv1_top8_to_lda'][k] = readout(spdvec(logm(C8[m].astype(np.float64))), ys)
            # 该被试自己的最优 8 维：对其类均值协方差做 PCA
            Cm = np.stack([C22[m][ys == c].mean(0) for c in range(4)])
            w_, v_ = np.linalg.eigh(Cm.mean(0))
            P8 = v_[:, ::-1][:, :8]
            Cp = np.einsum('jc,ncd,ld->njl', P8.T, C22[m], P8.T)
            basis['subject_pca8'][k] = readout(spdvec(logm(Cp.astype(np.float64))), ys)
        def gk(d):
            return [group_mean_low(d), group_mean_high(d)]
        bsum = {k: ({'per_subject': v, 'low4_high5': gk(v)} if isinstance(v, dict) else v)
                for k, v in basis.items()}
        for k in ['all22', 'conv1_22dir', 'conv1_top8_to_lda', 'subject_pca8']:
            print(f"  I-2' {k:22s} 低{bsum[k]['low4_high5'][0]*100:5.1f} 高{bsum[k]['low4_high5'][1]*100:5.1f}", flush=True)
        print(f"  I-2' random8 全均={basis['random8_mean']*100:5.1f}±{basis['random8_std']*100:.1f}"
              f" 低{basis['random8_low4_mean']*100:5.1f}±{basis['random8_low4_std']*100:.1f}"
              f" 高{basis['random8_high5_mean']*100:5.1f}±{basis['random8_high5_std']*100:.1f}", flush=True)

        # ---- I-3' SPDNorm ----
        logM = net.spdnorm.running_mean.detach().cpu().numpy().astype(np.float64)
        gam = float(net.spdnorm.gamma.detach().cpu().numpy().reshape(-1)[0])
        bet = float(net.spdnorm.beta.detach().cpu().numpy().reshape(-1)[0])
        A = np.concatenate(covs['ract1'], axis=0).astype(np.float64)
        if A.ndim == 3:
            A = A.reshape(-1, M_win, A.shape[-2], A.shape[-1])
        n_, M_, Cc, _ = A.shape
        L = logm(A.reshape(-1, Cc, Cc)).reshape(n_, M_, Cc, Cc)
        pop = L.mean(axis=(0, 1))
        i3 = {}
        for s in range(9):
            m = sub_all == s; k = f'S{s+1:02d}'
            Lm = L[m].mean(axis=(0, 1))
            dev = L[m] - Lm
            i3[k] = {'dist_to_running_mean': float(np.linalg.norm(Lm - logM)),
                     'dist_to_population_mean': float(np.linalg.norm(Lm - pop)),
                     'dist_run_minus_pop': float(np.linalg.norm(Lm - logM) - np.linalg.norm(Lm - pop)),
                     'log_spread': float(np.sqrt((dev ** 2).sum(axis=(-1, -2)).mean()))}
        ld = np.mean([i3[f'S{_i:02d}']['dist_to_running_mean'] for _i in LOW4])
        hd = np.mean([i3[f'S{_i:02d}']['dist_to_running_mean'] for _i in HIGH5])
        ls = np.mean([i3[f'S{_i:02d}']['log_spread'] for _i in LOW4])
        hs = np.mean([i3[f'S{_i:02d}']['log_spread'] for _i in HIGH5])
        print(f"  I-3' 到running_mean距离 低{ld:.3f} 高{hd:.3f} | log域散布 低{ls:.3f} 高{hs:.3f} (比{ls/hs:.3f}) gamma={gam:.4f}", flush=True)

        chl = np.abs(W).sum(axis=0); chl = chl / chl.sum()
        res['runs'][seed_name] = {'checkpoint': ck.name, 'trace': trace,
            'basis_comparison': bsum, 'conv1_filter_cos_to_lda': cosj,
            'top8_filters': [int(x) for x in order[:8]],
            'spdnorm': {'gamma': gam, 'beta': bet, 'running_mean': logM.tolist(),
                        'per_subject': i3, 'low4_dist': float(ld), 'high5_dist': float(hd),
                        'low4_spread': float(ls), 'high5_spread': float(hs)},
            'conv1_channel_load': dict(zip(CH, chl.tolist()))}
        print(f'{seed_name} 完成 {time.time()-t0:.0f}s', flush=True)
        del covs, lvl, C22, Cw_full, A, L

    (OUT / 'pipeline_attribution.json').write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f'written  {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()