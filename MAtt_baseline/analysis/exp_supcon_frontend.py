# -*- coding: utf-8 -*-
"""T-1: SupCon 接到 mAtt 前端(一次性训练实验,exp_ 前缀 = 会训练)。

## 为什么做
历史 SupCon 69.7 只在**固定的 253 维 LE 特征**上训练编码器;深度模型侧端到端 CE 只有
58.82(62.5 Hz 新口径)。论文把「SupCon 接到模型前端」列为唯一可能同时拿到「共享+精度」
的路:LE 上 SupCon 把身份纯度从 1.000 压到 0.713 且精度升到 69.7,深度前端从未试过。

## 做法
前端 = mAtt(62.5 Hz, win=94, step=47, conv2 深度可分离)本体;`_features` 输出 180 维。
在 180 维上接投影头(180→128→64)做 SupCon(正样本 = batch 内同类不同试次),
投影头丢弃、前端冻结,再用**闭式 LW-LDA**(训练折解析拟合,无 SGD 头)读出并评估。

## 臂(每臂 3 seed;对照 = CE 训练前端 + 同款闭式头 = 57.9)
  S1 tau0.1 单视图   S2 tau0.1 + 加噪第二视图   S3 tau0.2 单视图   S4 tau0.1 lr1e-3

## 判据(事先登记)
  ① pooled 是否 > 62.9(真共享零训练基线);② 身份纯度 kNN(9) 是否显著 < 1.000;
  ③ 拆成 9 个逐人 LDA 后涨多少(涨得多 = 假共享)。三者必须同时看。

产物 results/supcon_frontend/{supcon_frontend.json, supcon_frontend.log}
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.signal import resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

from utils.dataset import load_preprocessed_dataset, split_train_valid_test  # noqa: E402
from utils.functions import build_matt_model                                 # noqa: E402

OUT = BASE / 'results' / 'supcon_frontend'
DATA = BASE.parent / 'saved_data' / 'preprocessed_data_bci2a_raw_ica.pkl'
WIN, STEP, DOWN = 94, 47, 4            # 62.5 Hz:1.5 s 窗 / 0.75 s 步
SUBJ = ['S%02d' % i for i in range(1, 10)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
ARGS = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=True)
SMOKE = '--smoke' in sys.argv
QUICK = '--quick' in sys.argv
KEY = '--key' in sys.argv          # 只跑关键臂(S0/S5/S6/S7),跳过已证明无效的从零训臂
SWEEP_MODE = '--sweep' in sys.argv  # 只跑 epoch/tau 扫参臂(S8-S11)
SEEDS = (42,) if SMOKE or QUICK else (42, 2024, 7)
EPOCHS = 8 if SMOKE else 100        # 与 train_pair 的 iterations 同口径:每 epoch 遍历全训练集
ARMS = {
    'S1_tau0.1': dict(tau=0.1, lr=5e-4, noise=0.0),
    'S2_tau0.1_noise': dict(tau=0.1, lr=5e-4, noise=0.15),
    'S3_tau0.2': dict(tau=0.2, lr=5e-4, noise=0.0),
    'S4_tau0.1_lr1e-3': dict(tau=0.1, lr=1e-3, noise=0.0),
    # 以下从已训 CE 前端(新口径 58.82)继续 SupCon:起点是 57.9(闭式头),不是随机初始化
    'S5_ceinit_lr1e-4': dict(tau=0.1, lr=1e-4, noise=0.0, init='ce'),
    'S6_ceinit_lr5e-4': dict(tau=0.1, lr=5e-4, noise=0.0, init='ce'),
    'S7_ceinit_lr1e-4_ep300': dict(tau=0.1, lr=1e-4, noise=0.0, init='ce', epochs=300),
    # 关键对照:CE 前端 + 未训练的投影头(0 epoch) ⇒ 分离「SupCon 的贡献」与「64 维投影本身」
    'S0_ceinit_0ep': dict(tau=0.1, lr=0.0, noise=0.0, init='ce', epochs=0),
    # 扫参:epoch 与 tau(全部从 CE 前端出发、lr 1e-4)
    'S8_ep50': dict(tau=0.1, lr=1e-4, noise=0.0, init='ce', epochs=50),
    'S9_ep150': dict(tau=0.1, lr=1e-4, noise=0.0, init='ce', epochs=150),
    'S10_tau0.05': dict(tau=0.05, lr=1e-4, noise=0.0, init='ce'),
    'S11_tau0.15': dict(tau=0.15, lr=1e-4, noise=0.0, init='ce'),
}
SWEEP = ('S8_ep50', 'S9_ep150', 'S10_tau0.05', 'S11_tau0.15')
CE_CKPT = {42: ('sep62_full_s42', 'sep62f42'), 2024: ('sep62_full_s2024', 'sep62f2024'),
           7: ('sep62_full_s7', 'sep62f7')}


class Proj(nn.Module):
    def __init__(self, d_in=180, d=128, z=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d, z))

    def forward(self, x):
        return self.net(x)


def supcon(z, y, tau):
    """有监督对比:正样本 = batch 内同类不同试次(照 analyze_contrastive_repr.supcon)。"""
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / tau
    B = z.shape[0]
    mask = (y[:, None] == y[None, :]).float()
    mask.fill_diagonal_(0)
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    exp_sim = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
    log_prob = sim - torch.log(exp_sim.sum(1, keepdim=True) + 1e-12)
    cnt = mask.sum(1).clamp(min=1)
    return -((log_prob * mask).sum(1) / cnt).mean()


def pack(ok, sub):
    per = {SUBJ[i]: float(ok[sub == i].mean()) for i in range(9)}
    return {'pooled': float(ok.mean()),
            'low4': float(np.mean([per[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([per[SUBJ[i]] for i in HIGH])),
            'per_subject': per}, ok


def shared_lda(Ftr, Fte, ytr, yte, sub_te):
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr, ytr)
    return pack(clf.predict(Fte) == yte, sub_te)


def per_subject_lda(Ftr, Fte, ytr, yte, sub_tr, sub_te):
    """共享性判据:拆成 9 个逐人 LDA(仅诊断,用 subject_id)。"""
    ok = np.zeros(len(yte), dtype=bool)
    for i in range(9):
        tr, te = sub_tr == i, sub_te == i
        clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto').fit(Ftr[tr], ytr[tr])
        ok[te] = clf.predict(Fte[te]) == yte[te]
    return pack(ok, sub_te)


def identity_purity(Ftr, Fte, sub_tr, sub_te):
    """身份纯度:训练折拟合 kNN(9) 预测 subject_id,报 test 准确率。"""
    knn = KNeighborsClassifier(n_neighbors=9).fit(Ftr, sub_tr)
    return float((knn.predict(Fte) == sub_te).mean())


def feats(net, X, bs=128):
    net.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            t = torch.Tensor(X[i:i + bs]).unsqueeze(1).to(DEV)
            out.append(net._features(t).cpu().numpy())
    return np.concatenate(out, 0)


def embed_proj(proj, F):
    """投影头输出(64 维) —— 与历史 SupCon(LE→MLP→64→LDA)对等的读出维度。"""
    proj.eval()
    with torch.no_grad():
        out = [proj(torch.tensor(F[i:i + 1024], dtype=torch.float32).to(DEV)).cpu().numpy()
               for i in range(0, len(F), 1024)]
    return np.concatenate(out, 0)


def train_arm(Xtr, ytr, tau, lr, noise, seed, init=None, epochs=None, wd=0.02, bs=128):
    """SupCon 训前端 + 投影头;返回 (前端, 投影头, 逐 epoch loss)。"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = build_matt_model(WIN, STEP, ARGS)
    if init == 'ce':
        d, tag = CE_CKPT[seed]
        ck = BASE / 'results' / d / ('%s_cross_mix_best_acc.pt' % tag)
        net.load_state_dict(torch.load(str(ck), map_location='cpu'))
        print('    [init] 从 CE 前端继续:%s' % ck.name, flush=True)
    proj = Proj().to(DEV)
    params = list(net.parameters()) + list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    epochs = EPOCHS if epochs is None else epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    Xt = torch.tensor(Xtr, dtype=torch.float32)
    yt = torch.tensor(ytr, dtype=torch.long)
    n = len(Xt)
    rng = np.random.default_rng(seed)
    hist = []
    for _ in range(epochs):
        net.train()
        perm = torch.tensor(rng.permutation(n), dtype=torch.long)
        tot, cnt = 0.0, 0
        for i in range(0, n - bs + 1, bs):
            idx = perm[i:i + bs]
            xb = Xt[idx].to(DEV)
            yb = yt[idx].to(DEV)
            zb = proj(net._features(xb.unsqueeze(1)))
            if noise > 0:
                xb2 = xb + noise * xb.std() * torch.randn_like(xb)
                zb = torch.cat([zb, proj(net._features(xb2.unsqueeze(1)))], 0)
                yb = torch.cat([yb, yb], 0)
            loss = supcon(zb, yb, tau)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            tot += float(loss)
            cnt += 1
        sched.step()
        hist.append(tot / max(cnt, 1))
    return net, proj, hist


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    subject_data, _ = load_preprocessed_dataset(str(DATA))
    y_all = np.concatenate([s[1] for s in subject_data]).astype(np.int64)
    sub_all = np.concatenate([np.full(len(s[1]), i, dtype=np.int64)
                              for i, s in enumerate(subject_data)])
    X62 = np.concatenate([resample_poly(np.asarray(s[0], dtype=np.float32), 1, DOWN,
                                        axis=-1) for s in subject_data], 0)
    idx = np.arange(len(y_all), dtype=np.int64)
    ret = split_train_valid_test(X62, y_all, 0.70, 0.15, random_state=42,
                                 others=[sub_all, idx])
    itr, ite_ = ret[9], ret[11]
    ytr, yte, sub_tr, sub_te = ret[1], ret[5], ret[6], ret[8]
    print('[数据] %s | train %d / test %d' % (X62.shape, len(itr), len(ite_)), flush=True)
    rec = {'note': 'SupCon 训 mAtt 前端(非固定 LE);冻结 + 闭式 LW-LDA 读出',
           'epochs': EPOCHS, 'seeds': list(SEEDS), 'arms': {}}
    if QUICK:
        arms = {'S5_ceinit_lr1e-4': ARMS['S5_ceinit_lr1e-4']}
    elif SWEEP_MODE:
        arms = {k: ARMS[k] for k in SWEEP}
    elif KEY:
        arms = {k: ARMS[k] for k in ('S0_ceinit_0ep', 'S5_ceinit_lr1e-4',
                                     'S6_ceinit_lr5e-4', 'S7_ceinit_lr1e-4_ep300')}
    else:
        arms = ARMS
    for arm, cfg in arms.items():
        rows = {}
        for seed in SEEDS:
            t0 = time.time()
            net, proj, hist = train_arm(X62[itr], ytr, cfg['tau'], cfg['lr'],
                                        cfg['noise'], seed, init=cfg.get('init'),
                                        epochs=cfg.get('epochs'))
            Ftr, Fte = feats(net, X62[itr]), feats(net, X62[ite_])       # 180 维前端
            Ztr, Zte = embed_proj(proj, Ftr), embed_proj(proj, Fte)      # 64 维投影
            sh, ok64 = shared_lda(Ztr, Zte, ytr, yte, sub_te)
            ps, _ = per_subject_lda(Ztr, Zte, ytr, yte, sub_tr, sub_te)
            idp = identity_purity(Ztr, Zte, sub_tr, sub_te)
            sh180, ok180 = shared_lda(Ftr, Fte, ytr, yte, sub_te)
            id180 = identity_purity(Ftr, Fte, sub_tr, sub_te)
            ck = OUT / 'ckpt'
            ck.mkdir(exist_ok=True)
            torch.save({'arm': arm, 'seed': seed, 'config': cfg,
                        'net': net.state_dict(), 'proj': proj.state_dict()},
                       str(ck / ('%s_s%d.pt' % (arm, seed))))
            rows[str(seed)] = {'shared': sh, 'per_subject_lda': ps,
                               'identity_purity': idp,
                               'ok64': [bool(v) for v in ok64],
                               'ok180': [bool(v) for v in ok180],
                               'shared_front180': sh180,
                               'identity_purity_front180': id180,
                               'loss_first': float(np.mean(hist[:5])),
                               'loss_last': float(np.mean(hist[-3:])),
                               'sec': round(time.time() - t0, 1)}
            msg = '  %-16s s%d | 64D pooled %.1f (l4 %.1f h5 %.1f) split9 %.1f id %.3f'
            msg += ' | 180D pooled %.1f id %.3f | loss %.2f->%.2f | %.0fs'
            print(msg % (arm, seed, 100 * sh['pooled'], 100 * sh['low4'], 100 * sh['high5'],
                         100 * ps['pooled'], idp, 100 * sh180['pooled'], id180,
                         rows[str(seed)]['loss_first'], rows[str(seed)]['loss_last'],
                         rows[str(seed)]['sec']), flush=True)
        ps_m = {s: float(np.mean([rows[str(x)]['shared']['per_subject'][s] for x in SEEDS]))
                for s in SUBJ}
        rec['arms'][arm] = {
            'config': cfg,
            'pooled': float(np.mean([rows[str(x)]['shared']['pooled'] for x in SEEDS])),
            'low4': float(np.mean([ps_m[SUBJ[i]] for i in LOW])),
            'high5': float(np.mean([ps_m[SUBJ[i]] for i in HIGH])),
            'per_subject_lda_pooled': float(np.mean(
                [rows[str(x)]['per_subject_lda']['pooled'] for x in SEEDS])),
            'identity_purity': float(np.mean([rows[str(x)]['identity_purity'] for x in SEEDS])),
            'shared_front180': float(np.mean([rows[str(x)]['shared_front180']['pooled']
                                              for x in SEEDS])),
            'identity_purity_front180': float(np.mean(
                [rows[str(x)]['identity_purity_front180'] for x in SEEDS])),
            'per_subject': ps_m, 'per_seed': rows}
    fp = OUT / 'supcon_frontend.json'
    if fp.exists():                      # 增量合并:多次运行各自只覆盖自己跑过的臂
        old = json.loads(fp.read_text(encoding='utf-8'))
        old.get('arms', {}).update(rec['arms'])
        rec['arms'] = old['arms']
    fp.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print('\n=== T-1 SupCon 前端(冻结 + 闭式 LW-LDA) ===')
    hdr = '%-18s %8s %7s %7s %8s %8s %9s'
    print(hdr % ('arm', 'pooled64', 'low4', 'high5', 'split9', 'id64', 'front180'))
    for arm, r in rec['arms'].items():
        print('%-18s %7.1f%% %6.1f%% %6.1f%% %7.1f%% %8.3f %8.1f%%'
              % (arm, 100 * r['pooled'], 100 * r['low4'], 100 * r['high5'],
                 100 * r['per_subject_lda_pooled'], r['identity_purity'],
                 100 * r['shared_front180']))
    print('对照: CE 前端 + 同款闭式头 = 57.9 (low4 40.9 / high5 71.5, split9 +4.9);')
    print('      LE 上 SupCon = 69.7 (id 0.713); 真共享零训练基线 = 62.9')
    print('[OK] %s/supcon_frontend.json' % OUT)


if __name__ == '__main__':
    main()