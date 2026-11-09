# -*- coding: utf-8 -*-
"""官方数据(128Hz + EA)跨 session:前端各级的可训练读出定位。

口径 = 当前正式口径(材料「官方数据复现与共享代价」§5):
  数据 saved_data/matt_official_data/BCIC_S0X_{T,E}.mat
  对齐 逐被试 EA(全部 576 试次) | 128Hz | 全长 562 点
  模型 build_matt_model(192,96): conv2=20 / d_attn_out=18 / 通道先验 / dropout 0.3
  切分 T 训练 -> E 测试(跨 session);1 seed
读出 各级特征 -> SupCon(64d, 1 seed) + 闭式 LW-LDA;同时给线性 LDA 对照
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from scipy import io

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'MAtt_baseline'))
sys.path.insert(0, str(ROOT / 'preprocess'))
from utils.functions import build_matt_model                       # noqa: E402
from alignment import compute_ea_matrix_from_numpy, apply_ea_to_numpy  # noqa: E402

DATA = ROOT / 'saved_data' / 'matt_official_data' / 'data' / 'BCICIV_2a_mat'
OUT = ROOT / 'MAtt_baseline' / 'results' / 'official_frontend_loss'
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
WIN, STEP = 192, 96
SUBJ = [f'S{i+1:02d}' for i in range(9)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]
ARGS = dict(n_classes=4, d_attn_in=20, d_attn_out=18, local_radius=6,
            anchor_stride=None, n_heads=1, dropout=0.3, conv2_separable=False)


def cov_tn(x):
    xc = x - x.mean(-1, keepdims=True)
    C = np.einsum('bct,bet->bce', xc, xc) / xc.shape[-1]
    return C * (C.shape[1] / np.trace(C, axis1=1, axis2=2)[:, None, None])


def logm(C):
    w, V = np.linalg.eigh(C.astype(np.float64))
    w = np.maximum(w, 1e-12)
    return np.einsum('nij,nj,nkj->nik', V, np.log(w), V)


def spdvec(L):
    d = L.shape[-1]
    iu = np.triu_indices(d)
    v = L[:, iu[0], iu[1]].copy()
    v[:, iu[0] != iu[1]] *= np.sqrt(2)
    return v


class Enc(nn.Module):
    def __init__(self, d_in, d=256, z=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, d), nn.LayerNorm(d), nn.GELU(),
                                 nn.Linear(d, d), nn.LayerNorm(d), nn.GELU(),
                                 nn.Linear(d, z))

    def forward(self, x):
        return self.net(x)


def supcon(z, y, tau=0.1):
    z = Fn.normalize(z, dim=1)
    sim = z @ z.t() / tau
    eye = torch.eye(len(z), dtype=torch.bool, device=z.device)
    pos = (y[:, None] == y[None, :]) & ~eye
    sim = sim - sim.max(1, keepdim=True).values.detach()
    ex = torch.exp(sim) * (~eye)
    logp = sim - torch.log(ex.sum(1, keepdim=True) + 1e-12)
    loss = -(logp * pos).sum(1)
    cnt = pos.sum(1).clamp(min=1)
    return (loss / cnt)[cnt > 0].mean()


def grp(per):
    return (float(np.mean([per[SUBJ[i]] for i in LOW])),
            float(np.mean([per[SUBJ[i]] for i in HIGH])))


def readout(A, ya, B, yb, sb):
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
    ok = LDA(solver='lsqr', shrinkage='auto').fit(A, ya).predict(B) == yb
    return float(ok.mean()), {SUBJ[i]: float(ok[sb == i].mean()) for i in range(9)}


def load_official():
    per = []
    for s in range(1, 10):
        Xs, ys = [], []
        for part, xk, yk in (('T', 'x_train', 'y_train'), ('E', 'x_test', 'y_test')):
            m = io.loadmat(str(DATA / ('BCIC_S%02d_%s.mat' % (s, part))))
            Xs.append(np.asarray(m[xk], dtype=np.float64))
            ys.append(np.asarray(m[yk]).reshape(-1).astype(np.int64))
        X = np.concatenate(Xs, 0)
        H = apply_ea_to_numpy(X, compute_ea_matrix_from_numpy(X))
        per.append((H[:288], ys[0], H[288:], ys[1]))
    return per


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    per = load_official()
    Xtr = np.concatenate([p[0] for p in per], 0).astype(np.float32)
    ytr = np.concatenate([p[1] for p in per], 0)
    Xte = np.concatenate([p[2] for p in per], 0).astype(np.float32)
    yte = np.concatenate([p[3] for p in per], 0)
    subte = np.concatenate([np.full(288, i, dtype=np.int64) for i in range(9)])
    print('T', Xtr.shape, 'E', Xte.shape, flush=True)

    # T 内 85/15 选模
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(Xtr))
    nva = int(0.15 * len(Xtr))
    iva, itr = idx[:nva], idx[nva:]

    torch.manual_seed(42)
    net = build_matt_model(WIN, STEP, ARGS).to(DEV)
    opt = torch.optim.AdamW(net.parameters(), lr=5e-4, weight_decay=0.02)
    Xt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    Xv = torch.tensor(Xtr[iva]).to(DEV)
    yv = torch.tensor(ytr[iva]).to(DEV)
    best = (1e10, None)
    t0 = time.time()
    for ep in range(100):
        net.train()
        pm = rng.permutation(itr)
        for i in range(0, len(pm) - 127, 128):
            j = pm[i:i + 128]
            loss = Fn.cross_entropy(net(Xt[j].to(DEV).unsqueeze(1)), yt[j].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 5 == 4 or ep == 99:
            net.eval()
            with torch.no_grad():
                vl = float(Fn.cross_entropy(net(Xv.unsqueeze(1)), yv))
            if vl < best[0]:
                best = (vl, {k: t.clone() for k, t in net.state_dict().items()})
    net.load_state_dict(best[1]); net.eval()
    with torch.no_grad():
        pr = torch.cat([net(torch.tensor(Xte[i:i + 128]).to(DEV).unsqueeze(1)).argmax(1).cpu()
                        for i in range(0, len(Xte), 128)]).numpy()
    ce_acc = float((pr == yte).mean())
    print('CE 跨session test %.2f%%  (训练 %.0fs)' % (100 * ce_acc, time.time() - t0), flush=True)
    torch.save(net.state_dict(), OUT / 'official_s42_ce.pt')

    store = {}
    def rec(k):
        def h(m, i, o):
            store[k] = o.detach()
        return h
    hs = [net.conv1.register_forward_hook(rec('conv1')),
          net.conv2.register_forward_hook(rec('conv2')),
          net.ract1.register_forward_hook(rec('ract1')),
          net.spdnorm.register_forward_hook(rec('spdnorm'))]

    def extract(Xa):
        fe = {k: [] for k in ('conv1', 'conv2', 'ract1', 'spdnorm', 'tangent')}
        with torch.no_grad():
            for b in range(0, len(Xa), 128):
                xb = torch.tensor(Xa[b:b + 128], dtype=torch.float32).unsqueeze(1).to(DEV)
                spd = net._shared_backbone(xb)
                a2, sh2 = net.att2(spd)
                a2 = net.tangent(net.ract2(a2))
                flat = net.flat(a2.view(sh2[0], sh2[1], -1))
                fe['conv1'].append(store['conv1'].squeeze(2).cpu().numpy().astype(np.float32))
                fe['conv2'].append(store['conv2'].squeeze(2).cpu().numpy().astype(np.float32))
                fe['ract1'].append(store['ract1'].cpu().numpy().astype(np.float32))
                fe['spdnorm'].append(store['spdnorm'].cpu().numpy().astype(np.float32))
                fe['tangent'].append(flat.cpu().numpy().astype(np.float32))
        return {k: np.concatenate(v, 0) for k, v in fe.items()}

    Ftr, Fte = extract(Xtr), extract(Xte)
    for h in hs:
        h.remove()

    M_win = (Xtr.shape[-1] + 12 - 12 + 1 - WIN) // STEP + 1
    level = {'入口_LE253': (spdvec(logm(cov_tn(Xtr[:, :, 124:562]))),
                            spdvec(logm(cov_tn(Xte[:, :, 124:562])))),
             'conv1_LE253': (spdvec(logm(cov_tn(Ftr['conv1'][:, :, 124:562]))),
                             spdvec(logm(cov_tn(Fte['conv1'][:, :, 124:562])))),
             'conv2_LE190': (spdvec(logm(cov_tn(Ftr['conv2'][:, :, 124:562]))),
                             spdvec(logm(cov_tn(Fte['conv2'][:, :, 124:562]))))}
    for key, name in (('ract1', 'E2R'), ('spdnorm', 'SPDNorm')):
        for side, raw in (('tr', Ftr[key]), ('te', Fte[key])):
            a = raw if raw.ndim == 4 else raw.reshape(-1, M_win, raw.shape[-2], raw.shape[-1])
            n_, M_, Cc, _ = a.shape
            lm = logm(a.reshape(-1, Cc, Cc)).reshape(n_, M_, Cc, Cc)
            v = spdvec(lm.mean(axis=1).astype(np.float64)).astype(np.float32)
            if side == 'tr':
                lv_tr = v
            else:
                lv_te = v
        level[name] = (lv_tr, lv_te)
    level['后端_684'] = (Ftr['tangent'], Fte['tangent'])

    out = {'ce_crosssession': ce_acc, 'levels': {}}
    for name, (A, B) in level.items():
        mu, sg = A.mean(0, keepdims=True), A.std(0, keepdims=True) + 1e-6
        As = ((A - mu) / sg).astype(np.float32)
        Bs = ((B - mu) / sg).astype(np.float32)
        ovl, per_l = readout(As, ytr, Bs, yte, subte)
        lo_l, hi_l = grp(per_l)
        torch.manual_seed(0)
        enc = Enc(As.shape[1]).to(DEV)
        o = torch.optim.AdamW(enc.parameters(), lr=1e-3, weight_decay=1e-4)
        Xa, Ya = torch.tensor(As), torch.tensor(ytr)
        rr = np.random.default_rng(0)
        for _ in range(120):
            enc.train(); pm = rr.permutation(len(Xa))
            for i in range(0, len(Xa) - 511, 512):
                ix = torch.tensor(pm[i:i + 512])
                ls = supcon(enc(Xa[ix].to(DEV)), Ya[ix].to(DEV))
                o.zero_grad(); ls.backward(); o.step()
        enc.eval()
        with torch.no_grad():
            Ztr = torch.cat([enc(torch.tensor(As[i:i + 2048]).to(DEV))
                             for i in range(0, len(As), 2048)]).cpu().numpy()
            Zte = torch.cat([enc(torch.tensor(Bs[i:i + 2048]).to(DEV))
                             for i in range(0, len(Bs), 2048)]).cpu().numpy()
        ov, per = readout(Ztr, ytr, Zte, yte, subte)
        lo, hi = grp(per)
        out['levels'][name] = {'dim': int(A.shape[1]), 'linear': ovl, 'supcon': ov,
                               'low4': lo, 'high5': hi}
        print('%-12s in=%3d  线性 %.2f (低%.1f/高%.1f) | SupCon %.2f (低四 %.1f / 高五 %.1f)'
              % (name, A.shape[1], 100 * ovl, 100 * lo_l, 100 * hi_l, 100 * ov, 100 * lo, 100 * hi),
              flush=True)
    (OUT / 'official_frontend_loss.json').write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')
    print('[OK] %s/official_frontend_loss.json' % OUT)


if __name__ == '__main__':
    main()
