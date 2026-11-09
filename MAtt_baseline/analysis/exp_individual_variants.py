"""三变体单变量实验: (B) 按人协方差中心化 (C) 整段单窗池化 (A) 按人空间投影

动机: 被试内 CSP/切空间在低四人上读得出 45-65%, 而共享深度模型只有 33-42% (差 7-25pp)。
      本脚本逐个检验候选机制, 与 base 臂走完全相同的切分/seed/迭代/评测。

用法: python analysis/exp_individual_variants.py --arm base|B_center|C_pool|A_persubj \
        --seeds 42,2024,7 --iters 100 --tag main
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent/'preprocess'))
import torch, torch.nn as nn
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from mAtt.mAtt import _SparseAttentionBase

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS = 250
CUE = 250
log = lambda *a: print(*a, flush=True)


def load_split():
    data,_ = load_preprocessed_dataset(str(BASE.parent/'saved_data'/'preprocessed_data_bci2a_raw_ica.pkl'))
    X = np.concatenate([data[s][0] for s in range(len(data))],axis=0)
    y = np.concatenate([data[s][1] for s in range(len(data))],axis=0)
    sub = np.concatenate([np.full(len(data[s][1]),s,dtype=np.int64) for s in range(len(data))])
    o = split_train_valid_test(X,y,train_ratio=.70,valid_ratio=.15,random_state=42,others=[sub])
    return (o[0],o[1],o[6]), (o[2],o[3],o[7]), (o[4],o[5],o[8])


def make_net(cfg, n_subj=9):
    T = cfg["T"]
    M = (T - cfg["WIN"]) // cfg["STEP"] + 1
    kw = dict(d_attn_in=8, d_attn_out=8, local_radius=6, anchor_stride=None,
              n_heads=1, n_windows=M, use_hann=True, use_channel_weight=True,
              n_spatial_filters=22, channel_gaussian=False, gaussian_sigma=0.62,
              conv1_kernel=22, conv1_out=22, conv2_kernel=12, conv2_pad=6, conv2_out=8,
              conv2_separable=False, conv2_multiscale=False, manifold_change=None,
              attn_change=None, attn_prior=None, dropout=0.3, score_original=True, norm_type="in")
    if cfg["persubj_conv1"]:
        return NetA(T, cfg["WIN"], cfg["STEP"], n_classes=4, n_ps_filt=cfg["n_ps_filt"], n_subj=n_subj, **kw)
    return Net(T, cfg["WIN"], cfg["STEP"], n_classes=4, **kw)


class Net(_SparseAttentionBase):
    """base / B_center / C_pool 共用; center_spd 打开时为 B 臂"""
    def __init__(self, T, WIN, STEP, n_classes=4, center_spd=False, **kw):
        self.center_spd = bool(center_spd)
        self.subj_ref = None
        super().__init__(WIN, STEP, n_classes=n_classes, **kw)

    def forward(self, x, sidx=None):
        x = self._features(x, sidx)
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)

    def _features(self, x, sidx=None):
        spd = self._shared_backbone(x)          # (B,M,C,C)
        if self.center_spd and self.subj_ref is not None and sidx is not None:
            spd = self._center(spd, sidx)
        x, shape = self.att2(spd)
        x = self.ract2(x)
        x = self.tangent(x)
        return self.flat(x.view(shape[0], shape[1], -1))

    def _center(self, spd, sidx):
        out = spd.clone()
        for s in torch.unique(sidx):
            m = sidx == s
            if int(s) in self.subj_ref:
                out[m] = self.subj_ref[int(s)](spd[m])
        return out


class NetA(Net):
    """A 臂: conv1 换成按被试的空间投影(每人 22->4), 其余不变"""
    def __init__(self, T, WIN, STEP, n_classes=4, n_ps_filt=4, n_subj=9, **kw):
        self.n_ps_filt = n_ps_filt
        self.n_subj = n_subj
        super().__init__(T, WIN, STEP, n_classes=n_classes, **kw)
        # 用按被试的投影替换共享 conv1
        self.ps_conv1 = None

    def build_persubj(self):
        self.ps_conv1 = nn.ParameterList([
            nn.Parameter(torch.randn(self.n_ps_filt, 1, 22, 1) * 0.1) for _ in range(self.n_subj)])
        # 主路 conv2 的输入通道数改为 n_ps_filt
        self.conv1 = nn.Conv2d(1, self.n_ps_filt, (22, 1))
        self.In1 = nn.InstanceNorm2d(self.n_ps_filt, affine=True)
        self.conv2 = nn.Conv2d(self.n_ps_filt, 8, (1, 12), padding=(0, 6))
        self.In2 = nn.InstanceNorm2d(8, affine=True)

    def _shared_backbone(self, x, sidx=None):
        if self.channel_weight is not None: x = self.channel_weight(x)
        if self.ps_conv1 is None or sidx is None:
            x = self.In1(self.conv1(x))
        else:
            outs = []
            for s in range(self.n_subj):
                m = (sidx == s).nonzero(as_tuple=True)[0]
                if len(m) == 0: continue
                xs = x[m]
                w = self.ps_conv1[s].expand(xs.shape[0], -1, -1, -1)
                outs.append((xs * w).sum(dim=2, keepdim=True))
            x = torch.cat(outs, 0)
            # 按原顺序还原
            order = torch.argsort(torch.cat([ (sidx==s).nonzero(as_tuple=True)[0] for s in range(self.n_subj) if (sidx==s).any() ]))
            x = x[order]
            x = self.In1(x) if x.shape[1] == 22 else x
        x = self.In2(self.conv2(x))
        spd = self.spdnorm(self.ract1(x))
        return self.manifold_change(spd)


def fit_center_refs(net, Xtr, str_, bs=256, random_ref=False):
    """只用训练折, 按被试算 log 域平均协方差, 建立中心化映射"""
    net.eval()
    refs = {}
    with torch.no_grad():
        for s in np.unique(str_):
            m = str_ == s
            covs = []
            for i in range(0, int(m.sum()), bs):
                xb = torch.as_tensor(Xtr[m][i:i+bs], dtype=torch.float32, device=DEV).unsqueeze(1)
                spd = net._shared_backbone(xb)          # (B,M,C,C)
                covs.append(spd.mean(dim=1))            # 窗间平均 -> (B,C,C)
            C = torch.cat(covs, 0).mean(0)              # 该被试平均协方差
            if random_ref:                              # 判决对照: 换成随机 SPD
                A = torch.randn(C.shape[0], C.shape[1])
                C = (A @ A.T + torch.eye(C.shape[0]) * 0.5).float()
            refs[int(s)] = C.cpu()
    def make(C):
        def _f(spd):                                    # spd: (B,M,C,C)
            Cr = C.to(spd.device)
            e, V = torch.linalg.eigh(Cr)
            W = V @ torch.diag(e.clamp_min(1e-12).rsqrt()) @ V.transpose(-1, -2)
            def ap(M):
                return W @ M @ W.transpose(-1, -2)
            return ap(spd)
        return _f
    return {k: make(v) for k, v in refs.items()}


ARMS = {
    "base":      dict(T=1126, WIN=375, STEP=187, center_spd=False, persubj_conv1=False, slice=None),
    "B_center":  dict(T=1126, WIN=375, STEP=187, center_spd=True,  persubj_conv1=False, slice=None),
    "C_pool":    dict(T=751,  WIN=751, STEP=187, center_spd=False, persubj_conv1=False, slice=(375,1126)),
    "A_persubj": dict(T=1126, WIN=375, STEP=187, center_spd=False, persubj_conv1=True,  slice=None, n_ps_filt=4),
    "B_random":  dict(T=1126, WIN=375, STEP=187, center_spd=True,  persubj_conv1=False, slice=None, random_ref=True),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    cfg = ARMS[a.arm]
    (Xtr,ytr,str_), (Xva,yva,sva), (Xte,yte,ste) = load_split()
    if cfg["slice"] is not None:
        s0, s1 = cfg["slice"]
        Xtr, Xva, Xte = Xtr[:,:,s0:s1], Xva[:,:,s0:s1], Xte[:,:,s0:s1]
    log(f"[{a.arm}] Xtr{Xtr.shape} Xva{Xva.shape} Xte{Xte.shape} T={cfg['T']} WIN={cfg['WIN']}")
    out = {"arm": a.arm, "cfg": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
           "iters": a.iters, "runs": {}}
    for sd in [int(v) for v in a.seeds.split(",")]:
        torch.manual_seed(sd); np.random.seed(sd)
        net = make_net(cfg).to(DEV)
        if cfg["persubj_conv1"]:
            net.build_persubj(); net = net.to(DEV)
        if cfg["center_spd"]:
            net.subj_ref = fit_center_refs(net, Xtr, str_, random_ref=cfg.get("random_ref", False))
            log(f"  [B] 训练折按人参考协方差已建立: {len(net.subj_ref)} 人")
        npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
        CE = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=0.02)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
        def mk(X, y, s, sh):
            ds = torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1),
                                                torch.Tensor(y).long(),
                                                torch.Tensor(s).long())
            return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=sh, num_workers=0)
        tr = mk(Xtr,ytr,str_,True); va = mk(Xva,yva,sva,False); te = mk(Xte,yte,ste,False)
        rec = {"iter":[], "train_loss":[], "val_acc":[], "test_acc":[]}
        t0 = time.time()
        for it in range(a.iters):
            net.train(); ls = n = 0
            for xb, yb, sb in tr:
                xb, yb, sb = xb.to(DEV), yb.to(DEV), sb.to(DEV)
                loss = CE(net(xb, sb), yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
                ls += loss.item()*len(yb); n += len(yb)
            sch.step(); net.eval()
            def ev(ld):
                ok = tot = 0
                with torch.no_grad():
                    for xb, yb, sb in ld:
                        xb, yb, sb = xb.to(DEV), yb.to(DEV), sb.to(DEV)
                        ok += (net(xb, sb).argmax(1) == yb).sum().item(); tot += len(yb)
                return ok/tot
            rec["iter"].append(it); rec["train_loss"].append(ls/n)
            rec["val_acc"].append(ev(va)); rec["test_acc"].append(ev(te))
            if it % 20 == 0 or it == a.iters-1:
                log(f"  seed{sd} it{it:3d} loss={ls/n:.4f} val={rec['val_acc'][-1]*100:.2f}% test={rec['test_acc'][-1]*100:.2f}%")
        net.eval(); per = {}
        with torch.no_grad():
            ps = []; ys = []
            for xb, yb, sb in te:
                xb = xb.to(DEV); sb = sb.to(DEV)
                ps.append(net(xb, sb).argmax(1).cpu().numpy()); ys.append(yb.numpy())
        pred = np.concatenate(ps); yy = np.concatenate(ys)
        for s in np.unique(ste):
            m = ste == s; per[f"S{int(s)+1:02d}"] = [int(m.sum()), float((pred[m]==yy[m]).mean())]
        rec["per_subject"] = per; rec["secs"] = round(time.time()-t0,1); rec["n_params"] = int(npar)
        log(f"  seed{sd} 完成 {rec['secs']}s params={npar} 最终test={rec['test_acc'][-1]*100:.2f}% 峰值={max(rec['test_acc'])*100:.2f}%")
        out["runs"][str(sd)] = rec
    od = BASE/"results"/f"individual_variants_{a.tag}"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od/f"{a.arm}.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[落盘] {od}/{a.arm}.json")


if __name__ == "__main__":
    main()