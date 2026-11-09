"""conv2 时间滤波 vs 带通: 把可学 conv2 换成固定 FIR 带通 (梯子实验)

问题: conv2 的 1x12 可学时间卷积, 与一个设计好的带通真的一样吗?
      若一样 => 换成固定带通应当无损; 且加长固定带通应能提高频率选择性。

臂:
  base        现状 (conv2 = 可学 1x12)
  fixed12     换成固定 1x12 FIR 带通        (与现状同抽头数, 只去掉"可学")
  fixed24     换成固定 1x24 FIR 带通        (匹配原文 128Hz 下 12 抽头的等效选择性)
  fixed96     换成固定 1x96 FIR 带通(按人IAF)(高选择性, 接近 1/Δf 要求)

所有臂 SPD 维度保持 8, 后端与参数量完全一致, 只差时间滤波。
固定滤波不参与训练(不需要梯度), 输出经 1x1 投影回 8 通道(可学, 保持容量可比)。

用法: python analysis/exp_fixed_conv2.py --arm fixed24 --seeds 42,2024,7 --iters 100 --tag main
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent/'preprocess'))
import torch, torch.nn as nn, torch.nn.functional as F
from scipy.signal import welch, firwin
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from mAtt.mAtt import _SparseAttentionBase

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FS, WIN, STEP, NCH = 250, 375, 187, 22
CENTRAL = [7,8,9,10,11,14,15,16]
IAF_FIXED = {0:11.0, 1:9.0, 2:11.0, 3:11.0, 4:9.0, 5:12.0, 6:10.0, 7:11.0, 8:10.0}
log = lambda *a: print(*a, flush=True)

ARMS = {
  "base":    dict(taps=None, n_filt=8, per_subject=False),
  "fixed12": dict(taps=12,  n_filt=8, per_subject=False),
  "fixed24": dict(taps=24,  n_filt=8, per_subject=False),
  "fixed96": dict(taps=96,  n_filt=8, per_subject=True),
}


def load_split():
    data,_ = load_preprocessed_dataset(str(BASE.parent/'saved_data'/'preprocessed_data_bci2a_raw_ica.pkl'))
    X = np.concatenate([data[s][0] for s in range(len(data))],axis=0)
    y = np.concatenate([data[s][1] for s in range(len(data))],axis=0)
    sub = np.concatenate([np.full(len(data[s][1]),s,dtype=np.int64) for s in range(len(data))])
    o = split_train_valid_test(X,y,train_ratio=.70,valid_ratio=.15,random_state=42,others=[sub])
    return (o[0],o[1],o[6]), (o[2],o[3],o[7]), (o[4],o[5],o[8])


def iaf_from_train(Xtr, str_):
    out = {}
    for s in np.unique(str_):
        rest = Xtr[str_==s][:, CENTRAL, :250]
        f,P = welch(rest, fs=FS, nperseg=250, axis=-1)
        P = P.mean(axis=(0,1)); m = (f>=7)&(f<=13)
        out[int(s)] = float(f[m][P[m].argmax()])
    return out


def design(taps, lo, hi):
    h = firwin(taps, [lo, hi], pass_zero=False, fs=FS, window="hamming")
    h = h - h.mean()
    return (h/np.abs(h).sum()).astype(np.float32)


class Net(_SparseAttentionBase):
    """base: 可学 conv2;  其余: 固定 FIR + 可学 1x1 投影"""
    def __init__(self, arm, taps=None, n_filt=8, per_subject=False, **kw):
        self.arm = arm; self.taps = taps; self.n_filt = n_filt; self.per_subject = per_subject
        super().__init__(WIN, STEP, **kw)
        if taps is not None:
            self.proj = nn.Conv2d(NCH, n_filt, (1,1))
            self.In2 = nn.InstanceNorm2d(n_filt, affine=True)
            k = design(taps, 8.0, 13.0) if not per_subject else None
            if not per_subject:
                self.register_buffer("fir", torch.tensor(k).view(1,1,1,-1), persistent=False)

    def _conv2_fixed(self, x, sidx):
        B, C, _, T = x.shape
        P = self.taps // 2
        xp = F.pad(x, (P, self.taps - 1 - P))          # (B,C,1,T+taps-1)
        L = xp.shape[-1]
        xr = xp.reshape(B * C, 1, 1, L)                # 把 (B,C) 合并成 batch, groups=B*C 做同核深度卷积
        if self.per_subject:
            outs = torch.zeros(B, C, 1, T, device=x.device, dtype=x.dtype)
            for s in torch.unique(sidx):
                m = (sidx == s)
                if int(s) not in self.subj_fir: continue
                h = self.subj_fir[int(s)].to(x.device).view(1, 1, 1, -1)
                xs = xp[m].reshape(-1, 1, 1, L)
                y = F.conv2d(xs, h)
                outs[m] = y.reshape(int(m.sum()), C, 1, T)
            y = outs
        else:
            h = self.fir.to(x.device).view(1, 1, 1, -1)
            y = F.conv2d(xr, h).reshape(B, C, 1, T)
        return self.In2(self.proj(y))

    def set_subject_firs(self, iaf):
        self.subj_fir = {k: torch.tensor(design(self.taps, max(1.0,v-1.5), v+1.5)).to(DEV)
                         for k, v in iaf.items()}

    def _shared_backbone(self, x, sidx=None):
        if self.channel_weight is not None: x = self.channel_weight(x)
        x = self.In1(self.conv1(x))
        if self.taps is None:
            x = self.In2(self.conv2(x))
        else:
            x = self._conv2_fixed(x, sidx)
        spd = self.spdnorm(self.ract1(x))
        return self.manifold_change(spd)

    def forward(self, x, sidx=None):
        spd = self._shared_backbone(x, sidx)
        x, shape = self.att2(spd); x = self.ract2(x); x = self.tangent(x)
        x = self.flat(x.view(shape[0], shape[1], -1))
        if self.dropout is not None: x = self.dropout(x)
        return self.linear(x)


def make_net(arm, iaf=None):
    c = ARMS[arm]
    kw = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6, anchor_stride=None,
              n_heads=1, n_windows=5, use_hann=True, use_channel_weight=True,
              n_spatial_filters=22, channel_gaussian=False, gaussian_sigma=0.62,
              conv1_kernel=22, conv1_out=22, conv2_kernel=12, conv2_pad=6, conv2_out=8,
              conv2_separable=False, conv2_multiscale=False, manifold_change=None,
              attn_change=None, attn_prior=None, dropout=0.3, score_original=True, norm_type="in")
    net = Net(arm, taps=c["taps"], n_filt=c["n_filt"], per_subject=c["per_subject"], **kw)
    if c["per_subject"]:
        net.subj_fir = {}
        net.set_subject_firs(iaf)
    return net


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    (Xtr,ytr,str_), (Xva,yva,sva), (Xte,yte,ste) = load_split()
    iaf = iaf_from_train(Xtr, str_)
    log(f"[{a.arm}] taps={ARMS[a.arm]['taps']} per_subject={ARMS[a.arm]['per_subject']}  IAF=" +
        " ".join(f"S{k+1:02d}={v:.0f}" for k,v in sorted(iaf.items())))
    out = {"arm": a.arm, "cfg": ARMS[a.arm], "iters": a.iters,
           "iaf_hz": {f"S{k+1:02d}": v for k,v in sorted(iaf.items())}, "runs": {}}
    for sd in [int(v) for v in a.seeds.split(",")]:
        torch.manual_seed(sd); np.random.seed(sd)
        net = make_net(a.arm, iaf).to(DEV)
        npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
        CE = nn.CrossEntropyLoss()
        opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=5e-4, weight_decay=0.02)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
        def mk(X,y,s,sh):
            ds = torch.utils.data.TensorDataset(torch.Tensor(X).unsqueeze(1),
                torch.Tensor(y).long(), torch.Tensor(s).long())
            return torch.utils.data.DataLoader(ds, batch_size=128, shuffle=sh, num_workers=0)
        tr, va, te = mk(Xtr,ytr,str_,True), mk(Xva,yva,sva,False), mk(Xte,yte,ste,False)
        rec = {"iter":[], "train_loss":[], "val_acc":[], "test_acc":[]}
        t0 = time.time()
        for it in range(a.iters):
            net.train(); ls = n = 0
            for xb, yb, sb in tr:
                xb, yb, sb = xb.to(DEV), yb.to(DEV), sb.to(DEV)
                loss = CE(net(xb, sb), yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_([p for p in net.parameters() if p.requires_grad], 1.0)
                opt.step()
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
                xb, sb = xb.to(DEV), sb.to(DEV)
                ps.append(net(xb, sb).argmax(1).cpu().numpy()); ys.append(yb.numpy())
        pred = np.concatenate(ps); yy = np.concatenate(ys)
        for s in np.unique(ste):
            m = ste == s; per[f"S{int(s)+1:02d}"] = [int(m.sum()), float((pred[m]==yy[m]).mean())]
        rec["per_subject"] = per; rec["secs"] = round(time.time()-t0,1); rec["n_params"] = int(npar)
        log(f"  seed{sd} 完成 {rec['secs']}s params={npar} 最终test={rec['test_acc'][-1]*100:.2f}% 峰值={max(rec['test_acc'])*100:.2f}%")
        out["runs"][str(sd)] = rec
    od = BASE/"results"/f"fixed_conv2_{a.tag}"; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od/f"{a.arm}.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
    log(f"[落盘] {od}/{a.arm}.json")


if __name__ == "__main__":
    main()