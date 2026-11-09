"""窄带并联前端 A/B (平衡精度与计算量)

设计:
  基线臂 base : channel_weight -> conv1(22->8) -> IN -> conv2(1x12,8->8) -> IN -> E2R -> 原后端
  窄带臂 iaf  : 同上, 并联合成一路:
                 x_iaf = FIR96(x)                [固定, 每被试中心=训练折估计的 IAF, 带宽±1.5Hz]
                 z_iaf = conv1_iaf(22->4) -> IN  [可学, 独立空间滤波]
                 z_iaf = conv2_iaf(1x12,4->4) -> IN
                 通道维拼接 (8+4=12) -> 同一个 E2R/SPD/注意力后端 (维度自适应)
  计算量:   base 14.4M MAC/样本 ; iaf 18.4M (1.28x)
  滤波器:   固定(不学); IAF 只从训练折静息段估计 -> 无测试泄漏

用法: python analysis/exp_narrowband_frontend.py --arm base --seeds 42 --iters 40 --tag pilot
"""
import sys, json, time, argparse
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent/'preprocess'))
import torch, torch.nn as nn
from scipy.signal import welch, firwin
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from mAtt.mAtt import _SparseAttentionBase, ChannelPriorWeighting, BCI2A_CHANNELS

DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FS, WIN, STEP, NCH, TAPS, BAND = 250, 375, 187, 22, 96, 1.5
CENTRAL = [7,8,9,10,11,14,15,16]
IAF_FIXED = {0:11.0, 1:9.0, 2:11.0, 3:11.0, 4:9.0, 5:12.0, 6:10.0, 7:11.0, 8:10.0}  # 训练折估计值
log = lambda *a: print(*a, flush=True)


def load_split():
    data,_ = load_preprocessed_dataset(str(BASE.parent/'saved_data'/'preprocessed_data_bci2a_raw_ica.pkl'))
    X = np.concatenate([data[s][0] for s in range(len(data))],axis=0)
    y = np.concatenate([data[s][1] for s in range(len(data))],axis=0)
    sub = np.concatenate([np.full(len(data[s][1]),s,dtype=np.int64) for s in range(len(data))])
    o = split_train_valid_test(X,y,train_ratio=.70,valid_ratio=.15,random_state=42,others=[sub])
    return o


def make_iaf_data(X, sub):
    """对每个被试按自己的 IAF 做固定 FIR 窄带滤波 -> (N,22,T)"""
    out = np.empty_like(X)
    for s in np.unique(sub):
        h = firwin(TAPS, [max(1.,IAF_FIXED[int(s)]-BAND), IAF_FIXED[int(s)]+BAND],
                   pass_zero=False, fs=FS, window='hamming')
        h = h - h.mean(); h = h/np.abs(h).sum()
        m = sub==s
        out[m] = np.apply_along_axis(lambda v: np.convolve(v, h, mode='same'), -1, X[m])
    return out.astype(np.float32)


class Net(_SparseAttentionBase):
    def __init__(self, use_iaf, n_iaf_out=4, **kw):
        self.use_iaf = bool(use_iaf)
        self.n_iaf_out = n_iaf_out
        width = kw['d_attn_in']                      # 宽带路宽度(8), 也是 conv2 输出通道数
        total = width + (n_iaf_out if use_iaf else 0)  # 拼接后总通道(12): E2R/SPD/注意力都看这个
        # 基类里 d_attn_in 同时决定 conv2 输出通道数与 SPDNorm/注意力维度。
        # 先按 total 建(让 SPDNorm/注意力看到 12 通道), 再把 conv2 输出改回 width(8):
        # 宽带路出 8, 另外 4 由窄带支路补足, 拼接后正好 total。
        super().__init__(WIN, STEP, **{**kw, 'd_attn_in': total})
        self.conv2 = nn.Conv2d(kw['conv1_out'], width, (1, 12), padding=(0, 6))
        self.In2 = nn.InstanceNorm2d(width, affine=True)
        if use_iaf:
            self.conv1b = nn.Conv2d(1, n_iaf_out, (22, 1))
            self.In1b = nn.InstanceNorm2d(n_iaf_out, affine=True)
            self.conv2b = nn.Conv2d(n_iaf_out, n_iaf_out, (1, 12), padding=(0, 6))
            self.In2b = nn.InstanceNorm2d(n_iaf_out, affine=True)
            self.w_iaf = nn.Parameter(torch.ones(total, 1, 1))
        else:
            for k in ['conv1b','In1b','conv2b','In2b','w_iaf']: setattr(self, k, None)

    def _shared_backbone(self, x, xb_iaf=None):
        if self.channel_weight is not None:
            x = self.channel_weight(x)
            if xb_iaf is not None: xb_iaf = self.channel_weight(xb_iaf)
        a = self.In2(self.conv2(self.In1(self.conv1(x))))
        if self.use_iaf and xb_iaf is not None:
            b = self.In2b(self.conv2b(self.In1b(self.conv1b(xb_iaf))))
            out = torch.cat([a, b], 1) * self.w_iaf
        else:
            out = a
        spd = self.spdnorm(self.ract1(out))
        return self.manifold_change(spd)

    def _features(self, x, xb_iaf=None):
        spd = self._shared_backbone(x, xb_iaf)
        x, shape = self.att2(spd)
        return self.flat(self.tangent(self.ract2(x)).view(shape[0], shape[1], -1))

    def forward(self, x, xb_iaf=None):
        f = self._features(x, xb_iaf)
        if self.dropout is not None: f = self.dropout(f)
        return self.linear(f)


def _infer_M():
    """conv2 padding=6 -> 输出 T+1 点, E2R 滑窗数"""
    return (1126 + 1 - WIN)//STEP + 1


def make_net(use_iaf, n_iaf_out=4):
    kw = dict(n_classes=4, d_attn_in=8, d_attn_out=8, local_radius=6, anchor_stride=None,
              n_heads=1, n_windows=_infer_M(), use_hann=True, use_channel_weight=True,
              n_spatial_filters=22, channel_gaussian=False, gaussian_sigma=0.62,
              conv1_kernel=22, conv1_out=22, conv2_kernel=12, conv2_pad=6,
              conv2_separable=False, conv2_multiscale=False, manifold_change=None,
              attn_change=None, attn_prior=None, dropout=0.3, score_original=True, norm_type='in')
    return Net(use_iaf, n_iaf_out, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', choices=['base','iaf'], required=True)
    ap.add_argument('--seeds', default='42'); ap.add_argument('--iters', type=int, default=40)
    ap.add_argument('--tag', default='pilot'); ap.add_argument('--bs', type=int, default=128)
    a = ap.parse_args()
    o = load_split()
    Xtr,ytr,str_,Xva,yva,sva,Xte,yte,ste = o[0],o[1],o[6],o[2],o[3],o[7],o[4],o[5],o[8]
    if a.arm=='iaf':
        t0=time.time(); Xtr_f = make_iaf_data(Xtr,str_); Xva_f = make_iaf_data(Xva,sva); Xte_f = make_iaf_data(Xte,ste)
        log(f'[窄带数据] 预滤波完成 {time.time()-t0:.1f}s')
    else:
        Xtr_f=Xva_f=Xte_f=None
    def mk(X,Xf,y,sh):
        t=[torch.Tensor(X).unsqueeze(1)]
        if Xf is not None: t.append(torch.Tensor(Xf).unsqueeze(1))
        t.append(torch.Tensor(y).long())
        return torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*t), batch_size=a.bs,
                                           shuffle=sh, num_workers=0)
    tr,va,te = mk(Xtr,Xtr_f,ytr,True), mk(Xva,Xva_f,yva,False), mk(Xte,Xte_f,yte,False)
    out = {'arm':a.arm,'iters':a.iters,'bs':a.bs,'iaf_hz':{f'S{k+1:02d}':v for k,v in IAF_FIXED.items()},
           'taps':TAPS,'band':BAND,'runs':{}}
    for sd in [int(v) for v in a.seeds.split(',')]:
        torch.manual_seed(sd); np.random.seed(sd)
        net = make_net(a.arm=='iaf', n_iaf_out=4).to(DEV)
        npar = sum(p.numel() for p in net.parameters() if p.requires_grad)
        CE = nn.CrossEntropyLoss(); opt = torch.optim.Adam(net.parameters(), lr=5e-4, weight_decay=0.02)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.iters)
        rec={'iter':[],'train_loss':[],'val_acc':[],'test_acc':[]}; t0=time.time()
        for it in range(a.iters):
            net.train(); ls=n=0
            for batch in tr:
                xb = batch[0].to(DEV); yb = batch[-1].to(DEV)
                xf = batch[1].to(DEV) if a.arm=='iaf' else None
                loss = CE(net(xb, xf), yb)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.0); opt.step()
                ls += loss.item()*len(yb); n += len(yb)
            sch.step(); net.eval()
            def ev(ld):
                ok=tot=0
                with torch.no_grad():
                    for batch in ld:
                        xb=batch[0].to(DEV); yb=batch[-1].to(DEV)
                        xf = batch[1].to(DEV) if a.arm=='iaf' else None
                        ok += (net(xb,xf).argmax(1)==yb).sum().item(); tot += len(yb)
                return ok/tot
            rec['iter'].append(it); rec['train_loss'].append(ls/n)
            rec['val_acc'].append(ev(va)); rec['test_acc'].append(ev(te))
            if it%10==0 or it==a.iters-1:
                log(f'  seed{sd} it{it:3d} loss={ls/n:.4f} val={rec["val_acc"][-1]*100:.2f}% test={rec["test_acc"][-1]*100:.2f}%')
        net.eval(); per={}
        with torch.no_grad():
            ps=[];ys=[]
            for batch in te:
                xb=batch[0].to(DEV); yb=batch[-1]
                xf = batch[1].to(DEV) if a.arm=='iaf' else None
                ps.append(net(xb,xf).argmax(1).cpu().numpy()); ys.append(yb.numpy())
        pred=np.concatenate(ps); yy=np.concatenate(ys)
        for s in np.unique(ste):
            m=ste==s; per[f'S{int(s)+1:02d}']=[int(m.sum()), float((pred[m]==yy[m]).mean())]
        rec['per_subject']=per; rec['secs']=round(time.time()-t0,1); rec['n_params']=int(npar)
        log(f'  seed{sd} 完成 {rec["secs"]}s  params={npar}  最终test={rec["test_acc"][-1]*100:.2f}% 峰值={max(rec["test_acc"])*100:.2f}%')
        out['runs'][str(sd)]=rec
    od = BASE/'results'/f'narrowband_{a.tag}'; od.mkdir(parents=True,exist_ok=True)
    json.dump(out, open(od/f'{a.arm}.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
    log(f'[落盘] {od}/{a.arm}.json')

if __name__=='__main__': main()
