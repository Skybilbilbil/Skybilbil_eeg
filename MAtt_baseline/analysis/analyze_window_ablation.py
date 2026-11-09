"""窗级特征消融: 时间维到底是不是瓶颈(只读已有 ckpt, 不训练)"""
import sys, json, time
from pathlib import Path
import numpy as np
BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE.parent / 'preprocess'))
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils.dataset import load_preprocessed_dataset, split_train_valid_test
from utils.functions import build_matt_model
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WIN, STEP = 375, 187
CKPTS = [('seed42', BASE/'results/verify_samepool_70_15_15/verify70_cross_mix_best_acc.pt'),
         ('seed2024', BASE/'results/verify70_seed2024/v70s2024_cross_mix_best_acc.pt'),
         ('seed7', BASE/'results/verify70_seed7/v70s7_cross_mix_best_acc.pt')]
RECORDED = {'seed42': 56.17, 'seed2024': 54.24, 'seed7': 55.91}

def load_all():
    data, _ = load_preprocessed_dataset(str(BASE.parent/'saved_data'/'preprocessed_data_bci2a_raw_ica.pkl'))
    X = np.concatenate([data[s][0] for s in range(len(data))], axis=0)
    y = np.concatenate([data[s][1] for s in range(len(data))], axis=0)
    sub = np.concatenate([np.full(len(data[s][1]), s, dtype=np.int64) for s in range(len(data))])
    o = split_train_valid_test(X, y, train_ratio=0.70, valid_ratio=0.15, random_state=42, others=[sub])
    return (o[0],o[1],o[6]), (o[2],o[3],o[7]), (o[4],o[5],o[8])

def build_net():
    return build_matt_model(WIN, STEP, {'n_classes':4,'d_attn_in':8,'d_attn_out':8,'local_radius':6,
        'anchor_stride':None,'n_heads':1,'n_windows':None,'conv2_separable':False,'conv2_multiscale':False,
        'n_spatial_filters':22,'channel_gaussian':False,'gaussian_sigma':0.62,'attn_change':None,
        'attn_prior':None,'manifold_change':None,'dropout':0.3,'norm_type':'in'})

@torch.no_grad()
def window_features(net, X, bs=128):
    net.eval(); feats, logits, atts = [], [], []
    for i in range(0, len(X), bs):
        xb = torch.as_tensor(X[i:i+bs], dtype=torch.float32, device=DEVICE).unsqueeze(1)
        spd = net._shared_backbone(xb)
        try:
            att = net.att2.attention_prob(spd); atts.append(att.mean(0).cpu().numpy())
        except Exception:
            pass
        x, shape = net.att2(spd)
        f = net.tangent(net.ract2(x)).reshape(shape[0], shape[1], -1)
        feats.append(f.cpu().numpy()); logits.append(net.linear(f.flatten(1)).cpu().numpy())
    return np.concatenate(feats), np.concatenate(logits), (np.mean(atts,0) if atts else None)

def variants(M):
    v = {'all_windows_拼接': ('all', None), 'mean_over_windows_平均': ('mean', None),
         'first+last': ('pick',[0,M-1]), 'w2+w3+w4': ('pick', list(range(1,min(4,M))))}
    for m in range(M): v[f'only_w{m+1}'] = ('pick',[m])
    return v

def av(F, kind, idx):
    if kind=='all': return F.reshape(len(F),-1)
    if kind=='mean': return F.mean(axis=1)
    return F[:, idx, :].reshape(len(F),-1)

def fit_probe(Ftr,ytr,Fva,yva):
    sc = StandardScaler().fit(Ftr); A, B = sc.transform(Ftr), sc.transform(Fva)
    best,bC,bclf = -1,None,None
    for C in [0.003,0.01,0.03,0.1,0.3,1.0,3.0]:
        clf = LogisticRegression(C=C, max_iter=4000).fit(A,ytr)
        a = accuracy_score(yva, clf.predict(B))
        if a > best: best,bC,bclf = a,C,clf
    return bclf, bC, sc

def main():
    t0=time.time()
    (Xtr,ytr,_),(Xva,yva,_),(Xte,yte,ste) = load_all()
    print(f'[数据] train {Xtr.shape} valid {Xva.shape} test {Xte.shape} dev={DEVICE}', flush=True)
    out={'meta':{'n_test_trials':int(len(yte)),'split':'samepool 70/15/15 rs=42',
        'probe':'StandardScaler+LogReg(C by valid)',
        'note':'绝对精度低于 pair_training.json 记录(前向有未查明偏差); 变体间同一前向, 相对比较有效'},'runs':{}}
    for tag, path in CKPTS:
        if not path.exists(): print(f'[跳过] {path}'); continue
        net = build_net(); net.load_state_dict(torch.load(path, map_location='cpu'), strict=False); net.to(DEVICE)
        Ftr,Ltr,A = window_features(net,Xtr); Fva,_,_ = window_features(net,Xva); Fte,Lte,_ = window_features(net,Xte)
        M = Fte.shape[1]; native = accuracy_score(yte, Lte.argmax(1))
        attc = float(np.trace(A)/A.sum()) if A is not None else float('nan')
        print(f'\n=== {tag}  M={M}  (记录 {RECORDED[tag]}%) ===', flush=True)
        print(f'  [对照] 原生头 test={native*100:.2f}%   注意力对角集中度={attc:.3f} (1/M={1/M:.3f}=均匀)', flush=True)
        rec={'native_head_test':float(native),'M':int(M),'attn_conc':attc,'recorded_test':RECORDED[tag],'variants':{}}
        for name,(kind,idx) in variants(M).items():
            Vtr,Vva,Vte = av(Ftr,kind,idx), av(Fva,kind,idx), av(Fte,kind,idx)
            clf,C,sc = fit_probe(Vtr,ytr,Vva,yva)
            pred = clf.predict(sc.transform(Vte)); a = accuracy_score(yte,pred)
            per = {f'S{int(s)+1:02d}':[int((ste==s).sum()), float(accuracy_score(yte[ste==s],pred[ste==s]))] for s in np.unique(ste)}
            rec['variants'][name]={'test':float(a),'C':float(C),'dim':int(Vte.shape[1]),'per_subject':per}
            print(f'  {name:24s} dim={Vte.shape[1]:4d}  test={a*100:5.2f}%  C={C}', flush=True)
        out['runs'][tag]=rec
        del net
        if DEVICE.type=='cuda': torch.cuda.empty_cache()
    out['elapsed_s']=round(time.time()-t0,1)
    od = BASE/'results'/'window_ablation'; od.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(od/'window_ablation.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n[落盘] {od}/window_ablation.json  {out["elapsed_s"]}s', flush=True)

if __name__ == '__main__': main()
