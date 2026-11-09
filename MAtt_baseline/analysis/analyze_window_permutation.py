"""窗信息的配对检验: (1) 窗口顺序置换 (2) 逐被试配对差
判决: 若打乱窗的时间顺序后精度不掉, 说明"哪一窗"不携带判别信息。"""
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
def feats(net, X, bs=128):
    net.eval(); out=[]
    for i in range(0,len(X),bs):
        xb = torch.as_tensor(X[i:i+bs],dtype=torch.float32,device=DEVICE).unsqueeze(1)
        spd = net._shared_backbone(xb); x, shape = net.att2(spd)
        out.append(net.tangent(net.ract2(x)).reshape(shape[0],shape[1],-1).cpu().numpy())
    return np.concatenate(out)

def probe(Vtr,ytr,Vva,yva,Vte,seed=0):
    sc = StandardScaler().fit(Vtr); A,B,C = sc.transform(Vtr),sc.transform(Vva),sc.transform(Vte)
    best,bclf = -1,None
    for c in [0.003,0.01,0.03,0.1,0.3,1.0,3.0]:
        clf = LogisticRegression(C=c,max_iter=4000).fit(A,ytr)
        a = accuracy_score(yva,clf.predict(B))
        if a>best: best,bclf = a,clf
    return bclf.predict(C)

def main():
    t0=time.time(); rng=np.random.default_rng(0)
    (Xtr,ytr,str_),(Xva,yva,sva),(Xte,yte,ste) = load_all()
    M = 5
    acc_all, acc_mean, acc_shuf = {}, {}, {}
    per_subj = {}
    for tag,path in CKPTS:
        net = build_net(); net.load_state_dict(torch.load(path,map_location='cpu'),strict=False); net.to(DEVICE)
        Ftr,Fva,Fte = feats(net,Xtr),feats(net,Xva),feats(net,Xte)
        del net
        if DEVICE.type=='cuda': torch.cuda.empty_cache()
        # 自然顺序: 拼接 与 平均
        p_all  = probe(Ftr.reshape(len(Ftr),-1),ytr, Fva.reshape(len(Fva),-1),yva, Fte.reshape(len(Fte),-1))
        p_mean = probe(Ftr.mean(1),ytr, Fva.mean(1),yva, Fte.mean(1))
        # 置换: 每个样本在 train/valid/test 上按同一规则打乱窗顺序(同一 rng 序列长度=各折样本数)
        def shuf(F, r):
            G = F.copy()
            for i in range(len(G)): G[i] = G[i][r.permutation(M)]
            return G
        r1 = np.random.default_rng(12345); ftr_s = shuf(Ftr,r1)
        r2 = np.random.default_rng(12345); fva_s = shuf(Fva,r2)
        r3 = np.random.default_rng(12345); fte_s = shuf(Fte,r3)
        p_shuf = probe(ftr_s.reshape(len(ftr_s),-1),ytr, fva_s.reshape(len(fva_s),-1),yva, fte_s.reshape(len(fte_s),-1))
        for nm,pred in [('all',p_all),('mean',p_mean),('shuffled',p_shuf)]:
            per = {f'S{int(s)+1:02d}': float(accuracy_score(yte[ste==s],pred[ste==s])) for s in np.unique(ste)}
            per_subj.setdefault(nm,{})[tag]=per
        acc_all[tag]=accuracy_score(yte,p_all); acc_mean[tag]=accuracy_score(yte,p_mean); acc_shuf[tag]=accuracy_score(yte,p_shuf)
        print(f'{tag}: all={acc_all[tag]*100:.2f}%  mean={acc_mean[tag]*100:.2f}%  shuffled={acc_shuf[tag]*100:.2f}%', flush=True)
    print()
    print('=== 逐 seed 配对差 (pp) ===')
    for tag in acc_all:
        print(f'  {tag}: all-mean={100*(acc_all[tag]-acc_mean[tag]):+.2f}   all-shuffled={100*(acc_all[tag]-acc_shuf[tag]):+.2f}   mean-shuffled={100*(acc_mean[tag]-acc_shuf[tag]):+.2f}')
    print()
    print('=== 跨 9 被试 (3 seed x 9 人 = 27 配对) ===')
    for a,b,lab in [('all','mean','all - mean'),('all','shuffled','all - shuffled'),('mean','shuffled','mean - shuffled')]:
        d=[]
        for tag in per_subj[a]:
            for s in per_subj[a][tag]: d.append(100*(per_subj[a][tag][s]-per_subj[b][tag][s]))
        d=np.array(d)
        se = d.std(ddof=1)/np.sqrt(len(d))
        print(f'  {lab:16s}: 均值={d.mean():+6.2f}pp  SD={d.std(ddof=1):5.2f}  n={len(d)}  SE={se:.2f}  t={d.mean()/se:+.2f}')
    od = BASE/'results'/'window_ablation'
    json.dump({'per_seed':{'all':acc_all,'mean':acc_mean,'shuffled':acc_shuf},
               'per_subject':per_subj,'elapsed_s':round(time.time()-t0,1)},
              open(od/'window_permutation.json','w',encoding='utf-8'), ensure_ascii=False, indent=1)
    print(f'\n[落盘] {od}/window_permutation.json  {round(time.time()-t0,1)}s')

if __name__=='__main__': main()
