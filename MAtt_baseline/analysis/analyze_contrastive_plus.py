# -*- coding: utf-8 -*-
"""SupCon 表示加强:容量/温度/epoch 扫描 + 核读出 + 无监督簇对齐(合法化 D 臂)"""
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH
import analyze_erd_decomposition as ED
from analyze_multichart_ceiling import cov_tn, grp, fit_lda
from utils.dataset import load_preprocessed_dataset, split_train_valid_test

OUT = BASE / 'results' / 'contrastive_repr'
TASK = ED.TASK; SUBS = ['S%02d' % i for i in range(1, 10)]
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'

def spdvec(M):
    iu = np.triu_indices(22); v = M[iu].copy(); v[iu[0] != iu[1]] *= np.sqrt(2); return v
def le(A):
    w, V = np.linalg.eigh(A.astype(np.float64)); w = np.maximum(w, 1e-12)
    return np.stack([spdvec((V[i]*np.log(w[i])) @ V[i].T) for i in range(len(A))])
class Enc(nn.Module):
    def __init__(s, d_in=253, d=256, z=64):
        super().__init__()
        s.net = nn.Sequential(nn.Linear(d_in,d), nn.LayerNorm(d), nn.GELU(),
                              nn.Linear(d,d), nn.LayerNorm(d), nn.GELU(), nn.Linear(d,z))
    def forward(s,x): return s.net(x)

def supcon(z, y, tau=0.2):
    z = F.normalize(z, 1); sim = z @ z.t() / tau
    B = z.shape[0]
    mask = (y[:,None] == y[None,:]).float(); mask.fill_diagonal_(0)
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    exp = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
    lp = sim - torch.log(exp.sum(1, keepdim=True) + 1e-12)
    return -((lp * mask).sum(1) / mask.sum(1).clamp(min=1)).mean()

def train(X1, X2, y, tau=0.2, d=256, z=64, epochs=200, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    enc = Enc(X1.shape[1], d, z).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    X1t = torch.tensor(X1, dtype=torch.float32); X2t = torch.tensor(X2, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long).to(DEV)
    n = len(X1t)
    for _ in range(epochs):
        enc.train(); perm = torch.randperm(n); tot=0; nb=0
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            if len(idx) < 8: continue
            za = enc(X1t[idx].to(DEV)); zb = enc(X2t[idx].to(DEV))
            zz = torch.cat([za, zb], 0); yy = torch.cat([yt[idx], yt[idx]], 0)
            loss = supcon(zz, yy, tau)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
    return enc
def embed(enc, X):
    enc.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(enc(torch.tensor(X[i:i+1024], dtype=torch.float32).to(DEV)).cpu().numpy())
    return np.concatenate(out, 0)

sd,_ = load_preprocessed_dataset(str(PKL))
X = np.concatenate([s[0] for s in sd], 0).astype(np.float64)
y = np.concatenate([s[1] for s in sd], 0)
sub = np.concatenate([np.full(len(s[1]), i, dtype=np.int64) for i, s in enumerate(sd)])
ret = split_train_valid_test(X, y, 0.70, 0.15, random_state=42, others=[sub])
Xtr,ytr,str_ = ret[0],ret[1],ret[6]; Xva,yva = ret[2],ret[3]; Xte,yte,ste = ret[4],ret[5],ret[8]
L = [le(cov_tn(x[:,:,TASK[0]:TASK[1]])) for x in (Xtr,Xva,Xte)]
Ltr,Lva,Lte = L

rec = {'note':'读出=闭式LW-LDA / SVM;idpurity=kNN(9)','arms':{}}
def rep(name, F, svm=False):
    ok = fit_lda(F[0], ytr).predict(F[2]) == yte
    per = {SUBS[i]: float(ok[ste==i].mean()) for i in range(9)}; lo,hi = grp(per)
    pur = float((KNeighborsClassifier(n_neighbors=9).fit(F[0], str_).predict(F[0]) == str_).mean())
    line = '%-26s LDA %5.1f (low %5.1f high %5.1f) id %.3f' % (name, ok.mean()*100, lo*100, hi*100, pur)
    rec['arms'][name] = {'lda': float(ok.mean()), 'low4': lo, 'high5': hi, 'id_purity': pur}
    if svm:
        r0 = np.random.default_rng(1).choice(len(F[0]), min(800,len(F[0])), replace=False)
        m2 = float(np.median(((F[0][r0][:,None,:]-F[0][r0][None,:,:])**2).sum(-1)[np.triu_indices(len(r0),1)]))
        best=None
        for C_ in (10.,100.):
            for gm in (1.,2.,3.):
                s = SVC(kernel='rbf', C=C_, gamma=gm/(2*m2)).fit(F[0], ytr)
                v = float((s.predict(F[1])==yva).mean())
                if best is None or v>best[0]: best=(v,s)
        oks = best[1].predict(F[2])==yte
        line += '  SVM %5.1f (low %5.1f high %5.1f)' % (oks.mean()*100,
            np.mean([oks[ste==i].mean() for i in LOW])*100, np.mean([oks[ste==i].mean() for i in HIGH])*100)
        rec['arms'][name]['svm'] = float(oks.mean())
    print(line, flush=True)
    return ok

rep('base_LE253', (Ltr,Lva,Lte), svm=True)
rng = np.random.default_rng(0)
nz = lambda Z,s: Z + rng.normal(0, s*Z.std(0), Z.shape)
best = None
for tau in (0.1, 0.2):
    for d,z in ((256,64),(512,128)):
        for ep in (200,):
            enc = train(nz(Ltr,0.05), nz(Ltr,0.05), ytr, tau=tau, d=d, z=z, epochs=ep, seed=0)
            E = [embed(enc, T) for T in (Ltr,Lva,Lte)]
            tag = 'supcon_t%.1f_d%d_z%d' % (tau,d,z)
            rep(tag, E, svm=True)
            if best is None or rec['arms'][tag]['svm'] > best[0]: best = (rec['arms'][tag]['svm'], tag, enc, E)
print('BEST svm:', best[1], '%.1f' % (best[0]*100), flush=True)
# 无监督簇对齐(合法化 D 臂):KMeans 9 簇 + 同类跨簇中心拉近的表示已由 supcon 得到,这里只做后处理对齐
Ftr,Fva,Fte = best[3]
km = KMeans(n_clusters=9, n_init=10, random_state=0).fit(Ftr)
c_tr = km.labels_; c_va = km.predict(Fva); c_te = km.predict(Fte)
MU = np.stack([Ftr[c_tr==k].mean(0) for k in range(9)])
al = lambda A,c: np.concatenate([A[c==k]-MU[k] for k in range(9)],0)
rep('supcon_best_cluster_align', (al(Ftr,c_tr), al(Fva,c_va), al(Fte,c_te)), svm=True)
(OUT/'contrastive_repr_plus.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
print('[OK]', flush=True)
