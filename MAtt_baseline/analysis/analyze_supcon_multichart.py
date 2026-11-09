# -*- coding: utf-8 -*-
"""SupCon 表示 ⊕ 多切图:两种增益是否可叠加(多切图=曲率/多参考点;SupCon=类内紧凑/身份压缩)

问:多切图线性 68.6 与 SupCon 表示 69.7 的增益是否独立?
  可叠加 ⇒ 拼接后应显著 > 73.5(经典核读出);
  不可叠加 ⇒ 二者吃的是同一份信息(与推进十八「多切图核 vs 单切图核不可加」同型)。
判据:test > 73.5;同时报身份纯度(衡量共享性)。
"""
import json, sys
from pathlib import Path
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE)); sys.path.insert(0, str(BASE / 'analysis'))
from analyze_mirror_commonality import PKL, SUBJ, LOW, HIGH
import analyze_erd_decomposition as ED
from analyze_multichart_ceiling import cov_tn, chart_feats, grp, fit_lda
from utils.dataset import load_preprocessed_dataset, split_train_valid_test

OUT = BASE / 'results' / 'supcon_multichart'
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
def supcon_loss(z, y, tau=0.2):
    z = F.normalize(z,1); sim = z @ z.t() / tau
    B = z.shape[0]
    mask = (y[:,None]==y[None,:]).float(); mask.fill_diagonal_(0)
    sim = sim - sim.max(1, keepdim=True)[0].detach()
    exp = torch.exp(sim) * (1 - torch.eye(B, device=z.device))
    lp = sim - torch.log(exp.sum(1,keepdim=True)+1e-12)
    return -((lp*mask).sum(1)/mask.sum(1).clamp(min=1)).mean()
def train_supcon(X1, X2, y, tau=0.1, d=256, z=64, epochs=200, bs=512, lr=1e-3, seed=0):
    torch.manual_seed(seed)
    enc = Enc(X1.shape[1], d, z).to(DEV)
    opt = torch.optim.AdamW(enc.parameters(), lr=lr, weight_decay=1e-4)
    X1t = torch.tensor(X1, dtype=torch.float32); X2t = torch.tensor(X2, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long).to(DEV)
    n = len(X1t)
    for _ in range(epochs):
        enc.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i+bs]
            if len(idx) < 8: continue
            za = enc(X1t[idx].to(DEV)); zb = enc(X2t[idx].to(DEV))
            loss = supcon_loss(torch.cat([za,zb],0), torch.cat([yt[idx],yt[idx]],0), tau)
            opt.zero_grad(); loss.backward(); opt.step()
    return enc
def embed(enc, X):
    enc.eval(); out=[]
    with torch.no_grad():
        for i in range(0, len(X), 1024):
            out.append(enc(torch.tensor(X[i:i+1024], dtype=torch.float32).to(DEV)).cpu().numpy())
    return np.concatenate(out,0)

sd,_ = load_preprocessed_dataset(str(PKL))
X = np.concatenate([s[0] for s in sd],0).astype(np.float64)
y = np.concatenate([s[1] for s in sd],0)
sub = np.concatenate([np.full(len(s[1]),i,dtype=np.int64) for i,s in enumerate(sd)])
ret = split_train_valid_test(X,y,0.70,0.15,random_state=42,others=[sub])
Xtr,ytr,str_ = ret[0],ret[1],ret[6]; Xva,yva = ret[2],ret[3]; Xte,yte,ste = ret[4],ret[5],ret[8]
L = [le(cov_tn(x[:,:,TASK[0]:TASK[1]])) for x in (Xtr,Xva,Xte)]
Ltr,Lva,Lte = L
C = [cov_tn(x[:,:,TASK[0]:TASK[1]]) for x in (Xtr,Xva,Xte)]
rec = {'note':'读出=闭式LW-LDA;SVM 用 γ 网格{1,2,3};idpurity=kNN(9)','arms':{}}

def stats(name, F, svm=False):
    ok = fit_lda(F[0], ytr).predict(F[2]) == yte
    per = {SUBS[i]: float(ok[ste==i].mean()) for i in range(9)}; lo,hi = grp(per)
    pur = float((KNeighborsClassifier(n_neighbors=9).fit(F[0], str_).predict(F[0]) == str_).mean())
    line = '%-28s LDA %5.1f (low %5.1f high %5.1f) id %.3f' % (name, ok.mean()*100, lo*100, hi*100, pur)
    rec['arms'][name] = {'lda': float(ok.mean()), 'low4': lo, 'high5': hi, 'id': pur}
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

stats('LE253', (Ltr,Lva,Lte), svm=True)
rng = np.random.default_rng(0)
nz = lambda Z,s: Z + rng.normal(0, s*Z.std(0), Z.shape)
enc = train_supcon(nz(Ltr,0.05), nz(Ltr,0.05), ytr, tau=0.1, d=256, z=64, epochs=200, seed=0)
torch.save(enc.state_dict(), OUT.parent / 'contrastive_repr' / 'supcon_enc.pt')
S = [embed(enc, T) for T in (Ltr,Lva,Lte)]
stats('supcon64', S, svm=True)
# 多切图 K=4/8
for K in (4,8):
    r = np.random.default_rng(0)
    refs = [C[0][i] for i in r.choice(len(ytr), K, replace=False)]
    MC = [np.concatenate([chart_feats(c, R)[0] for R in refs], 1) for c in C]
    stats('MC%d_%d' % (K, MC[0].shape[1]), MC, svm=False)
    CAT = [np.concatenate([S[i], MC[i]], 1) for i in range(3)]
    stats('SUP+MC%d_cat' % K, CAT, svm=True)
(OUT).mkdir(parents=True, exist_ok=True)
(OUT/'supcon_multichart.json').write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
print('[OK]', flush=True)
