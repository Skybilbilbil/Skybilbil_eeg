"""A/B 结果分析: 窄带并联前端是否把低四人单独抬起"""
import json, numpy as np
from pathlib import Path
B = Path("/mnt/d/EEG_Analysis/Riemannian_Trajectories/MAtt_baseline/results/narrowband_main")
base = json.load(open(B/"base.json", encoding="utf-8"))
iaf  = json.load(open(B/"iaf.json",  encoding="utf-8"))
LOW  = ["S02","S04","S05","S06"]
HIGH = ["S01","S03","S07","S08","S09"]
seeds = sorted(base["runs"].keys(), key=int)
print("=== 每 seed 总体测试精度 (最后迭代 / 峰值) ===")
rows = {}
for tag, d in [("base", base), ("iaf", iaf)]:
    for sd in seeds:
        r = d["runs"][sd]
        rows[(tag, sd)] = (r["test_acc"][-1], max(r["test_acc"]), r["secs"], r.get("n_params"))
        print(f"  {tag:5s} seed{sd:5s} final={r['test_acc'][-1]*100:5.2f}%  peak={max(r['test_acc'])*100:5.2f}%  {r['secs']}s params={r.get('n_params')}")
print()
print("=== 逐被试测试精度 (最后迭代) ===")
hdr = "seed  " + "".join(f"{s:>7s}" for s in ["S01","S02","S03","S04","S05","S06","S07","S08","S09"]) + "   LOW均值  HIGH均值  全组"
for tag, d in [("base", base), ("iaf", iaf)]:
    print(f"--- {tag} ---")
    print(hdr)
    for sd in seeds:
        per = d["runs"][sd]["per_subject"]
        v = [per[s][1] for s in ["S01","S02","S03","S04","S05","S06","S07","S08","S09"]]
        lo = np.mean([per[s][1] for s in LOW]); hi = np.mean([per[s][1] for s in HIGH])
        print(f"{sd:5s} " + "".join(f"{x*100:7.2f}" for x in v) + f"  {lo*100:7.2f} {hi*100:8.2f} {np.mean(v)*100:6.2f}")
print()
print("=== 配对差 iaf - base (pp), 按 seed x 被试 ===")
dl, dh, da = [], [], []
per_seed = {}
for sd in seeds:
    pb = base["runs"][sd]["per_subject"]; pi = iaf["runs"][sd]["per_subject"]
    d = {s: 100*(pi[s][1]-pb[s][1]) for s in pi}
    per_seed[sd] = d
    lo = np.mean([d[s] for s in LOW]); hi = np.mean([d[s] for s in HIGH])
    dl.append(lo); dh.append(hi); da.append(np.mean(list(d.values())))
    print(f"  seed{sd:5s} 低四人={lo:+6.2f}pp  高五人={hi:+6.2f}pp  全组={np.mean(list(d.values())):+6.2f}pp   逐人: " + " ".join(f"{s}{d[s]:+5.1f}" for s in ["S01","S02","S03","S04","S05","S06","S07","S08","S09"]))
print()
print("=== 汇总 (3 seed) ===")
allL = [per_seed[sd][s] for sd in seeds for s in LOW]
allH = [per_seed[sd][s] for sd in seeds for s in HIGH]
print(f"  低四人 iaf-base: 均值={np.mean(allL):+6.2f}pp  SD={np.std(allL,ddof=1):5.2f}  n={len(allL)}  SE={np.std(allL,ddof=1)/np.sqrt(len(allL)):.2f}  t={np.mean(allL)/(np.std(allL,ddof=1)/np.sqrt(len(allL))):+.2f}")
print(f"  高五人 iaf-base: 均值={np.mean(allH):+6.2f}pp  SD={np.std(allH,ddof=1):5.2f}  n={len(allH)}  SE={np.std(allH,ddof=1)/np.sqrt(len(allH)):.2f}  t={np.mean(allH)/(np.std(allH,ddof=1)/np.sqrt(len(allH))):+.2f}")
print(f"  低-高 差: {np.mean(allL)-np.mean(allH):+.2f}pp  (判据: 低四人应显著大于高五人)")
print(f"  总体 base={np.mean([base['runs'][s]['test_acc'][-1] for s in seeds])*100:.2f}%  iaf={np.mean([iaf['runs'][s]['test_acc'][-1] for s in seeds])*100:.2f}%")