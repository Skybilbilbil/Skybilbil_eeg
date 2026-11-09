"""六臂统一分析: 每种机制对低四人/高五人的配对效应"""
import json, numpy as np
from pathlib import Path
B = Path(__file__).resolve().parents[1] / "results"   # = MAtt_baseline/results/
LOW  = ["S02","S04","S05","S06"]
HIGH = ["S01","S03","S07","S08","S09"]
ALL  = HIGH[:1] + LOW + HIGH[1:]
ARMS = {
  "base":      (B/"individual_variants_main/base.json",       "base"),
  "B_random":  (B/"individual_variants_main/B_random.json",   "B_random"),
  "A_persubj": (B/"individual_variants_main/A_persubj.json",  "A_persubj"),
  "C_pool":    (B/"individual_variants_main/C_pool.json",     "C_pool"),
  "iaf":       (B/"narrowband_main/iaf.json",                 "iaf"),
}
base_path = B/"individual_variants_main/base.json"
if not base_path.exists():
    base_path = B/"narrowband_main/base.json"
D = {}
D["base"] = json.load(open(base_path, encoding="utf-8"))
for k,(p,_) in ARMS.items():
    if k=="base": continue
    if p.exists(): D[k] = json.load(open(p, encoding="utf-8"))
seeds = sorted(D["base"]["runs"].keys(), key=int)
print("=== 每臂总体 test（最后迭代，3 seed） ===")
print("臂            " + " ".join(f"{s:>8s}" for s in seeds) + "      均值    峰值均值")
for k,d in D.items():
    f = [d["runs"][s]["test_acc"][-1] for s in seeds]
    pk = [max(d["runs"][s]["test_acc"]) for s in seeds]
    print(f"{k:13s} " + " ".join(f"{x*100:8.2f}" for x in f) + f"  {np.mean(f)*100:8.2f}  {np.mean(pk)*100:8.2f}")
print()
print("=== 相对 base 的配对差（按 seed×被试，pp） ===")
print("臂           低四人均值   SD   t      高五人均值   SD   t    低-高")
base = D["base"]
for k,d in D.items():
    if k=="base": continue
    dl, dh = [], []
    for s in seeds:
        pb = base["runs"][s]["per_subject"]; pv = d["runs"][s]["per_subject"]
        for sub in LOW:  dl.append(100*(pv[sub][1]-pb[sub][1]))
        for sub in HIGH: dh.append(100*(pv[sub][1]-pb[sub][1]))
    dl, dh = np.array(dl), np.array(dh)
    tl = dl.mean()/(dl.std(ddof=1)/np.sqrt(len(dl))) if dl.std(ddof=1)>0 else 0
    th = dh.mean()/(dh.std(ddof=1)/np.sqrt(len(dh))) if dh.std(ddof=1)>0 else 0
    print(f"{k:13s} {dl.mean():+8.2f}  {dl.std(ddof=1):5.2f} {tl:+6.2f}   {dh.mean():+8.2f}  {dh.std(ddof=1):5.2f} {th:+6.2f}  {dl.mean()-dh.mean():+6.2f}")
print()
print("=== 逐被试（各臂 3-seed 均值） ===")
print("臂            " + "".join(f"{s:>7s}" for s in ["S01","S02","S03","S04","S05","S06","S07","S08","S09"]))
for k,d in D.items():
    v = [np.mean([d["runs"][s]["per_subject"][sub][1] for s in seeds])*100 for sub in ["S01","S02","S03","S04","S05","S06","S07","S08","S09"]]
    print(f"{k:13s} " + "".join(f"{x:7.2f}" for x in v))
print()
print("判据: 低-高 明显为正 = 低四人被单独抬起；接近 0 = 同步平移（判死）；<5pp 算噪声")