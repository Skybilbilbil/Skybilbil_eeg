# -*- coding: utf-8 -*-
"""固定 conv2 梯子: 极小预算跑 (硬性总时长上限 300s)"""
import subprocess, time, sys, os
from pathlib import Path
BASE = Path(__file__).resolve().parents[1]   # = MAtt_baseline/
ARMS = ["base", "fixed12", "fixed24", "fixed96"]
SEEDS = "42,2024"
ITERS = 25
BUDGET_S = 300
t0 = time.time()
for arm in ARMS:
    el = time.time() - t0
    if el > BUDGET_S:
        print(f"[超预算 {el:.0f}s] 跳过 {arm}", flush=True); continue
    cmd = f"cd {BASE} && source ~/miniconda3/etc/profile.d/conda.sh && conda activate eeg_analysis && " \
          f"timeout {int(BUDGET_S-el)} python analysis/exp_fixed_conv2.py --arm {arm} " \
          f"--seeds {SEEDS} --iters {ITERS} --tag quick"
    print(f"[{el:5.0f}s] 开始 {arm}", flush=True)
    r = subprocess.run(["bash","-lc",cmd], capture_output=True, text=True)
    tail = [l for l in r.stdout.split(chr(10)) if ("完成" in l or "落盘" in l)]
    for l in tail: print("   " + l, flush=True)
print(f"总时长 {time.time()-t0:.0f}s", flush=True)