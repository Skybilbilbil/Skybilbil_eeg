# -*- coding: utf-8 -*-
"""BCI 官方口径（跨 session 主协议）：9 支深度族 + 经典，权重在 val 上选。

已有预测（results/mixopt_*/pred_*.npz，均为官方 BCIC_S0X + EA、跨 session）：
  d12 × {42,2024,7}、d14 × {42,2024,7}、d18pred × {42,2024,7}  ⇒ 9 支
论文 §5.2 的 6 支（d12+d18）为 64.93；本脚本检验 9 支（加 d14）能否再降方差。

产物：results/bci_official_family9/bci_official_family9.json
"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / 'MAtt_baseline' / 'results'
OUT = RES / 'bci_official_family9'
D12 = [(RES / 'mixopt_d12' / f'pred_{s}.npz') for s in (42, 2024, 7)]
D14 = [(RES / 'mixopt_d14' / f'pred_{s}.npz') for s in (42, 2024, 7)]
D18 = [(RES / 'mixopt_d18pred' / f'pred_{s}.npz') for s in (42, 2024, 7)]
LOW, HIGH = [1, 3, 4, 5], [0, 2, 6, 7, 8]


def sc(ok, sb):
    per = {i: float(ok[sb == i].mean()) for i in range(9)}
    return (float(ok.mean()),
            float(np.mean([per[i] for i in LOW])),
            float(np.mean([per[i] for i in HIGH])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d0 = np.load(D12[0])
    yte, sb, yva = d0['yte'], d0['sb'], d0['yva']
    rec = {'n_test': int(len(yte)), 'n_va': int(len(yva)), 'sets': {}}

    def run(tag, paths):
        ds = [np.load(p) for p in paths if p.exists()]
        p_deep = np.mean([d['p_deep'] for d in ds], 0)
        p_cls = np.mean([d['p_cls'] for d in ds], 0)
        # ⚠ 各 seed 的验证折索引不同（rng=default_rng(seed)），只能把**同一 seed** 的几支平均
        pairs = [(d, p) for d, p in zip(ds, paths) if p.exists()]
        va42 = [d for d, p in pairs if p.name == 'pred_42.npz']
        fam_va = np.mean([d['p_d_va'] for d in va42], 0)
        cls_va = va42[0]['p_c_va']
        a = sc(p_deep.argmax(1) == yte, sb)
        # 融合权重在 val 上选
        bw, bv = 0.5, -1.0
        for w in np.arange(0, 1.0001, 0.02):
            v = float(((w * fam_va + (1 - w) * cls_va).argmax(1) == yva).mean())
            if v > bv:
                bv, bw = v, float(w)
        p = bw * p_deep + (1 - bw) * p_cls
        f = sc(p.argmax(1) == yte, sb)
        rec['sets'][tag] = {'n': len(ds), 'deep_family': a, 'w_deep': bw, 'val': bv,
                            'fused': f}
        print(f'[{tag:6s} n={len(ds)}] 深度族 {a[0]*100:5.2f} | w_deep={bw:.2f} | '
              f'val {bv*100:5.2f} | 融合 {f[0]*100:5.2f} '
              f'(低四 {f[1]*100:.1f} / 高五 {f[2]*100:.1f})', flush=True)

    run('d12', D12)
    run('d12+d18', D12 + D18)
    run('d12+d14', D12 + D14)
    run('all9', D12 + D14 + D18)

    (OUT / 'bci_official_family9.json').write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'[OK] {OUT}/bci_official_family9.json', flush=True)


if __name__ == '__main__':
    main()
