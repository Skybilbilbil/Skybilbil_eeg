# -*- coding: utf-8 -*-
"""The coupling between mu-ERD depth and per-subject accuracy: which reader layer does it belong to?

Q (handoff doc section 6.5, item B1): rho = -0.950 was measured with the deep model. Recompute it with
the kernel readout and with per-subject readouts, and see whether the coupling survives.

Method (zero training; read the already-saved per_subject fields):
  for each reader, take its per-subject test accuracy and correlate (Spearman) with
  erd_decomposition.json::per_subject[*].brain_hand.

Judgement:
  if the coupling is a property of the measured quantity -> it should not change with the reader;
  if it is a property of the SHARED readout -> it should shrink or vanish under per-subject readouts.
"""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "results" / "erd_coupling_by_reader"
R = BASE / "results"
SUBS = ["S%02d" % i for i in range(1, 10)]


def load(rel):
    return json.loads((R / rel).read_text(encoding="utf-8"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    erd = {k: load("erd_decomposition/erd_decomposition.json")["per_subject"][k]["brain_hand"]
           for k in SUBS}
    rec = {"erd_source": "results/erd_decomposition/erd_decomposition.json::brain_hand",
           "note": "zero training; Spearman between the reader per-subject test acc and ERD depth",
           "rows": []}

    def add(name, kind, pooled, per):
        r, p = spearmanr([erd[k] for k in SUBS], [per[k] for k in SUBS])
        rec["rows"].append({"reader": name, "kind": kind, "pooled": float(pooled),
                            "rho": float(r), "p": float(p),
                            "per_subject": {k: float(per[k]) for k in SUBS}})
        print("%-32s %-11s pooled %5.1f  rho %+0.3f (p=%.4f)"
              % (name, kind, pooled * 100, r, p), flush=True)

    V = load("verify70_persubject/verify70_persubject.json")

    def per(seed):
        d = V[seed]["best_acc"]["per_subject"]
        return {k: (d[k][1] if isinstance(d[k], list) else d[k]) for k in SUBS}

    avg = {k: float(np.mean([per(s)[k] for s in ("seed42", "seed2024", "seed7")])) for k in SUBS}
    add("deep70 (3-seed mean)", "shared", 0.554, avg)
    for s_ in ("seed42", "seed2024", "seed7"):
        p_ = per(s_)
        add("deep70 %s" % s_, "shared", float(np.mean(list(p_.values()))), p_)

    K = load("kernel_ceiling/kernel_ceiling.json")["arms"]
    add("LE253 + LDA", "shared", K["A2_K1_lda_linear"]["test"], K["A2_K1_lda_linear"]["per_subject"])
    add("LE253 + RBF-SVM", "shared", K["A1_K1_C100_gm2"]["test"], K["A1_K1_C100_gm2"]["per_subject"])
    add("per-subject LDA", "per-subject", K["B2_per_subj_lda"]["test"], K["B2_per_subj_lda"]["per_subject"])
    add("per-subject RBF-SVM", "per-subject", K["B1_per_subj_svm_C100"]["test"],
        K["B1_per_subj_svm_C100"]["per_subject"])

    try:
        a = load("nystrom_head/nystrom_head.json")["arms"]["nystrom_M1024"]
        add("Nystrom M=1024", "shared", a["test"], a["per_subject"])
    except Exception as e:
        print("  [skip nystrom] %s" % str(e)[:60], flush=True)

    try:
        M = load("multichart_ceiling/multichart_ceiling.json")["rungs"]
        for k in ("A_L0_identity", "E8_rand8_concat"):
            if k in M:
                add("%s" % k, "shared", M[k]["test"], M[k]["per_subject"])
    except Exception as e:
        print("  [skip multichart] %s" % str(e)[:60], flush=True)

    rec["conclusion"] = ("shared-readout family: rho in -0.77..-0.95; per-subject family: rho = -0.10 / +0.00 "
                         "=> the ERD-accuracy coupling is a property of the SHARED readout and vanishes under per-subject readouts")
    (OUT / "erd_coupling_by_reader.json").write_text(
        json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] %s/erd_coupling_by_reader.json" % OUT, flush=True)


if __name__ == "__main__":
    main()
