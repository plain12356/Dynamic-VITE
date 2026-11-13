"""Derives every number quoted in Sec. IV C (stronger baselines) from the result files."""
import csv, os, numpy as np
RES = os.path.join(os.path.dirname(__file__), "..", "results")
rows = list(csv.DictReader(open(os.path.join(RES, "strong_summary.csv"))))
orig = list(csv.DictReader(open(os.path.join(RES, "summary_models.csv"))))
f = lambda x: float(x) if x not in ("", "nan") else np.nan
spin, h4 = ["mfim_12", "mfim2d_3x3", "mfim2d_3x4"], ["h4_1.5", "h4_2.0"]
for t in ("0.001", "0.0001", "1e-06"):
    print(f"--- target {t}")
    for k in spin + h4:
        R = [(f(r[f"ratio_{t}"]), r["method"]) for r in rows if r["system"] == k]
        Rf = [x for x in R if np.isfinite(x[0])]
        o = next((r for r in orig if r["system"] == k and abs(float(r["tol"]) - float(t)) < 1e-12), None)
        oo = []
        if o:
            for c, lab in (("avqite_over_dyn", "AVQITE sched"), ("adapt_over_dyn", "ADAPT sched"), ("uccsd_over_dyn", "ADAPT UCCSD")):
                if np.isfinite(f(o[c])):
                    oo.append((f(o[c]), lab))
        print(f"{k:11s} strong min {min(Rf) if Rf else None}  max {max(Rf) if Rf else None} | all-baselines min {min(Rf + oo) if Rf + oo else None}")
print("--- operators to reach 1e-4 (spin, single-string pools)")
ops = [(f(r["ops_0.0001"]), r["system"], r["method"]) for r in rows if r["system"] in spin]
print(sorted(ops))
print("--- TETRIS vs ADAPT (qubit) at 1e-4: ADAPT/TETRIS")
for k in spin:
    a = next(f(r["ratio_0.0001"]) for r in rows if r["system"] == k and r["method"] == "ADAPT, qubit")
    b = next(f(r["ratio_0.0001"]) for r in rows if r["system"] == k and r["method"] == "TETRIS, qubit")
    print(k, round(a / b, 2))
for k in h4:
    a = next(f(r["ratio_0.0001"]) for r in rows if r["system"] == k and r["method"] == "ADAPT, QEB")
    b = next(f(r["ratio_0.0001"]) for r in rows if r["system"] == k and r["method"] == "TETRIS, QEB")
    print(k, "ADAPT-QEB / TETRIS-QEB", round(a / b, 2))
print("--- absolute cutoff vs relative AVQITE (same pool)")
for k in spin + h4:
    rel = next(r for r in rows if r["system"] == k and r["method"] in ("AVQITE, qubit", "AVQITE, QEB"))
    ab = next((r for r in rows if r["system"] == k and "cut" in r["method"]), None)
    if ab:
        print(k, [(t, f(rel[f"ratio_{t}"]), f(ab[f"ratio_{t}"])) for t in ("0.001", "0.0001", "1e-06")])
print("--- reached 1e-6 on tori?")
for k in ("mfim2d_3x3", "mfim2d_3x4"):
    print(k, [(r["method"], r["ratio_1e-06"], r["final_rel_err"]) for r in rows if r["system"] == k])
