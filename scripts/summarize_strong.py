"""Table of the stronger baselines relative to dynamic VITE (eps = 0.9).

    python scripts/summarize_strong.py   # results/table_strong.tex, results/strong_summary.csv
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from make_model_figures import systems, first
import ensembles as ens

RES = os.path.join(os.path.dirname(__file__), "..", "results")
ORDER = ["mfim_12", "mfim2d_3x3", "mfim2d_3x4", "h4_1.5", "h4_2.0"]
TOLS = (1e-3, 1e-4, 1e-6)
COLS = [("adapt", "qubit", None, "ADAPT, qubit"), ("adapt", "qubit", "tetris", "TETRIS, qubit"),
        ("avqite", "qubit", None, "AVQITE, qubit"),
        ("adapt", "qeb", None, "ADAPT, QEB"), ("adapt", "qeb", "tetris", "TETRIS, QEB"),
        ("adapt", "uccsd", "tetris", "TETRIS, UCCSD"), ("avqite", "qeb", None, "AVQITE, QEB")]


def main():
    S = systems()
    B = json.load(open(os.path.join(RES, "strong_baselines.json")))
    A = []
    for f in ("strong_baselines_abs.json", "strong_baselines_abs_spin.json"):
        if os.path.exists(os.path.join(RES, f)):
            A += json.load(open(os.path.join(RES, f)))
    rows, tex = [], []
    for k in ORDER:
        if k not in S:
            continue
        dyn = S[k]["dyn"]
        runs = [r for r in B if r["system"] == k]
        for kind, pool, batch, lab in COLS:
            r = next((r for r in runs if r["kind"] == kind and r["params"].get("pool") == pool
                      and r["params"].get("batch") == batch), None)
            if r is None:
                continue
            psize = r["pool_size"]
            if kind == "adapt" and ens.has(k, kind, r["params"]):        # seeded ensemble: medians
                ratios = [ens.median_cost(k, kind, r["params"], t) / first(dyn, t) for t in TOLS]
                ops = [ens.median_cost(k, kind, r["params"], t, key="n_layers") for t in TOLS]
                r = ens.representative(k, kind, r["params"])
            else:
                ratios = [first(r, t) / first(dyn, t) for t in TOLS]
                ops = [first(r, t, "n_layers") for t in TOLS]
            fin = max((r["energy"][-1] - r["E0"]) / abs(r["E0"]), 1e-16)
            rows.append([k, lab, psize, *ratios, *ops, fin, len(r["layers"])])
        # AVQITE with the original absolute cutoff: best cutoff in hindsight, per target
        ab = [r for r in A if r["system"] == k]
        if ab:
            pool = ab[0]["params"]["pool"]
            ratios, ops, cuts = [], [], []
            for t in TOLS:
                c = [(first(r, t) / first(dyn, t), first(r, t, "n_layers"), r["params"]["abs_cut"]) for r in ab]
                c = [x for x in c if np.isfinite(x[0])]
                b = min(c) if c else (np.nan, np.nan, None)
                ratios.append(b[0]); ops.append(b[1]); cuts.append(b[2])
            fin = min(max((r["energy"][-1] - r["E0"]) / abs(r["E0"]), 1e-16) for r in ab)
            lab = "AVQITE, " + ("QEB" if pool == "qeb" else "qubit") + ", $L^2_{\\rm cut}$"
            rows.append([k, lab, ab[0]["pool_size"], *ratios, *ops, fin, len(ab)])
            print(k, "abs cuts used per target:", cuts, "of", sorted(r["params"]["abs_cut"] for r in ab))
    with open(os.path.join(RES, "strong_summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "method", "pool_size"] + [f"ratio_{t:g}" for t in TOLS] +
                   [f"ops_{t:g}" for t in TOLS] + ["final_rel_err", "final_ops"])
        w.writerows(rows)
    labels = {"mfim_12": "Mixed-field 1D, $N=12$", "mfim2d_3x3": "Mixed-field 2D, $3\\times3$",
              "mfim2d_3x4": "Mixed-field 2D, $3\\times4$", "h4_1.5": "H$_4$, 1.5 \\AA", "h4_2.0": "H$_4$, 2.0 \\AA"}
    fmt = lambda x: "--" if not np.isfinite(x) else (f"{x:.2f}" if x < 10 else f"{x:.0f}")
    fo = lambda x: "--" if not np.isfinite(x) else f"{int(x)}"
    tex += ["\\begin{tabular}{llccccccc}", "\\hline\\hline",
            "System & Method & pool & \\multicolumn{3}{c}{circuits / dynamic VITE} & \\multicolumn{3}{c}{operators} \\\\",
            " & & & $10^{-3}$ & $10^{-4}$ & $10^{-6}$ & $10^{-3}$ & $10^{-4}$ & $10^{-6}$ \\\\", "\\hline"]
    prev = None
    for row in rows:
        k, lab, ps = row[:3]
        if prev is not None and k != prev:
            tex.append("\\hline")
        tex.append(f"{labels[k] if k != prev else ''} & {lab} & {ps} & " + " & ".join(fmt(x) for x in row[3:6]) +
                   " & " + " & ".join(fo(x) for x in row[6:9]) + " \\\\")
        prev = k
    tex += ["\\hline\\hline", "\\end{tabular}"]
    open(os.path.join(RES, "table_strong.tex"), "w").write("\n".join(tex) + "\n")
    for row in rows:
        print(f"{row[0]:11s} {row[1]:14s} pool {row[2]:4d} | ratios " + " ".join(fmt(x).rjust(6) for x in row[3:6]) +
              " | ops " + " ".join(fo(x).rjust(4) for x in row[6:9]) + f" | final {row[9]:.1e} ({row[10]} ops)")


if __name__ == "__main__":
    main()
