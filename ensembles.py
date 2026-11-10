"""Access to the seeded ADAPT-type ensembles (results/adapt_ensembles.json).

Tables use the median, over seeds, of the circuits needed to reach each target; figures use the
representative run, the seed whose cost to reach 1e-4 is the median (lower median for even counts).

    python scripts/ensembles.py     # results/ensembles.csv, results/table_ensembles.tex
"""
from __future__ import annotations
import csv, json, os
from functools import lru_cache
import numpy as np

RES = os.path.join(os.path.dirname(__file__), "..", "results")
TOLS = (1e-3, 1e-4, 1e-6)


def _rel(r):
    return np.maximum((np.array(r["energy"]) - r["E0"]) / abs(r["E0"]), 1e-16)


def _first(r, tol, key="circuits"):
    h = np.nonzero(_rel(r) < tol)[0]
    return float(r[key][h[0]]) if h.size else np.nan


@lru_cache(maxsize=None)
def _load():
    p = os.path.join(RES, "adapt_ensembles.json")
    return json.load(open(p)) if os.path.exists(p) else []


def _key(params):
    return json.dumps(params, sort_keys=True)


def runs(system, kind, params):
    return [r for r in _load() if r["system"] == system and r["kind"] == kind and _key(r["params"]) == _key(params)]


def has(system, kind, params):
    return len(runs(system, kind, params)) > 0


def costs(system, kind, params, tol, key="circuits"):
    return [_first(r, tol, key) for r in runs(system, kind, params)]


def median_cost(system, kind, params, tol, key="circuits"):
    c = [x for x in costs(system, kind, params, tol, key) if np.isfinite(x)]
    rs = runs(system, kind, params)
    if not rs or len(c) < len(rs) / 2:          # target missed by most seeds
        return np.nan
    return float(np.median(c))


def cost_range(system, kind, params, tol):
    c = [x for x in costs(system, kind, params, tol) if np.isfinite(x)]
    return (min(c), max(c)) if c else (np.nan, np.nan)


def representative(system, kind, params, tol=1e-4):
    rs = runs(system, kind, params)
    if not rs:
        return None
    c = [(_first(r, tol), r["seed"], r) for r in rs]
    c = sorted([x for x in c if np.isfinite(x[0])], key=lambda x: (x[0], x[1]))
    return c[(len(c) - 1) // 2][2] if c else rs[0]


ORDER = ["mfim_12", "mfim2d_3x3", "mfim2d_3x4", "h4_1.5", "h4_2.0", "tfim1d_12", "tfim2d_3x3", "tfim2d_3x4", "hubbard_4"]
LABELS = {"mfim_12": "Mixed-field 1D, $N=12$", "mfim2d_3x3": "Mixed-field 2D, $3\\times3$",
          "mfim2d_3x4": "Mixed-field 2D, $3\\times4$", "h4_1.5": "H$_4$, 1.5 \\AA", "h4_2.0": "H$_4$, 2.0 \\AA",
          "tfim1d_12": "TFIM 1D, $N=12$", "tfim2d_3x3": "TFIM 2D, $3\\times3$", "tfim2d_3x4": "TFIM 2D, $3\\times4$",
          "hubbard_4": "Hubbard, $L=4$"}
CONFIGS = [("adapt", {}, "ADAPT-VQE, schedule"), ("adapt", {"pool": "uccsd", "max_ops": 40}, "ADAPT-VQE, UCCSD"),
           ("adapt", {"pool": "qubit"}, "ADAPT-VQE, qubit"), ("adapt", {"pool": "qubit", "batch": "tetris"}, "TETRIS, qubit"),
           ("adapt", {"pool": "qeb"}, "ADAPT-VQE, QEB"), ("adapt", {"pool": "qeb", "batch": "tetris"}, "TETRIS, QEB"),
           ("adapt", {"pool": "uccsd", "batch": "tetris"}, "TETRIS, UCCSD")]


def main():
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from make_model_figures import systems, first
    S = systems()
    rows, tex = [], []
    for g in ORDER:
        if g not in S:
            continue
        dyn = S[g]["dyn"]
        for kind, p, mlab in CONFIGS:
            if not has(g, kind, p):
                continue
            n = len(runs(g, kind, p))
            row = [g, mlab, n]
            for t in TOLS:
                c = first(dyn, t)
                lo, hi = cost_range(g, kind, p, t)
                row += [median_cost(g, kind, p, t) / c, lo / c, hi / c]
            rows.append(row)
    with open(os.path.join(RES, "ensembles.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "method", "seeds"] + [f"{s}_{t:g}" for t in TOLS for s in ("median", "min", "max")])
        w.writerows(rows)
    fmt = lambda x: "--" if not np.isfinite(x) else (f"{x:.2f}" if x < 10 else f"{x:.0f}")
    tex += ["\\begin{tabular}{llcccc}",
            "System & Method & seeds & $10^{-3}$ & $10^{-4}$ & $10^{-6}$ \\\\", "\\hline"]
    prev = None
    for r in rows:
        if prev is not None and r[0] != prev:
            tex.append("\\hline")
        lab = LABELS[r[0]] if r[0] != prev else ""
        cells = ["--" if not np.isfinite(r[3 + 3 * i]) else f"{fmt(r[3 + 3 * i])} [{fmt(r[4 + 3 * i])}, {fmt(r[5 + 3 * i])}]"
                 for i in range(3)]
        tex.append(f"{lab} & {r[1]} & {r[2]} & " + " & ".join(cells) + " \\\\")
        prev = r[0]
    tex += ["\\end{tabular}"]
    open(os.path.join(RES, "table_ensembles.tex"), "w").write("\n".join(tex) + "\n")
    for r in rows:
        print(f"{r[0]} {r[1]:22s} n={r[2]} " + "  ".join(f"{t:g}: {fmt(r[3+3*i])} [{fmt(r[4+3*i])},{fmt(r[5+3*i])}]"
                                                      for i, t in enumerate(TOLS)))


if __name__ == "__main__":
    main()
