"""Print the key numbers for each new-model benchmark (results/models_*.json)."""
import json, os, sys, glob
import numpy as np

TOLS = (1e-3, 1e-4, 1e-6)
CHEM = 1.6e-3      # Hartree


def rel(r):
    return np.maximum((np.array(r["energy"]) - r["E0"]) / abs(r["E0"]), 1e-16)


def first(r, tol, key="circuits", absolute=False):
    e = np.array(r["energy"]) - r["E0"] if absolute else rel(r)
    h = np.nonzero(e < tol)[0]
    return r[key][h[0]] if h.size else np.nan


def label(r):
    k, p = r["kind"], r["params"]
    return {"dynamic": f"dyn eps={p.get('eps')}", "static": f"static L={p.get('L')}", "avqite": "AVQITE",
            "adapt": "ADAPT(" + p.get("pool", "sched") + ")"}[k]


if __name__ == "__main__":
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(os.path.dirname(__file__), "..", "results", "models_*.json")))
    for f in files:
        R = json.load(open(f))
        S = R[0]
        print(f"== {S['system']}  n={S['n']}  E0={S['E0']:.6f}  group={S.get('group_size')}  K={S.get('sched_len')}")
        mol = S["system"].startswith("H4")
        for r in sorted(R, key=lambda r: (r["kind"] != "dynamic", r["kind"], r["params"].get("L", 0))):
            L = r["n_layers"][-1] if r["kind"] != "adapt" else len(r["layers"])
            cols = [f"{label(r):16s}", f"final={rel(r)[-1]:.1e}", f"L={L:3d}", f"F={r['fidelity']:.4f}"]
            for t in TOLS:
                c, l = first(r, t), first(r, t, "n_layers")
                cols.append(f"{t:.0e}:" + (f"{c:.2g}/L{int(l)}" if np.isfinite(c) else "--"))
            if mol:
                c = first(r, CHEM, absolute=True)
                cols.append("chem:" + (f"{c:.2g}" if np.isfinite(c) else "--"))
            cols.append(f"wall={r['wall']:.0f}s")
            print("  " + " ".join(cols))
