"""Size scaling on the 1D mixed-field Ising ring: cost of the stronger baselines relative to dynamic VITE.

    python scripts/make_scaling_figure.py   # paper/figures/fig_scaling.pdf, results/scaling.csv
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
METHODS = [("avqite", None, "AVQITE (qubit pool)", "#CC79A7", "o"),
           ("adapt", "tetris", "TETRIS-ADAPT-VQE (qubit pool)", "#D55E00", "s"),
           ("adapt", None, "ADAPT-VQE (qubit pool)", "#8B4513", "^")]
TOLS = (1e-3, 1e-4, 1e-6)


def rel(r):
    return np.maximum((np.array(r["energy"]) - r["E0"]) / abs(r["E0"]), 1e-16)


def first(r, tol, key="circuits"):
    if r is None:
        return np.nan
    h = np.nonzero(rel(r) < tol)[0]
    return float(r[key][h[0]]) if h.size else np.nan


def bcost(n, d, kind, batch, tol, key="circuits"):
    """Baseline cost: median of the seeded ensemble where one exists (scripts/ensembles.py), else the single run."""
    import ensembles as ens
    params = {"pool": "qubit", "batch": batch} if batch else {"pool": "qubit"}
    if kind == "adapt" and ens.has(f"mfim_{n}", "adapt", params):
        return ens.median_cost(f"mfim_{n}", "adapt", params, tol, key=key)
    return first(d.get((kind, batch)), tol, key)


def load():
    """N -> {'dynamic': run, ('avqite', None): run, ...}"""
    out = {}
    for r in json.load(open(os.path.join(RES, "mfim_eps.json"))):
        if r["params"]["eps"] == 0.9:
            out.setdefault(r["n"], {})["dynamic"] = r
    runs = json.load(open(os.path.join(RES, "scaling.json"))) if os.path.exists(os.path.join(RES, "scaling.json")) else []
    runs += [d for d in json.load(open(os.path.join(RES, "strong_baselines.json"))) if d["system"] == "mfim_12"]
    for d in runs:
        n = int(d["system"].split("_")[1])
        if d["kind"] == "dynamic":
            out.setdefault(n, {})["dynamic"] = d
        elif d["params"].get("pool") == "qubit":
            out.setdefault(n, {})[(d["kind"], d["params"].get("batch"))] = d
    return dict(sorted(out.items()))


def main():
    D = load()
    rows = []
    for n, d in D.items():
        dyn = d.get("dynamic")
        for kind, batch, lab, _, _ in METHODS:
            r = d.get((kind, batch))
            if dyn is None or r is None:
                continue
            rows.append([n, lab] + [bcost(n, d, kind, batch, t) / first(dyn, t) for t in TOLS] +
                        [first(dyn, t, "n_layers") for t in TOLS] + [bcost(n, d, kind, batch, t, "n_layers") for t in TOLS] +
                        [first(dyn, t) for t in TOLS])
    with open(os.path.join(RES, "scaling.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "method"] + [f"ratio_{t:g}" for t in TOLS] + [f"dyn_layers_{t:g}" for t in TOLS] +
                   [f"ops_{t:g}" for t in TOLS] + [f"dyn_circuits_{t:g}" for t in TOLS])
        w.writerows(rows)
    from symmetry_cost import reduced_cost
    fig, A = plt.subplots(2, 2, figsize=(3.4, 3.4), gridspec_kw=dict(wspace=0.42, hspace=0.55))
    axs = A[0]
    for ax, tol in zip(axs, (1e-4, 1e-6)):
        for kind, batch, lab, col, mk in METHODS:
            pts = [(n, bcost(n, d, kind, batch, tol) / first(d.get("dynamic"), tol)) for n, d in D.items()
                   if d.get("dynamic") is not None and d.get((kind, batch)) is not None]
            pts = [(n, y) for n, y in pts if np.isfinite(y)]
            if pts:
                ax.plot(*zip(*pts), marker=mk, ms=3.5, color=col, label=lab)
        # dynamic VITE with symmetry-reduced M and V, relative to its unreduced cost
        Nd = [n for n, d in D.items() if d.get("dynamic") is not None]
        red = []
        for n in Nd:
            r = D[n]["dynamic"]
            k = int(np.nonzero(rel(r) < tol)[0][0])
            red.append(reduced_cost(r, upto=k)[k] / first(r, tol))
        ax.plot(Nd, red, marker="D", ms=3.2, color="#0072B2", ls="--", mfc="white")
        ax.axhline(1, color="#0072B2", lw=1.2)
        ax.text(0.97, 0.88, "dynamic VITE", transform=ax.get_yaxis_transform(), ha="right", va="top",
                fontsize=6.2, color="#0072B2")
        ax.set_yscale("log")
        ax.set_xlabel("$N$")
        ax.set_xticks(sorted(D))
        ax.set_title(f"target $10^{{{int(np.log10(tol))}}}$", fontsize=7.5, pad=3)
        ax.grid(alpha=0.25, lw=0.4)
    axs[0].set_ylabel("circuits / dynamic VITE")
    lo = min(a.get_ylim()[0] for a in axs); hi = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(min(lo, 0.8), hi)
    # absolute circuits to reach the target, including dynamic VITE with symmetry-reduced M and V
    absrows = []
    for ax, tol in zip(A[1], (1e-4, 1e-6)):
        Ns = [n for n, d in D.items() if d.get("dynamic") is not None]
        dyn = [first(D[n]["dynamic"], tol) for n in Ns]
        red = []
        for n in Ns:
            r = D[n]["dynamic"]
            k = int(np.nonzero(rel(r) < tol)[0][0])
            red.append(reduced_cost(r, upto=k)[k])
        ax.plot(Ns, dyn, marker="D", ms=3.2, color="#0072B2", label="dynamic VITE")
        ax.plot(Ns, red, marker="D", ms=3.2, color="#0072B2", ls="--", mfc="white",
                label="dynamic VITE, symmetry-reduced")
        for kind, batch, lab, col, mk in METHODS:
            pts = [(n, bcost(n, D[n], kind, batch, tol)) for n in Ns if D[n].get((kind, batch)) is not None]
            pts = [(n, y) for n, y in pts if np.isfinite(y)]
            ax.plot(*zip(*pts), marker=mk, ms=3.2, color=col)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns]); ax.minorticks_off()
        ax.set_xlabel("$N$")
        ax.grid(alpha=0.25, lw=0.4)
        fit = lambda y: np.polyfit(np.log(Ns), np.log(y), 1)[0]
        absrows.append((tol, Ns, dyn, red, fit(dyn), fit(red)))
    A[1][0].set_ylabel("circuits to target")
    h, l = axs[0].get_legend_handles_labels()
    h2, l2 = A[1][0].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.965), fontsize=5.8)
    for lab, ax, tol in zip("abcd", A.flat, (1e-4, 1e-6, 1e-4, 1e-6)):
        ax.set_title(f"({lab}) target $10^{{{int(np.log10(tol))}}}$", fontsize=7.5, pad=3)
    fig.savefig(os.path.join(FIG, "fig_scaling.pdf"))
    plt.close(fig)
    for tol, Ns, dyn, red, a, b in absrows:
        print(f"target {tol:g}: dyn {['%.2g' % x for x in dyn]} slope {a:.2f} | symmetry-reduced "
              f"{['%.2g' % x for x in red]} slope {b:.2f} | reduction {['%.0f' % (x / y) for x, y in zip(dyn, red)]}")
        with open(os.path.join(RES, f"scaling_symmetry_{tol:g}.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["N", "dyn_circuits", "dyn_symmetry_reduced"])
            w.writerows(zip(Ns, dyn, red))
    for r in rows:
        print(f"N={r[0]:2d} {r[1]:32s} ratios " + " ".join(f"{x:6.3g}" for x in r[2:5]) +
              " | dyn L " + " ".join(f"{x:3.0f}" for x in r[5:8]) + " | ops " + " ".join(f"{x:4.0f}" for x in r[8:11]))


if __name__ == "__main__":
    main()
