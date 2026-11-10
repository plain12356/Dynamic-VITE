"""Cross-model figures and summary table (1D, 2D, fermionic).

    python scripts/make_model_figures.py   # paper/figures/fig_models.pdf, fig_depth.pdf, results/table_summary.tex
"""
from __future__ import annotations
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys as _sys
_sys.path.insert(0, os.path.dirname(__file__))
import ensembles as ens

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES, FIG = os.path.join(ROOT, "results"), os.path.join(ROOT, "paper", "figures")
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 6.2,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2, "axes.linewidth": 0.6,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42})
COL = {"dyn": "#0072B2", "static": "0.55", "avqite": "#CC79A7", "adapt": "#D55E00", "ucc": "#8B4513",
       "tetris_fine": "#009E73", "avqite_fine": "#CC79A7"}
TOLS = (1e-3, 1e-4, 1e-5, 1e-6)


def load(f):
    p = os.path.join(RES, f)
    return json.load(open(p)) if os.path.exists(p) else []


def rel(r):
    return np.maximum((np.array(r["energy"]) - r["E0"]) / abs(r["E0"]), 1e-16)


def first(r, tol, key="circuits"):
    if r is None:
        return np.nan
    e = rel(r)
    h = np.nonzero(e < tol)[0]
    return float(r[key][h[0]]) if h.size else np.nan


def systems():
    """name -> dict(label, n, group, dyn, dyn01, static[list], avqite, adapt, ucc)."""
    out = {}
    # 1D TFIM, N = 12 (critical ring): static scan = depths N/2-1, N/2
    main, bench, eps = load("main.json"), load("bench.json"), load("eps.json")
    if main:
        n = 12
        g = lambda m: next((r for r in bench if r["n"] == n and r["method"] == m), None)
        st = [r for r in bench if r["n"] == n and r["method"].startswith("Static")]
        for r in st:
            r["params"] = {"L": r["n_layers"][-1]}
        out["tfim1d_12"] = dict(label="TFIM 1D, $N=12$", cls="1D", n=n, group=2 * n,
                                dyn=next(r for r in main if r["n"] == n),
                                dyn01=next((r for r in eps if r["params"]["eps"] == 0.1), None),
                                static=st, avqite=g("AVQITE"), adapt=g("ADAPT-VQE"), ucc=None)
    me, ms, ma, mad = load("mfim_eps.json"), load("mfim_static.json"), load("mfim_avqite.json"), load("mfim_adapt.json")
    if me:
        n = 12
        out["mfim_12"] = dict(label="Mixed-field Ising 1D, $N=12$", cls="1D", n=n, group=2 * n,
                              dyn=next(r for r in me if r["n"] == n and r["params"]["eps"] == 0.9),
                              dyn01=next(r for r in me if r["n"] == n and r["params"]["eps"] == 0.1),
                              static=[r for r in ms if r["n"] == n], avqite=next((r for r in ma if r["n"] == n), None),
                              adapt=next((r for r in mad if r["n"] == n), None), ucc=None)
    labels = {"tfim2d_3x3": ("TFIM 2D, $3\\times3$", "2D"), "tfim2d_3x4": ("TFIM 2D, $3\\times4$", "2D"),
              "mfim2d_3x3": ("Mixed-field Ising 2D, $3\\times3$", "2D"),
              "mfim2d_3x4": ("Mixed-field Ising 2D, $3\\times4$", "2D"),
              "hubbard_4": ("Hubbard, $L=4$", "fermionic"), "hubbard_6": ("Hubbard, $L=6$", "fermionic"),
              "h4_1.5": ("H$_4$, 1.5 \\AA", "fermionic"), "h4_2.0": ("H$_4$, 2.0 \\AA", "fermionic")}
    for name, (lab, cls) in labels.items():
        R = load(f"models_{name}.json") + load(f"models_{name}_static.json") + load(f"models_{name}_adapt40.json")
        if not R:
            continue
        pick = lambda k, **p: next((r for r in R if r["kind"] == k and all(r["params"].get(a) == b for a, b in p.items())), None)
        out[name] = dict(label=lab, cls=cls, n=R[0]["n"], group=R[0].get("group_size"),
                         dyn=pick("dynamic", eps=0.9), dyn01=pick("dynamic", eps=0.1),
                         static=[r for r in R if r["kind"] == "static"], avqite=pick("avqite"),
                         adapt=next((r for r in R if r["kind"] == "adapt" and "pool" not in r["params"]), None),
                         ucc=pick("adapt", pool="uccsd"))

    # ADAPT-type costs are sensitive to rounding-level perturbations: where a seeded ensemble exists,
    # tables use its median and figures its representative run (scripts/ensembles.py)
    for name, d in out.items():
        for key, params in (("adapt", {}), ("ucc", {"pool": "uccsd", "max_ops": 40})):
            if ens.has(name, "adapt", params):
                d[key] = ens.representative(name, "adapt", params)
                d[key + "_ens"] = (name, "adapt", params)
    return out


def cost(d, key, tol, what="circuits"):
    """Circuits (or layers) to reach tol: ensemble median where an ensemble exists, else the single run."""
    if d.get(key + "_ens"):
        return ens.median_cost(*d[key + "_ens"], tol, key=what)
    return first(d.get(key), tol, what)


def best_static(st, tol):
    c = [(first(r, tol), r["params"]["L"]) for r in st]
    c = [x for x in c if np.isfinite(x[0])]
    if not c:
        return np.nan, None, None
    b = min(c)
    return b[0], b[1], min(L for _, L in c)


def strong_runs():
    """system -> {"tetris_fine": run, "avqite_fine": run} from results/strong_baselines.json
    (fine pool = single Pauli strings for the spin models, qubit excitations (QEB) for H4)."""
    p = os.path.join(RES, "strong_baselines.json")
    out = {}
    for r in (json.load(open(p)) if os.path.exists(p) else []):
        pool = r["params"].get("pool")
        if pool not in ("qubit", "qeb"):
            continue
        if r["kind"] == "adapt" and r["params"].get("batch") == "tetris":
            if ens.has(r["system"], "adapt", r["params"]):
                r = ens.representative(r["system"], "adapt", r["params"])
            out.setdefault(r["system"], {})["tetris_fine"] = r
        elif r["kind"] == "avqite" and not r["params"].get("abs_cut"):
            out.setdefault(r["system"], {})["avqite_fine"] = r
    return out


def fig_models(S):
    order = ["mfim_12", "mfim2d_3x3", "mfim2d_3x4", "h4_1.5", "h4_2.0", "tfim2d_3x3", "tfim2d_3x4",
             "hubbard_4", "hubbard_6"]
    order = [k for k in order if k in S]
    SB = strong_runs()
    fig, axs = plt.subplots(3, 3, figsize=(7.0, 6.3), gridspec_kw=dict(wspace=0.32, hspace=0.45, top=0.9))
    for ax, k, lab in zip(axs.flat, order, "abcdefghi"):
        d = S[k]
        Ls = sorted(r["params"]["L"] for r in d["static"])
        show = [Ls[0], Ls[len(Ls) // 2], Ls[-1]]
        for r in d["static"]:
            if r["params"]["L"] in show:
                ax.loglog(np.maximum(r["circuits"], 1), rel(r), color=COL["static"], lw=0.8,
                          alpha=0.4 + 0.6 * show.index(r["params"]["L"]) / 2)
                ax.text(r["circuits"][-1] * 1.05, rel(r)[-1], f"{r['params']['L']}", fontsize=5, color="0.4",
                        va="center", clip_on=True)
        for key, name, ls in (("avqite", "AVQITE (schedule pool)", "-"), ("adapt", "ADAPT-VQE (schedule pool)", "-"),
                              ("ucc", "ADAPT-VQE (UCCSD pool)", "--")):
            r = d.get(key)
            if r:
                ax.loglog(np.maximum(r["circuits"], 1), rel(r), color=COL[key], ls=ls, lw=1.0, label=name)
        for key, name, ls in (("tetris_fine", "TETRIS-ADAPT-VQE (qubit/QEB pool)", "-"),
                              ("avqite_fine", "AVQITE (qubit/QEB pool)", (0, (1.2, 1.2)))):
            r = SB.get(k, {}).get(key)
            if r:
                ax.loglog(np.maximum(r["circuits"], 1), rel(r), color=COL[key], ls=ls, lw=1.1, label=name)
        ax.loglog(np.maximum(d["dyn"]["circuits"], 1), rel(d["dyn"]), color=COL["dyn"], lw=1.6, label="dynamic VITE")
        ax.set_ylim(1e-9, 2)
        ax.set_title(f"({lab}) " + d["label"].replace("\\AA", "Å"), loc="left", fontsize=7.5, pad=3)
        if lab in "adg":
            ax.set_ylabel(r"$(E-E_0)/|E_0|$")
        if lab in "ghi":
            ax.set_xlabel("cumulative circuits")
    h, l = [], []
    for ax in axs.flat:
        for hh, ll in zip(*ax.get_legend_handles_labels()):
            if ll not in l:
                h.append(hh); l.append(ll)
    h.append(plt.Line2D([], [], color=COL["static"], lw=0.8)); l.append("static VITE, $L$ layers (labelled)")
    fig.legend(h, l, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.savefig(os.path.join(FIG, "fig_models.pdf"))
    plt.close(fig)


def fig_depth(S):
    order = ["tfim1d_12", "mfim_12", "tfim2d_3x3", "tfim2d_3x4", "mfim2d_3x3", "mfim2d_3x4", "hubbard_4",
             "h4_1.5", "h4_2.0"]
    order = [k for k in order if k in S]
    mk = {"1D": "o", "2D": "s", "fermionic": "D"}
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, k in enumerate(order):
        d = S[k]
        y = [first(d["dyn"], t, "n_layers") for t in TOLS]
        ymin = [best_static(d["static"], t)[2] for t in TOLS]
        c = cmap(i)
        ax.plot(TOLS, y, marker=mk[d["cls"]], ms=3.2, color=c, lw=1, label=d["label"].replace("\\AA", "Å"))
        ax.plot(TOLS, [np.nan if v is None else v for v in ymin], color=c, ls=":", lw=0.8)
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_xlabel("target relative error"); ax.set_ylabel("layers when target reached")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=5.5, ncol=3, loc="lower center",
               bbox_to_anchor=(0.55, 0.97), handlelength=1.5, columnspacing=0.8)
    fig.savefig(os.path.join(FIG, "fig_depth.pdf"))
    plt.close(fig)


def table(S):
    order = ["mfim_12", "mfim2d_3x3", "mfim2d_3x4", "h4_1.5", "h4_2.0",            # advantage systems
             "tfim1d_12", "tfim2d_3x3", "tfim2d_3x4", "hubbard_4", "hubbard_6"]    # validation and limits
    order = [k for k in order if k in S]
    rows, csvrows = [], []
    fmt = lambda x: "--" if not np.isfinite(x) else (f"{x:.2f}" if x < 10 else f"{x:.0f}")
    sci = lambda x: "--" if not np.isfinite(x) else f"${x / 10 ** int(np.floor(np.log10(x))):.1f}\\!\\times\\!10^{{{int(np.floor(np.log10(x)))}}}$"
    targets = {"hubbard_6": (1e-2,)}
    for k in order:
        d = S[k]
        tl = targets.get(k, (1e-4, 1e-6))
        for tol in tl:
            c = first(d["dyn"], tol)
            Ld = first(d["dyn"], tol, "n_layers")
            bs, bL, mL = best_static(d["static"], tol)
            r01 = first(d["dyn01"], tol) / c if d["dyn01"] else np.nan
            ra = first(d["avqite"], tol) / c
            rd = cost(d, "adapt", tol) / c
            ru = cost(d, "ucc", tol) / c if d["ucc"] else np.nan
            csvrows.append([k, tol, c, Ld, bs, bL, mL, bs / c, r01, ra, rd, ru])
            lab = d["label"] if tol == tl[0] else ""
            rows.append(f"{lab} & {d['n']} & $10^{{{int(round(np.log10(tol)))}}}$ & {'--' if not np.isfinite(Ld) else int(Ld)} & "
                        f"{'--' if mL is None else mL} & {sci(c)} & {fmt(ra)} & {fmt(rd)} & "
                        f"{fmt(ru) if d['ucc'] else ''} & {fmt(bs / c)} & {fmt(r01)} \\\\")
        rows.append("\\hline\\hline" if k == "h4_2.0" else "\\hline")
    hdr = ["\\begin{tabular}{lcccccccccc}",
           " & & & \\multicolumn{2}{c}{layers} & dyn.\\ VITE & \\multicolumn{5}{c}{circuits / dynamic VITE} \\\\",
           "System & $N_q$ & target & $L_{\\rm dyn}$ & $L_{\\min}$ & circuits & AVQITE & ADAPT & UCCSD & static$^\\ast$ & $\\epsilon{=}0.1$ \\\\",
           "\\hline"]
    open(os.path.join(RES, "table_summary.tex"), "w").write("\n".join(hdr + rows[:-1] + ["\\end{tabular}"]) + "\n")
    with open(os.path.join(RES, "summary_models.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "tol", "dyn_circuits", "dyn_layers", "best_static_circuits", "best_static_L",
                    "min_static_L", "static_over_dyn", "eps01_over_dyn", "avqite_over_dyn", "adapt_over_dyn",
                    "uccsd_over_dyn"])
        w.writerows(csvrows)


if __name__ == "__main__":
    S = systems()
    fig_models(S); fig_depth(S); table(S)
    print("systems:", list(S))
