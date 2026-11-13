"""Make all paper figures and tables from results/*.json.

    python scripts/make_figures.py            # writes paper/figures/*.pdf and results/*.csv, *.tex
"""
from __future__ import annotations
import json, os, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "paper", "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7, "lines.linewidth": 1.2,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL1, COL2 = 3.4, 7.0
# Okabe-Ito
C = {"Look-ahead": "#CC79A7", "No floor": "#000000", "Dynamic VITE": "#0072B2", "Static VITE (L=N/2)": "#009E73", "Static VITE (L=N/2-1)": "#999999",
     "Stagnation trigger": "#E69F00", "AVQITE": "#CC79A7", "ADAPT-VQE": "#D55E00",
     "Dynamic VITE (local set)": "#56B4E9"}
MARK = {"Look-ahead": "D", "No floor": "x", "Dynamic VITE": "o", "Static VITE (L=N/2)": "s", "Static VITE (L=N/2-1)": "v",
        "Stagnation trigger": "^", "AVQITE": "D", "ADAPT-VQE": "P"}
LABEL = {"Look-ahead": "Dynamic, look-ahead", "No floor": r"Dynamic, $\kappa=0$", "Static VITE (L=N/2)": r"Static VITE, $L=N/2$", "Static VITE (L=N/2-1)": r"Static VITE, $L=N/2-1$",
         "Dynamic VITE": "Dynamic VITE (this work)", "Stagnation trigger": "Energy-stagnation trigger",
         "AVQITE": "AVQITE (pool screening)", "ADAPT-VQE": "ADAPT-VQE (BFGS)"}
TARGET = 1e-6
EPS_TFIM = 0.9


def load(name):
    p = os.path.join(RES, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else []


def relerr(r):
    e = np.array(r["energy"])
    return np.maximum((e - r["E0"]) / abs(r["E0"]), 1e-16)


def cost_to(r, tol=TARGET, key="circuits"):
    e = relerr(r)
    hit = np.nonzero(e < tol)[0]
    return float(r[key][hit[0]]) if hit.size else np.nan


def tau_to(r, tol=TARGET):
    e = relerr(r)
    hit = np.nonzero(e < tol)[0]
    return float(r["tau"][hit[0]]) if hit.size else np.nan


# ----------------------------------------------------------------------------- Fig. 2: dynamics
def fig_dynamics(main, light):
    light = light or main
    sizes = sorted({r["n"] for r in main})
    show = [n for n in sizes if n in (8, 12, 16)] or sizes
    cmap = plt.get_cmap("viridis")
    col = {n: cmap(0.1 + 0.75 * i / max(len(sizes) - 1, 1)) for i, n in enumerate(sizes)}
    fig, ax = plt.subplots(3, 1, figsize=(COL1, 4.9), gridspec_kw=dict(hspace=0.42))
    for n in show:
        r = next(x for x in main if x["n"] == n)
        t = np.array(r["tau"])
        ax[0].semilogy(t, relerr(r), color=col[n], label=f"$N={n}$")
        rl = next(x for x in light if x["n"] == n)
        tl = np.array(rl["tau"]); zoom = tl <= 4.0
        rho = np.array(rl.get("rho_pre", rl["rho"]))
        ax[1].semilogy(tl[zoom], np.maximum(rho[zoom], 1e-16), color=col[n], lw=1.0, label=f"$N={n}$")
    for n in sizes:
        r = next(x for x in light if x["n"] == n)
        ev = [e for e in r["events"] if e[2].startswith("G") and e[2][1:].isdigit()]
        ax[2].plot([int(e[2][1:]) for e in ev], [e[1] for e in ev], marker="o", ms=3, lw=0.8, color=col[n],
                   label=f"$N={n}$")
    ax[1].axhline(EPS_TFIM, color="k", ls="--", lw=0.7)
    ax[1].text(3.0, EPS_TFIM * 1.25, rf"$\epsilon={EPS_TFIM}$", fontsize=7)
    ax[0].set_ylabel(r"$(E-E_0)/|E_0|$"); ax[0].set_xlabel(r"imaginary time $\tau$")
    ax[0].legend(loc="upper right", frameon=False)
    ax[1].set_ylabel(r"$\rho$ tested by trigger"); ax[1].set_xlabel(r"$\tau$ (growth window)")
    ax[1].set_ylim(1e-6, 2.5); ax[1].set_xlim(0, 4.0)
    ax[2].set_xlabel(r"range $r$ of appended generator $G_r$")
    ax[2].set_ylabel(r"growth time $\tau_r$")
    ax[2].legend(frameon=False, ncol=2, loc="upper left", handlelength=1.2)
    for a, lab in zip(ax, "abc"):
        a.set_title(f"({lab})", loc="left", fontsize=8, pad=3)
    fig.savefig(os.path.join(FIG, "fig_dynamics.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 3: benchmarks
def panel(a, lab):
    a.set_title(f"({lab})", loc="left", fontsize=8, pad=3)


def fig_benchmarks(main, bench):
    runs = main + bench
    methods = ["Dynamic VITE", "Static VITE (L=N/2)", "AVQITE", "Stagnation trigger", "ADAPT-VQE",
               "Static VITE (L=N/2-1)"]
    ref = {r["n"]: cost_to(r) for r in main}
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.25), gridspec_kw=dict(wspace=0.42))
    for m in methods:
        rs = sorted([r for r in runs if r["method"] == m], key=lambda r: r["n"])
        if not rs:
            continue
        ns = [r["n"] for r in rs]
        cost = np.array([cost_to(r) for r in rs])
        kw = dict(marker=MARK[m], ms=3.5, color=C[m], label=LABEL[m])
        if np.isfinite(cost).any():
            ax[0].semilogy(ns, cost, **kw)
            ax[1].semilogy(ns, cost / np.array([ref[n] for n in ns]), **kw)
        ax[2].semilogy(ns, [relerr(r)[-1] for r in rs], **kw)
    for a in ax:
        a.set_xlabel("$N$"); a.set_xticks(sorted(ref))
    ax[0].set_ylabel(r"circuits to $10^{-6}$ rel. error")
    ax[1].set_ylabel("cost relative to dynamic VITE")
    ax[1].set_yticks([0.05, 0.1, 0.2, 0.5, 1, 2])
    ax[1].set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1", "2"])
    ax[1].axhline(1, color="k", lw=0.5)
    ax[2].set_ylabel("final relative energy error")
    ax[2].axhline(TARGET, color="k", ls=":", lw=0.7)
    h, l = ax[2].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    for a, lab in zip(ax, "abc"):
        panel(a, lab)
    fig.savefig(os.path.join(FIG, "fig_benchmarks.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 4: diagnostics
def fig_diagnostics(main, bench, ablation, n=12):
    pick = lambda rs, m: next((r for r in rs if r["method"] == m and r["n"] == n), None)
    d, st, un = pick(main, "Dynamic VITE"), pick(bench, "Stagnation trigger"), pick(bench, "Static VITE (L=N/2-1)")
    loc = pick(ablation, "Dynamic VITE (local set)")
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.15), gridspec_kw=dict(wspace=0.4))
    for r, m in ((d, "Dynamic VITE"), (st, "Stagnation trigger")):
        if r is None:
            continue
        ax[0].step(r["tau"], r["n_layers"], where="post", color=C[m], label=LABEL[m])
        ax[1].semilogy(r["tau"], relerr(r), color=C[m], label=LABEL[m])
    ax[0].axhline(n // 2, color="k", ls=":", lw=0.7)
    ax[0].text(7.5, n // 2 + 0.35, r"exact depth $N/2$", fontsize=6.5)
    ax[0].set_xlabel(r"$\tau$"); ax[0].set_ylabel("layers $L$")
    ax[0].set_yticks(range(0, n // 2 + 13, 3))
    ax[0].legend(frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.35), fontsize=5.5)
    ax[1].set_xlabel(r"$\tau$"); ax[1].set_ylabel(r"$(E-E_0)/|E_0|$")
    for r, m, lab in ((d, "Dynamic VITE", "dynamic, ladder"),
                      (un, "Static VITE (L=N/2-1)", r"static, ladder $L=N/2-1$"),
                      (loc, "Dynamic VITE (local set)", "dynamic, local set")):
        if r is None:
            continue
        t = np.array(r["tau"]); keep = t > 0
        ax[2].semilogy(t[keep], np.maximum(np.array(r.get("rho_pre", r["rho"]))[keep], 1e-16), color=C[m], label=lab)
    ax[2].axhline(EPS_TFIM, color="k", ls="--", lw=0.7)
    ax[2].set_xscale("log"); ax[2].set_xlim(0.1, 15)
    ax[2].set_ylim(1e-16, 5); ax[2].set_xlabel(r"$\tau$"); ax[2].set_ylabel(r"$\rho$")
    ax[2].legend(frameon=False, loc="center", bbox_to_anchor=(0.6, 0.42))
    for a, lab in zip(ax, "abc"):
        panel(a, lab)
    fig.savefig(os.path.join(FIG, "fig_diagnostics.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- Fig. 5: noise
def fig_noise(noise):
    if not noise:
        return
    shots = sorted({r["params"]["shots"] for r in noise})
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.75), gridspec_kw=dict(wspace=0.55))
    variants = [("noisy_dynamic", "Dynamic VITE", 0.85), ("noisy_lookahead", "Look-ahead", 0.95),
                ("noisy_nofloor", "No floor", 1.05), ("noisy_static", "Static VITE (L=N/2)", 1.15)]
    for kind, m, off in variants:
        if not any(r["kind"] == kind for r in noise):
            continue
        med, lo, hi, dep_m, dep_lo, dep_hi = [], [], [], [], [], []
        for S in shots:
            rs = [r for r in noise if r["kind"] == kind and r["params"]["shots"] == S]
            e = np.array([np.median(relerr(r)[-20:]) for r in rs])
            L = np.array([r["n_layers"][-1] for r in rs])
            med.append(np.median(e)); lo.append(np.percentile(e, 25)); hi.append(np.percentile(e, 75))
            dep_m.append(np.median(L)); dep_lo.append(L.min()); dep_hi.append(L.max())
        x = np.array(shots) * off
        lab = "Dynamic, variance test" if kind == "noisy_dynamic" else LABEL[m]
        ax[0].errorbar(x, med, yerr=[np.array(med) - lo, np.array(hi) - med], marker=MARK[m], ms=3.5,
                       color=C[m], capsize=2, lw=1, label=lab)
        if kind != "noisy_static":
            ax[1].errorbar(x, dep_m, yerr=[np.array(dep_m) - dep_lo, np.array(dep_hi) - dep_m], marker=MARK[m],
                           ms=3.5, color=C[m], capsize=2, lw=1)
    n = noise[0]["n"]
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("shots per circuit $S$"); ax[0].set_ylabel("rel. energy error")
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=6,
               handlelength=1.2, columnspacing=0.8)
    ax[1].axhline(n // 2, color="k", ls=":", lw=0.7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("shots per circuit $S$"); ax[1].set_ylabel("selected depth $L$")
    ax[1].set_ylim(0, n // 2 + 13)
    for a, lab in zip(ax, "ab"):
        panel(a, lab)
    fig.savefig(os.path.join(FIG, "fig_noise.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------- tables
def tables(main, bench, eps, ablation, noise):
    runs = main + bench
    rows = []
    for r in sorted(runs, key=lambda r: (r["n"], r["method"])):
        scr = r.get("by_kind", {})
        rows.append({"N": r["n"], "method": r["method"], "final_layers": r["n_layers"][-1],
                     "layers": " ".join(r["layers"]),
                     "growth_tau": ";".join(f"{e[1]:.1f}" for e in r["events"]),
                     "final_rel_error": relerr(r)[-1], "fidelity": r["fidelity"],
                     "circuits_to_1e-6": cost_to(r), "cnot_circuits_to_1e-6": cost_to(r, key="cnots"),
                     "tau_to_1e-6": tau_to(r), "total_circuits": r["circuits"][-1], "wall_s": r["wall"],
                     "M_circuits": scr.get("M", 0), "V_circuits": scr.get("V", 0),
                     "H2_circuits": scr.get("H2", 0), "E_circuits": scr.get("energy", 0)})
    with open(os.path.join(RES, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # LaTeX table: circuits to 1e-6 per method and N
    methods = ["Dynamic VITE", "Static VITE (L=N/2)", "Stagnation trigger", "AVQITE", "ADAPT-VQE"]
    short = {"Dynamic VITE": "Dynamic VITE", "Static VITE (L=N/2)": r"Static, $L{=}N/2$",
             "Stagnation trigger": "Stagnation", "AVQITE": "AVQITE", "ADAPT-VQE": "ADAPT-VQE"}
    sizes = sorted({r["n"] for r in main})

    def fmt(x):
        if not np.isfinite(x):
            return "--"
        e = int(np.floor(np.log10(x)))
        return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"
    lines = [r"\begin{tabular}{l" + "c" * len(sizes) + "}", r"\hline\hline",
             "Method & " + " & ".join(f"$N={n}$" for n in sizes) + r" \\", r"\hline"]
    for m in methods:
        vals = []
        for n in sizes:
            r = next((x for x in runs if x["method"] == m and x["n"] == n), None)
            vals.append(fmt(cost_to(r)) if r else "--")
        lines.append(short[m] + " & " + " & ".join(vals) + r" \\")
    lines.append(r"\hline")
    # final depth row for the dynamic method
    dep = [str(next(x for x in main if x["n"] == n)["n_layers"][-1]) for n in sizes]
    lines.append(r"Depth selected by trigger & " + " & ".join(dep) + r" \\")
    stl = [str(next((x for x in bench if x["n"] == n and x["method"] == "Stagnation trigger"), {"n_layers": ["--"]})["n_layers"][-1]) for n in sizes]
    lines.append(r"Depth, stagnation trigger & " + " & ".join(stl) + r" \\")
    lines += [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(RES, "table_cost.tex"), "w").write("\n".join(lines) + "\n")

    # threshold sensitivity
    if eps:
        with open(os.path.join(RES, "eps.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["eps", "N", "final_layers", "growth_tau", "final_rel_error", "circuits_to_1e-6", "tau_to_1e-6"])
            for r in sorted(eps, key=lambda r: r["params"]["eps"]):
                w.writerow([r["params"]["eps"], r["n"], r["n_layers"][-1],
                            ";".join(f"{e[1]:.1f}" for e in r["events"]), relerr(r)[-1], cost_to(r), tau_to(r)])
    if ablation:
        with open(os.path.join(RES, "ablation.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["N", "final_layers", "final_rel_error", "final_rho", "fidelity"])
            for r in sorted(ablation, key=lambda r: r["n"]):
                w.writerow([r["n"], r["n_layers"][-1], relerr(r)[-1], r["rho"][-1], r["fidelity"]])
    if noise:
        with open(os.path.join(RES, "noise.csv"), "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["kind", "shots", "median_rel_error", "q25", "q75", "depths"])
            for kind in ("noisy_dynamic", "noisy_lookahead", "noisy_nofloor", "noisy_static"):
                for S in sorted({r["params"]["shots"] for r in noise}):
                    rs = [r for r in noise if r["kind"] == kind and r["params"]["shots"] == S]
                    e = np.array([np.median(relerr(r)[-20:]) for r in rs])
                    w.writerow([kind, S, np.median(e), np.percentile(e, 25), np.percentile(e, 75),
                                " ".join(str(r["n_layers"][-1]) for r in rs)])


if __name__ == "__main__":
    main, bench, eps, abl, noise = (load(k) for k in ("main", "bench", "eps", "ablation", "noise"))
    if main:
        fig_dynamics(main, load("lightcone"))
    if main and bench:
        fig_benchmarks(main, bench)
        fig_diagnostics(main, bench, abl)
    fig_noise(noise)
    if main and bench:
        tables(main, bench, eps, abl, noise)
    print("figures:", sorted(os.listdir(FIG)))


# ============================================================================= mixed-field Ising
def _mf(name):
    return load(f"mfim_{name}")


def _first(r, tol, key="circuits"):
    e = relerr(r)
    h = np.nonzero(e < tol)[0]
    return float(r[key][h[0]]) if h.size else np.nan


def best_static(static, n, tol, hz=None):
    rs = [r for r in static if r["n"] == n and (hz is None or r["params"].get("hz") == hz)]
    c = [(_first(r, tol), r["params"]["L"]) for r in rs]
    c = [x for x in c if np.isfinite(x[0])]
    if not c:
        return np.nan, None, None
    best = min(c)
    return best[0], best[1], min(L for _, L in c)


def fig_mfim(eps_runs, static, avq, adapt, n=12, eps_main=0.9):
    dyn = next(r for r in eps_runs if r["n"] == n and r["params"]["eps"] == eps_main)
    fig, ax = plt.subplots(1, 3, figsize=(COL2, 2.25), gridspec_kw=dict(wspace=0.42))
    # (a) error vs cumulative circuits: dynamic against every fixed depth
    sts = sorted([r for r in static if r["n"] == n], key=lambda r: r["params"]["L"])
    show = [r for r in sts if r["params"]["L"] in (2, 4, 7, 12, 20)]
    cmap = plt.get_cmap("Greys")
    for i, r in enumerate(show):
        ax[0].loglog(np.maximum(r["circuits"], 1), relerr(r), color=cmap(0.35 + 0.6 * i / len(show)), lw=0.9,
                     label=f"static $L={r['params']['L']}$")
    ax[0].loglog(np.maximum(dyn["circuits"], 1), relerr(dyn), color=C["Dynamic VITE"], lw=1.5,
                 label=rf"dynamic, $\epsilon={eps_main}$")
    ax[0].set_xlabel("cumulative circuits"); ax[0].set_ylabel(r"$(E-E_0)/|E_0|$")
    ax[0].set_ylim(1e-9, 2); ax[0].set_xlim(1e3, None)
    ax[0].legend(frameon=False, fontsize=5.2, loc="lower left", handlelength=1.4)
    # (b) depth reached at each accuracy: dynamic vs smallest sufficient fixed depth
    tols = [1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for nn, mk in zip(sorted({r["n"] for r in eps_runs}), "os^"):
        d = next(r for r in eps_runs if r["n"] == nn and r["params"]["eps"] == eps_main)
        ax[1].semilogx(tols, [_first(d, t, "n_layers") for t in tols], marker=mk, ms=3.5, color=C["Dynamic VITE"],
                       lw=0.9, alpha=0.5 + 0.25 * (nn - 8) / 4, label=f"dynamic, $N={nn}$")
    ax[1].semilogx(tols, [best_static(static, n, t)[2] for t in tols], color="k", ls=":", marker="x", ms=4,
                   lw=0.9, label=f"min. fixed depth, $N={n}$")
    ax[1].invert_xaxis(); ax[1].set_xlabel("target rel. error"); ax[1].set_ylabel("layers when target reached")
    ax[1].legend(frameon=False, fontsize=5.5, loc="upper left")
    # (c) cost to 1e-6 vs N
    tol = 1e-6
    sizes = sorted({r["n"] for r in eps_runs})
    series = [("Dynamic VITE", [_first(next(r for r in eps_runs if r["n"] == m and r["params"]["eps"] == eps_main), tol) for m in sizes],
               rf"dynamic VITE, $\epsilon={eps_main}$"),
              ("Static VITE (L=N/2)", [best_static(static, m, tol)[0] for m in sizes], "best fixed depth (hindsight)")]
    if avq:
        series.append(("AVQITE", [_first(next((r for r in avq if r["n"] == m), None), tol) if any(r["n"] == m for r in avq) else np.nan for m in sizes], "AVQITE (pool screening)"))
    if adapt:
        series.append(("ADAPT-VQE", [_first(next((r for r in adapt if r["n"] == m), None), tol) if any(r["n"] == m for r in adapt) else np.nan for m in sizes], "ADAPT-VQE (BFGS)"))
    for m, ys, lab in series:
        ax[2].semilogy(sizes, ys, marker=MARK[m], ms=3.5, color=C[m], label=lab)
    ax[2].set_xticks(sizes); ax[2].set_xlabel("$N$"); ax[2].set_ylabel(r"circuits to $10^{-6}$ rel. error")
    ax[2].legend(frameon=False, fontsize=5.5, loc="center left", bbox_to_anchor=(0.0, 0.62))
    for a, lab in zip(ax, "abc"):
        panel(a, lab)
    fig.savefig(os.path.join(FIG, "fig_mfim.pdf"))
    plt.close(fig)


def fig_hz(hz_runs):
    if not hz_runs:
        return
    hzs = sorted({r["params"]["hz"] for r in hz_runs})
    dyn = {r["params"]["hz"]: r for r in hz_runs if r["kind"] == "dynamic"}
    stat = [r for r in hz_runs if r["kind"] == "static"]
    n = hz_runs[0]["n"]
    fig, ax = plt.subplots(1, 2, figsize=(COL1, 1.95), gridspec_kw=dict(wspace=0.6))
    xs = np.arange(len(hzs))
    for tol, mk, ls in ((1e-4, "o", "-"), (1e-6, "s", "--")):
        lab = rf"$10^{{{int(np.log10(tol))}}}$"
        ax[0].plot(xs, [_first(dyn[h], tol, "n_layers") for h in hzs], marker=mk, ms=3.2, ls=ls,
                   color=C["Dynamic VITE"], label=f"dynamic, {lab}")
        ax[0].plot(xs, [best_static(stat, n, tol, hz=h)[2] for h in hzs], marker="x", ms=3.5, ls=":",
                   color="k" if tol == 1e-4 else "0.5", label=f"min. fixed, {lab}")
        ratio = [_first(dyn[h], tol) / best_static(stat, n, tol, hz=h)[0] for h in hzs]
        ax[1].plot(xs, ratio, marker=mk, ms=3.2, ls=ls, color=C["Dynamic VITE"], label=lab)
    ax[0].axhline(n // 2, color="0.6", lw=0.6, ls="--")
    ax[0].set_xlabel("$h_z$"); ax[0].set_ylabel("layers at target")
    for a in ax:
        a.set_xticks(xs); a.set_xticklabels([f"{h:g}" for h in hzs], fontsize=6)
    h0, l0 = ax[0].get_legend_handles_labels()
    fig.legend(h0, l0, frameon=False, ncol=2, fontsize=5.5, loc="lower center", bbox_to_anchor=(0.5, 0.97),
               handlelength=1.6, columnspacing=0.8)
    ax[1].axhline(1, color="k", lw=0.5)
    ax[1].set_xlabel("$h_z$"); ax[1].set_ylabel("cost / best fixed depth")
    ax[1].set_ylim(0, 1.2)
    for a, lab in zip(ax, "ab"):
        panel(a, lab)
    fig.savefig(os.path.join(FIG, "fig_hz.pdf"))
    plt.close(fig)


def mfim_tables(eps_runs, static, avq, adapt, hz_runs, adaptn=()):
    sizes = sorted({r["n"] for r in eps_runs})
    rows = []
    for tol in (1e-4, 1e-6):
        for m in sizes:
            for e in sorted({r["params"]["eps"] for r in eps_runs}):
                d = next(r for r in eps_runs if r["n"] == m and r["params"]["eps"] == e)
                rows.append(dict(tol=tol, N=m, method=f"dynamic eps={e}", circuits=_first(d, tol),
                                 layers=_first(d, tol, "n_layers"), tau=_first(d, tol, "tau")))
            c, L, Lmin = best_static(static, m, tol)
            rows.append(dict(tol=tol, N=m, method="best static", circuits=c, layers=L, tau=np.nan))
            rows.append(dict(tol=tol, N=m, method="min static depth", circuits=np.nan, layers=Lmin, tau=np.nan))
            for lab, rs in (("AVQITE", avq), ("ADAPT-VQE", adapt), ("ADAPT-VQE (normalized)", adaptn)):
                r = next((x for x in rs if x["n"] == m), None)
                if r:
                    rows.append(dict(tol=tol, N=m, method=lab, circuits=_first(r, tol), layers=_first(r, tol, "n_layers"),
                                     tau=_first(r, tol, "tau")))
    with open(os.path.join(RES, "mfim_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def fmt(x):
        if not np.isfinite(x):
            return "--"
        e = int(np.floor(np.log10(x)))
        return f"${x / 10 ** e:.2f}\\times10^{{{e}}}$"
    get = lambda m, meth, tol: next((r for r in rows if r["N"] == m and r["method"] == meth and r["tol"] == tol), None)
    lines = [r"\begin{tabular}{l" + "cc" * len(sizes) + "}", r"\hline\hline",
             " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{$N={m}$}}" for m in sizes) + r" \\",
             "Method & " + " & ".join(r"$10^{-4}$ & $10^{-6}$" for _ in sizes) + r" \\", r"\hline"]
    for meth, lab in (("dynamic eps=0.9", r"Dynamic VITE, $\epsilon=0.9$"), ("dynamic eps=0.1", r"Dynamic VITE, $\epsilon=0.1$"),
                      ("best static", "Best fixed depth (hindsight)"), ("AVQITE", r"AVQITE, $\epsilon=0.9$"),
                      ("ADAPT-VQE", "ADAPT-VQE"), ("ADAPT-VQE (normalized)", "ADAPT-VQE, normalized")):
        vals = []
        for m in sizes:
            for tol in (1e-4, 1e-6):
                r = get(m, meth, tol)
                vals.append(fmt(r["circuits"]) if r else "--")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    lines.append(r"\hline")
    for meth, lab in (("dynamic eps=0.9", r"Layers, dynamic $\epsilon=0.9$"), ("min static depth", "Smallest sufficient fixed depth"),
                      ("AVQITE", "Layers, AVQITE"), ("ADAPT-VQE", "Operators, ADAPT-VQE")):
        vals = []
        for m in sizes:
            for tol in (1e-4, 1e-6):
                r = get(m, meth, tol)
                vals.append(str(int(r["layers"])) if r and np.isfinite(r["layers"] if r["layers"] is not None else np.nan) else "--")
        lines.append(lab + " & " + " & ".join(vals) + r" \\")
    lines += [r"\hline\hline", r"\end{tabular}"]
    open(os.path.join(RES, "table_mfim.tex"), "w").write("\n".join(lines) + "\n")


def mfim_main():
    eps_runs, static, avq, adapt, hz_runs = (_mf(k) for k in ("eps", "static", "avqite", "adapt", "hz"))
    if eps_runs and static:
        fig_mfim(eps_runs, static, avq, adapt)
        mfim_tables(eps_runs, static, avq, adapt, hz_runs, _mf("adaptnorm"))
    fig_hz(hz_runs)


if __name__ == "__main__":
    mfim_main()
    print("mfim figures:", sorted(f for f in os.listdir(FIG) if "mfim" in f or "hz" in f))


# ============================================================================= noisy comparison
def noisecmp_stats(R, tol=1e-4):
    """Per (model, n, kind, S): median late error, median total circuits, median circuits to tol."""
    out = {}
    for key in sorted({(r["model"], r["n"], r["kind"], r["shots"]) for r in R}):
        rs = [r for r in R if (r["model"], r["n"], r["kind"], r["shots"]) == key]
        late = []
        hit = []
        for r in rs:
            e = np.maximum((np.array(r["energy"]) - r["E0"]) / abs(r["E0"]), 1e-16)
            late.append(np.median(e[-20:]))
            h = np.nonzero(e < tol)[0]
            hit.append(r["circuits"][h[0]] if h.size else np.nan)
        hit = np.array(hit)
        out[key] = dict(err=np.median(late), err_lo=np.percentile(late, 25), err_hi=np.percentile(late, 75),
                        circ=np.median([r["circuits"][-1] for r in rs]),
                        hit_frac=np.mean(np.isfinite(hit)), circ_tol=np.nanmedian(hit) if np.isfinite(hit).any() else np.nan,
                        ops=np.median([len(r["layers"]) for r in rs]), n=len(rs))
    return out


def fig_noisecmp(R):
    if not R:
        return
    st = noisecmp_stats(R)
    systems = [("tfim", 8, "TFIM, $N=8$"), ("tfim", 12, "TFIM, $N=12$"), ("mfim", 8, "mixed field, $N=8$")]
    systems = [s for s in systems if any((s[0], s[1]) == (k[0], k[1]) for k in st)]
    meth = [("dynamic", "Dynamic VITE, variance test", C["Dynamic VITE"], "o"),
            ("dynamic_la", "Dynamic VITE, look-ahead", C["Look-ahead"], "D"),
            ("adam", "ADAPT-VQE (Adam)", "#D55E00", "P"),
            ("spsa", "ADAPT-VQE (SPSA)", "#E69F00", "X")]
    fig, ax = plt.subplots(2, len(systems), figsize=(COL2, 3.9), gridspec_kw=dict(wspace=0.35, hspace=0.45))
    ax = np.atleast_2d(ax)
    for j, (m, n, title) in enumerate(systems):
        for kind, lab, col, mk in meth:
            ks = sorted(k for k in st if k[:3] == (m, n, kind))
            if not ks:
                continue
            S = np.array([k[3] for k in ks], float)
            e = np.array([st[k]["err"] for k in ks]); lo = np.array([st[k]["err_lo"] for k in ks]); hi = np.array([st[k]["err_hi"] for k in ks])
            ax[0, j].errorbar(S, e, yerr=[e - lo, hi - e], color=col, marker=mk, ms=3.5, capsize=2, lw=1, label=lab)
            ax[1, j].loglog(S * np.array([st[k]["circ"] for k in ks]), e, color=col, marker=mk, ms=3.5, lw=1, label=lab)
        ax[0, j].set_xscale("log"); ax[0, j].set_yscale("log")
        ax[0, j].set_xlabel("shots per circuit $S$"); ax[0, j].set_title(title, fontsize=7.5)
        ax[1, j].set_xlabel("total shots")
        ax[0, j].set_xticks([1e3, 1e5, 1e7])
    ax[0, 0].set_ylabel("final rel. error"); ax[1, 0].set_ylabel("final rel. error")
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.97))
    for a, lab in zip(ax.flat, "abcdef"):
        a.text(0.02, 1.04, f"({lab})", transform=a.transAxes, fontsize=8)
    fig.savefig(os.path.join(FIG, "fig_noisecmp.pdf"))
    plt.close(fig)
    with open(os.path.join(RES, "noisecmp.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "N", "method", "shots", "runs", "median_final_rel_error", "q25", "q75",
                    "median_total_circuits", "median_total_shots", "median_ops_or_layers", "frac_reaching_1e-4",
                    "median_circuits_to_1e-4"])
        for k, v in sorted(st.items()):
            w.writerow([k[0], k[1], k[2], k[3], v["n"], v["err"], v["err_lo"], v["err_hi"], v["circ"],
                        v["circ"] * k[3], v["ops"], v["hit_frac"], v["circ_tol"]])


if __name__ == "__main__":
    fig_noisecmp(load("noisecmp"))
    print("noisecmp figure written")
