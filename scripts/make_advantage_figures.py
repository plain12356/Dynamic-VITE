"""Figures for the origin of the advantage over pool-based methods.

    python scripts/make_advantage_figures.py
      paper/figures/fig_hzscan.pdf     cost relative to dynamic VITE versus h_z (3x3 torus)
      paper/figures/fig_breakdown.pdf  where each method's circuits go, at the first 1e-4 crossing
      results/hzscan.csv, results/breakdown.csv
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
COL = {"dynamic": "#0072B2", "avqite": "#CC79A7", "adapt": "#D55E00", "ucc": "#8B4513"}
HZ = (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)
# cost categories: (ledger kinds, label, color)
CATS = [(("energy", "H2"), r"$E$, $\langle H^2\rangle$", "#B0B0B0"), (("V",), "force $V$", "#009E73"),
        (("M",), "metric $M$", "#56B4E9"), (("screen",), "screening", "#E69F00"),
        (("opt",), "optimizer (BFGS)", "#7A5195")]


def rel(r):
    return np.maximum((np.array(r["energy"]) - r["E0"]) / abs(r["E0"]), 1e-16)


def hit(r, tol):
    h = np.nonzero(rel(r) < tol)[0]
    return int(h[0]) if h.size else None


def first(r, tol, key="circuits"):
    i = hit(r, tol)
    return np.nan if i is None else float(r[key][i])


def hz_name(hz):
    return f"mfim2d_3x3_hz{hz:g}"


def load_scan():
    out = {}
    for hz in HZ:
        p = os.path.join(RES, f"models_{hz_name(hz)}.json")
        if os.path.exists(p):
            R = json.load(open(p))
            out[hz] = {r["kind"]: r for r in R}
    return out


def load_scan_strong():
    """h_z -> {"tetris": run, "avqite_q": run, "adapt_q": run} with the single-string (qubit) pool."""
    out = {}
    files = [("strong_baselines_hz.json", None), ("strong_baselines.json", "mfim2d_3x3")]
    for f, only in files:
        p = os.path.join(RES, f)
        if not os.path.exists(p):
            continue
        for r in json.load(open(p)):
            sysn = r["system"]
            if only is not None and sysn != only:
                continue
            if not sysn.startswith("mfim2d_3x3") or r["params"].get("pool") != "qubit" or r["params"].get("abs_cut"):
                continue
            hz = 0.5 if sysn == "mfim2d_3x3" else float(sysn.split("_hz")[1])
            key = "avqite_q" if r["kind"] == "avqite" else ("tetris" if r["params"].get("batch") == "tetris" else "adapt_q")
            out.setdefault(hz, {})[key] = r
    return out


SCAN_SYSTEM = {0.0: "tfim2d_3x3", 0.5: "mfim2d_3x3"}     # scan points that are also main systems
SCAN_PARAMS = {"adapt": {}, "tetris": {"pool": "qubit", "batch": "tetris"}, "adapt_q": {"pool": "qubit"}}


def scan_cost(h, k, kq, r, tol, key="circuits"):
    """Cost of a scan point: seeded-ensemble median where one exists, else the single run."""
    import ensembles as ens
    name = SCAN_SYSTEM.get(h)
    params = SCAN_PARAMS.get(k if k else kq)
    if name and params is not None and (k in ("adapt", None)) and ens.has(name, "adapt", params):
        return ens.median_cost(name, "adapt", params, tol, key=key)
    return first(r, tol, key)


def fig_hzscan():
    import matplotlib.transforms as mtrans
    S = load_scan()
    Q = load_scan_strong()
    hzs = [h for h in HZ if h in S and all(k in S[h] for k in ("dynamic", "avqite", "adapt"))]
    rows = []
    fig, axs = plt.subplots(1, 2, figsize=(3.4, 1.85), gridspec_kw=dict(wspace=0.42))
    series = [("avqite", None, "AVQITE (schedule pool)", COL["avqite"], "-", "o"),
              ("adapt", None, "ADAPT-VQE (schedule pool)", COL["adapt"], "-", "o"),
              (None, "tetris", "TETRIS-ADAPT-VQE (qubit pool)", "#009E73", "-", "s"),
              (None, "avqite_q", "AVQITE (qubit pool)", COL["avqite"], (0, (1.2, 1.2)), "D")]
    for ax, tol in zip(axs, (1e-4, 1e-6)):
        missing = []
        for k, kq, lab, col, ls, mk in series:
            xs, ys = [], []
            for h in hzs:
                r = S[h][k] if k else Q.get(h, {}).get(kq)
                if r is None:
                    continue
                y = scan_cost(h, k, kq, r, tol) / first(S[h]["dynamic"], tol)
                xs.append(h); ys.append(y)            # NaN breaks the line at targets not reached
                if not np.isfinite(y):
                    missing.append((h, col, mk))
            ax.plot(xs, ys, marker=mk, ms=3.2, color=col, ls=ls, label=lab, mfc=col if k else "white")
        ax.axhline(1, color="#0072B2", lw=1.2)
        ax.set_yscale("log")
        ax.set_xlabel("$h_z$")
        ax.set_title(f"target $10^{{{int(np.log10(tol))}}}$", fontsize=7.5, pad=3)
        ax.grid(alpha=0.25, lw=0.4, which="major")
        ax._missing = missing
    axs[0].set_ylabel("circuits / dynamic VITE")
    lo = min(a.get_ylim()[0] for a in axs); hi = max(a.get_ylim()[1] for a in axs)
    for a in axs:
        a.set_ylim(lo, hi * 2.5)
        tr = mtrans.blended_transform_factory(a.transData, a.transAxes)
        for i, (h, col, mk) in enumerate(a._missing):      # target not reached within 150 operators
            a.plot([h], [0.96 - 0.05 * sum(1 for m in a._missing[:i] if m[0] == h)], marker="^", ms=3.4,
                   color=col, mfc="white", transform=tr, clip_on=False, ls="none")
    h, l = axs[0].get_legend_handles_labels()
    h.append(plt.Line2D([], [], color="#0072B2", lw=1.2)); l.append("dynamic VITE (reference, = 1)")
    h.append(plt.Line2D([], [], color="0.4", marker="^", mfc="white", ls="none", ms=3.4)); l.append("not reached (150 operators)")
    fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.98), fontsize=5.8)
    fig.savefig(os.path.join(FIG, "fig_hzscan.pdf"))
    plt.close(fig)
    for h in hzs:
        d = S[h]
        row = [h]
        for tol in (1e-3, 1e-4, 1e-6):
            c = first(d["dynamic"], tol)
            row += [c, first(d["dynamic"], tol, "n_layers"), first(d["avqite"], tol) / c,
                    first(d["avqite"], tol, "n_layers"), scan_cost(h, "adapt", None, d["adapt"], tol) / c,
                    scan_cost(h, "adapt", None, d["adapt"], tol, "n_layers")]
            for kq in ("tetris", "avqite_q", "adapt_q"):
                r = Q.get(h, {}).get(kq)
                row += [scan_cost(h, None, kq, r, tol) / c if r else np.nan,
                        scan_cost(h, None, kq, r, tol, "n_layers") if r else np.nan]
        row += [rel(d[k])[-1] for k in ("dynamic", "avqite", "adapt")]
        rows.append(row)
    with open(os.path.join(RES, "hzscan.csv"), "w", newline="") as f:
        w = csv.writer(f)
        hdr = ["hz"]
        for t in ("1e-3", "1e-4", "1e-6"):
            hdr += [f"dyn_circ_{t}", f"dyn_L_{t}", f"avq_ratio_{t}", f"avq_L_{t}", f"adapt_ratio_{t}", f"adapt_L_{t}",
                    f"tetris_q_ratio_{t}", f"tetris_q_ops_{t}", f"avqite_q_ratio_{t}", f"avqite_q_ops_{t}",
                    f"adapt_q_ratio_{t}", f"adapt_q_ops_{t}"]
        w.writerow(hdr + ["final_dyn", "final_avq", "final_adapt"])
        w.writerows(rows)


def breakdown_runs():
    """system label -> list of (method label, color key, run) with by_kind_hist."""
    S = load_scan()
    B = json.load(open(os.path.join(RES, "breakdown.json"))) if os.path.exists(os.path.join(RES, "breakdown.json")) else []
    pick = lambda sysn, kind, **p: next((r for r in B if r["system"] == sysn and r["kind"] == kind and
                                         all(r["params"].get(a) == b for a, b in p.items()) and
                                         (kind != "adapt" or ("pool" in r["params"]) == ("pool" in p))), None)
    SBp = os.path.join(RES, "strong_baselines.json")
    SB = json.load(open(SBp)) if os.path.exists(SBp) else []

    import ensembles as ens

    def fine(sysn, kind, pool, batch=None):
        return next((r for r in SB if r["system"] == sysn and r["kind"] == kind and r["params"].get("pool") == pool
                     and r["params"].get("batch") == batch and not r["params"].get("abs_cut")), None)
    out = []
    if 0.0 in S:
        out.append(("TFIM 2D, $3\\times3$", [("dynamic VITE", "dynamic", S[0.0]["dynamic"]), ("AVQITE", "avqite", S[0.0]["avqite"]),
                                               ("ADAPT-VQE", "adapt", ens.representative("tfim2d_3x3", "adapt", {}) or S[0.0]["adapt"])]))
    if 0.5 in S:
        out.append(("Mixed-field 2D, $3\\times3$", [("dynamic VITE", "dynamic", S[0.5]["dynamic"]), ("AVQITE", "avqite", S[0.5]["avqite"]),
                                                    ("ADAPT-VQE", "adapt", ens.representative("mfim2d_3x3", "adapt", {}) or S[0.5]["adapt"]),
                                                    ("AVQITE, qubit", "avqite", fine("mfim2d_3x3", "avqite", "qubit")),
                                                    ("TETRIS, qubit", "adapt", ens.representative("mfim2d_3x3", "adapt", {"pool": "qubit", "batch": "tetris"})
                                                     or fine("mfim2d_3x3", "adapt", "qubit", "tetris"))]))
    if B:
        out.append(("Mixed-field 1D, $N=12$", [("dynamic VITE", "dynamic", pick("mfim_12", "dynamic")),
                                               ("AVQITE", "avqite", pick("mfim_12", "avqite")),
                                               ("ADAPT-VQE", "adapt", ens.representative("mfim_12", "adapt", {}) or pick("mfim_12", "adapt")),
                                               ("AVQITE, qubit", "avqite", fine("mfim_12", "avqite", "qubit")),
                                               ("TETRIS, qubit", "adapt", ens.representative("mfim_12", "adapt", {"pool": "qubit", "batch": "tetris"})
                                                or fine("mfim_12", "adapt", "qubit", "tetris"))]))
        out.append(("H$_4$, 2.0 Å", [("dynamic VITE", "dynamic", pick("h4_2.0", "dynamic")), ("AVQITE", "avqite", pick("h4_2.0", "avqite")),
                                     ("ADAPT-VQE", "adapt", ens.representative("h4_2.0", "adapt", {}) or pick("h4_2.0", "adapt")),
                                     ("ADAPT (UCCSD)", "ucc", ens.representative("h4_2.0", "adapt", {"pool": "uccsd", "max_ops": 40})
                                      or pick("h4_2.0", "adapt", pool="uccsd")),
                                     ("AVQITE, QEB", "avqite", fine("h4_2.0", "avqite", "qeb")),
                                     ("TETRIS, QEB", "adapt", ens.representative("h4_2.0", "adapt", {"pool": "qeb", "batch": "tetris"})
                                      or fine("h4_2.0", "adapt", "qeb", "tetris"))]))
    return out


def fig_breakdown(tol=1e-4):
    systems = breakdown_runs()
    if not systems:
        return
    rows = []
    fig, axs = plt.subplots(len(systems), 1, figsize=(3.4, 0.62 * sum(len(m) for _, m in systems) / 3 + 0.5 * len(systems)),
                            gridspec_kw=dict(hspace=0.75, top=0.955))
    axs = np.atleast_1d(axs)
    for ax, (slab, methods) in zip(axs, systems):
        cdyn = first(methods[0][2], tol)
        ylabels = []
        for j, (mlab, ck, r) in enumerate(methods):
            i = hit(r, tol) if r else None
            ylabels.append(mlab)
            if i is None:
                ax.text(0.01, j, "target not reached", va="center", fontsize=6.5, color="0.35")
                rows.append([slab, mlab, np.nan] + [np.nan] * len(CATS))
                continue
            bk = r["by_kind_hist"][i]
            tot = sum(bk.values())
            assert abs(tot - r["circuits"][i]) < 1e-6 * max(tot, 1)
            left = 0.0
            fr = []
            for kinds, clab, col in CATS:
                v = sum(bk.get(k, 0.0) for k in kinds) / tot
                fr.append(v)
                if v > 0:
                    ax.barh(j, v, left=left, height=0.62, color=col, edgecolor="white", linewidth=1.0,
                            label=clab if j == 0 or not any(clab == t.get_label() for t in ax.patches) else None)
                left += v
            x = tot / cdyn
            ax.text(1.03, j, "1×" if not j else (f"{x:.2g}×" if x < 10 else f"{x:.0f}×"), va="center", fontsize=6.8)
            rows.append([slab, mlab, tot / cdyn] + fr)
        ax.set_yticks(range(len(methods)), ylabels, fontsize=6.8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100%"] if ax is axs[-1] else [])
        ax.tick_params(axis="y", length=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_title(slab, loc="left", fontsize=7.5, pad=2)
    axs[-1].set_xlabel(f"share of circuits up to relative error $10^{{{int(np.log10(tol))}}}$")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in CATS]
    fig.legend(handles, [l for _, l, _ in CATS], frameon=False, ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, 0.99), fontsize=6.5, handlelength=1.2, columnspacing=0.8)
    fig.text(1.0, 0.995, "cost vs\ndynamic", ha="right", va="bottom", fontsize=6.5, color="0.3",
             transform=fig.transFigure)
    fig.savefig(os.path.join(FIG, "fig_breakdown.pdf"))
    plt.close(fig)
    with open(os.path.join(RES, "breakdown.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "method", "cost_over_dyn"] + [l for _, l, _ in CATS])
        w.writerows(rows)


if __name__ == "__main__":
    fig_hzscan()
    fig_breakdown()
