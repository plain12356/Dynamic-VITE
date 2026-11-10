"""Stronger baselines on the systems where dynamic VITE beats both pool-based methods (paper, Sec. IV F).

Spin models (1D mixed-field ring N=12, 2D mixed-field tori): ADAPT-VQE and TETRIS-ADAPT-VQE with the
2-local qubit pool, and AVQITE with the same pool. H4: ADAPT-VQE and TETRIS-ADAPT-VQE with the
qubit-excitation (QEB) pool, TETRIS-ADAPT-VQE with the UCCSD pool, and AVQITE with the QEB pool.
Same cost model, integrator and stopping rules as scripts/run_models.py.

    python scripts/run_strong_baselines.py --systems mfim2d_3x3 h4_1.5 h4_2.0 mfim_12 mfim2d_3x4
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"          # single-threaded linear algebra: bitwise-reproducible results
import argparse, json, os, sys, time
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dynamic_vite.methods import vite, adapt_vqe
from dynamic_vite.pools import qubit_pool_spin, qeb_pool

RES = os.path.join(os.path.dirname(__file__), "..", "results")
MAX_OPS = 150          # cap on operators (ADAPT) or layers (AVQITE) for single-string pools
VAR_STOP = 1e-12
STOP_REL = 1e-7        # stop once the relative error is below 1e-7 (all targets are >= 1e-6; costs up to them unchanged)


def load(name):
    """-> H, Hm, E0, psi0, gs, pools dict, (dtau, steps)."""
    if name.startswith("mfim_"):                 # 1D mixed-field ring, 'mfim_<N>'
        import run_mfim
        H, Hm, E0, gs, psi0, sched, _ = run_mfim.system(int(name.split("_")[1]))
        return H, Hm, E0, psi0, gs, {"qubit": qubit_pool_spin(H)}, (run_mfim.DTAU, int(round(run_mfim.TAU_MAX / run_mfim.DTAU)))
    import run_models
    from dynamic_vite.systems import get_system
    S, Hm, sched, _, _, ucc = get_system(name)
    pools = {"qubit": qubit_pool_spin(S.H)} if ucc is None else {"qeb": qeb_pool(ucc, S.H.n), "uccsd": ucc}
    return S.H, Hm, S.E0, S.psi0, S.gs, pools, run_models.SETTINGS[name.split("_")[0]]


CUTS = {"h4": (1e-3, 5e-4, 1e-4), "spin": (1e-1, 1e-2, 1e-3)}    # absolute L2 cutoffs; 5e-4 = Gomes et al. (H4)


def specs_abs(name):
    """AVQITE with the original absolute cutoff L2_min > L2_cut, scanned over CUTS."""
    if name.startswith("h4"):
        return [(name, "avqite", {"pool": "qeb", "abs_cut": c}) for c in CUTS["h4"]]
    return [(name, "avqite", {"pool": "qubit", "abs_cut": c}) for c in CUTS["spin"]]


def specs_for(name):
    if name.startswith("h4"):
        return [(name, "adapt", {"pool": "qeb"}), (name, "adapt", {"pool": "qeb", "batch": "tetris"}),
                (name, "adapt", {"pool": "uccsd", "batch": "tetris"}), (name, "avqite", {"pool": "qeb"})]
    return [(name, "adapt", {"pool": "qubit"}), (name, "adapt", {"pool": "qubit", "batch": "tetris"}),
            (name, "avqite", {"pool": "qubit"})]


def task(spec):
    name, kind, p = spec
    H, Hm, E0, psi0, gs, pools, (dt, steps) = load(name)
    pool = pools[p["pool"]]
    if kind == "adapt":
        lab = ("TETRIS-" if p.get("batch") else "") + f"ADAPT-VQE ({p['pool']} pool)"
        r = adapt_vqe(H, Hm, E0, psi0, pool, gs=gs, grad_tol=1e-5, max_ops=MAX_OPS, batch=p.get("batch"),
                      tie_round=1e-8, stop_rel=STOP_REL, adjoint=True, method_name=lab,
                      jitter=p.get("jitter", 0.0), seed=p.get("seed"))
    else:
        r = vite(H, Hm, E0, psi0, pool, mode="avqite", pool=pool, eps=0.9, max_layers=MAX_OPS, dtau=dt,
                 n_steps=2 * steps, gs=gs, var_stop=VAR_STOP,      # twice the VITE step budget: only helps AVQITE
                 abs_cut=p.get("abs_cut"), stop_rel=STOP_REL,
                 method_name=f"AVQITE ({p['pool']} pool" + (f", L2_cut={p['abs_cut']:g})" if p.get("abs_cut") else ")"))
    d = {k: getattr(r, k) for k in ("method", "n", "E0", "energy", "n_layers", "circuits", "layers", "wall",
                                    "by_kind", "by_kind_hist")}
    d.update(system=name, kind=kind, params=p, pool_size=len(pool))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--abs", action="store_true", help="AVQITE with absolute cutoffs only")
    ap.add_argument("--tag", default="", help="output-file suffix (for concurrent runs)")
    a = ap.parse_args()
    out = os.path.join(RES, ("strong_baselines_abs" if a.abs else "strong_baselines") + a.tag + ".json")
    res = json.load(open(out)) if os.path.exists(out) else []
    specs = [s for n in a.systems for s in (specs_abs(n) if a.abs else specs_for(n))]
    res = [d for d in res if not any(d["system"] == s[0] and d["kind"] == s[1] and d["params"] == s[2] for s in specs)]
    with Pool(a.workers) as pool:
        for d in pool.imap_unordered(task, specs, chunksize=1):
            res.append(d)
            json.dump(res, open(out, "w"))
            print(f"  {d['system']} {d['method']}: {len(d['layers'])} ops, {d['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
