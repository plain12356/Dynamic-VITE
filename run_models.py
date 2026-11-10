"""Benchmarks on 2D (transverse- and mixed-field Ising) and fermionic models (paper, Sec. IV).

    python scripts/run_models.py --systems tfim2d_3x3 mfim2d_3x3 hubbard_4 h4_1.5 h4_2.0 tfim2d_3x4 mfim2d_3x4 hubbard_6 --workers 2

For each system: dynamic VITE (eps = 0.9 and 0.1), static VITE with the first L scheduled layers,
AVQITE (pool = the schedule, operators reusable, depth capped at the schedule length), ADAPT-VQE with
the schedule as pool, and for molecules ADAPT-VQE with the UCCSD pool. Writes results/models_<system>.json.
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"          # single-threaded linear algebra: bitwise-reproducible results
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.systems import get_system, SCHED_LEN
from dynamic_vite.methods import vite, adapt_vqe

# integration settings per system class: (dtau, n_steps)
SETTINGS = {"tfim2d": (0.1, 150), "mfim2d": (0.1, 150), "hubbard": (0.1, 300), "h4": (0.5, 200)}
STATIC = {"tfim2d_3x3": (1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 30, 40),
          "tfim2d_3x4": (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40),
          "mfim2d_3x3": (1, 2, 3, 4, 6, 8, 10, 12, 15, 20, 30, 40),
          "mfim2d_3x4": (2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 40),
          "hubbard_4": (2, 4, 6, 8, 10, 12, 15, 18, 20, 22, 25, 30, 40),
          "hubbard_6": (5, 10, 20, 30, 40, 50, 60, 70, 80),
          "h4_1.5": (2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40),
          "h4_2.0": (2, 4, 6, 8, 10, 12, 15, 18, 20, 25, 30, 40)}
VAR_STOP = 1e-12
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def summarise(r, kind, params, S, entries=None):
    d = {k: getattr(r, k) for k in ("method", "n", "E0", "tau", "energy", "var", "rho", "n_layers", "circuits",
                                    "cnots", "layers", "fidelity", "wall", "by_kind", "by_kind_hist")}
    d["rho_pre"] = getattr(r, "rho_pre", [])
    d["events"] = [list(e) for e in r.events]
    d["n_halvings"] = getattr(r, "n_halvings", 0)
    d.update(kind=kind, params=params, system=S.name, E_ref=S.extra.get("E_HF"))
    if entries is not None:
        d["schedule"] = [[e.level, e.support, e.label, e.n_strings] for e in entries]
    return d


def task(spec):
    name, kind, p = spec
    S, Hm, sched, entries, gsize, pool_ucc = get_system(name)
    dt, steps = SETTINGS[name.split("_")[0]]
    common = dict(dtau=dt, n_steps=steps, gs=S.gs, var_stop=VAR_STOP)
    if kind == "dynamic":
        r = vite(S.H, Hm, S.E0, S.psi0, sched, mode="dynamic", eps=p["eps"],
                 method_name=f"Dynamic VITE (eps={p['eps']})", **common)
    elif kind == "static":
        r = vite(S.H, Hm, S.E0, S.psi0, sched[:p["L"]], mode="static", method_name=f"Static VITE (L={p['L']})", **common)
    elif kind == "avqite":
        r = vite(S.H, Hm, S.E0, S.psi0, sched, mode="avqite", pool=sched, eps=0.9, max_layers=len(sched),
                 method_name="AVQITE", **common)
    elif kind == "adapt":
        pool = pool_ucc if p.get("pool") == "uccsd" else sched
        r = adapt_vqe(S.H, Hm, S.E0, S.psi0, pool, gs=S.gs, grad_tol=1e-5, max_ops=p.get("max_ops", len(sched)),
                      jitter=p.get("jitter", 0.0), seed=p.get("seed"),
                      method_name="ADAPT-VQE (UCCSD pool)" if p.get("pool") == "uccsd" else "ADAPT-VQE")
    else:
        raise ValueError(kind)
    d = summarise(r, kind, p, S, entries if kind == "dynamic" else None)
    d["group_size"] = gsize
    d["sched_len"] = len(sched)
    return d


def specs_for(name):
    sp = [(name, "dynamic", {"eps": 0.9}), (name, "dynamic", {"eps": 0.1}), (name, "avqite", {}),
          (name, "adapt", {})]
    if "_hz" in name:                                   # field scan: no static scan, eps = 0.9 only
        return [(name, "dynamic", {"eps": 0.9}), (name, "avqite", {}), (name, "adapt", {})]
    if name.startswith("h4"):
        sp.append((name, "adapt", {"pool": "uccsd", "max_ops": 40}))
    sp += [(name, "static", {"L": L}) for L in STATIC[name] if L <= SCHED_LEN.get(name, 40)]
    return sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None, help="kinds to run")
    ap.add_argument("--tag", default="", help="suffix of the output file (for parallel partial runs)")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    for name in a.systems:
        t0 = time.time()
        sp = [s for s in specs_for(name) if not a.only or s[1] in a.only]
        sp.sort(key=lambda s: -(s[2].get("L", 0) if s[1] == "static" else 1000))   # slow first
        out = os.path.join(OUT, f"models_{name}{a.tag}.json")
        old = json.load(open(out)) if os.path.exists(out) and a.only else []
        keep = [d for d in old if not any(d["kind"] == s[1] and d["params"] == s[2] for s in sp)]
        res = list(keep)
        with Pool(a.workers) as p:
            for d in p.imap_unordered(task, sp, chunksize=1):
                res.append(d)
                json.dump(res, open(out, "w"))
                print(f"  {name} {d['kind']} {d['params']}: done ({d['wall']:.0f}s)", flush=True)
        print(f"[{name}] {len(sp)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
