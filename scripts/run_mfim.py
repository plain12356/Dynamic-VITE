"""Non-integrable mixed-field Ising experiments with the automatic commutator schedule.

    python scripts/run_mfim.py --sizes 8 10 12 --workers 2 [--only eps static avqite adapt hz]

Model: H = J sum Z_j Z_{j+1} + hx sum X_j + hz sum Z_j on a ring, (J, hx, hz) = (-1, 1.05, 0.5)
(strongly non-integrable point) unless stated. Initial state |->^N (ground state of hx sum X).
Schedule: first SCHED_LEN layers of commutator_schedule(H, hx sum X).

Groups
  eps     dynamic VITE for thresholds eps in EPS_LIST
  static  static VITE with the first L schedule layers, L in STATIC_DEPTHS
  avqite  AVQITE: same threshold, pool = the SCHED_LEN schedule layers (screened at each growth;
          operators may be reused), capped at SCHED_LEN layers like the dynamic schedule
  adapt   ADAPT-VQE with the same pool
  adaptnorm  ADAPT-VQE ranking candidates by gradient / ||G|| (size-normalized)
  hz      integrability breaking at hx = 1 (critical TFIM + longitudinal field), one N
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"          # single-threaded linear algebra: bitwise-reproducible results
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np
import scipy.sparse.linalg as sla

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.models import mfim, field_parent, minus_state
from dynamic_vite.schedule import commutator_schedule
from dynamic_vite.methods import vite, adapt_vqe

J, HX, HZ = -1.0, 1.05, 0.5
DTAU, TAU_MAX = 0.1, 10.0
SCHED_LEN = 40
EPS_LIST = (0.1, 0.3, 0.5, 0.7, 0.9, 0.97)
EPS_MAIN = 0.9
STATIC_DEPTHS = (1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40)
VAR_STOP = 1e-12
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def system(n, hx=HX, hz=HZ):
    H = mfim(n, J, hx, hz)
    Hm = H.to_sparse().tocsr()
    w, v = sla.eigsh(Hm, k=1, which="SA", tol=1e-12) if Hm.shape[0] > 64 else np.linalg.eigh(Hm.toarray())
    gs = v[:, 0]
    sched, entries = commutator_schedule(H, field_parent(n, hx), max_layers=SCHED_LEN, max_levels=8,
                                         return_entries=True)
    return H, Hm, float(w[0]), gs, minus_state(n), sched, entries


def to_dict(r, entries=None):
    d = dict(r.__dict__)
    d["events"] = [list(e) for e in r.events]
    if entries is not None:
        d["schedule"] = [[e.level, e.support, e.label] for e in entries]
    return d


def task(spec):
    kind, n, p = spec
    H, Hm, E0, gs, psi0, sched, entries = system(n, p.get("hx", HX), p.get("hz", HZ))
    steps = int(round(p.get("tau_max", TAU_MAX) / DTAU))
    common = dict(dtau=DTAU, n_steps=steps, gs=gs, var_stop=VAR_STOP)
    if kind == "dynamic":
        r = vite(H, Hm, E0, psi0, sched, mode="dynamic", eps=p["eps"],
                 method_name=f"Dynamic VITE (eps={p['eps']})", **common)
    elif kind == "static":
        r = vite(H, Hm, E0, psi0, sched[:p["L"]], mode="static", method_name=f"Static VITE (L={p['L']})", **common)
    elif kind == "avqite":
        r = vite(H, Hm, E0, psi0, sched, mode="avqite", pool=sched, eps=p["eps"], max_layers=SCHED_LEN,
                 method_name=f"AVQITE (eps={p['eps']})", **common)
    elif kind == "adapt":
        norm = p.get("normalize", False)
        r = adapt_vqe(H, Hm, E0, psi0, sched, gs=gs, grad_tol=p.get("grad_tol", 1e-5), max_ops=SCHED_LEN,
                      normalize=norm, method_name="ADAPT-VQE (normalized)" if norm else "ADAPT-VQE",
                      jitter=p.get("jitter", 0.0), seed=p.get("seed"))
    else:
        raise ValueError(kind)
    d = to_dict(r, entries if kind == "dynamic" else None)
    d.update(kind=kind, params=p)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 10, 12])
    ap.add_argument("--hz-n", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    groups = {
        "eps": [("dynamic", n, {"eps": e}) for n in a.sizes for e in EPS_LIST],
        "static": [("static", n, {"L": L}) for n in a.sizes for L in STATIC_DEPTHS],
        "avqite": [("avqite", n, {"eps": EPS_MAIN}) for n in a.sizes],
        "adapt": [("adapt", n, {}) for n in a.sizes],
        "adaptnorm": [("adapt", n, {"normalize": True}) for n in a.sizes],
        "hz": [(k, a.hz_n, {"hx": 1.0, "hz": hz, "eps": EPS_MAIN, "L": L})
               for hz in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0)
               for k, L in [("dynamic", None)] + [("static", L) for L in (2, 3, 4, 5, 6, 8, 10, 12, 15, 20)]],
    }
    for name, specs in groups.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        specs = sorted(specs, key=lambda s: -s[1])
        with Pool(a.workers) as pool:
            res = pool.map(task, specs, chunksize=1)
        with open(os.path.join(OUT, f"mfim_{name}.json"), "w") as f:
            json.dump(res, f)
        print(f"[mfim_{name}] {len(res)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
