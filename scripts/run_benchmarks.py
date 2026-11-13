"""Run every experiment in the paper and write JSON results to results/.

    python scripts/run_benchmarks.py --sizes 6 8 10 12 14 16 --workers 2

Experiments
  main      dynamic VITE (commutator ladder, pool-free residual trigger) for each N
  bench     baselines for each N: static VITE (depth N/2 and N/2-1), energy-stagnation trigger,
            AVQITE (pool screening), ADAPT-VQE (pool, BFGS re-optimisation)
  eps       threshold sensitivity at one N
  ablation  dynamic VITE with the original local gate set as the fixed schedule
  field     dynamic VITE away from the critical point (h/J != 1) at one N
  lightcone growth window at dtau = 0.02 to resolve the growth times tau_r
  stall_tol tolerance sensitivity of the energy-stagnation baseline
  noise     Gaussian shot-noise model: dynamic VITE (with and without the noise floor, and with the
            variance-free look-ahead test) vs static
            depth-N/2 VITE over seeds
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
from dynamic_vite.models import tfim, minus_state, free_fermion_ground_energy, exact_ground_energy
from dynamic_vite.ansatz import ladder_schedule, local_schedule, operator_pool
from dynamic_vite.methods import vite, adapt_vqe, ShotNoise

J, H_FIELD = 1.0, 1.0
DTAU, TAU_MAX = 0.1, 15.0
EPS = 0.9
VAR_STOP = 1e-13
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def system(n, h=H_FIELD):
    H = tfim(n, J, h, periodic=True)
    Hm = H.to_sparse().tocsr()
    E0 = free_fermion_ground_energy(n, J, h)
    w, v = sla.eigsh(Hm, k=1, which="SA") if Hm.shape[0] > 64 else np.linalg.eigh(Hm.toarray())
    gs = v[:, 0]
    if Hm.shape[0] > 64:
        assert abs(w[0] - E0) < 1e-8, (w[0], E0)
    return H, Hm, E0, gs, minus_state(n)


def run_to_dict(r):
    d = dict(r.__dict__)
    d["events"] = [list(e) for e in r.events]
    return d


def task(spec):
    kind, n, params = spec
    H, Hm, E0, gs, psi0 = system(n, params.get("h", H_FIELD))
    steps = int(round(TAU_MAX / DTAU))
    # Schedule: the ladder G_1..G_{N/2}, followed by the 12 local shared layers. Ladder layers beyond
    # N/2 would be redundant (G_{N-r} = P G_r with P = prod X = +1 on the even-parity sector), so the
    # extension uses genuinely different layers to test whether the trigger stops by itself.
    long_ladder = ladder_schedule(n) + local_schedule(n)
    if kind == "dynamic":
        dt = params.get("dtau", DTAU)
        r = vite(H, Hm, E0, psi0, long_ladder, mode="dynamic", eps=params.get("eps", EPS),
                 dtau=dt, n_steps=int(round(params.get("tau_max", TAU_MAX) / dt)), gs=gs, var_stop=VAR_STOP,
                 method_name=params.get("name", "Dynamic VITE"))
    elif kind == "static":
        depth = params["depth"]
        r = vite(H, Hm, E0, psi0, ladder_schedule(n, max_range=depth), mode="static", dtau=DTAU,
                 n_steps=steps, gs=gs, var_stop=VAR_STOP, method_name=params["name"])
    elif kind == "stall":
        r = vite(H, Hm, E0, psi0, long_ladder, mode="stall", dtau=DTAU, n_steps=steps, gs=gs,
                 var_stop=VAR_STOP, stall_tol=params.get("tol", 1e-4), method_name="Stagnation trigger")
    elif kind == "avqite":
        r = vite(H, Hm, E0, psi0, long_ladder, mode="avqite", pool=operator_pool(n), eps=EPS,
                 dtau=DTAU, n_steps=steps, gs=gs, var_stop=VAR_STOP, method_name="AVQITE")
    elif kind == "adapt":
        r = adapt_vqe(H, Hm, E0, psi0, operator_pool(n), gs=gs, grad_tol=1e-4,
                      jitter=params.get("jitter", 0.0), seed=params.get("seed"))
    elif kind == "local":
        r = vite(H, Hm, E0, psi0, local_schedule(n), mode="dynamic", eps=EPS, dtau=DTAU,
                 n_steps=steps, gs=gs, var_stop=VAR_STOP, method_name="Dynamic VITE (local set)")
    elif kind in ("noisy_dynamic", "noisy_static", "noisy_nofloor", "noisy_lookahead"):
        S, seed = params["shots"], params["seed"]
        sum_h2 = sum(abs(c) ** 2 for c, x, z in H.items())
        sum_hh2 = sum(abs(c) ** 2 for c, x, z in (H @ H).items() if (x | z))
        nz = ShotNoise(S, sum_h2, sum_hh2, np.random.default_rng(seed))
        if kind == "noisy_lookahead":
            r = vite(H, Hm, E0, psi0, long_ladder, mode="dynamic", eps=EPS, dtau=DTAU, n_steps=steps,
                     gs=gs, noise=nz, lookahead=1, lookahead_every=4, method_name="Dynamic VITE (look-ahead)")
        elif kind in ("noisy_dynamic", "noisy_nofloor"):
            r = vite(H, Hm, E0, psi0, long_ladder, mode="dynamic", eps=EPS, dtau=DTAU, n_steps=steps,
                     gs=gs, noise=nz, kappa=0.0 if kind == "noisy_nofloor" else 3.0,
                     method_name="Dynamic VITE" + (" (no floor)" if kind == "noisy_nofloor" else ""))
        else:
            r = vite(H, Hm, E0, psi0, ladder_schedule(n), mode="static", dtau=DTAU, n_steps=steps,
                     gs=gs, noise=nz, method_name="Static VITE (L=N/2)")
    else:
        raise ValueError(kind)
    d = run_to_dict(r)
    d["params"] = params
    d["kind"] = kind
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[6, 8, 10, 12, 14, 16])
    ap.add_argument("--eps-n", type=int, default=12)
    ap.add_argument("--noise-n", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    groups = {}
    groups["main"] = [("dynamic", n, {}) for n in a.sizes]
    groups["bench"] = []
    for n in a.sizes:
        groups["bench"] += [("static", n, {"depth": n // 2, "name": "Static VITE (L=N/2)"}),
                            ("static", n, {"depth": n // 2 - 1, "name": "Static VITE (L=N/2-1)"}),
                            ("stall", n, {}), ("avqite", n, {}), ("adapt", n, {})]
    groups["eps"] = [("dynamic", a.eps_n, {"eps": e, "name": f"eps={e}"})
                     for e in (0.01, 0.03, 0.1, 0.3, 0.6, 0.9)]
    groups["lightcone"] = [("dynamic", n, {"dtau": 0.02, "tau_max": 4.0, "name": "lightcone"}) for n in a.sizes]
    groups["stall_tol"] = [("stall", a.eps_n, {"tol": t}) for t in (1e-3, 1e-4, 1e-5)]
    groups["field"] = [("dynamic", a.eps_n, {"h": h, "name": f"h={h}"}) for h in (0.3, 0.6, 1.0, 1.5, 3.0)]
    groups["ablation"] = [("local", n, {}) for n in a.sizes if n <= 12]
    groups["noise"] = [(k, a.noise_n, {"shots": S, "seed": s})
                       for S in (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)
                       for s in range(a.seeds) for k in ("noisy_dynamic", "noisy_static", "noisy_nofloor", "noisy_lookahead")]

    for name, specs in groups.items():
        if a.only and name not in a.only:
            continue
        t0 = time.time()
        specs = sorted(specs, key=lambda s: -s[1])           # big systems first
        with Pool(a.workers) as p:
            res = p.map(task, specs, chunksize=1)
        with open(os.path.join(OUT, f"{name}.json"), "w") as f:
            json.dump(res, f)
        print(f"[{name}] {len(res)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
