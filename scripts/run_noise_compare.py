"""Noisy comparison: dynamic VITE vs ADAPT-VQE (Adam, SPSA) under the same Gaussian shot-noise model.

    python scripts/run_noise_compare.py --workers 2 [--seeds 5]

Systems: critical TFIM ring N=8 and N=12 (schedule: ladder + 12 local layers; pool: the same layers),
mixed-field Ising ring N=8 at (J, hx, hz) = (-1, 1.05, 0.5) (schedule = pool = 40 commutator layers).
Shots per circuit S in SHOTS; SEEDS seeds per point. By default every cell uses all S and 5 seeds.
In the paper the TFIM N=12 SPSA cell is incomplete: runs that accumulated many operators took more
than 30 minutes of simulation each and were stopped; the results file contains S = 1e3, 1e4, 1e5
(5 seeds) and S = 1e7 (seed 1). Use --only-cell, --shots-slow and --seeds-slow to run that cell
separately (it writes to --out and skips runs already present in --keep). Hyperparameters of the ADAPT optimisers were
chosen on TFIM N=8, S=1e5 by scripts/tune_noisy_adapt.py (lowest median final error): Adam step 0.001 with
40 iterations per growth, SPSA first-step size 0.01 with 200 iterations per growth.
Writes results/noisecmp.json.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
from dynamic_vite.models import tfim, minus_state, free_fermion_ground_energy
from dynamic_vite.ansatz import ladder_schedule, local_schedule
from dynamic_vite.methods import vite, noisy_adapt_vqe, ShotNoise

SHOTS = (10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7)
ADAM = dict(optimizer="adam", lr=0.001, iters_per_growth=40)
SPSA = dict(optimizer="spsa", spsa_step=0.01, iters_per_growth=200, trust=0.1)
OUT = os.path.join(os.path.dirname(__file__), "..", "results")


def system(name, n):
    if name == "tfim":
        H = tfim(n)
        sched = ladder_schedule(n) + local_schedule(n)
        return H, H.to_sparse().tocsr(), free_fermion_ground_energy(n), minus_state(n), sched, 150, 3 * n
    import run_mfim as R
    H, Hm, E0, gs, psi0, sched, _ = R.system(n)
    return H, Hm, E0, psi0, sched, 100, 40


def task(spec):
    model, n, method, S, seed = spec
    H, Hm, E0, psi0, sched, steps, max_ops = system(model, n)
    sum_h2 = sum(abs(c) ** 2 for c, x, z in H.items())
    sum_hh2 = sum(abs(c) ** 2 for c, x, z in (H @ H).items() if (x | z))
    nz = ShotNoise(S, sum_h2, sum_hh2, np.random.default_rng(1000 * seed + int(np.log10(S))))
    if method == "dynamic":
        r = vite(H, Hm, E0, psi0, sched, mode="dynamic", eps=0.9, dtau=0.1, n_steps=steps, noise=nz,
                 method_name="Dynamic VITE")
    elif method == "dynamic_la":
        r = vite(H, Hm, E0, psi0, sched, mode="dynamic", eps=0.9, dtau=0.1, n_steps=steps, noise=nz,
                 lookahead=1, lookahead_every=4, method_name="Dynamic VITE (look-ahead)")
    elif method == "adam":
        r = noisy_adapt_vqe(H, Hm, E0, psi0, sched, nz, max_ops=max_ops, method_name="ADAPT-VQE (Adam)", **ADAM)
    else:
        r = noisy_adapt_vqe(H, Hm, E0, psi0, sched, nz, max_ops=max_ops, method_name="ADAPT-VQE (SPSA)", **SPSA)
    d = {k: v for k, v in r.__dict__.items() if k in ("method", "n", "E0", "energy", "circuits", "n_layers",
                                                        "layers", "wall")}
    d.update(model=model, shots=S, seed=seed, kind=method)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seeds-slow", type=int, default=5)
    ap.add_argument("--shots-slow", type=float, nargs="+", default=[float(x) for x in SHOTS])
    ap.add_argument("--out", default="noisecmp.json")
    ap.add_argument("--only-cell", nargs=3, default=None, metavar=("MODEL", "N", "METHOD"),
                    help="run only this cell, e.g. tfim 12 spsa")
    ap.add_argument("--keep", default=None, help="json with finished runs to keep (skipped here)")
    a = ap.parse_args()
    # cheap cells first; the expensive TFIM N=12 SPSA cell last and with fewer seeds
    specs = [(m, n, meth, S, s) for (m, n) in (("tfim", 8), ("mfim", 8), ("tfim", 12))
             for meth in ("dynamic", "dynamic_la", "adam", "spsa")
             for S in (tuple(int(x) for x in a.shots_slow) if (m, n, meth) == ("tfim", 12, "spsa") else SHOTS)
             for s in range(a.seeds_slow if (m, n, meth) == ("tfim", 12, "spsa") else a.seeds)]
    specs.sort(key=lambda sp: (sp[:3] == ("tfim", 12, "spsa"), sp[3]))
    if a.only_cell:
        cell = (a.only_cell[0], int(a.only_cell[1]), a.only_cell[2])
        specs = [sp for sp in specs if sp[:3] == cell]
    t0 = time.time()
    res = []
    if a.keep:
        res = json.load(open(a.keep))
        done = {(r["model"], r["n"], r["kind"], r["shots"], r["seed"]) for r in res}
        specs = [sp for sp in specs if sp not in done]
    with Pool(a.workers) as p:
        for d in p.imap_unordered(task, specs, chunksize=1):
            res.append(d)
            print(f"{len(res)} runs saved ({d['model']} N={d['n']} {d['kind']} S={d['shots']:g}), {time.time() - t0:.0f}s",
                  flush=True)
            json.dump(res, open(os.path.join(OUT, a.out), "w"))
    json.dump(res, open(os.path.join(OUT, a.out), "w"))
    print(f"[noisecmp] {len(res)} runs in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
