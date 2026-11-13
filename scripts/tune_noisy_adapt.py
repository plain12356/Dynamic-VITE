"""Hyperparameter choice for noisy ADAPT-VQE (paper, Sec. II.E): TFIM N=8, S=1e5, two seeds.

    python scripts/tune_noisy_adapt.py      # writes results/tune_noisy_adapt.csv

Adam step and SPSA first-step size from {0.003, 0.01, 0.03, 0.1}; the setting with the lowest median
final error (median of the last 20 recorded steps) is used in run_noise_compare.py.
"""
import csv, os, sys, time
from multiprocessing import Pool
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.models import tfim, minus_state, free_fermion_ground_energy
from dynamic_vite.ansatz import operator_pool
from dynamic_vite.methods import noisy_adapt_vqe, ShotNoise

N, S = 8, 10 ** 5
H = tfim(N); E0 = free_fermion_ground_energy(N); Hm = H.to_sparse().tocsr(); psi0 = minus_state(N)
SH2 = sum(abs(c) ** 2 for c, x, z in H.items())
SHH2 = sum(abs(c) ** 2 for c, x, z in (H @ H).items() if (x | z))
GRID = (0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)


def job(spec):
    opt, par, seed = spec
    kw = dict(lr=par, iters_per_growth=40) if opt == "adam" else dict(spsa_step=par, iters_per_growth=200, trust=0.1)
    r = noisy_adapt_vqe(H, Hm, E0, psi0, operator_pool(N), ShotNoise(S, SH2, SHH2, np.random.default_rng(100 + seed)),
                        optimizer=opt, max_ops=3 * N, **kw)
    e = (np.median(r.energy[-20:]) - E0) / abs(E0)
    return opt, par, seed, e, len(r.layers), r.circuits[-1]


if __name__ == "__main__":
    specs = [(o, p, s) for o in ("adam", "spsa") for p in GRID for s in range(2)]
    with Pool(2) as P:
        rows = P.map(job, specs, chunksize=1)
    out = os.path.join(os.path.dirname(__file__), "..", "results", "tune_noisy_adapt.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["optimizer", "step", "seed", "final_rel_error", "operators", "circuits"])
        w.writerows(rows)
    for o in ("adam", "spsa"):
        med = {p: np.median([r[3] for r in rows if r[0] == o and r[1] == p]) for p in GRID}
        print(o, {p: f"{v:.1e}" for p, v in med.items()}, "best:", min(med, key=med.get))
