"""Force on each queued layer at a stalled circuit (paper, Sec. III.G, 'remaining limit').

    python scripts/diagnose_stall.py      # writes results/stall_forces.csv

Mixed-field ring N=8: the first L scheduled layers (L = 3, 4, 5) are integrated noiselessly for 80
steps (dtau = 0.1) to their stalled state; for each of the next 8 scheduled layers the orthogonalised
force g_c is compared with its sampling std at S = 1e7 shots per circuit (V_c noise only).
"""
import csv, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import run_mfim as R
from dynamic_vite.methods import ShotNoise
from dynamic_vite.geometry import snapshot, candidate_column, pinv_psd

H, Hm, E0, gs, psi0, sched, _ = R.system(8)
sh2 = sum(abs(c) ** 2 for c, x, z in H.items())
nz = ShotNoise(10 ** 7, sh2, 0.0, np.random.default_rng(0))
rows = []
for L in (3, 4, 5):
    th = np.zeros(L)
    for _ in range(80):
        s = snapshot(sched[:L], th, psi0, Hm)
        th = th + 0.1 * pinv_psd(s.M) @ s.V
    s = snapshot(sched[:L], th, psi0, Hm)
    A = pinv_psd(s.M)
    for j, c in enumerate(sched[L:L + 8], start=1):
        mc, mcc, vc = candidate_column(s, c)
        g = vc - mc @ A @ s.V
        rows.append([L, (s.E - E0) / abs(E0), s.rho(), j, c.name, abs(g) / nz.column_stds(sched[:L], c)[2]])
out = os.path.join(os.path.dirname(__file__), "..", "results", "stall_forces.csv")
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["stalled_layers", "rel_error", "rho", "queue_position", "layer", "force_over_sigma"])
    w.writerows(rows)
for r in rows:
    if r[0] == 4:
        print(r[3], r[4], f"{r[5]:.1f} sigma")
