"""Size scaling on the 1D mixed-field Ising ring: dynamic VITE (eps = 0.9) against the stronger
baselines (ADAPT-VQE, TETRIS-ADAPT-VQE and AVQITE with the 2-local qubit pool), N = 8 ... 16.

Dynamic VITE for N <= 12 is taken from results/mfim_eps.json and the N = 12 baselines from
results/strong_baselines.json; this script runs the rest and writes results/scaling.json.

    python scripts/run_scaling.py --sizes 8 10 14
    python scripts/run_scaling.py --sizes 16 --skip adapt
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"          # single-threaded linear algebra: bitwise-reproducible results
import argparse, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_mfim, run_strong_baselines as rsb

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "scaling.json")


def task(spec):
    kind, n, p = spec
    if kind == "dynamic":
        d = run_mfim.task(("dynamic", n, {"eps": 0.9}))
        keep = ("method", "n", "E0", "energy", "n_layers", "circuits", "layers", "wall", "by_kind", "by_kind_hist")
        d = {k: d.get(k) for k in keep}
        d.update(system=f"mfim_{n}", kind="dynamic", params={"eps": 0.9})
        return d
    return rsb.task((f"mfim_{n}", kind, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", required=True)
    ap.add_argument("--skip", nargs="*", default=[], help="e.g. adapt (plain ADAPT-VQE)")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    specs = []
    for n in a.sizes:
        if n > 12:
            specs.append(("dynamic", n, {"eps": 0.9}))
        if n != 12:
            specs += [("avqite", n, {"pool": "qubit"}), ("adapt", n, {"pool": "qubit", "batch": "tetris"})]
            if "adapt" not in a.skip:
                specs.append(("adapt", n, {"pool": "qubit"}))
    specs.sort(key=lambda s: (-s[1], s[0] != "adapt"))          # largest and slowest first
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    with Pool(a.workers) as pool:
        for d in pool.imap_unordered(task, specs, chunksize=1):
            res = [r for r in res if not (r["system"] == d["system"] and r["kind"] == d["kind"] and r["params"] == d["params"])]
            res.append(d)
            json.dump(res, open(OUT, "w"))
            print(f"  {d['system']} {d['method']}: {len(d['layers'])} ops/layers, {d['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
