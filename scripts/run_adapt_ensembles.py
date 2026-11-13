"""Seeded ensembles for ADAPT-type runs (paper, Sec. III B and Appendix E).

The number of BFGS evaluations ADAPT-VQE needs to reach its gradient tolerance is sensitive to
rounding-level differences, which multithreaded linear algebra introduces from run to run. Each
configuration is therefore run with SEEDS seeded perturbations of size JITTER (radians) added to the
BFGS starting point at every re-optimization; this keeps the operator sequence unchanged and samples
the spread of the cost. Tables use the median over seeds; figures use the run whose cost to reach
1e-4 is the median. Writes results/adapt_ensembles.json.

    python scripts/run_adapt_ensembles.py --group h4
    python scripts/run_adapt_ensembles.py --group spin --seeds 3
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"          # single-threaded linear algebra: bitwise-reproducible results
import argparse, json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import run_models, run_strong_baselines as rsb, run_scaling

RES = os.path.join(os.path.dirname(__file__), "..", "results")
OUT = os.path.join(RES, "adapt_ensembles.json")
JITTER = 1e-12
GROUPS = {
    "h4": [(src, g, kind, p) for g in ("h4_1.5", "h4_2.0") for src, kind, p in (
        ("models", "adapt", {}), ("models", "adapt", {"pool": "uccsd", "max_ops": 40}),
        ("strong", "adapt", {"pool": "qeb"}), ("strong", "adapt", {"pool": "qeb", "batch": "tetris"}),
        ("strong", "adapt", {"pool": "uccsd", "batch": "tetris"}))],
    # every spin-model ADAPT-type entry of Tables III and IV (Hubbard L=6 excluded: capped, never reaches its target)
    "tables": ([("models", g, "adapt", {}) for g in ("mfim2d_3x4", "tfim2d_3x4", "mfim2d_3x3", "tfim2d_3x3", "hubbard_4")] +
               [("mfim", "mfim_12", "adapt", {}), ("bench", "tfim1d_12", "adapt", {})] +
               [("strong", g, "adapt", p) for g in ("mfim2d_3x4", "mfim_12", "mfim2d_3x3")
                for p in ({"pool": "qubit"}, {"pool": "qubit", "batch": "tetris"})]),
    "spin": [("models", "mfim2d_3x3", "adapt", {}),
             ("strong", "mfim2d_3x3", "adapt", {"pool": "qubit", "batch": "tetris"}),
             ("strong", "mfim_12", "adapt", {"pool": "qubit", "batch": "tetris"})],
}


def task(spec):
    src, name, kind, p, seed = spec
    q = dict(p, jitter=JITTER, seed=seed)
    if src == "models":
        d = run_models.task((name, kind, q))
    elif src == "strong":
        d = rsb.task((name, kind, q))
    elif src == "mfim":
        import run_mfim
        d = run_mfim.task((kind, int(name.split("_")[1]), q))
    elif src == "bench":
        import run_benchmarks
        d = run_benchmarks.task((kind, int(name.split("_")[1]), q))
    else:
        raise ValueError(src)
    keep = ("method", "n", "E0", "energy", "n_layers", "circuits", "layers", "wall", "by_kind", "by_kind_hist")
    out = {k: d.get(k) for k in keep}
    out.update(src=src, system=name, kind=kind, params=p, seed=seed)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True, choices=sorted(GROUPS))
    ap.add_argument("--seeds", type=int, default=7)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    specs = [(src, g, kind, p, s) for src, g, kind, p in GROUPS[a.group] for s in range(a.seeds)]
    res = json.load(open(OUT)) if os.path.exists(OUT) else []
    done = {(r["src"], r["system"], r["kind"], json.dumps(r["params"], sort_keys=True), r["seed"]) for r in res}
    specs = [sp for sp in specs if (sp[0], sp[1], sp[2], json.dumps(sp[3], sort_keys=True), sp[4]) not in done]
    print(f"{len(specs)} runs to do", flush=True)          # resumes after an interruption
    with Pool(a.workers) as pool:
        for d in pool.imap_unordered(task, specs, chunksize=1):
            res = [r for r in res if not (r["system"] == d["system"] and r["kind"] == d["kind"] and
                                          r["params"] == d["params"] and r["seed"] == d["seed"] and r["src"] == d["src"])]
            res.append(d)
            json.dump(res, open(OUT, "w"))
            print(f"  {d['system']} {d['method']} seed {d['seed']}: {d['wall']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
