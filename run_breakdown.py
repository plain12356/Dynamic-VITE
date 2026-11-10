"""Re-runs the main comparison on selected systems with per-step cost history (by_kind_hist), for the
cost-breakdown figure. Totals are identical to the original runs (checked by this script).

    python scripts/run_breakdown.py      # writes results/breakdown.json
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"          # single-threaded linear algebra: bitwise-reproducible results
import json, os, sys
from multiprocessing import Pool
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import run_mfim, run_models

RES = os.path.join(os.path.dirname(__file__), "..", "results")
SPECS = ([("mfim_12", k, p) for k, p in (("dynamic", {"eps": 0.9}), ("avqite", {"eps": 0.9}), ("adapt", {}))] +
         [("h4_2.0", k, p) for k, p in (("dynamic", {"eps": 0.9}), ("avqite", {}), ("adapt", {}),
                                          ("adapt", {"pool": "uccsd", "max_ops": 40}))])


def task(spec):
    sysname, kind, p = spec
    if sysname == "mfim_12":
        d = run_mfim.task((kind, 12, p))
    else:
        d = run_models.task((sysname, kind, p))
    keep = ("method", "n", "E0", "energy", "circuits", "n_layers", "by_kind", "by_kind_hist", "params", "kind")
    out = {k: d.get(k) for k in keep}
    out["system"] = sysname
    out["kind"], out["params"] = kind, p
    return out


if __name__ == "__main__":
    with Pool(2) as pool:
        res = pool.map(task, SPECS, chunksize=1)
    json.dump(res, open(os.path.join(RES, "breakdown.json"), "w"))
    print("done", len(res))
