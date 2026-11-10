"""Overlap of the Hartree-Fock reference with the exact ground state of linear H4 (STO-3G) versus bond length
(paper, Sec. III A: 0.98 near equilibrium, 0.87 at 1.5 A, 0.69 at 2.0 A).

    python scripts/h4_overlap.py      # results/h4_overlap.json
"""
from __future__ import annotations
import os as _os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    _os.environ[_v] = "1"
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.models2 import molecule
from dynamic_vite.systems import finalize

BOND_LENGTHS = (0.9, 1.5, 2.0)      # Angstrom; 0.9 is close to the equilibrium bond length of the chain


def main():
    out = []
    for R in BOND_LENGTHS:
        S = molecule([("H", (0, 0, i * R)) for i in range(4)], name=f"H4_{R:g}")
        finalize(S)
        ov = float(abs(np.vdot(S.psi0, S.gs)))
        out.append(dict(bond_length=R, overlap=ov, weight=ov ** 2))
        print(f"R = {R:.1f} A: |<HF|gs>| = {ov:.3f}   weight = {ov ** 2:.3f}   outside HF = {1 - ov ** 2:.2f}")
    json.dump(out, open(os.path.join(os.path.dirname(__file__), "..", "results", "h4_overlap.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
