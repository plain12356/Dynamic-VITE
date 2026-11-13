"""Symmetry-reduced cost of dynamic VITE on the mixed-field ring (paper, Sec. IV C, Fig. 3).

Every scheduled layer is a sum over a dihedral orbit, so the state stays dihedral-symmetric and
symmetry-related entries are equal: M_ij needs one circuit per orbit of string pairs (P, Q),
P in G_i, Q in G_j, under simultaneous group action, and V_l one per orbit of strings in G_l
(times g(H)), instead of |G_i||G_j| and |G_l| g(H). This script replays the recorded layer sequence
of each run, checks that the unreduced replay reproduces the recorded M and V circuit totals
exactly, and returns the reduced cumulative cost.
"""
from __future__ import annotations
import os, sys
from functools import lru_cache
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))
import run_mfim
from dynamic_vite.models import mfim, field_parent
from dynamic_vite.schedule import commutator_schedule
from dynamic_vite.schedule_general import permute_mask, closure
from dynamic_vite.cost import CostModel


@lru_cache(maxsize=None)
def setup(n):
    H = mfim(n, run_mfim.J, run_mfim.HX, run_mfim.HZ)
    sched = commutator_schedule(H, field_parent(n, run_mfim.HX), max_layers=run_mfim.SCHED_LEN, max_levels=8)
    G = closure([tuple((i + 1) % n for i in range(n)), tuple((-i) % n for i in range(n))], n)
    strings = [[(x, z) for _, x, z in L.G.items()] for L in sched]
    return sched, G, strings, CostModel.for_hamiltonian(H).gH


def _canon(G, pair):
    return min(tuple((permute_mask(x, g), permute_mask(z, g)) for x, z in pair) for g in G)


@lru_cache(maxsize=None)
def pair_orbits(n, i, j):
    _, G, S, _ = setup(n)
    return len({_canon(G, (a, b)) for a in S[i] for b in S[j]})


@lru_cache(maxsize=None)
def string_orbits(n, l):
    _, G, S, _ = setup(n)
    return len({_canon(G, (a,)) for a in S[l]})


def tri(n, L, reduced):
    _, _, S, _ = setup(n)
    if reduced:
        return sum(pair_orbits(n, i, j) for i in range(L) for j in range(i, L))
    return sum(len(S[i]) * len(S[j]) for i in range(L) for j in range(i, L))


def vsum(n, L, reduced):
    _, _, S, gH = setup(n)
    return gH * sum(string_orbits(n, l) if reduced else len(S[l]) for l in range(L))


def reduced_cost(run, upto=None):
    """Cumulative circuits of a dynamic-VITE run with symmetry-reduced M and V, at each recorded step
    up to index `upto`. Checks the unreduced replay against the run's recorded totals."""
    n = run["n"]
    Ls = run["n_layers"]
    K = len(Ls) if upto is None else upto + 1
    if upto is None or upto == len(Ls) - 1:
        Mraw = sum(tri(n, L, False) for L in Ls)
        Vraw = sum(vsum(n, L, False) for L in Ls)
        assert abs(Mraw - run["by_kind"]["M"]) < 0.5 and abs(Vraw - run["by_kind"]["V"]) < 0.5, \
            (n, Mraw, run["by_kind"]["M"], Vraw, run["by_kind"]["V"])
    out, cM, cV, rM, rV = [], 0, 0, 0, 0
    for k in range(K):
        L = Ls[k]
        cM += tri(n, L, False); cV += vsum(n, L, False)
        rM += tri(n, L, True); rV += vsum(n, L, True)
        out.append(run["circuits"][k] - cM - cV + rM + rV)
    return out
