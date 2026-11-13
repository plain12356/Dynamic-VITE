"""Benchmark systems used in the paper, with their schedules (cached per process)."""
from __future__ import annotations
from functools import lru_cache
from .models2 import tfim_2d, hubbard_chain, molecule, finalize, uccsd_pool
from .schedule_general import general_schedule, symmetry_group

SCHED_LEN = {"tfim2d_3x3": 40, "tfim2d_3x4": 40, "mfim2d_3x3": 40, "mfim2d_3x4": 40, "hubbard_4": 40, "hubbard_6": 80, "hubbardT_4": 40,
             "hubbardT_6": 40, "h4_1.5": 40, "h4_2.0": 40}


@lru_cache(maxsize=None)
def get_system(name: str, sched_len: int | None = None):
    """name: 'tfim2d_3x3', 'tfim2d_3x4', 'hubbard_4', 'hubbard_6', 'h4_1.5', 'h4_2.0'."""
    kind, par = name.split("_", 1)
    order = "support"
    if kind == "tfim2d":
        Lx, Ly = map(int, par.split("x"))
        S = tfim_2d(Lx, Ly)
    elif kind == "mfim2d":                      # 'mfim2d_3x3' (h_z = 0.5) or 'mfim2d_3x3_hz0.25'
        lat, _, hzs = par.partition("_hz")
        Lx, Ly = map(int, lat.split("x"))
        S = tfim_2d(Lx, Ly, hz=float(hzs) if hzs else 0.5)
    elif kind in ("hubbard", "hubbardT"):
        S = hubbard_chain(int(par), U=4.0)
    elif kind == "h4":
        R = float(par)
        S = molecule([("H", (0, 0, i * R)) for i in range(4)], name=f"H4_{R:g}")
        order = "weight"
    else:
        raise ValueError(name)
    Hm = finalize(S)
    G, gens = symmetry_group(S.H, S.H0, S.candidate_gens) if S.candidate_gens else ([tuple(range(S.H.n))], [])
    K = sched_len or SCHED_LEN.get(name, 40)
    grouping = S.extra.get("grouping") if kind == "hubbardT" else None
    sched, entries = general_schedule(S.H, S.H0, G, merge_u1=S.merge_u1, max_layers=K,
                                      return_entries=True, order=order, grouping=grouping)
    pool_ucc = uccsd_pool(S.extra["nel"], S.H.n) if kind == "h4" else None
    return S, Hm, sched, entries, len(G), pool_ucc
