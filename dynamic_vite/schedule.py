"""Automatic, measurement-free growth schedules from nested commutators.

For a real Hamiltonian H (no Y operators) and an initial state that is the ground state of a
real "parent" Hamiltonian H0, the operators

    C_1 = i [H, H0],        C_{n+1} = -[H, [H, C_n]]

are Hermitian with purely imaginary matrix elements (odd number of Y per Pauli string), so
exp(-i theta C) is real orthogonal and keeps real states real. C_n collects the directions the
imaginary-time flow e^{-tau H} opens at successive orders. Each C_n is split into orbits of
Pauli strings under the symmetry group of H (here translations and reflections of a ring), and
every orbit not seen at an earlier level becomes one shared-parameter layer

    G_o = sum_{P in orbit o} P .

Layers are ordered by (level, support, -|coefficient|, canonical string). The order is fixed
before the run and uses no measurement; the residual trigger only decides when the next layer
is appended.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .paulis import PauliSum
from .ansatz import Layer


# ----------------------------------------------------------------------------- ring symmetry
def _rot(m: int, n: int, k: int) -> int:
    k %= n
    full = (1 << n) - 1
    return ((m << k) | (m >> (n - k))) & full


def _refl(m: int, n: int) -> int:
    out = 0
    for j in range(n):
        if (m >> j) & 1:
            out |= 1 << ((-j) % n)
    return out


def dihedral_images(x: int, z: int, n: int, reflect: bool = True):
    imgs = []
    for xx, zz in ([(x, z), (_refl(x, n), _refl(z, n))] if reflect else [(x, z)]):
        for k in range(n):
            imgs.append((_rot(xx, n, k), _rot(zz, n, k)))
    return imgs


def canonical(x: int, z: int, n: int, reflect: bool = True):
    return min(dihedral_images(x, z, n, reflect))


def arc_support(x: int, z: int, n: int) -> int:
    """Length of the shortest arc of the ring that contains the support of the string."""
    occ = [(((x | z) >> j) & 1) for j in range(n)]
    sites = [j for j in range(n) if occ[j]]
    if not sites:
        return 0
    if len(sites) == 1:
        return 1
    gaps = [(sites[(i + 1) % len(sites)] - sites[i]) % n for i in range(len(sites))]
    return n - max(gaps) + 1


def orbit_sum(x: int, z: int, n: int, reflect: bool = True) -> PauliSum:
    G = PauliSum(n)
    for xx, zz in set(dihedral_images(x, z, n, reflect)):
        G.add(1.0, xx, zz)
    return G


# ----------------------------------------------------------------------------- commutators
def commutator(A: PauliSum, B: PauliSum, tol: float = 1e-12) -> PauliSum:
    """[A, B] using anticommutation of Pauli strings (only anticommuting pairs contribute)."""
    out = PauliSum(A.n)
    for ca, xa, za in A.items():
        for cb, xb, zb in B.items():
            # P_a P_b = -P_b P_a  iff  symplectic product is odd
            if (bin(xa & zb).count("1") + bin(za & xb).count("1")) % 2 == 0:
                continue
            ph, x, z = PauliSum._mul_strings(xa, za, xb, zb)
            out.add(2 * ca * cb * ph, x, z)
    return PauliSum(A.n, [(c, x, z) for c, x, z in out.items() if abs(c) > tol])


def _truncate(C: PauliSum, max_support: int, rel_tol: float) -> PauliSum:
    if not len(C):
        return C
    cmax = max(abs(c) for c, _, _ in C.items())
    keep = [(c / cmax, x, z) for c, x, z in C.items()
            if abs(c) > rel_tol * cmax and arc_support(x, z, C.n) <= max_support]
    return PauliSum(C.n, keep)


@dataclass
class ScheduleEntry:
    level: int
    support: int
    weight: float
    canon: tuple
    label: str


def _label(x, z, n):
    """String read along the shortest arc containing its support (site order, I for identity)."""
    best = None
    for k in range(n):
        xx, zz = _rot(x, n, k), _rot(z, n, k)
        s = "".join("I" if not (((xx | zz) >> j) & 1) else
                    ("X" if (xx >> j) & 1 and not (zz >> j) & 1 else ("Z" if not (xx >> j) & 1 else "Y"))
                    for j in range(n)).strip("I")
        if best is None or (len(s), s) < (len(best), best):
            best = s
    return best


def commutator_schedule(H: PauliSum, H0: PauliSum, *, max_layers: int = 40, max_levels: int = 8,
                        max_support: int | None = None, rel_tol: float = 1e-9,
                        reflect: bool = True, return_entries: bool = False):
    """Build the fixed layer sequence (see module docstring)."""
    n = H.n
    max_support = max_support if max_support is not None else n - 1
    C = commutator(H, H0).scale(1j)
    C = _truncate(PauliSum(n, [(np.real(c), x, z) for c, x, z in C.items()]), max_support, rel_tol)
    seen: set = set()
    entries: list[ScheduleEntry] = []
    for level in range(1, max_levels + 1):
        new = {}
        for c, x, z in C.items():
            if bin(x & z).count("1") % 2 == 0:
                raise AssertionError("nested commutator produced a non-real generator")
            key = canonical(x, z, n, reflect)
            if key in seen:
                continue
            new[key] = max(new.get(key, 0.0), abs(c))
        lev = sorted(new.items(), key=lambda kv: (arc_support(*kv[0], n), -round(kv[1], 9), kv[0]))
        for key, w in lev:
            seen.add(key)
            entries.append(ScheduleEntry(level, arc_support(*key, n), w, key, _label(*key, n)))
        if len(entries) >= max_layers or level == max_levels:
            break
        C = commutator(H, commutator(H, C)).scale(-1)
        C = _truncate(PauliSum(n, [(np.real(c), x, z) for c, x, z in C.items()]), max_support, rel_tol)
        if not len(C):
            break
    entries = entries[:max_layers]
    layers = [Layer(f"L{k + 1}:{e.label}", orbit_sum(*e.canon, n, reflect)) for k, e in enumerate(entries)]
    return (layers, entries) if return_entries else layers
