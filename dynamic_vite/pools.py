"""Operator pools for the stronger baselines (one Pauli string or one excitation per operator).

* qubit_pool_spin(H): all 1- and 2-local real generators on the interaction graph of H, i.e. Y_i on
  every site and Y_aZ_b, Z_aY_b, Y_aX_b, X_aY_b on every bond (a, b) (bonds = weight-2 terms of H).
  Strings with an odd number of Y are the real generators (exp(-i theta P) real orthogonal), which
  suffice for a real Hamiltonian and real reference. This is the 2-local qubit pool used for spin
  models with ADAPT-VQE and AVQITE; no symmetry is imposed.
* qeb_pool(ucc_pool, n): qubit-excitation-based (QEB) pool: each spin-conserving single or double
  excitation of the UCCSD pool with its Jordan-Wigner Z strings removed. Each operator conserves
  particle number (it is a qubit excitation), and is the pool of QEB-ADAPT-VQE.
"""
from __future__ import annotations
from .paulis import PauliSum, label_to_masks, weight
from .ansatz import Layer


def _bonds(H: PauliSum):
    out = set()
    for c, x, z in H.items():
        m = x | z
        if bin(m).count("1") == 2:
            a, b = [i for i in range(H.n) if (m >> i) & 1]
            out.add((a, b))
    return sorted(out)


def qubit_pool_spin(H: PauliSum):
    n = H.n
    pool = []
    for i in range(n):
        pool.append(Layer(f"Y{i}", PauliSum(n, [(1.0, *label_to_masks({i: "Y"}))])))
    for a, b in _bonds(H):
        for pa, pb in (("Y", "Z"), ("Z", "Y"), ("Y", "X"), ("X", "Y")):
            pool.append(Layer(f"{pa}{a}{pb}{b}", PauliSum(n, [(1.0, *label_to_masks({a: pa, b: pb}))])))
    return pool


def qeb_pool(ucc_pool, n):
    pool, seen = [], set()
    for L in ucc_pool:
        terms = {}
        for c, x, z in L.G.items():
            z2 = z & x                      # drop Z on sites without X/Y (the Jordan-Wigner strings)
            terms[(x, z2)] = terms.get((x, z2), 0.0) + c
        terms = [(c, x, z) for (x, z), c in terms.items() if abs(c) > 1e-12]
        key = tuple(sorted((x, z, round(c, 9)) for c, x, z in terms))
        if not terms or key in seen:
            continue
        seen.add(key)
        cm = max(abs(c) for c, _, _ in terms)
        pool.append(Layer("Q" + L.name, PauliSum(n, [(c / cm, x, z) for c, x, z in terms])))
    return pool


def support(layer: Layer) -> int:
    m = 0
    for _, x, z in layer.G.items():
        m |= x | z
    return m
