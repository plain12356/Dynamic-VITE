"""Nearest-neighbour transverse-field Ising model and reference states."""
from __future__ import annotations
import numpy as np
import scipy.sparse.linalg as sla
from .paulis import PauliSum, label_to_masks


def tfim(n: int, J: float = 1.0, h: float = 1.0, periodic: bool = True) -> PauliSum:
    """H = J sum_j Z_j Z_{j+1} + h sum_j X_j  (critical at |h/J| = 1)."""
    H = PauliSum(n)
    bonds = n if periodic else n - 1
    for j in range(bonds):
        H.add(J, *label_to_masks({j: "Z", (j + 1) % n: "Z"}))
    for j in range(n):
        H.add(h, *label_to_masks({j: "X"}))
    return H


def tfim_parts(n: int, J: float = 1.0, h: float = 1.0, periodic: bool = True):
    Hzz, Hx = PauliSum(n), PauliSum(n)
    for j in range(n if periodic else n - 1):
        Hzz.add(J, *label_to_masks({j: "Z", (j + 1) % n: "Z"}))
    for j in range(n):
        Hx.add(h, *label_to_masks({j: "X"}))
    return Hzz, Hx


def exact_ground_energy(H: PauliSum) -> float:
    Hs = H.to_sparse()
    if Hs.shape[0] <= 64:
        return float(np.linalg.eigvalsh(Hs.toarray())[0])
    return float(sla.eigsh(Hs, k=1, which="SA")[0][0])


def exact_gap(H: PauliSum, sector_state: np.ndarray | None = None) -> float:
    """Gap between the two lowest levels of the full spectrum."""
    Hs = H.to_sparse()
    if Hs.shape[0] <= 64:
        ev = np.linalg.eigvalsh(Hs.toarray())
    else:
        ev = np.sort(sla.eigsh(Hs, k=4, which="SA")[0])
    return float(ev[1] - ev[0])


def free_fermion_ground_energy(n: int, J: float = 1.0, h: float = 1.0) -> float:
    """Closed form for the periodic chain, even n, even-parity sector (antiperiodic fermions).

    E0 = -sum_k sqrt(J^2 + h^2 + 2 J h cos k),  k = pi(2m+1)/n, m = 0..n-1.
    The sign of J does not matter on an even ring (sublattice rotation).
    """
    k = np.pi * (2 * np.arange(n) + 1) / n
    return float(-np.sum(np.sqrt(J ** 2 + h ** 2 + 2 * abs(J) * h * np.cos(k))))


def minus_state(n: int) -> np.ndarray:
    """|-->^n, the ground state of the field term h sum X (h > 0). Real amplitudes."""
    minus = np.array([1.0, -1.0]) / np.sqrt(2)
    psi = np.array([1.0])
    for _ in range(n):
        psi = np.kron(minus, psi)
    return psi.astype(complex)


def mfim(n: int, J: float = -1.0, hx: float = 1.05, hz: float = 0.5, periodic: bool = True) -> PauliSum:
    """Mixed-field Ising ring H = J sum Z_j Z_{j+1} + hx sum X_j + hz sum Z_j (non-integrable for hz != 0).

    Default (J, hx, hz) = (-1, 1.05, 0.5) is the strongly non-integrable point of Banuls, Cirac and
    Hastings, PRL 106, 050405 (2011), up to the sign of hz (a global spin flip, prod_j X_j).
    """
    H = tfim(n, J, hx, periodic)
    for j in range(n):
        if hz:
            H.add(hz, *label_to_masks({j: "Z"}))
    return H


def field_parent(n: int, hx: float = 1.0) -> PauliSum:
    """H0 = hx sum X_j; its ground state is |->^n for hx > 0."""
    H0 = PauliSum(n)
    for j in range(n):
        H0.add(hx, *label_to_masks({j: "X"}))
    return H0
