"""Shared-parameter layers, fixed growth schedules and operator pools.

Every layer is U_l(theta_l) = exp(-i theta_l G_l), where G_l is a translation-invariant sum of
Pauli strings sharing one real parameter. The circuit is U(theta) = U_L ... U_1 acting on |psi0>.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply
from .paulis import PauliSum, label_to_masks, weight


@dataclass
class Layer:
    name: str
    G: PauliSum
    mat: sp.csr_matrix = field(repr=False, default=None)

    def __post_init__(self):
        if self.mat is None:
            self.mat = self.G.to_sparse().tocsr()
        self.n_strings = len(self.G)
        self.weights = [weight(x, z) for _, x, z in self.G.items()]
        # CNOTs for one first-order product of string exponentials (CNOT ladder: 2(w-1) per string)
        self.cnot = int(sum(2 * (w - 1) for w in self.weights))
        # Real generator: exp(-i theta G) is real orthogonal, keeps real states real
        self.real = self.G.is_real_antisymmetric_generator()
        if self.real:
            self.A = (-1j * self.mat).real.tocsr()   # exp(-i theta G) = exp(theta A), A real antisymmetric
        # single Pauli string G = c P: exp(-i theta G) = cos(c theta) - i sin(c theta) P exactly (P^2 = 1)
        self.single_c = float(np.real(next(iter(self.G.items()))[0])) if self.n_strings == 1 else None

    def apply(self, theta: float, psi: np.ndarray) -> np.ndarray:
        if theta == 0.0:
            return psi
        if self.single_c is not None:
            c = self.single_c
            return np.cos(c * theta) * psi - 1j * np.sin(c * theta) / c * (self.mat @ psi)
        if self.real and np.all(np.abs(psi.imag) < 1e-15):
            return expm_multiply(theta * self.A, psi.real).astype(complex)
        return expm_multiply(-1j * theta * self.mat, psi)

    def gen(self, psi: np.ndarray) -> np.ndarray:
        """-i G psi"""
        return -1j * (self.mat @ psi)


# ----------------------------------------------------------------------------- generators
def ladder_generator(n: int, r: int, periodic: bool = True) -> PauliSum:
    """G_r = sum_j ( Y_j X_{j+1..j+r-1} Z_{j+r} + Z_j X_{j+1..j+r-1} Y_{j+r} ).

    G_1 = -(1/2) i [H_ZZ, H_X] / (J h): the first imaginary-time direction from |-->^n.
    The G_r are quadratic in Jordan-Wigner fermions, real (one Y per string) and mutually commuting.
    """
    G = PauliSum(n)
    sites = range(n) if periodic else range(n - r)
    for j in sites:
        mid = {(j + k) % n: "X" for k in range(1, r)}
        a = dict(mid); a[j] = "Y"; a[(j + r) % n] = "Z"
        b = dict(mid); b[j] = "Z"; b[(j + r) % n] = "Y"
        G.add(1.0, *label_to_masks(a))
        G.add(1.0, *label_to_masks(b))
    return G


def local_generator(n: int, label: str, periodic: bool = True) -> PauliSum:
    """Translation-invariant shared generator sum_j P_j (one letter) or sum_j P_j Q_{j+1} (two letters)."""
    G = PauliSum(n)
    if len(label) == 1:
        for j in range(n):
            G.add(1.0, *label_to_masks({j: label}))
    else:
        for j in range(n if periodic else n - 1):
            G.add(1.0, *label_to_masks({j: label[0], (j + 1) % n: label[1]}))
    return G


LOCAL_ORDER = ["X", "Y", "Z", "XX", "YY", "ZZ", "XY", "YX", "XZ", "ZX", "YZ", "ZY"]


# ----------------------------------------------------------------------------- schedules
def ladder_schedule(n: int, max_range: int | None = None, periodic: bool = True) -> list[Layer]:
    """Fixed pool-free schedule G_1, G_2, ..., G_{n/2} (by default)."""
    rmax = max_range if max_range is not None else n // 2
    return [Layer(f"G{r}", ladder_generator(n, r, periodic)) for r in range(1, rmax + 1)]


def local_schedule(n: int, periodic: bool = True) -> list[Layer]:
    """Ablation: the original manuscript's local gate set, in its fixed order."""
    return [Layer(lab, local_generator(n, lab, periodic)) for lab in LOCAL_ORDER]


def hva_schedule(n: int, p: int, Hzz: PauliSum, Hx: PauliSum) -> list[Layer]:
    """Ablation: Hamiltonian-variational schedule ZZ, X, ZZ, X, ... (p blocks)."""
    out = []
    mzz, mx = Hzz.to_sparse().tocsr(), Hx.to_sparse().tocsr()
    for k in range(p):
        out.append(Layer(f"ZZ{k + 1}", Hzz, mzz))
        out.append(Layer(f"X{k + 1}", Hx, mx))
    return out


def operator_pool(n: int, periodic: bool = True) -> list[Layer]:
    """Pool for pool-based baselines (AVQITE, ADAPT-VQE): ladder generators + local shared set."""
    return ladder_schedule(n, periodic=periodic) + local_schedule(n, periodic)


# ----------------------------------------------------------------------------- states
def prepare(layers: list[Layer], thetas: np.ndarray, psi0: np.ndarray) -> list[np.ndarray]:
    """Forward states s_0 = psi0, s_l = U_l s_{l-1}."""
    states = [psi0]
    for L, t in zip(layers, thetas):
        states.append(L.apply(float(t), states[-1]))
    return states


def tangents(layers: list[Layer], thetas: np.ndarray, psi0: np.ndarray):
    """Return (psi, T) with T[l] = d psi / d theta_l (exact, no finite differences)."""
    states = prepare(layers, thetas, psi0)
    L = len(layers)
    T = np.empty((L, psi0.size), dtype=complex)
    for l in range(L):
        v = layers[l].gen(states[l + 1])
        for m in range(l + 1, L):
            v = layers[m].apply(float(thetas[m]), v)
        T[l] = v
    return states[-1], T
