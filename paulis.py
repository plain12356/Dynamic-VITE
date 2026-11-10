"""Pauli strings and sums on n qubits, stored as (coef, xmask, zmask).

Convention: qubit q corresponds to bit q of the computational-basis index
(little-endian, as in Qiskit). A string with bit q set in both masks is Y_q.
"""
from __future__ import annotations
import numpy as np
import scipy.sparse as sp

_PARITY_CACHE: dict[int, np.ndarray] = {}


def _parity_table(n: int) -> np.ndarray:
    if n not in _PARITY_CACHE:
        idx = np.arange(2 ** n, dtype=np.int64)
        par = np.zeros(2 ** n, dtype=np.int8)
        x = idx.copy()
        while np.any(x):
            par ^= (x & 1).astype(np.int8)
            x >>= 1
        _PARITY_CACHE[n] = par
    return _PARITY_CACHE[n]


def label_to_masks(label: dict[int, str]) -> tuple[int, int]:
    """label: {qubit: 'X'|'Y'|'Z'}"""
    x = z = 0
    for q, p in label.items():
        if p in "XY":
            x |= 1 << q
        if p in "ZY":
            z |= 1 << q
    return x, z


def weight(x: int, z: int) -> int:
    return bin(x | z).count("1")


def masks_to_str(x: int, z: int, n: int) -> str:
    """Qiskit-ordered label (qubit n-1 first)."""
    out = []
    for q in reversed(range(n)):
        bx, bz = (x >> q) & 1, (z >> q) & 1
        out.append("I" if not (bx or bz) else ("X" if bx and not bz else ("Z" if bz and not bx else "Y")))
    return "".join(out)


class PauliSum:
    def __init__(self, n: int, terms=None):
        self.n = n
        self.terms: dict[tuple[int, int], complex] = {}
        for c, x, z in (terms or []):
            self.add(c, x, z)

    def add(self, c, x, z):
        key = (x, z)
        self.terms[key] = self.terms.get(key, 0) + c
        if abs(self.terms[key]) < 1e-14:
            del self.terms[key]

    def __len__(self):
        return len(self.terms)

    def items(self):
        return [(c, x, z) for (x, z), c in self.terms.items()]

    # ---- algebra -------------------------------------------------------
    @staticmethod
    def _mul_strings(x1, z1, x2, z2):
        """P1 P2 = phase * P(x1^x2, z1^z2) with P = i^{|x&z|} X^x Z^z."""
        # write P = i^{y} X^x Z^z, y = popcount(x&z)
        y1 = bin(x1 & z1).count("1")
        y2 = bin(x2 & z2).count("1")
        # Z^z1 X^x2 = (-1)^{|z1&x2|} X^x2 Z^z1
        sign = (-1) ** bin(z1 & x2).count("1")
        x, z = x1 ^ x2, z1 ^ z2
        y = bin(x & z).count("1")
        phase = (1j ** (y1 + y2 - y)) * sign
        return phase, x, z

    def __matmul__(self, other: "PauliSum") -> "PauliSum":
        out = PauliSum(self.n)
        for c1, x1, z1 in self.items():
            for c2, x2, z2 in other.items():
                ph, x, z = self._mul_strings(x1, z1, x2, z2)
                out.add(c1 * c2 * ph, x, z)
        return out

    def __add__(self, other):
        out = PauliSum(self.n, self.items())
        for c, x, z in other.items():
            out.add(c, x, z)
        return out

    def scale(self, a):
        return PauliSum(self.n, [(a * c, x, z) for c, x, z in self.items()])

    def commutes_with(self, other) -> bool:
        comm = (self @ other) + (other @ self).scale(-1)
        return len(comm) == 0

    # ---- matrices ------------------------------------------------------
    def to_sparse(self) -> sp.csr_matrix:
        n = self.n
        dim = 2 ** n
        par = _parity_table(n)
        idx = np.arange(dim, dtype=np.int64)
        rows, cols, vals = [], [], []
        for c, x, z in self.items():
            y = bin(x & z).count("1")
            src = idx ^ x                      # P|src> -> |idx>
            ph = (1j ** y) * (1 - 2 * par[src & z].astype(np.int64))
            rows.append(idx)
            cols.append(src)
            vals.append(c * ph)
        m = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(dim, dim))
        m.sum_duplicates()
        m.eliminate_zeros()
        return m

    def is_real_antisymmetric_generator(self) -> bool:
        """True if exp(-i theta G) is a real orthogonal matrix (odd number of Y in each string)."""
        return all(bin(x & z).count("1") % 2 == 1 and abs(np.imag(c)) < 1e-14 for c, x, z in self.items())


def qwc_groups(ps: PauliSum) -> int:
    """Number of qubit-wise-commuting groups found by greedy colouring (largest weight first)."""
    strings = sorted([(x, z) for c, x, z in ps.items() if (x | z)], key=lambda s: -weight(*s))
    groups: list[list[tuple[int, int]]] = []

    def qwc(a, b):
        xa, za = a
        xb, zb = b
        both = (xa | za) & (xb | zb)
        return ((xa ^ xb) & both) == 0 and ((za ^ zb) & both) == 0

    for s in strings:
        for g in groups:
            if all(qwc(s, t) for t in g):
                g.append(s)
                break
        else:
            groups.append([s])
    return len(groups)
