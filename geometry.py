"""McLachlan quantities for variational imaginary-time evolution.

With projected tangents |t_i> = (1 - |phi><phi|) d_i|phi> (removes the global-phase direction;
identical to McArdle et al. for real ansatze, where <phi|d_i phi> = 0):

    M_ij = Re <t_i|t_j>,     V_i = -Re <t_i|(H - E)|phi>,
    theta_dot = M^+ V,       L^2_min = Var(H) - V^T M^+ V,      rho = L^2_min / Var(H).

rho in [0, 1] is the fraction of the exact imaginary-time direction that the tangent space misses.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ansatz import Layer, tangents


def pinv_psd(M: np.ndarray, rcond: float = 1e-6) -> np.ndarray:
    w, U = np.linalg.eigh(0.5 * (M + M.T))
    if w.size == 0:
        return np.zeros_like(M)
    cut = rcond * max(w.max(), 1e-300)
    winv = np.where(w > cut, 1.0 / np.where(w > cut, w, 1.0), 0.0)
    return (U * winv) @ U.T


@dataclass
class Snapshot:
    psi: np.ndarray
    E: float
    var: float
    V: np.ndarray
    M: np.ndarray
    T: np.ndarray          # projected tangents
    r: np.ndarray          # (H - E)|psi>

    def residual(self, M=None, V=None, var=None):
        M = self.M if M is None else M
        V = self.V if V is None else V
        var = self.var if var is None else var
        if V.size == 0:
            return var
        return float(var - V @ pinv_psd(M) @ V)

    def rho(self, **kw):
        L2 = self.residual(**kw)
        var = kw.get("var", self.var)
        return float(L2 / var) if var > 0 else 0.0


def snapshot(layers: list[Layer], thetas: np.ndarray, psi0: np.ndarray, Hmat) -> Snapshot:
    if len(layers):
        psi, T = tangents(layers, thetas, psi0)
    else:
        psi, T = psi0, np.zeros((0, psi0.size), complex)
    Hpsi = Hmat @ psi
    E = float(np.real(np.vdot(psi, Hpsi)))
    r = Hpsi - E * psi
    var = float(np.real(np.vdot(r, r)))
    if len(T):
        T = T - np.outer(T @ psi.conj(), psi)      # project out |phi>
    V = -np.real(T.conj() @ r)
    M = np.real(T.conj() @ T.T)
    return Snapshot(psi, E, var, V, M, T, r)


def candidate_residual(snap: Snapshot, cand: Layer) -> float:
    """L^2_min after appending `cand` at the end of the circuit with theta = 0."""
    t = cand.gen(snap.psi)
    t = t - np.vdot(snap.psi, t) * snap.psi
    Vc = -np.real(np.vdot(t, snap.r))
    Mcol = np.real(snap.T.conj() @ t) if len(snap.T) else np.zeros(0)
    Mcc = np.real(np.vdot(t, t))
    L = snap.M.shape[0]
    M2 = np.zeros((L + 1, L + 1))
    M2[:L, :L] = snap.M
    M2[:L, L] = M2[L, :L] = Mcol
    M2[L, L] = Mcc
    V2 = np.append(snap.V, Vc)
    return float(snap.var - V2 @ pinv_psd(M2) @ V2)


def energy_gradient_at_zero(snap: Snapshot, cand: Layer) -> float:
    """dE/dtheta for `cand` appended at theta = 0  (= -2 V_c)."""
    t = cand.gen(snap.psi)
    return float(2 * np.real(np.vdot(t, snap.r)))


def candidate_column(snap: Snapshot, cand: Layer):
    """Exact entries for appending `cand` at theta = 0: (M column vs existing layers, M_cc, V_c)."""
    t = cand.gen(snap.psi)
    t = t - np.vdot(snap.psi, t) * snap.psi
    Mcol = np.real(snap.T.conj() @ t) if len(snap.T) else np.zeros(0)
    return Mcol, float(np.real(np.vdot(t, t))), float(-np.real(np.vdot(t, snap.r)))


def lookahead_rates(M, V, Mcol, Mcc, Vc, reg=0.0):
    """Captured descent rates R = V^T M^+ V and R' (with the candidate appended).

    rho_next = 1 - R/R' is a lower bound on rho = 1 - R/Var(H), because R' <= Var(H);
    unlike rho it does not involve Var(H).
    """
    L = len(V)
    M2 = np.zeros((L + 1, L + 1))
    M2[:L, :L] = M
    M2[:L, L] = M2[L, :L] = Mcol
    M2[L, L] = Mcc
    V2 = np.append(V, Vc)
    R = float(V @ pinv_psd(M + reg * np.eye(L)) @ V) if L else 0.0
    R2 = float(V2 @ pinv_psd(M2 + reg * np.eye(L + 1)) @ V2)
    return R, R2
