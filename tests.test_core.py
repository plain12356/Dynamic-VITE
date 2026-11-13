"""Correctness checks for the simulation core.  Run:  python -m pytest -q tests"""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.paulis import PauliSum, label_to_masks, masks_to_str
from dynamic_vite.models import tfim, tfim_parts, exact_ground_energy, free_fermion_ground_energy, minus_state
from dynamic_vite.ansatz import ladder_generator, ladder_schedule, local_schedule, Layer, tangents
from dynamic_vite.geometry import snapshot, candidate_residual, pinv_psd
from dynamic_vite.methods import vite


def test_pauli_matrices_match_qiskit():
    qi = pytest.importorskip("qiskit.quantum_info")
    rng = np.random.default_rng(1)
    n = 4
    for _ in range(20):
        lab = {q: "IXYZ"[rng.integers(4)] for q in range(n)}
        lab = {q: p for q, p in lab.items() if p != "I"}
        x, z = label_to_masks(lab)
        ours = PauliSum(n, [(1.0, x, z)]).to_sparse().toarray()
        ref = qi.SparsePauliOp(masks_to_str(x, z, n) if (x | z) else "I" * n).to_matrix()
        assert np.allclose(ours, ref)


@pytest.mark.parametrize("n", [4, 6, 8, 10])
def test_free_fermion_energy(n):
    assert abs(exact_ground_energy(tfim(n)) - free_fermion_ground_energy(n)) < 1e-9


@pytest.mark.parametrize("n", [6, 8, 10, 12, 14, 16])
def test_ladder_is_commutator_and_commuting(n):
    Hzz, Hx = tfim_parts(n)
    comm = ((Hzz @ Hx) + (Hx @ Hzz).scale(-1)).scale(1j)
    assert len(comm + ladder_generator(n, 1).scale(2)) == 0       # i[H_ZZ, H_X] = -2 G_1
    Gs = [ladder_generator(n, r) for r in range(1, n)]
    for a in range(len(Gs)):
        assert Gs[a].is_real_antisymmetric_generator()
        for b in range(a):
            assert Gs[a].commutes_with(Gs[b])


def test_exact_tangents_match_finite_differences():
    n = 6
    layers = ladder_schedule(n) + local_schedule(n)[:4]
    th = np.random.default_rng(0).uniform(-0.4, 0.4, len(layers))
    psi0 = minus_state(n)
    psi, T = tangents(layers, th, psi0)
    for l in range(len(layers)):
        e = 1e-6
        a, b = th.copy(), th.copy(); a[l] += e; b[l] -= e
        fd = (tangents(layers, a, psi0)[0] - tangents(layers, b, psi0)[0]) / (2 * e)
        assert np.allclose(fd, T[l], atol=1e-7)


def test_residual_bounds_and_monotone_under_growth():
    """Appending a layer at theta=0 leaves the state unchanged and can only lower L^2_min."""
    n = 8
    H = tfim(n); Hm = H.to_sparse().tocsr(); psi0 = minus_state(n)
    layers = ladder_schedule(n, max_range=2) + local_schedule(n)[:3]
    th = np.random.default_rng(3).uniform(-0.5, 0.5, len(layers))
    s = snapshot(layers, th, psi0, Hm)
    L2 = s.residual()
    assert -1e-10 <= L2 <= s.var + 1e-10
    for cand in ladder_schedule(n)[2:] + local_schedule(n):
        assert candidate_residual(s, cand) <= L2 + 1e-10


def test_full_ladder_is_exact_and_shallower_is_not():
    n = 8
    H = tfim(n); Hm = H.to_sparse().tocsr(); E0 = free_fermion_ground_energy(n); psi0 = minus_state(n)
    full = vite(H, Hm, E0, psi0, ladder_schedule(n), mode="static", dtau=0.1, n_steps=150)
    short = vite(H, Hm, E0, psi0, ladder_schedule(n, n // 2 - 1), mode="static", dtau=0.1, n_steps=150)
    assert abs(full.energy[-1] - E0) < 1e-9 and max(full.rho[1:]) < 1e-5
    assert short.energy[-1] - E0 > 1e-2 and short.rho[-1] > 0.9


def test_dynamic_trigger_selects_depth_n_over_2():
    for n in (6, 8, 10):
        H = tfim(n); Hm = H.to_sparse().tocsr(); E0 = free_fermion_ground_energy(n); psi0 = minus_state(n)
        r = vite(H, Hm, E0, psi0, ladder_schedule(n, max_range=n - 1), mode="dynamic", eps=0.1,
                 dtau=0.1, n_steps=150, var_stop=1e-13)
        assert r.n_layers[-1] == n // 2
        assert abs(r.energy[-1] - E0) < 1e-9


def test_layer_cnot_count_matches_qiskit_transpile():
    qk = pytest.importorskip("qiskit")
    from qiskit.circuit.library import PauliEvolutionGate
    from qiskit.quantum_info import SparsePauliOp
    n = 6
    for lay in ladder_schedule(n) + local_schedule(n)[3:6]:
        qc = qk.QuantumCircuit(n)
        for c, x, z in lay.G.items():
            qc.append(PauliEvolutionGate(SparsePauliOp(masks_to_str(x, z, n)), time=0.3), range(n))
        t = qk.transpile(qc, basis_gates=["u", "cx"], optimization_level=0)
        assert t.count_ops().get("cx", 0) == lay.cnot


@pytest.mark.parametrize("n", [6, 8, 10])
def test_ladder_beyond_half_is_redundant(n):
    """G_{N-r} = P G_r with P = prod_j X_j, so on the even-parity sector layers beyond N/2 add nothing."""
    P = PauliSum(n, [(1.0, (1 << n) - 1, 0)])
    for r in range(1, n):
        diff = ladder_generator(n, n - r) + (P @ ladder_generator(n, r)).scale(-1)
        assert len(diff) == 0
