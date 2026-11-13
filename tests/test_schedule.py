"""Tests for the automatic commutator schedule and the mixed-field model."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.models import tfim, mfim, field_parent, minus_state, exact_ground_energy
from dynamic_vite.ansatz import ladder_generator
from dynamic_vite.schedule import commutator_schedule, canonical, dihedral_images, commutator
from dynamic_vite.paulis import PauliSum


@pytest.mark.parametrize("n", [6, 8, 10, 12])
def test_schedule_reproduces_ladder_on_tfim(n):
    """On the TFIM the automatic schedule is exactly G_1, ..., G_{N/2}, in order."""
    for J in (1.0, -1.0):
        lay = commutator_schedule(tfim(n, J, 1.0), field_parent(n), max_layers=n // 2)
        for r, L in enumerate(lay, start=1):
            assert len(L.G + ladder_generator(n, r).scale(-1)) == 0


@pytest.mark.parametrize("n", [6, 8])
def test_schedule_layers_real_and_symmetric(n):
    H = mfim(n)
    lay = commutator_schedule(H, field_parent(n, 1.05), max_layers=30)
    keys = set()
    for L in lay:
        assert L.real                                   # odd number of Y in every string
        for c, x, z in L.G.items():                     # closed under translations and reflection
            for xx, zz in dihedral_images(x, z, n):
                assert (xx, zz) in L.G.terms
        k = canonical(*next(iter(L.G.terms)), n)
        assert k not in keys                            # no orbit repeated
        keys.add(k)


def test_commutator_matches_matrix_commutator():
    n = 5
    rng = np.random.default_rng(0)
    A = PauliSum(n, [(rng.normal(), int(rng.integers(32)), int(rng.integers(32))) for _ in range(6)])
    B = PauliSum(n, [(rng.normal(), int(rng.integers(32)), int(rng.integers(32))) for _ in range(6)])
    a, b = A.to_sparse().toarray(), B.to_sparse().toarray()
    assert np.allclose(commutator(A, B).to_sparse().toarray(), a @ b - b @ a)


def test_mfim_is_nonintegrable_level_count():
    """With hz != 0 the commutator structure no longer closes on the free-fermion ladder."""
    n = 8
    lay, ent = commutator_schedule(mfim(n), field_parent(n, 1.05), max_layers=12, return_entries=True)
    assert [e.label for e in ent[:2]] == ["Y", "YZ"]
    assert len({e.level for e in ent}) >= 3


def test_lookahead_bound_is_variance_free_lower_bound_on_rho():
    """rho_next = 1 - R/R' <= rho = 1 - R/Var for random states and every candidate."""
    from dynamic_vite.geometry import snapshot, candidate_column, lookahead_rates
    from dynamic_vite.ansatz import ladder_schedule, local_schedule
    n = 8
    H = mfim(n); Hm = H.to_sparse().tocsr(); psi0 = minus_state(n)
    sched = commutator_schedule(H, field_parent(n, 1.05), max_layers=10)
    rng = np.random.default_rng(5)
    for L in (1, 3, 6):
        th = rng.uniform(-0.4, 0.4, L)
        s = snapshot(sched[:L], th, psi0, Hm)
        rho = s.rho()
        for cand in sched[L:]:
            R, R2 = lookahead_rates(s.M, s.V, *candidate_column(s, cand))
            assert R2 >= R - 1e-10 and R2 <= s.var + 1e-10
            assert 1 - R / R2 <= rho + 1e-9


def test_lookahead_inactive_without_noise():
    from dynamic_vite.methods import vite
    from dynamic_vite.ansatz import ladder_schedule, local_schedule
    n = 8
    H = tfim(n); Hm = H.to_sparse().tocsr(); E0 = exact_ground_energy(H); psi0 = minus_state(n)
    sched = ladder_schedule(n) + local_schedule(n)
    a = vite(H, Hm, E0, psi0, sched, mode="dynamic", eps=0.9, dtau=0.1, n_steps=60)
    b = vite(H, Hm, E0, psi0, sched, mode="dynamic", eps=0.9, dtau=0.1, n_steps=60, lookahead=8)
    assert a.energy == b.energy and a.circuits == b.circuits


def test_lookahead_gain_equals_schur_complement_formula():
    """R' - R = g^2 / s with g = V_c - m_c^T M^+ V and s = M_cc - m_c^T M^+ m_c."""
    from dynamic_vite.geometry import snapshot, candidate_column, lookahead_rates, pinv_psd
    n = 6
    H = mfim(n); Hm = H.to_sparse().tocsr(); psi0 = minus_state(n)
    sched = commutator_schedule(H, field_parent(n, 1.05), max_layers=8)
    th = np.random.default_rng(2).uniform(-0.4, 0.4, 4)
    s = snapshot(sched[:4], th, psi0, Hm)
    A = pinv_psd(s.M)
    for cand in sched[4:]:
        mc, mcc, vc = candidate_column(s, cand)
        R, R2 = lookahead_rates(s.M, s.V, mc, mcc, vc)
        g, sc = vc - mc @ A @ s.V, mcc - mc @ A @ mc
        if sc > 1e-8:
            assert abs((R2 - R) - g * g / sc) < 1e-8 * max(1.0, R2)
