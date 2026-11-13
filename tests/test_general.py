"""Tests for the general schedule (2D, fermions) and the new models."""
import os, sys
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from dynamic_vite.models import tfim, field_parent
from dynamic_vite.ansatz import ladder_generator
from dynamic_vite.paulis import PauliSum
from dynamic_vite.schedule import commutator
from dynamic_vite.schedule_general import general_schedule, symmetry_group
from dynamic_vite.models2 import tfim_2d, hubbard_chain, finalize, sector_ground, basis_state


@pytest.mark.parametrize("n", [6, 8, 10])
def test_general_schedule_reproduces_ladder(n):
    for J in (1.0, -1.0):
        H, H0 = tfim(n, J), field_parent(n)
        ring = [[(i + 1) % n for i in range(n)], [(-i) % n for i in range(n)]]
        G, _ = symmetry_group(H, H0, ring)
        assert len(G) == 2 * n
        for r, L in enumerate(general_schedule(H, H0, G, max_layers=n // 2), start=1):
            Gr = ladder_generator(n, r)
            assert min(len(L.G + Gr.scale(-1)), len(L.G + Gr)) == 0      # equal up to sign


def test_2d_symmetry_groups():
    for (Lx, Ly), order in (((3, 3), 72), ((3, 4), 48)):
        S = tfim_2d(Lx, Ly)
        G, _ = symmetry_group(S.H, S.H0, S.candidate_gens)
        assert len(G) == order
        for L in general_schedule(S.H, S.H0, G, max_layers=10):
            assert L.real


@pytest.mark.parametrize("L", [4])
def test_hubbard_layers_conserve_particle_numbers(L):
    S = hubbard_chain(L)
    G, _ = symmetry_group(S.H, S.H0, S.candidate_gens)
    Nup = PauliSum(2 * L, [(1.0, 0, 1 << q) for q in range(L)])
    Ndn = PauliSum(2 * L, [(1.0, 0, 1 << (L + q)) for q in range(L)])
    for lay in general_schedule(S.H, S.H0, G, merge_u1=True, max_layers=40):
        assert lay.real
        assert len(commutator(lay.G, Nup)) == 0 and len(commutator(lay.G, Ndn)) == 0


def test_hubbard_matches_openfermion():
    of = pytest.importorskip("openfermion")
    from dynamic_vite.models2 import _qubitop_to_paulisum
    L = 4
    S = hubbard_chain(L); finalize(S)
    Hof = _qubitop_to_paulisum(of.jordan_wigner(of.fermi_hubbard(L, 1, 1.0, 4.0, periodic=False)), 2 * L)
    occ = [2 * i for i in range(L) if i % 2 == 0] + [2 * i + 1 for i in range(L) if i % 2 == 1]
    a, b = sum(1 << p for p in range(0, 2 * L, 2)), sum(1 << p for p in range(1, 2 * L, 2))
    e, _ = sector_ground(Hof.to_sparse().tocsr(), basis_state(2 * L, occ), [a, b])
    assert abs(e - S.E0) < 1e-9


def test_h4_sector_energy_is_fci():
    pytest.importorskip("pyscf"); pytest.importorskip("openfermion")
    from dynamic_vite.models2 import molecule
    S = molecule([("H", (0, 0, 1.5 * i)) for i in range(4)])
    finalize(S)
    assert abs(S.E0 - S.extra["E_FCI"]) < 1e-8


def test_mfim2d_schedule_and_symmetry():
    """2D mixed field: same lattice group as the TFIM, real odd-Y layers, single-site Y orbit first."""
    from dynamic_vite.models2 import tfim_2d
    from dynamic_vite.schedule_general import symmetry_group, general_schedule
    S = tfim_2d(3, 3, hz=0.5)
    G, _ = symmetry_group(S.H, S.H0, S.candidate_gens)
    assert len(G) == 72 and S.name == "MFIM2D_3x3"
    layers, ent = general_schedule(S.H, S.H0, G, max_layers=10, return_entries=True)
    assert ent[0].label.count("Y") == 1 and ent[0].support == 1 and ent[0].n_strings == 9
    for L in layers:
        for c, x, z in L.G.items():
            assert bin(x & z).count("1") % 2 == 1


def test_single_string_apply_and_adjoint_gradient():
    """Closed-form exponential of a single Pauli string and the O(L) adjoint gradient used by the
    stronger baselines agree with expm_multiply and with the exact tangent vectors."""
    import numpy as np
    from scipy.sparse.linalg import expm_multiply
    from dynamic_vite.models2 import tfim_2d
    from dynamic_vite.pools import qubit_pool_spin
    from dynamic_vite.ansatz import tangents
    S = tfim_2d(3, 3, hz=0.5)
    Hm = S.H.to_sparse().tocsr()
    pool = qubit_pool_spin(S.H)
    assert len(pool) == 9 + 18 * 4 and all(L.real and L.single_c is not None for L in pool)
    rng = np.random.default_rng(0)
    layers = [pool[i] for i in rng.integers(0, len(pool), 12)]
    th = rng.normal(size=12)
    v = S.psi0
    for L, t in zip(layers, th):
        a = L.apply(t, v)
        assert np.abs(a - expm_multiply(-1j * t * L.mat, v)).max() < 1e-12
        v = a
    psi, T = tangents(layers, th, S.psi0)
    g1 = 2 * np.real(T.conj() @ (Hm @ psi))
    phi, lam, g2 = psi.copy(), Hm @ psi, np.empty(12)
    for l in range(11, -1, -1):
        g2[l] = 2 * np.real(np.vdot(lam, layers[l].gen(phi)))
        phi, lam = layers[l].apply(-th[l], phi), layers[l].apply(-th[l], lam)
    assert np.abs(g1 - g2).max() < 1e-10


def test_qeb_pool_conserves_particle_number():
    import numpy as np
    import scipy.sparse as sp
    from dynamic_vite.models2 import uccsd_pool
    from dynamic_vite.pools import qeb_pool
    n = 8
    pool = qeb_pool(uccsd_pool(4, n), n)
    N = sp.diags(np.array([bin(i).count("1") for i in range(2 ** n)], float))
    assert len(pool) == 26
    for L in pool:
        assert abs(L.mat @ N - N @ L.mat).max() < 1e-12


def test_first_commutator_level_reproduces_first_imaginary_time_step():
    """-iC1|psi0> = -(H0 - e0)(H - E)|psi0>, and the level-1 classes together give rho = 0 at tau = 0
    (one class suffices for the TFIM; the mixed-field model needs both of its level-1 classes)."""
    import numpy as np
    from dynamic_vite.geometry import snapshot
    from dynamic_vite.models import mfim, field_parent, minus_state
    from dynamic_vite.schedule import commutator_schedule, commutator
    n = 6
    for hz, n_level1 in ((0.0, 1), (0.5, 2)):
        H, H0 = mfim(n, -1.0, 1.05, hz), field_parent(n, 1.05)
        Hm, H0m = H.to_sparse().tocsr(), H0.to_sparse().tocsr()
        psi0 = minus_state(n)
        C1 = commutator(H, H0).scale(1j).to_sparse().tocsr()
        e0, E = np.real(np.vdot(psi0, H0m @ psi0)), np.real(np.vdot(psi0, Hm @ psi0))
        r = Hm @ psi0 - E * psi0
        assert np.abs(-1j * (C1 @ psi0) + (H0m @ r - e0 * r)).max() < 1e-12
        sched, ent = commutator_schedule(H, H0, max_layers=8, return_entries=True)
        k = sum(e.level == 1 for e in ent)
        assert k == n_level1
        assert snapshot(sched[:k], np.zeros(k), psi0, Hm).rho() < 1e-10
        if k > 1:
            assert snapshot(sched[:1], np.zeros(1), psi0, Hm).rho() > 0.1
