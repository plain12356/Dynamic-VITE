"""2D and fermionic models, reference states, symmetry candidates and sector ground states.

Conventions as in paulis.py: qubit q is bit q of the basis index.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import scipy.sparse.linalg as sla
from .paulis import PauliSum, label_to_masks
from .ansatz import Layer


@dataclass
class System:
    name: str
    H: PauliSum
    H0: PauliSum
    psi0: np.ndarray
    candidate_gens: list
    merge_u1: bool
    sector_masks: list          # conserved popcounts (list of bit masks); [] = full space
    E0: float = None
    gs: np.ndarray = None
    extra: dict = None


def basis_state(n, occ_bits):
    psi = np.zeros(2 ** n, complex)
    psi[sum(1 << q for q in occ_bits)] = 1.0
    return psi


def minus_state(n):
    v = np.array([1.0])
    for _ in range(n):
        v = np.kron(np.array([1.0, -1.0]) / np.sqrt(2), v)
    return v.astype(complex)


def sector_ground(Hm, psi0, masks, k=1):
    """Lowest eigenpair of H restricted to basis states with the same popcounts as psi0 in each mask."""
    dim = Hm.shape[0]
    if not masks:
        idx = np.arange(dim)
    else:
        ref = int(np.argmax(np.abs(psi0)))
        idx = np.arange(dim)
        keep = np.ones(dim, bool)
        for m in masks:
            pc = np.array([bin(i & m).count("1") for i in range(dim)])
            keep &= pc == bin(ref & m).count("1")
        idx = idx[keep]
    Hs = Hm[idx][:, idx]
    if Hs.shape[0] <= 400:
        w, v = np.linalg.eigh(Hs.toarray())
    else:
        w, v = sla.eigsh(Hs, k=k, which="SA", tol=1e-12)
        o = np.argsort(w); w, v = w[o], v[:, o]
    gs = np.zeros(dim, complex)
    gs[idx] = v[:, 0]
    return float(w[0]), gs


# ----------------------------------------------------------------------------- 2D TFIM
def tfim_2d(Lx, Ly, J=-1.0, h=3.04, hz=0.0):
    """H = J sum_<ij> Z_i Z_j + h sum_i X_i + hz sum_i Z_i on an Lx x Ly torus (qubit = x + Lx*y).

    hz = 0: transverse-field Ising model; hz != 0: mixed-field Ising model (non-integrable, no Z2)."""
    n = Lx * Ly
    q = lambda x, y: (x % Lx) + Lx * (y % Ly)
    H = PauliSum(n)
    bonds = set()
    for x in range(Lx):
        for y in range(Ly):
            for (a, b) in ((q(x, y), q(x + 1, y)), (q(x, y), q(x, y + 1))):
                if a != b and (min(a, b), max(a, b)) not in bonds:
                    bonds.add((min(a, b), max(a, b)))
                    H.add(J, *label_to_masks({a: "Z", b: "Z"}))
    for i in range(n):
        H.add(h, *label_to_masks({i: "X"}))
        if hz:
            H.add(hz, *label_to_masks({i: "Z"}))
    H0 = PauliSum(n)
    for i in range(n):
        H0.add(abs(h), *label_to_masks({i: "X"}))
    gens = [[q(x + 1, y) for y in range(Ly) for x in range(Lx)],          # translations
            [q(x, y + 1) for y in range(Ly) for x in range(Lx)],
            [q(-x, y) for y in range(Ly) for x in range(Lx)],              # reflections
            [q(x, -y) for y in range(Ly) for x in range(Lx)]]
    if Lx == Ly:
        gens.append([q(y, x) for y in range(Ly) for x in range(Lx)])      # diagonal reflection
    # permutation g maps qubit i -> g[i]; the lists above are built in qubit order i = x + Lx*y
    return System(f"{'MFIM2D' if hz else 'TFIM2D'}_{Lx}x{Ly}", H, H0, minus_state(n), gens, False, [],
                  extra=dict(Lx=Lx, Ly=Ly, J=J, h=h, hz=hz, bonds=len(bonds)))


# ----------------------------------------------------------------------------- Hubbard chain
def hubbard_chain(L, t=1.0, U=4.0):
    """Open Fermi-Hubbard chain, block ordering (qubit i = i up, L+i = i down), Jordan-Wigner.

    H = -t sum_{i,s} (a+_{i s} a_{i+1 s} + h.c.) + U sum_i n_{i up} n_{i down}.
    With block ordering, nearest-neighbour hopping has no JW string. Reference: Neel product state
    (up on even sites, down on odd sites), ground state of the staggered field
    H0 = sum_i s_i (n_{i dn} - n_{i up}), s_i = (-1)^i.
    """
    n = 2 * L
    H = PauliSum(n)
    for s in (0, L):
        for i in range(L - 1):
            a, b = s + i, s + i + 1
            H.add(-t / 2, *label_to_masks({a: "X", b: "X"}))
            H.add(-t / 2, *label_to_masks({a: "Y", b: "Y"}))
    for i in range(L):                                  # U n_up n_dn = U/4 (1 - Z_u)(1 - Z_d)
        u, d = i, L + i
        H.add(U / 4, 0, 0)
        H.add(-U / 4, *label_to_masks({u: "Z"}))
        H.add(-U / 4, *label_to_masks({d: "Z"}))
        H.add(U / 4, *label_to_masks({u: "Z", d: "Z"}))
    H0 = PauliSum(n)                                    # n = (1 - Z)/2
    for i in range(L):
        si = (-1) ** i
        H0.add(-si / 2, *label_to_masks({L + i: "Z"}))
        H0.add(+si / 2, *label_to_masks({i: "Z"}))
    occ = [i for i in range(L) if i % 2 == 0] + [L + i for i in range(L) if i % 2 == 1]
    refl = [L - 1 - i for i in range(L)] + [L + (L - 1 - i) for i in range(L)]
    flip = [L + i for i in range(L)] + [i for i in range(L)]
    refl_flip = [flip[refl[i]] for i in range(n)]
    up, dn = sum(1 << i for i in range(L)), sum(1 << (L + i) for i in range(L))
    trans = [(i + 1) % L for i in range(L)] + [L + (i + 1) % L for i in range(L)]
    return System(f"Hubbard_L{L}_U{U:g}", H, H0, basis_state(n, occ), [refl, flip, refl_flip], True,
                  [up, dn], extra=dict(L=L, t=t, U=U, grouping=[trans]))


# ----------------------------------------------------------------------------- molecules
def _qubitop_to_paulisum(qop, n):
    P = PauliSum(n)
    for term, c in qop.terms.items():
        lab = {q: p for q, p in term}
        x, z = label_to_masks(lab)
        P.add(complex(c), x, z)
    return PauliSum(n, [(float(np.real(c)), x, z) for c, x, z in P.items() if abs(c) > 1e-12])


def molecule(geometry, basis="sto-3g", name="mol", charge=0, spin=0):
    """Molecular Hamiltonian (Jordan-Wigner, interleaved spin orbitals 2p = alpha, 2p+1 = beta),
    Hartree-Fock reference and Fock parent H0 = sum_p (eps_p - mu) n_p (mu between HOMO and LUMO)."""
    from pyscf import gto, scf, ao2mo, fci
    import openfermion as of
    mol = gto.M(atom=geometry, basis=basis, charge=charge, spin=spin, verbose=0)
    mf = scf.RHF(mol).run()
    C = mf.mo_coeff
    norb = C.shape[1]
    h1 = C.T @ mf.get_hcore() @ C
    eri = ao2mo.restore(1, ao2mo.kernel(mol, C), norb)               # (pq|rs)
    two = np.asarray(eri.transpose(0, 2, 3, 1), order="C")             # openfermion: [p,q,r,s] = (ps|qr)
    ob, tb = of.chem.molecular_data.spinorb_from_spatial(h1, two)
    iop = of.InteractionOperator(mol.energy_nuc(), ob, 0.5 * tb)
    n = 2 * norb
    H = _qubitop_to_paulisum(of.jordan_wigner(of.get_fermion_operator(iop)), n)
    nel = mol.nelectron
    eps = np.repeat(mf.mo_energy, 2)
    mu = 0.5 * (eps[nel - 1] + eps[nel])
    H0 = PauliSum(n)
    for p in range(n):
        H0.add(-(eps[p] - mu) / 2, *label_to_masks({p: "Z"}))
    alpha, beta = sum(1 << p for p in range(0, n, 2)), sum(1 << p for p in range(1, n, 2))
    e_fci = fci.FCI(mf).kernel()[0]
    sysm = System(name, H, H0, basis_state(n, range(nel)), [], True, [alpha, beta],
                  extra=dict(E_HF=float(mf.e_tot), E_FCI=float(e_fci), nel=nel, norb=norb))
    return sysm


def uccsd_pool(nel, n):
    """Spin-conserving singles and doubles (occupied -> virtual) for an interleaved JW ordering,
    as Hermitian generators G = i (tau - tau^dagger) with real coefficients on odd-Y strings."""
    import openfermion as of
    occ, vir = list(range(nel)), list(range(nel, n))
    spin = lambda p: p % 2
    ops = []
    for i in occ:
        for a in vir:
            if spin(i) == spin(a):
                ops.append(((a, 1), (i, 0)))
    for i, j in [(i, j) for i in occ for j in occ if i < j]:
        for a, b in [(a, b) for a in vir for b in vir if a < b]:
            if sorted([spin(i), spin(j)]) == sorted([spin(a), spin(b)]):
                ops.append(((a, 1), (b, 1), (j, 0), (i, 0)))
    pool = []
    for k, term in enumerate(ops):
        f = of.FermionOperator(term, 1.0)
        tau = f - of.hermitian_conjugated(f)
        G = _qubitop_to_paulisum(of.jordan_wigner(1j * tau), n)
        if len(G):
            lab = "".join(str(o[0]) for o in term)
            pool.append(Layer(f"U{k + 1}:{lab}", G))
    return pool


def finalize(sys: System):
    """Attach exact sector ground energy and state."""
    Hm = sys.H.to_sparse().tocsr()
    sys.E0, sys.gs = sector_ground(Hm, sys.psi0, sys.sector_masks)
    return Hm
