"""Dynamic VITE (pool-free, residual-triggered) and the baselines it is compared with.

All imaginary-time methods use the same explicit-Euler McLachlan step
    theta <- theta + dtau * M^+ V
and the same cost ledger, so differences come only from how the ansatz is built.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import time
import numpy as np
from scipy.optimize import minimize

from .ansatz import Layer, tangents, prepare
from .geometry import (snapshot, pinv_psd, candidate_residual, energy_gradient_at_zero,
                       candidate_column, lookahead_rates)
from .cost import CostModel, Ledger


# ----------------------------------------------------------------------------- sampling noise
@dataclass
class ShotNoise:
    """Gaussian model of finite sampling: every circuit estimate has std <= 1/sqrt(shots).

    An entry built from K circuits with coefficients c_k gets std = sqrt(sum c_k^2 / shots).
    """
    shots: int
    sum_h2: float        # sum_k h_k^2 over Pauli terms of H
    sum_hh2: float       # same for H^2
    rng: np.random.Generator
    last_E: float = 0.0

    def perturb(self, snap, layers):
        L = len(layers)
        s = 1.0 / np.sqrt(self.shots)
        sizes = np.array([lay.n_strings for lay in layers], float)
        Vn = snap.V + self.rng.normal(size=L) * s * np.sqrt(sizes * self.sum_h2)
        noise = self.rng.normal(size=(L, L)) * s * np.sqrt(np.outer(sizes, sizes))
        noise = np.triu(noise) + np.triu(noise, 1).T
        Mn = snap.M + noise
        E2 = snap.var + snap.E ** 2 + self.rng.normal() * s * np.sqrt(self.sum_hh2)
        En = snap.E + self.rng.normal() * s * np.sqrt(self.sum_h2)
        varn = max(E2 - En ** 2, 0.0)
        reg = s * np.sqrt(sizes.max() ** 2) if L else 0.0     # Tikhonov at the noise floor
        self.last_E = En
        return Mn, Vn, varn, reg

    def column_stds(self, layers, cand):
        """Standard deviations of the estimated M column, M_cc and V_c for candidate `cand`."""
        s = 1.0 / np.sqrt(self.shots)
        sizes = np.array([lay.n_strings for lay in layers], float)
        gc = cand.n_strings
        return s * np.sqrt(sizes * gc), s * gc, s * np.sqrt(gc * self.sum_h2)

    def stds(self, layers):
        """Standard deviations of the estimated M and V entries (as used in perturb)."""
        s = 1.0 / np.sqrt(self.shots)
        sizes = np.array([lay.n_strings for lay in layers], float)
        return s * np.sqrt(np.outer(sizes, sizes)), s * np.sqrt(sizes * self.sum_h2)

    def sigma_var(self, E: float) -> float:
        """Sampling std of the Var(H) estimate <H^2> - <H>^2 (propagated, first order)."""
        return float(np.sqrt((self.sum_hh2 + 4 * E ** 2 * self.sum_h2) / self.shots))


# ----------------------------------------------------------------------------- results
@dataclass
class Run:
    method: str
    n: int
    E0: float
    tau: list = field(default_factory=list)
    energy: list = field(default_factory=list)
    var: list = field(default_factory=list)
    residual: list = field(default_factory=list)       # L^2_min (exact, noiseless)
    rho: list = field(default_factory=list)            # residual / Var used for the step (after any growth)
    rho_pre: list = field(default_factory=list)        # residual / Var tested by the trigger (before growth)
    n_layers: list = field(default_factory=list)
    circuits: list = field(default_factory=list)
    cnots: list = field(default_factory=list)
    events: list = field(default_factory=list)          # (step, tau, layer name)
    layers: list = field(default_factory=list)
    fidelity: float = float("nan")
    wall: float = 0.0
    by_kind: dict = field(default_factory=dict)
    by_kind_hist: list = field(default_factory=list)     # cumulative by_kind at every recorded step

    def summary(self, gs=None):
        e = np.array(self.energy)
        rel = (e - self.E0) / abs(self.E0)
        return dict(method=self.method, n=self.n, final_energy=float(e[-1]), E0=self.E0,
                    rel_error=float(rel[-1]), fidelity=self.fidelity, final_layers=self.n_layers[-1],
                    layers=" ".join(self.layers), circuits=self.circuits[-1], cnots=self.cnots[-1],
                    wall=self.wall)


def _fid(psi, gs):
    return float(abs(np.vdot(gs, psi)) ** 2) if gs is not None else float("nan")


# ----------------------------------------------------------------------------- VITE family
def vite(Hps, Hmat, E0, psi0, schedule: list[Layer], *, dtau=0.05, n_steps=300,
         mode="dynamic", eps=0.1, pool: list[Layer] | None = None, initial_layers=0,
         var_floor=1e-10, patience=1, noise: ShotNoise | None = None, gs=None,
         method_name=None, max_layers=None, var_stop=0.0, kappa=3.0,
         stall_tol=1e-4, stall_wait=3, backtrack=None, max_halvings=8,
         lookahead=0, lookahead_every=1, n_boot=200, abs_cut=None, stop_rel=None) -> Run:
    """abs_cut (mode 'avqite' only): grow when L2_min > abs_cut instead of rho > eps, as in the original
    AVQITE (absolute McLachlan cutoff L2_cut); operators are added one at a time until L2_min <= abs_cut.

    mode = 'static'  : all schedule layers from tau = 0 (no growth).
              'dynamic' : pool-free growth, next schedule layer appended when rho > eps
                          (rho is set to 0 when the estimated Var(H) <= var_floor)
                          (and, with sampling noise, when the estimated L^2 exceeds kappa times
                          the sampling std of Var(H), so noise alone cannot trigger growth).
              'stall'   : pool-free growth on energy stagnation only (baseline without the residual):
                          next schedule layer appended when the relative energy decrease per step
                          falls below stall_tol, at least stall_wait steps after the last growth.
              'avqite'  : pool-based growth (Gomes et al. 2021): screen every pool operator and
                          append the one that minimises L^2, until rho <= eps.
    lookahead = w (dynamic mode, with noise; 0 = off): every lookahead_every steps in which the
    variance-based test does not fire, the next w scheduled layers are tested as candidates. For any
    candidate c, rho_next(c) = 1 - R/R'(c) is a variance-free lower bound on rho (R' <= Var(H)). A
    candidate certifies rho > eps if its (noise-debiased) rho_next exceeds eps and its orthogonalised
    force exceeds kappa_w times its sampling std (parametric bootstrap, n_boot resamples), with
    kappa_w the Bonferroni-corrected two-sided threshold for w tests at the level of kappa. If any
    candidate certifies, the scheduled layers are appended in order up to and including the first
    certifying one: the window certifies, it does not reorder. The M columns and V entries of all w
    candidates are charged at every evaluation. Without noise rho_next(c) <= rho, so the test can
    never fire when the variance test does not; it is not evaluated, and noiseless results are
    unchanged.
    backtrack (default: on without noise): explicit Euler step dtau, halved (up to max_halvings
    times) until the energy does not increase; each trial costs one energy estimate (g(H) circuits).
    The imaginary time recorded in run.tau is the sum of the accepted steps.
    """
    t0 = time.time()
    model = CostModel.for_hamiltonian(Hps)
    led = Ledger(model)
    run = Run(method_name or mode, Hps.n, E0)
    if mode == "static":
        layers = list(schedule)
    elif mode == "stall":
        layers = list(schedule[:max(initial_layers, 1)])
    else:
        layers = list(schedule[:initial_layers])
    need_h2 = mode in ("dynamic", "avqite")
    last_growth = 0
    thetas = np.zeros(len(layers))
    nxt = len(layers)
    over = 0
    max_layers = max_layers or 10 ** 9
    backtrack = (noise is None) if backtrack is None else backtrack
    tau = 0.0
    run.n_halvings = 0
    run.lookahead_growths = 0
    run.rho_next = []

    def estimates(snap):
        if noise is None:
            return snap.M, snap.V, snap.var, 0.0
        return noise.perturb(snap, layers)

    def rho_of(M, V, var, reg, E):
        if var <= var_floor:
            return 0.0, False
        L2 = var - (V @ pinv_psd(M + reg * np.eye(len(V))) @ V if len(V) else 0.0)
        rho = float(np.clip(L2 / var, 0.0, 1.0))
        significant = noise is None or L2 > kappa * noise.sigma_var(noise.last_E)
        return rho, significant

    for k in range(n_steps + 1):
        snap = snapshot(layers, thetas, psi0, Hmat)
        led.vite_step(layers, with_h2=need_h2)
        M, V, var, reg = estimates(snap)
        rho, sig = rho_of(M, V, var, reg, snap.E)
        run.rho_pre.append(rho)

        if mode == "stall":
            if (k - last_growth >= stall_wait and len(run.energy) and nxt < len(schedule)
                    and (run.energy[-1] - snap.E) / abs(snap.E) < stall_tol):
                cand = schedule[nxt]; nxt += 1
                led.growth_column(layers, cand)
                layers.append(cand); thetas = np.append(thetas, 0.0)
                run.events.append((k, tau, cand.name)); last_growth = k
                snap = snapshot(layers, thetas, psi0, Hmat)
                M, V, var, reg = estimates(snap)
                rho, sig = rho_of(M, V, var, reg, snap.E)
        elif mode != "static":
            over = over + 1 if (((rho * var > abs_cut) if abs_cut else (rho > eps)) and sig) else 0
            rn = np.nan
            if (lookahead and mode == "dynamic" and noise is not None and over < patience
                    and k % lookahead_every == 0 and nxt < len(schedule) and len(layers) < max_layers):
                window = schedule[nxt:nxt + lookahead]
                kw = _bonferroni_kappa(kappa, len(window))
                hit = None
                for j, cand in enumerate(window):
                    with led.tagged("lookahead"):
                        led.growth_column(layers, cand)
                    fire, r_j = _lookahead_test(snap, layers, cand, M, V, reg, noise, eps, kw, n_boot)
                    rn = r_j if j == 0 else rn
                    if fire and hit is None:
                        hit = j
                if hit is not None:
                    for cand in schedule[nxt:nxt + hit + 1]:
                        if len(layers) >= max_layers:
                            break
                        layers.append(cand)
                        thetas = np.append(thetas, 0.0)
                        run.events.append((k, tau, cand.name))
                        run.lookahead_growths += 1
                    nxt += hit + 1
                    snap = snapshot(layers, thetas, psi0, Hmat)
                    M, V, var, reg = estimates(snap)
                    rho, sig = rho_of(M, V, var, reg, snap.E)
            run.rho_next.append(rn)
            while over >= patience and len(layers) < max_layers:
                if mode == "dynamic":
                    if nxt >= len(schedule):
                        break
                    cand = schedule[nxt]
                    nxt += 1
                    led.growth_column(layers, cand)
                else:  # avqite: screen the whole pool
                    best, bestL2 = None, np.inf
                    for c in pool:
                        with led.tagged("screen"):
                            led.growth_column(layers, c)
                        L2c = candidate_residual(snap, c)
                        if L2c < bestL2 - 1e-12:
                            best, bestL2 = c, L2c
                    if best is None or bestL2 >= snap.residual() - 1e-12:
                        break
                    cand = best
                layers.append(cand)
                thetas = np.append(thetas, 0.0)
                run.events.append((k, tau, cand.name))
                snap = snapshot(layers, thetas, psi0, Hmat)
                M, V, var, reg = estimates(snap)
                rho, sig = rho_of(M, V, var, reg, snap.E)
                over = over + 1 if (((rho * var > abs_cut) if abs_cut else (rho > eps)) and sig) else 0

        run.tau.append(tau)
        run.energy.append(snap.E)
        run.var.append(snap.var)
        run.residual.append(snap.residual())
        run.rho.append(rho)
        run.n_layers.append(len(layers))
        c, x = led.snapshot()
        run.circuits.append(c)
        run.cnots.append(x)
        run.by_kind_hist.append(dict(led.by_kind))
        if k == n_steps or snap.var < var_stop or (stop_rel and (snap.E - E0) / abs(E0) < stop_rel):
            break
        if not len(V):
            tau += dtau
            continue
        step = pinv_psd(M + reg * np.eye(len(V))) @ V
        h = dtau
        if backtrack:
            tol = 1e-13 * max(abs(snap.E), 1.0)
            for _ in range(max_halvings):
                trial = thetas + h * step
                led.energy(layers)
                psi_t = prepare(layers, trial, psi0)[-1]
                if float(np.real(np.vdot(psi_t, Hmat @ psi_t))) <= snap.E + tol:
                    break
                h *= 0.5
                run.n_halvings += 1
        thetas = thetas + h * step
        tau += h

    run.layers = [l.name for l in layers]
    run.fidelity = _fid(snap.psi, gs)
    run.wall = time.time() - t0
    run.by_kind = dict(led.by_kind)
    return run


def _bonferroni_kappa(kappa, w):
    """Two-sided z threshold for w simultaneous tests at the family-wise level of a single kappa test."""
    from scipy.stats import norm
    alpha = 2 * norm.sf(kappa)
    return float(norm.isf(alpha / (2 * max(w, 1))))


def _lookahead_test(snap, layers, cand, M, V, reg, noise, eps, kappa, n_boot):
    """Variance-free growth test (see vite docstring). Returns (fire, rho_next estimate).

    With A = (M + reg)^+, the candidate's orthogonalised force g = V_c - m_c^T A V and Schur
    complement s = M_cc - m_c^T A m_c give R' - R = g^2 / s. Noise biases quadratic forms upward,
    so R and g^2 are debiased by their expected noise contributions (tr(A Sigma_V) and sigma_g^2).
    The test fires if rho_next = (R' - R) / R' > eps and |g| > kappa sigma_g, where sigma_g is the
    parametric-bootstrap std of g over the noise of all estimated entries.
    """
    rng = noise.rng
    Mcol, Mcc, Vc = candidate_column(snap, cand)
    sd_col, sd_cc, sd_vc = noise.column_stds(layers, cand)
    Mcol = Mcol + rng.normal(size=Mcol.size) * sd_col
    Mcc = Mcc + rng.normal() * sd_cc
    Vc = Vc + rng.normal() * sd_vc
    reg2 = max(reg, sd_cc)
    L = len(V)
    sd_M, sd_V = noise.stds(layers) if L else (np.zeros((0, 0)), np.zeros(0))

    def force(Mx, Vx, mc, mcc, vc):
        if L:
            A = pinv_psd(Mx + reg2 * np.eye(L))
            return vc - mc @ A @ Vx, mcc - mc @ A @ mc, A
        return vc, mcc, np.zeros((0, 0))

    g, schur, A = force(M, V, Mcol, Mcc, Vc)
    gb = []
    for _ in range(n_boot):
        e = rng.normal(size=M.shape) * sd_M
        gb.append(force(M + np.triu(e) + np.triu(e, 1).T, V + rng.normal(size=L) * sd_V,
                        Mcol + rng.normal(size=L) * sd_col, Mcc + rng.normal() * sd_cc,
                        Vc + rng.normal() * sd_vc)[0])
    sg = float(np.std(gb))
    if schur <= 0:
        return False, 0.0
    R = max(float(V @ A @ V) - float(np.sum(np.diag(A) * sd_V ** 2)), 0.0) if L else 0.0
    gain = max(g * g - sg * sg, 0.0) / schur
    rho_next = gain / (R + gain) if R + gain > 0 else 0.0
    return bool(rho_next > eps and abs(g) > kappa * sg), float(rho_next)


# ----------------------------------------------------------------------------- ADAPT-VQE
def adapt_vqe(Hps, Hmat, E0, psi0, pool: list[Layer], *, grad_tol=1e-3, max_ops=None,
              gs=None, gtol=1e-8, normalize=False, method_name="ADAPT-VQE", batch=None,
              tie_round=None, stop_rel=None, adjoint=False, jitter=0.0, seed=None) -> Run:
    """ADAPT-VQE with the same shared-parameter pool: append argmax |dE/dtheta| at theta = 0,
    then re-optimise all parameters with BFGS to convergence; stop when max |grad| < grad_tol.

    batch='tetris' (TETRIS-ADAPT-VQE, Anastasiou et al.): in each round append, in order of decreasing
    gradient, every operator with |grad| >= grad_tol whose qubit support is disjoint from the
    operators already chosen in that round; then re-optimise once.

    tie_round: if set, gradients are rounded to this relative precision before ranking and ties are
    broken by pool order, so that symmetry-equivalent operators are chosen reproducibly.

    jitter, seed: if jitter > 0, each BFGS re-optimization starts from the current parameters plus
    seeded Gaussian noise of this size (radians). The number of BFGS evaluations to reach gtol is
    sensitive to rounding-level differences; seeded jitter samples this sensitivity reproducibly."""
    from .pools import support
    t0 = time.time()
    led = Ledger(CostModel.for_hamiltonian(Hps))
    run = Run(method_name, Hps.n, E0)
    layers: list[Layer] = []
    thetas = np.zeros(0)
    max_ops = max_ops or 3 * Hps.n
    jit_rng = np.random.default_rng(seed)

    def f_and_g(th):
        if adjoint:      # O(L) backward pass; same gradient as the tangent vectors (to rounding)
            psi = prepare(layers, th, psi0)[-1]
            Hpsi = Hmat @ psi
            E = float(np.real(np.vdot(psi, Hpsi)))
            g = np.empty(len(layers))
            phi, lam = psi.copy(), Hpsi.copy()
            for l in range(len(layers) - 1, -1, -1):
                g[l] = 2 * np.real(np.vdot(lam, layers[l].gen(phi)))
                phi = layers[l].apply(-float(th[l]), phi)
                lam = layers[l].apply(-float(th[l]), lam)
        else:
            psi, T = tangents(layers, th, psi0)
            Hpsi = Hmat @ psi
            E = float(np.real(np.vdot(psi, Hpsi)))
            g = 2 * np.real(T.conj() @ Hpsi)
        with led.tagged("opt"):
            led.energy(layers)
            for l in range(len(layers)):
                led.V_entry(layers, l)
        return E, g

    it = 0
    while True:
        snap = snapshot(layers, thetas, psi0, Hmat)
        grads = []
        for c in pool:
            with led.tagged("screen"):
                led.gradient_entry(layers, c)
            g = abs(energy_gradient_at_zero(snap, c))
            grads.append(g / np.sqrt(c.n_strings) if normalize else g)
        run.tau.append(it)
        run.energy.append(snap.E)
        run.var.append(snap.var)
        run.residual.append(snap.residual())
        run.rho.append(snap.rho())
        run.n_layers.append(len(layers))
        c_, x_ = led.snapshot()
        run.circuits.append(c_)
        run.cnots.append(x_)
        run.by_kind_hist.append(dict(led.by_kind))
        gmax = max(grads)
        if gmax < grad_tol or len(layers) >= max_ops or (stop_rel and (snap.E - E0) / abs(E0) < stop_rel):
            break
        if tie_round:
            ga = np.asarray(grads)
            key = np.round(ga / max(ga.max(), 1e-300) / tie_round) if ga.max() > 0 else ga
            order = sorted(range(len(pool)), key=lambda i: (-key[i], i))
        else:
            order = list(np.argsort(grads)[::-1])
        if batch == "tetris":
            chosen, used = [], 0
            for idx in order:
                if grads[idx] < grad_tol or len(layers) + len(chosen) >= max_ops:
                    break
                sup = support(pool[idx])
                if sup & used:
                    continue
                chosen.append(pool[idx]); used |= sup
        else:
            chosen = [pool[order[0]] if tie_round else pool[int(np.argmax(grads))]]
        for best in chosen:
            layers.append(best)
            run.events.append((it, it, best.name))
            thetas = np.append(thetas, 0.0)
        x0 = thetas + (jit_rng.normal(scale=jitter, size=thetas.size) if jitter > 0 else 0.0)
        res = minimize(f_and_g, x0, jac=True, method="BFGS", options=dict(gtol=gtol, maxiter=2000))
        thetas = res.x
        it += 1
    run.layers = [l.name for l in layers]
    run.fidelity = _fid(snap.psi, gs)
    run.wall = time.time() - t0
    run.by_kind = dict(led.by_kind)
    return run


# ----------------------------------------------------------------------------- noisy ADAPT-VQE
def noisy_adapt_vqe(Hps, Hmat, E0, psi0, pool: list[Layer], noise: ShotNoise, *, optimizer="adam",
                    iters_per_growth=40, lr=0.05, max_ops=None, kappa=3.0, grad_tol=1e-5, gs=None,
                    spsa_step=0.05, spsa_c=0.1, spsa_calib=5, trust=0.1, method_name=None) -> Run:
    """ADAPT-VQE under the same Gaussian sampling model as noisy VITE (ShotNoise).

    Noise per estimate (S shots per circuit, s = 1/sqrt(S)):
      energy                       std = s sqrt(sum_k h_k^2)
      gradient dE/dtheta of layer  std = 2 s sqrt(|G| sum_k h_k^2)     (= 2 x the V-entry noise)
    Selection: candidate gradients at zero angle are estimated with noise; the largest |g| is
    appended if it exceeds max(grad_tol, kappa * std) (the same 3-sigma floor as noisy VITE);
    otherwise ADAPT stops. Re-optimisation after each growth (BFGS needs exact gradients, so):
      optimizer='adam': Adam on noisy full gradients, iters_per_growth iterations, step lr;
      optimizer='spsa': SPSA with standard gains (alpha=0.602, gamma=0.101), 2 noisy energies
                        per iteration, iters_per_growth iterations. The gain a is calibrated at
                        each growth (Spall's rule, as in Qiskit) from spsa_calib gradient samples
                        so that the first step has size spsa_step; calibration energies are counted.
                        Each update is clipped to |dtheta_l| <= trust (trust region, as in Qiskit),
                        which prevents divergence when the calibrated gain is large.
    The energy recorded after every optimiser iteration is the exact energy of the current state
    (for evaluation only). Costs follow the shared Ledger.
    """
    t0 = time.time()
    led = Ledger(CostModel.for_hamiltonian(Hps))
    run = Run(method_name or f"ADAPT-VQE ({optimizer})", Hps.n, E0)
    s = 1.0 / np.sqrt(noise.shots)
    sig_E = s * np.sqrt(noise.sum_h2)
    sig_g = lambda lay: 2 * s * np.sqrt(lay.n_strings * noise.sum_h2)
    rng = noise.rng
    max_ops = max_ops or 3 * Hps.n
    layers: list[Layer] = []
    thetas = np.zeros(0)
    m_adam = v_adam = np.zeros(0)
    step = 0

    def true_E(th):
        psi = prepare(layers, th, psi0)[-1]
        return float(np.real(np.vdot(psi, Hmat @ psi))), psi

    def record(E):
        run.tau.append(step)
        run.energy.append(E)
        run.n_layers.append(len(layers))
        c, x = led.snapshot()
        run.circuits.append(c)
        run.cnots.append(x)

    E, psi = true_E(thetas)
    record(E)
    while len(layers) < max_ops:
        snap = snapshot(layers, thetas, psi0, Hmat)
        g_noisy, ok = [], []
        for c in pool:
            led.gradient_entry(layers, c)
            g = energy_gradient_at_zero(snap, c) + rng.normal() * sig_g(c)
            g_noisy.append(abs(g))
            ok.append(abs(g) > max(grad_tol, kappa * sig_g(c)))
        if not any(ok):
            break
        best = int(np.argmax(np.where(ok, g_noisy, -1.0)))
        layers.append(pool[best])
        run.events.append((step, step, pool[best].name))
        thetas = np.append(thetas, 0.0)
        m_adam = np.append(m_adam, 0.0)
        v_adam = np.append(v_adam, 0.0)
        if optimizer == "spsa":
            A = 0.1 * iters_per_growth
            gm = []
            for _ in range(spsa_calib):
                delta = rng.choice([-1.0, 1.0], size=thetas.size)
                Ep = true_E(thetas + spsa_c * delta)[0] + rng.normal() * sig_E
                Em = true_E(thetas - spsa_c * delta)[0] + rng.normal() * sig_E
                led.energy(layers); led.energy(layers)
                gm.append(abs(Ep - Em) / (2 * spsa_c))
            spsa_a = spsa_step * (A + 1) ** 0.602 / max(np.mean(gm), 1e-12)
        for it in range(iters_per_growth):
            step += 1
            if optimizer == "adam":
                psi, T = tangents(layers, thetas, psi0)
                grad = 2 * np.real(T.conj() @ (Hmat @ psi))
                grad = grad + rng.normal(size=grad.size) * np.array([sig_g(l) for l in layers])
                for l in range(len(layers)):
                    led.V_entry(layers, l)
                m_adam = 0.9 * m_adam + 0.1 * grad
                v_adam = 0.999 * v_adam + 0.001 * grad ** 2
                k = it + 1
                mh, vh = m_adam / (1 - 0.9 ** k), v_adam / (1 - 0.999 ** k)
                thetas = thetas - lr * mh / (np.sqrt(vh) + 1e-8)
            elif optimizer == "spsa":
                ak = spsa_a / (it + 1 + A) ** 0.602
                ck = spsa_c / (it + 1) ** 0.101
                delta = rng.choice([-1.0, 1.0], size=thetas.size)
                Ep = true_E(thetas + ck * delta)[0] + rng.normal() * sig_E
                Em = true_E(thetas - ck * delta)[0] + rng.normal() * sig_E
                led.energy(layers); led.energy(layers)
                thetas = thetas - np.clip(ak * (Ep - Em) / (2 * ck) * delta, -trust, trust)
            else:
                raise ValueError(optimizer)
            E, psi = true_E(thetas)
            record(E)
    run.layers = [l.name for l in layers]
    run.fidelity = _fid(psi, gs)
    run.wall = time.time() - t0
    run.by_kind = dict(led.by_kind)
    return run
