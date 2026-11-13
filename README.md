# Dynamic VITE: pool-free, residual-triggered ansatz growth

Code for *Pool-free, residual-triggered ansatz growth for variational imaginary-time evolution: when a fixed
operator order beats adaptive screening* (Y.-H. Chen and R. Wong).

Variational imaginary-time evolution (VITE) follows the McLachlan equation of motion
`M θ̇ = V` for a shared-parameter circuit. The circuit grows along a **fixed schedule** of layers. The
next layer is appended whenever the relative McLachlan residual

    ρ = L²_min / Var(H),   L²_min = Var(H) − Vᵀ M⁺ V

exceeds a threshold ε. No operator pool is screened, and the ground-state energy is never used.
The schedule is built before the run from nested commutators of the Hamiltonian
(`C_1 = i[H, H0]`, `C_{n+1} = -[H,[H,C_n]]`), split into symmetry orbits; for the TFIM it reduces
exactly to the free-fermion ladder `G_r`.

Models: 1D — critical transverse-field Ising ring (N = 6-16), non-integrable mixed-field Ising ring
(N = 8-12); 2D — transverse-field and mixed-field (h_z = 0.5) Ising models on 3x3 and 3x4 tori; fermionic — open Fermi-Hubbard chain
(L = 4, 6; Jordan-Wigner) and the H4 molecule (STO-3G, 1.5 and 2.0 Å; PySCF + OpenFermion).

## Layout

| Path | Contents |
| --- | --- |
| `dynamic_vite/paulis.py` | Pauli strings and sums, sparse matrices, qubit-wise-commuting grouping |
| `dynamic_vite/models.py` | TFIM and mixed-field Ising ring, free-fermion ground energy, `\|−⟩^N` initial state |
| `dynamic_vite/schedule.py` | Automatic commutator schedule (nested commutators, dihedral orbits) |
| `dynamic_vite/schedule_general.py` | General schedule: automatic symmetry groups, U(1) (number-conserving) classes, commutator weights |
| `dynamic_vite/models2.py` | 2D TFIM and mixed-field Ising model, Hubbard chain, molecules, sector ground states, UCCSD pool |
| `dynamic_vite/pools.py` | Qubit and QEB pools for the stronger baselines |
| `dynamic_vite/systems.py` | The benchmark systems of the paper with their schedules |
| `dynamic_vite/ansatz.py` | Commutator-ladder generators `G_r`, the local ablation set, schedules, pool, exact tangent vectors |
| `dynamic_vite/geometry.py` | `M`, `V`, `Var(H)`, `L²_min`, `ρ` (projected tangents); candidate residuals |
| `dynamic_vite/cost.py` | Measurement-cost model (circuits and CNOT counts), shared by all methods; circuits are tagged by purpose (metric, force, screening, optimizer) |
| `dynamic_vite/methods.py` | Dynamic VITE, static VITE, energy-stagnation trigger, AVQITE, ADAPT-VQE (raw and normalized, and noisy with Adam or SPSA); step control; variance-free look-ahead test under noise (`lookahead=1, lookahead_every=4`); Gaussian shot-noise model |
| `scripts/run_benchmarks.py` | TFIM experiments, writes `results/*.json` |
| `scripts/run_mfim.py` | Mixed-field experiments, writes `results/mfim_*.json` |
| `scripts/run_models.py` | 2D and fermionic benchmarks, writes `results/models_*.json` |
| `scripts/make_model_figures.py` | Cross-model figures and the summary table |
| `scripts/run_breakdown.py` | Re-runs selected comparisons with per-step cost history, writes `results/breakdown.json` |
| `scripts/run_strong_baselines.py` | Stronger baselines on the five advantage systems: ADAPT-VQE, TETRIS-ADAPT-VQE and AVQITE with single-Pauli-string (qubit) and qubit-excitation (QEB) pools, AVQITE with absolute cutoffs (`--abs`) |
| `scripts/summarize_strong.py`, `scripts/check_strong_claims.py` | Table of the stronger baselines and the numbers quoted in the paper |
| `scripts/run_scaling.py`, `scripts/make_scaling_figure.py` | Size scaling on the mixed-field ring (N = 8-16) against the stronger baselines (`fig_scaling.pdf`) |
| `scripts/symmetry_cost.py` | Symmetry-reduced cost of dynamic VITE (one circuit per dihedral orbit of M and V entries), replayed from the recorded runs and checked against their unreduced totals |
| `scripts/run_adapt_ensembles.py`, `scripts/ensembles.py` | Seeded ensembles of ADAPT-type runs (rounding-level perturbations of the BFGS start); tables use medians, figures the median run (`table_ensembles.tex`) |
| `scripts/h4_overlap.py` | Hartree–Fock overlap with the ground state of H4 at 0.9, 1.5 and 2.0 Å (`results/h4_overlap.json`; Sec. III A) |
| `scripts/make_advantage_figures.py` | Longitudinal-field scan and cost-breakdown figures (`fig_hzscan.pdf`, `fig_breakdown.pdf`) |
| `scripts/summarize_models.py` | Prints key numbers per system |
| `scripts/run_noise_compare.py` | Shot-noise comparison (not in the paper): dynamic VITE vs ADAPT-VQE (Adam, SPSA), writes `results/noisecmp.json` |
| `scripts/tune_noisy_adapt.py` | Step-size choice for the noisy ADAPT-VQE optimizers, writes `results/tune_noisy_adapt.csv` |
| `scripts/diagnose_stall.py` | Forces on queued layers at a stalled circuit, writes `results/stall_forces.csv` |
| `scripts/run_all.sh` | Runs everything and makes the figures |
| `scripts/make_figures.py` | Builds `paper/figures/*.pdf`, `results/summary.csv`, `results/table_cost.tex` and other CSVs |
| `tests/` | Correctness tests (41) |

## Reproduce

All run scripts use single-threaded linear algebra, so every result is bitwise reproducible. ADAPT-type
costs are sensitive to rounding-level perturbations of the optimizer; the paper reports medians over seeded runs.


```bash
pip install -r requirements.txt
python -m pytest -q tests                                   # ~10 s
bash scripts/run_all.sh                                     # ~10-12 h on 2 cores
python scripts/make_figures.py
cd paper && latexmk -pdf main.tex                           # needs REVTeX 4.2
```

For the six-site Hubbard chain, ADAPT-VQE is capped at 40 operators (`results/models_hubbard_6_adapt40.json`);
an uncapped run did not finish within two hours.

The shot-noise code (Gaussian noise model, variance-free look-ahead test, noisy ADAPT-VQE with Adam
or SPSA; `scripts/run_noise_compare.py`, `scripts/tune_noisy_adapt.py`) is kept in the repository but is
not part of the paper, which reports noiseless simulations only. It is not run by `scripts/run_all.sh`.

Use `--only` to run selected groups: `main bench eps lightcone stall_tol field ablation noise` (TFIM)
and `eps static avqite adapt adaptnorm hz` (mixed field).

## What the tests check

- Pauli matrices match Qiskit.
- The TFIM ground energy matches the free-fermion formula.
- `i[H_ZZ, H_X] = −2 G_1`.
- The ladder generators are real and mutually commuting, and `G_{N−r} = P G_r`.
- Exact tangent vectors match finite differences.
- `0 ≤ L²_min ≤ Var(H)`, and `L²_min` never increases when a layer is appended at zero angle.
- The ladder is exact at depth `N/2` and not at `N/2 − 1`.
- The automatic schedule reproduces the ladder on the TFIM (both signs of J); its layers are real,
  symmetry-closed and non-repeating; the Pauli commutator matches matrix commutators.
- The trigger selects depth `N/2`.
- The variance-free bound `ρ_next = 1 − R/R′ ≤ ρ` holds for every candidate, `R′ − R = g²/s`
  (Schur complement), and the look-ahead leaves noiseless runs unchanged.
- Layer CNOT counts match Qiskit transpilation.

## Main parameters

`ε = 0.9`, `δτ = 0.1` with step halving when the energy would rise, 150 steps (TFIM) or 100 steps
(mixed field), pseudoinverse cutoff `1e-6` (relative), noise floor `κ = 3` (noisy runs only). Set at the top of
`scripts/run_benchmarks.py` and `scripts/run_mfim.py`.
