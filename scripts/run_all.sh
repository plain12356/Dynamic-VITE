#!/usr/bin/env bash
# Reproduce every result in the paper (several hours on 2 cores). The shot-noise study
# (scripts/tune_noisy_adapt.py, scripts/run_noise_compare.py) is not part of the paper and is not run here.
set -e
cd "$(dirname "$0")/.."
python3 scripts/run_mfim.py --sizes 8 10 12 --workers 2 --only eps static avqite hz adapt adaptnorm
python3 scripts/run_benchmarks.py --sizes 6 8 10 12 14 16 --workers 2
python3 scripts/run_models.py --systems tfim2d_3x3 mfim2d_3x3 hubbard_4 h4_1.5 h4_2.0 tfim2d_3x4 mfim2d_3x4 hubbard_6 --workers 2
python3 scripts/run_models.py --systems mfim2d_3x3_hz0 mfim2d_3x3_hz0.1 mfim2d_3x3_hz0.25 mfim2d_3x3_hz0.5 mfim2d_3x3_hz0.75 mfim2d_3x3_hz1 --workers 2
python3 scripts/run_breakdown.py
python3 scripts/run_strong_baselines.py --systems h4_2.0 h4_1.5 mfim2d_3x3 mfim_12 mfim2d_3x4
python3 scripts/run_strong_baselines.py --abs --systems h4_2.0 h4_1.5
python3 scripts/run_strong_baselines.py --abs --tag _spin --systems mfim2d_3x3 mfim_12 mfim2d_3x4
python3 scripts/run_strong_baselines.py --tag _hz --systems mfim2d_3x3_hz0 mfim2d_3x3_hz0.1 mfim2d_3x3_hz0.25 mfim2d_3x3_hz0.75 mfim2d_3x3_hz1
python3 scripts/run_scaling.py --sizes 14 10 8
python3 scripts/run_scaling.py --sizes 16
python3 scripts/run_adapt_ensembles.py --group h4 --seeds 7
python3 scripts/run_adapt_ensembles.py --group tables --seeds 5
python3 scripts/diagnose_stall.py
python3 scripts/make_figures.py
python3 scripts/ensembles.py
python3 scripts/make_model_figures.py
python3 scripts/make_advantage_figures.py
python3 scripts/summarize_strong.py
python3 scripts/make_scaling_figure.py
