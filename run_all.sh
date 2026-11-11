#!/usr/bin/env bash
# Full run: data -> experiments -> figures -> tables.
# QUICK=1 ./run_all.sh   smoke test (a few minutes)
# ./run_all.sh           main run (N_SIM=1000), ~13 min on 6 cores
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q tests/
python -m experiments.exp_0_eda
python -m experiments.exp_1_missing
python -m experiments.exp_2_outliers
python -m experiments.exp_3_sample_size
python -m experiments.exp_4_case_study
python -m experiments.make_figures
python -m experiments.make_tables
echo "done: results/*.csv, results/tables.md, figures/*.png"
