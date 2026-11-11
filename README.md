# Missing Values and Outliers in A/B Tests: When Dirty Data Changes the Decision

A simulation study on the open Hillstrom e-mail dataset: how the mechanism and rate of
missing values, the type and rate of outliers, and the sample size change the conclusions
of an A/B test (type I error, power, bias, CI coverage, share of flipped decisions).

- `paper.pdf` / `paper.md`: the paper (23 pages, results from the run stored in `results/`)
- `notebooks/report.ipynb`: executed report with all tables and figures
- `abtest_sim/`: simulation package; `experiments/`: experiment scripts
- `results/`: raw CSVs, `*.meta.json` (versions, run time, seed), `tables.md`
- `figures/`: all figures
- `PLAN.md`: one-page work plan

## Main findings

- Missingness that is the same in both arms keeps type I error nominal; it only costs power.
- Missingness in one arm that depends on the value is biased, and the bias does not shrink with n:
  5% loss takes the A/A false positive rate from 0.068 at 500 users per arm to 0.990 at 50,000.
  At 20% loss every significant result has the wrong sign.
- Mean imputation inflates type I error as 2*Phi_bar(1.96(1-r)), e.g. about 0.11 at r = 0.2.
- Post-stratification removes MAR bias completely; it does not help against MNAR.
- Winsorizing at the pooled Q99 turns a zero-inflated revenue metric into zeros; winsorizing at
  Q99 of positives fails once bots exceed 1% of nonzero values.
- An SRM check fires only above n* ~ 10.83(2-r)/r^2 per arm (about 8,400 for 5% loss).

## Reproduce in 30 minutes

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
QUICK=1 ./run_all.sh     # smoke test, a few minutes
./run_all.sh             # main run (N_SIM=1000): about 13 min on 6 cores
jupyter nbconvert --to notebook --execute --inplace notebooks/report.ipynb
python tools/build_pdf.py   # needs Google Chrome or Chromium
```

Environment variables: `QUICK=1`, `N_SIM=500`, `WORKERS=4`, `SOURCE=hillstrom|synthetic|cookie_cats`.

The Hillstrom CSV is downloaded automatically into `data/hillstrom.csv` (not committed).
SHA-256 of the version used: `0e5893329d8b93cefecc571777672028290ab69865718020c78c7284f291aece`.
The server can be slow; if the download fails, fetch the file from the URL in
`abtest_sim/data.py` manually. Cookie Cats (Kaggle) goes into `data/cookie_cats.csv` manually.

## Scripts and artifacts

| Step | Command | Artifacts |
|---|---|---|
| 0 | `python -m pytest -q tests/` | implementation checks |
| 1 | `python -m experiments.exp_0_eda` | `results/eda_*.csv`, `figures/fig1_distributions.png` |
| 2 | `python -m experiments.exp_1_missing` | `results/exp1_missing.csv` |
| 3 | `python -m experiments.exp_2_outliers` | `results/exp2_outliers.csv` |
| 4 | `python -m experiments.exp_3_sample_size` | `results/exp3_sample_size.csv` |
| 5 | `python -m experiments.exp_4_case_study` | `results/exp4_*.csv` |
| 6 | `python -m experiments.make_figures` | `figures/fig2..fig7*.png` |
| 7 | `python -m experiments.make_tables` | `results/tables.md` |
| 8 | `python tools/build_pdf.py` | `paper.pdf` |

Tested with Python 3.14.4, numpy 2.5.4, pandas 3.1.1, scipy 1.18.1, matplotlib 3.11.2.
