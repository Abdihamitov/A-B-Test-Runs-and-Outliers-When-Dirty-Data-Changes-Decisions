# Work plan

**Topic.** How missing values and outliers change the conclusions of an A/B test: a simulation on open data.

**Question.** Under which corruption mechanisms and rates does the standard analysis (Welch after
dropping missing rows, zero or mean imputation, winsorization, Mann-Whitney, post-stratification)
keep the type I error and an unbiased estimate, and under which does it change the decision?

**Data.** Main: Hillstrom E-Mail Analytics (64,000 customers, 3 randomized arms, open CSV).
The control group serves as the population for a semi-synthetic simulation. Optional: synthetic
data with a known covariate structure (MAR check) and Cookie Cats (Kaggle).

**Design (ADEMP).**
1. Two samples of size n are drawn with replacement from the control; in the effect scenario an
   effect sized for 80% power is injected into B.
2. Corruption: MCAR / MAR / MNAR missingness, in both arms or only in B, 5 to 30%;
   outliers: bots in both arms, bots in one arm, unit error x100, 0.05 to 1%.
3. Handling and tests: 9 handlers; Welch, z, Mann-Whitney, bootstrap, post-stratification.
4. Measures: type I error, power (with Wilson CI), bias, CI coverage and width, decision flip
   rate, type S and M errors, SRM detection.

**Experiments.**
- E0. Data description: zeros, quantiles, association with covariates.
- E1. Missingness: 5 mechanisms x 4 rates x 3 n x 3 metrics x (A/A, effect).
- E2. Outliers: 3 types x 4 rates x 3 n x 2 metrics x (A/A, effect).
- E3. Sample size from 500 to 50,000 under fixed corruption.
- E4. Real comparison, e-mail vs control: clean analysis and robustness to corruption.

**Analytical benchmarks** (to check the simulation): type I error under mean imputation
2*Phi_bar(1.96(1-r)); bias of zero imputation -r*mu; SRM threshold n* = 10.83(2-r)/r^2;
power under MCAR Phi(2.80 sqrt(1-r) - 1.96).

**Hypotheses.** H1 symmetric missingness is harmless for type I error; H2 differential missingness
is biased and type I error grows with n; H3 mean imputation inflates type I error; H4 symmetric
outliers kill power, one-arm outliers cause false positives; H5 SRM is blind for small n and r.

**Order.** Code and unit tests -> QUICK=1 smoke run -> main run -> tables and figures ->
paper -> executed notebook on GitHub -> PDF.

**Risks.** Weak covariate association in Hillstrom (MAR close to MCAR): covered by the synthetic
source. Very sparse spend metric: normal approximation checked with the bootstrap and with clean power.
