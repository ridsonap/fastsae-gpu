# fastsaegpu 0.1.0

### Initial CRAN Release

* Core estimation function `hb_area()` providing Hierarchical Bayesian area-level Small Area Estimation (SAE).
* GPU acceleration via 'NumPyro' and 'JAX' supporting Apple Silicon (Metal) and NVIDIA (CUDA), with automatic CPU fallback.
* Supported likelihood families:
  * Gaussian (`family = "gaussian"`) with direct sampling variance `vardir`.
  * Binomial (`family = "binomial"`) with area-level `trials`.
  * Poisson (`family = "poisson"`) with expected `exposure`.
  * Beta (`family = "beta"`) for rates and proportions bounded in (0, 1).
  * Negative Binomial (`family = "nbinomial"`) for overdispersed counts with estimated dispersion parameter.
  * Gamma (`family = "gamma"`) for strictly positive continuous survey indicators.
* Supported spatial random effect models:
  * Besag (ICAR) intrinsic autoregression.
  * BYM (Besag-York-Mollié: ICAR + IID convolution).
  * BYM2 (Scaled Besag-York-Mollié; Riebler et al., 2016).
  * Leroux CAR model with spatial correlation parameter.
* Supported temporal random effect dynamics:
  * AR(1) first-order autoregressive process.
  * RW(1) random walk with sum-to-zero identifiability constraint.
  * IID temporal effects.
* Spatio-temporal interaction structures:
  * Type I (unstructured space-time).
  * Type II (temporal structure, unstructured space).
  * Type III (spatial structure, unstructured time).
  * Type IV (fully structured spatio-temporal CAR x AR1/RW1).
  * Separable Kronecker interactions and domain-specific interactions.
* Python environment management:
  * `check_numpyro_available()`: tests Python/JAX/NumPyro installation and target device availability.
  * `setup_numpyro_env()`: automates virtual environment setup for Metal/CUDA/CPU.
* Complete S3 generic methods for fitted `fastsae_hb_area` objects:
  * `print()`, `summary()`, `coef()`, `fitted()`, and `residuals()`.
* Comprehensive empirical benchmarks and feature comparisons against `tipsae` (Stan MCMC) and `fastsae` (INLA baseline) across Beta, Spatial Beta, and Spatio-Temporal Beta models with structured datasets and visualizations in `benchmarks/`.

* Finite population Monte Carlo simulation study added (`benchmarks/run_finite_population_simulation.R`):
  * Comprehensive benchmark comparing `fastsaegpu`, `tipsae` (Stan NUTS), `fastsae` Frequentist EBLUP (REML), and `fastsae` Bayesian (INLA) across Non-Spatial and Spatial (Besag ICAR) models with $D = 50$ domains, $p = 3$ individual-level covariates, and $N \approx 150.000$ individuals.
  * Demonstrates exact statistical equivalence with state-of-the-art implementations (RRMSE 3.36% non-spatial, 3.10% spatial; CP95 93-94%; Pearson $r > 0.957$ and $r > 0.971$).
  * Fully documented with structured datasets (`simulation_summary.csv`, `simulation_domain_estimates.csv`) and 300 DPI publication-quality visualizations (`simulation_accuracy_comparison.png`, `simulation_runtime_comparison.png`).
