# fastsaegpu 0.1.0

### Bug fixes and hardening (pre-submission)

* `hb_area()` / `hb_twofold()` (Gaussian): the response is standardized internally for the sampler
  and draws are mapped back, so large-scale outcomes (e.g. incomes ~1e6) are no longer shrunk to 0
  by unit-scale priors.
* `hb_area()` benchmarking: the self-benchmark target is the weighted direct estimate over sampled
  domains on the scale of `hb` (previously binomial/Poisson used raw counts and any `NA` response
  turned all estimates into `NaN`); `benchmark_method` now defaults to `"logit"` for beta/binomial and
  `"optimal"` otherwise, and `"logit"` is rejected for unbounded families.
* `hb_area()`: `df_hb$y` is the direct estimate on the scale of `hb` (`y / trials` for binomial counts,
  `y / exposure` for Poisson/NB with exposure) so `residuals()` and `benchmark()` are consistent
  (**breaking**: previously raw counts). `estimated_total` / `estimated_count` use the benchmarked `hb`.
* Nested models (`hb_area(subarea=)`, `hb_twofold()`, `merf_area(subarea=)`): subarea codes that restart
  within each major area (`"01"`, `"02"`, ...) are no longer merged across areas, and levels are only
  auto-swapped when `domain` is actually nested within `subarea`.
* `merf_area()`: analytic MSE of the nested model now uses the total random-effect variance; missing
  values in the response/covariates give a clear error instead of misaligned `vardir`; ranger uses at
  most 2 threads under `_R_CHECK_LIMIT_CORES_`.
* `gvf_smooth()`: areas with observed `y` but zero/missing `vardir` receive the predicted variance; too few
  areas for the GVF model give a clear error instead of `NaN` variances.
* `benchmark()`: hierarchical external benchmarking skips unsampled (`NA`) direct estimates.
* Device handling: the jax-metal platform (`METAL`) is detected correctly, computation is pinned to the
  selected device, and requesting an unavailable `"metal"`/`"cuda"` device warns instead of silently
  running on CPU. `setup_numpyro_env(method = "conda")` now creates a conda environment.
* Horseshoe prior: the intercept is detected from the design matrix instead of assumed to be the first
  column (models without intercept now penalize every coefficient), and `shrinkage_factor` uses the
  regularized local scale `lambda_tilde` actually used by the prior.
* `merf_area()` spatial (SAR) model: likelihood and BLUP use the precision form (one Cholesky per
  evaluation, log-determinant from the eigenvalues of `W`) instead of dense inverses; identical
  estimates, ~6x faster for D = 400.
* `merf_area()`: the `randomForest` engine was removed (**breaking**); `ranger` moved from Suggests to
  Imports and `randomForest` was dropped from Suggests.
* `setup_numpyro_env()` installs pinned, tested versions (`jax==0.5.0`, `numpyro==0.19.0`; Metal:
  `jax==0.4.34`, `jaxlib==0.4.34`, `jax-metal==0.1.1`) and `device = "auto"` no longer selects Metal.
  jax-metal cannot compile NumPyro's NUTS (missing `popcnt`), so `device = "metal"` samples on CPU and
  now says so in its warning.
* Profiling-driven fixes:
  * `merf_area()` failed for formulas with transformed terms (`I(x^2)`, `log(x)`) because ranger's
    formula interface rejects such column names; the forest is now fitted through the x/y interface
    and `predict(newdata = )` rebuilds the model-matrix columns from the stored `terms`/`xlevels`.
  * Nested `merf_area()` assigned area BLUPs in alphabetical instead of first-appearance order
    (wrong area effects whenever the two orders differ, e.g. `P1, ..., P10`); the nested likelihood
    and BLUPs are now vectorized (per-area sums via one precomputed ordering) and shared with the bootstrap.
  * `merf_area()`: permutation importance is no longer computed in bootstrap refits and in-sample
    predictions only when needed (identical estimates; ~30-35% faster with bootstrap MSE).
  * NumPyro backend records only non-redundant per-draw arrays: `linear_pred` is derived from `hb_est`
    after sampling and the area/subarea split of the random effect is kept only for nested models
    (half the stored draws for non-nested models; identical estimates). Credible-interval quantiles are
    computed in one pass and the twofold area aggregation uses a sparse matrix instead of a per-area loop.
  * `hb_twofold()` area summaries use `rowsum()`/`tabulate()` instead of per-area loops; the unused
    fallback (which also ignored posterior covariance) was removed.
* Documentation: the Metal claims in DESCRIPTION/README/package help are corrected, `hb_area()` documents
  its return value and the self-benchmark target, `gvf_smooth()` no longer advertises formula inputs.
* Input validation: `NA` domain/subarea/time identifiers, non-positive `vardir`/`trials`/`exposure`,
  counts above `trials`, invalid benchmark weights/targets, and spatio-temporal interactions that would
  be silently dropped now fail with informative errors.

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

* Benchmarking & Calibration module added:
  * **In-Model Benchmarking in `hb_area(..., benchmark = TRUE)`:** Integrates Self-Benchmarking and External Benchmarking directly into the MCMC trace sampling on GPU NumPyro/JAX. Every MCMC sample draw is calibrated, providing exact Bayesian posterior distributions, posterior SDs, and 95% credible intervals under the benchmarking constraint without post-hoc approximations.
  * **Post-Hoc Benchmarking via `benchmark()` & `benchmark_sae()`:** Provides calibration of existing fitted models to guarantee aggregate coherence with survey direct totals (`target = NULL`) or known administrative targets (`target = value`).
  * Offers 4 calibration methods: `"logit"` (bounds estimates strictly in (0, 1) for Beta/Binomial models), `"optimal"` (MSE-weighted quadratic calibration), `"ratio"` (proportional scaling), and `"difference"` (additive shift).
  * Full S3 methods: `print()`, `summary()`, and `plot()` with ggplot2 diagnostic visualization.
  * Empirically validated on finite population simulation: external benchmarking eliminates aggregate national bias to 0.0000% and reduces both domain-level Absolute Relative Bias (ARB) and RRMSE.

* Advanced Methodological SAE Enhancements (Features A, B, C):
  * **Feature A: Generalized Variance Functions (GVF) Smoothing (`gvf_smooth()` & `smooth_vardir = TRUE`):**
    * Implements variance smoothing based on Wolter (2007), Otto & Bell (1995), and Rivest & Vandal (2003) to stabilize noisy direct sampling variances (`vardir`) and prevent artificial over-shrinkage.
    * Supports `log_linear` (with log-normal mean expectation correction), `power`, `ratio` (CV-squared), and `loess` smoothing.
    * S3 methods: `print.fastsaegpu_gvf()` and `plot.fastsaegpu_gvf()`.
  * **Feature B: Regularized Horseshoe Prior (`prior_beta = "horseshoe"`):**
    * Implements Finnish Regularized Horseshoe prior (Carvalho et al. 2010; Piironen & Vehtari 2017) in NumPyro/JAX for high-dimensional sparse covariates.
    * Unpenalized weakly informative normal intercept with Half-Cauchy local shrinkage and Inverse-Gamma slab.
    * Provides variable-specific shrinkage weights $\kappa_j \in [0, 1]$ directly in `estcoef` table (demonstrated 95-97.5% pruning of pure noise covariates in simulation).
  * **Feature C: Heavy-Tailed Student-$t$ Random Effects (`robust = TRUE`):**
    * Implements outlier-robust area random effects based on Bell & Huang (2006) and Gershunskaya & Lahiri (2018) using non-centered Student-$t$ innovations with estimated degrees of freedom $\nu_u \sim \mathcal{U}(2.5, 30.0)$.
    * Empirically shown to achieve lowest outlier domain error (ARB 4.29% & RRMSE 5.39%) by preventing localized regional shocks from dragging surrounding areas.
  * **Empirical Simulation Study (`benchmarks/simulate_advanced_features.R`):**
    * Comprehensive study across 50 domains, 15 covariates (3 signal, 12 noise), 4 outlier domains, and noisy variances.
    * Demonstrates that Full Synergy (A + B + C + Benchmarking) yields the lowest overall error (ARB 11.51% vs 13.18% direct, RRMSE 32.14% vs 39.56% direct — a 19% relative error reduction).

* Two-Level Nested Sub-Area SAE Model (`subarea`):
  * Implements the two-level nested hierarchical area model based on Torabi & Rao (2014) *Survey Methodology / JMA*, Fuller & Goyeneche (1998), and Rao & Molina (2015, Chapter 8).
  * Formulation: $\theta_{jk} = \mathbf{x}_{jk}^\top \boldsymbol{\beta} + u_j + v_{jk}$, where $u_j \sim \mathcal{N}(0, \sigma_{\text{area}}^2)$ is the major area (e.g., province) level effect and $v_{jk} \sim \mathcal{N}(0, \sigma_{\text{subarea}}^2)$ is the nested sub-area (e.g., district/kabupaten) level effect.
  * Added `subarea` argument to `hb_area()` alongside `domain`.
  * Intelligent hierarchy auto-detection: compares category cardinalities and gracefully aligns coarser levels to major area and finer levels to subarea.
  * Estimates hyperparameter `sigma2_subarea` and Intra-Cluster Correlation (ICC) diagnostic: $\text{ICC}_{\text{nested}} = \sigma_{\text{area}}^2 / (\sigma_{\text{area}}^2 + \sigma_{\text{subarea}}^2)$.
  * Full S3 methods support: `print.fastsaegpu_hb_area()` displays hierarchical levels and ICC.

* Hierarchical Bayes Twofold Subarea Model (`hb_twofold()`, Torabi & Rao 2014):
  * Dedicated GPU NumPyro implementation of the twofold subarea-level HB model, following
    Torabi & Rao (2014) *JMVA*, Rao & Molina (2015, Ch. 8), Mohadjer et al. (2007), and
    Erciulescu et al. (2019): `y_jk | theta_jk ~ N(theta_jk, psi_jk)`,
    `theta_jk = x_jk' beta + v_j + u_jk`.
  * Estimates subarea means (`df_hb`/`df_subarea`) and aggregate area means (`df_area`) simultaneously;
    area means aggregated from full posterior draws as `theta_j. = sum_k W_jk theta_jk`
    with normalized weights `W_jk = w_jk / sum_k w_jk`.
  * Supports non-sampled subareas (`y = NA`): predicted from the linking model only.
  * Returns fastsae-compatible structure plus saeHB.twofold-compatible aliases (`Mean`/`SD`/`CV`/`MSE` in `Est_sub`/`Est_area`, `coefficient`, `refVar`, `hb`/`df_eblup`).
  * Inherits `fastsae_hb_area` class: `benchmark()`, `coef()`, `fitted()`, `residuals()` work directly.

* Mixed Effects Random Forest Small Area Estimation (`merf_area()`):
  * Implements semi-parametric Mixed Effects Random Forest (MERF) and Fay-Herriot Random Forest (FH-RF) based on Krennmair & Schmid (2022) *JRSS-C*, Hajjem et al. (2014) *JSCS*, and Bukhari et al. (2025) *IPB University*.
  * Replaces linear fixed effects $X\beta$ with non-parametric Random Forest ensemble $f(X)$, capturing complex non-linearities and high-order interactions without manual feature engineering.
  * Fast C++ multithreaded engine powered by `ranger`.
  * Expectation-Maximization (EM) estimation: alternates between Random Forest fitting on adjusted response $y - u$ and profile log-likelihood optimization of area variance $\sigma_u^2$ with shrinkage $u_i = \gamma_i r_i$.
  * Parametric Bootstrap MSE (`mse_type = "bootstrap"`): computes empirical bootstrap MSE, RSE, and 95% confidence intervals based on Krennmair & Schmid (2022).
  * Extensions:
    * Two-Level Nested MERF (`subarea`): models macro-area $u_j$ and nested sub-area $v_{jk}$ with analytical profile likelihood and ICC diagnostics.
    * Spatial MERF (`spatial = W`): models spatial autoregressive SAR error correlation $\rho$ on area random effects.
    * GVF Variance Smoothing (`smooth_vardir = TRUE`): integrates with `gvf_smooth()` to stabilize noisy direct sampling variances before forest training.
    * Benchmarking & Calibration: full interoperability with `benchmark()` and `benchmark_sae()`.
  * S3 methods: `print()`, `summary()`, `plot()` (supporting `"importance"` and `"estimates"`), `coef()`, `fitted()`, `residuals()`.


