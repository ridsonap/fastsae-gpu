# CRAN Submission Comments

## Package: fastsaegpu 0.1.0

### Submission Type
Initial release.

### Test environments
* local macOS (Apple Silicon M-series, macOS 26.6), R 4.6.1
* GitHub Actions:
  * Ubuntu 24.04 (R-devel, R-release, R-oldrel-1)
  * macOS latest (R-release)
  * Windows Server latest (R-release)
* win-builder (R-devel, R-release)

### R CMD check results
There were 0 ERRORS, 0 WARNINGS.

Status: 1 NOTE
* checking CRAN incoming feasibility ... NOTE
  Maintainer: ‘Ridson Al Farizal P <alfrzlp@gmail.com>’
  New submission
  (Expected for an initial package submission to CRAN.)

### Notes for the CRAN Team
* `fastsaegpu` provides GPU-accelerated Hierarchical Bayesian Small Area Estimation using 'NumPyro' and 'JAX'.
* Python, 'NumPyro', and 'JAX' are optional backend dependencies interfaced via the 'reticulate' package.
* All exported functions check for Python/NumPyro availability before execution and provide informative messages if the environment is not yet configured.
* All unit tests requiring Python/NumPyro use `testthat::skip_on_cran()` and `testthat::skip_if_not()` and gracefully skip on environments without Python/NumPyro, ensuring `R CMD check` passes cleanly on all CRAN check servers.
* Examples executing MCMC estimation are placed inside `\donttest{}` blocks and guarded by `check_numpyro_available()` to keep check times within CRAN policy limits (< 5 seconds) and to avoid assuming an external Python/GPU environment on CRAN check servers.
