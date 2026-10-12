## Submission

Initial submission of fastsaegpu 0.1.0.

## Test environments

* local: macOS 26.6 (Apple Silicon), R 4.6.1 -- with and without the optional
  Python backend
* GitHub Actions: Ubuntu 24.04 (R-devel, R-release, R-oldrel-1), macOS
  (R-release), Windows Server (R-release)

## R CMD check results

0 errors | 0 warnings | 1 note

* checking CRAN incoming feasibility ... NOTE
  New submission

  Words such as 'Herriot', 'NumPyro', 'JAX', 'CUDA', 'SAE', 'MCMC' and 'NUTS'
  may be flagged as possibly misspelled; they are names and standard acronyms.

## Notes on the optional Python backend

* `hb_area()` and `hb_twofold()` run MCMC through 'NumPyro'/'JAX' via
  'reticulate'. Python is optional (see SystemRequirements) and is never
  installed or downloaded during checks: `check_numpyro_available()` returns
  `FALSE` when no configured Python with 'numpyro' and 'jax' is found, and
  explicitly prevents 'reticulate' from provisioning its managed ('uv') Python.
* Examples that run MCMC are wrapped in `\donttest{}` (they take more than 5 s
  when the backend is available) and guarded by `if (check_numpyro_available())`.
* `setup_numpyro_env()` creates a Python environment and installs packages, so
  its example is wrapped in `\dontrun{}`.
* Tests that need the Python backend use `skip_on_cran()`; `ranger` uses at
  most 2 threads when `_R_CHECK_LIMIT_CORES_` is set.
