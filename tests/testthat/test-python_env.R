test_that("setup_numpyro_env honours method = 'conda' and validates arguments", {
  skip_if_not_installed("reticulate")
  calls <- character()
  rec <- function(name) function(...) {
    calls <<- c(calls, name)
    invisible(TRUE)
  }
  local_mocked_bindings(
    conda_create = rec("conda_create"), conda_install = rec("conda_install"),
    use_condaenv = rec("use_condaenv"), virtualenv_create = rec("virtualenv_create"),
    virtualenv_install = rec("virtualenv_install"), use_virtualenv = rec("use_virtualenv"),
    .package = "reticulate"
  )
  suppressMessages(setup_numpyro_env("fastsaegpu-test", method = "conda", device = "cpu"))
  expect_equal(calls, c("conda_create", "conda_install", "use_condaenv"))

  calls <- character()
  suppressMessages(setup_numpyro_env("fastsaegpu-test", method = "virtualenv", device = "cpu"))
  expect_equal(calls, c("virtualenv_create", "virtualenv_install", "use_virtualenv"))

  expect_error(setup_numpyro_env(method = "pip"), "should be one of")
  expect_error(setup_numpyro_env(device = "gpu"), "should be one of")
  expect_error(check_numpyro_available(device = "gpu"), "should be one of")
})

test_that("setup_numpyro_env pins tested jax / jax-metal / numpyro versions", {
  skip_if_not_installed("reticulate")
  pkgs <- list()
  local_mocked_bindings(
    virtualenv_create = function(...) invisible(TRUE),
    virtualenv_install = function(envname, packages, ...) {
      pkgs[[length(pkgs) + 1]] <<- packages
      invisible(TRUE)
    },
    use_virtualenv = function(...) invisible(TRUE),
    .package = "reticulate"
  )
  suppressMessages(setup_numpyro_env("x", device = "metal"))
  suppressMessages(setup_numpyro_env("x", device = "cpu"))
  suppressMessages(setup_numpyro_env("x", device = "cuda"))
  expect_true(all(c("jax==0.4.34", "jaxlib==0.4.34", "jax-metal==0.1.1", "numpyro==0.19.0") %in% pkgs[[1]]))
  expect_true(all(c("jax==0.5.0", "jaxlib==0.5.0", "numpyro==0.19.0") %in% pkgs[[2]]))
  expect_true(all(c("jax[cuda12]==0.5.0", "numpyro==0.19.0") %in% pkgs[[3]]))
  # auto never selects Metal (NUTS cannot run there)
  skip_if(nzchar(Sys.which("nvidia-smi")), "CUDA GPU present")
  suppressMessages(setup_numpyro_env("x", device = "auto"))
  expect_false(any(grepl("jax-metal", pkgs[[4]])))
})
