test_that("merf_area runs basic area model and produces valid estimates", {
  set.seed(42)
  D <- 25
  x1 <- stats::rnorm(D)
  x2 <- stats::runif(D)
  y <- 2 * sin(pi * x1) + 1.5 * x2^2 + stats::rnorm(D, 0, 0.2)
  vardir <- rep(0.04, D)
  df <- data.frame(
    domain = paste0("area_", 1:D),
    y = y,
    vardir = vardir,
    x1 = x1,
    x2 = x2
  )

  fit <- merf_area(
    formula = y ~ x1 + x2,
    data = df,
    vardir = "vardir",
    domain = "domain",
    num_trees = 50,
    mse_type = "none",
    seed = 123
  )

  expect_s3_class(fit, "fastsaegpu_merf")
  expect_s3_class(fit, "fastsaegpu_model")
  expect_equal(nrow(fit$estimates), D)
  expect_true(all(c("domain", "y", "merf", "rf_pred", "random_effect", "gamma", "sd", "mse", "rse") %in% names(fit$estimates)))

  # Shrinkage factor should be strictly between 0 and 1
  expect_true(all(fit$estimates$gamma > 0 & fit$estimates$gamma < 1))

  # Correlation between direct y and merf should be very strong
  expect_true(stats::cor(fit$estimates$y, fit$estimates$merf) > 0.85)

  # Hyperparameters
  expect_true(fit$hyperparams$sigma2_u > 0)
  expect_true(fit$hyperparams$iterations >= 1)
})

test_that("merf_area computes parametric bootstrap MSE correctly", {
  set.seed(42)
  D <- 20
  x1 <- stats::rnorm(D)
  y <- 1.5 * x1 + stats::rnorm(D, 0, 0.3)
  vardir <- rep(0.02, D)
  df <- data.frame(domain = paste0("d", 1:D), y = y, vardir = vardir, x1 = x1)

  fit_boot <- merf_area(
    formula = y ~ x1,
    data = df,
    vardir = "vardir",
    domain = "domain",
    num_trees = 40,
    mse_type = "bootstrap",
    B = 10,
    seed = 456
  )

  expect_equal(fit_boot$B, 10)
  expect_true(all(fit_boot$estimates$mse > 0))
  expect_true(all(fit_boot$estimates$sd > 0))
  expect_true(all(fit_boot$estimates$rse > 0))
  expect_true(all(fit_boot$estimates$ci_lower < fit_boot$estimates$merf))
  expect_true(all(fit_boot$estimates$ci_upper > fit_boot$estimates$merf))
})

test_that("merf_area extracts variable importance", {
  set.seed(42)
  D <- 30
  x_sig <- stats::rnorm(D)
  x_noise <- stats::rnorm(D)
  y <- 3 * x_sig + stats::rnorm(D, 0, 0.2)
  vardir <- rep(0.03, D)
  df <- data.frame(domain = paste0("d", 1:D), y = y, vardir = vardir, x_sig = x_sig, x_noise = x_noise)

  fit <- merf_area(y ~ x_sig + x_noise, data = df, vardir = "vardir", num_trees = 60, seed = 789)

  expect_named(fit$importance)
  expect_equal(length(fit$importance), 2)
  # Signal covariate should have higher importance than noise
  expect_true(fit$importance["x_sig"] > fit$importance["x_noise"])
  expect_equal(coef(fit), fit$importance)
})

test_that("merf_area supports two-level nested sub-area structure", {
  set.seed(42)
  J <- 4
  K <- 4
  D <- J * K
  prov <- rep(paste0("Prov_", 1:J), each = K)
  kab <- paste0("Kab_", 1:D)
  u_prov <- stats::rnorm(J, 0, 0.6)
  v_kab <- stats::rnorm(D, 0, 0.3)
  x <- stats::rnorm(D)
  y <- 1.8 * x + u_prov[as.numeric(factor(prov))] + v_kab + stats::rnorm(D, 0, 0.1)
  vardir <- rep(0.015, D)
  df <- data.frame(prov = prov, kab = kab, y = y, vardir = vardir, x = x)

  fit_nested <- merf_area(
    formula = y ~ x,
    data = df,
    vardir = "vardir",
    domain = "prov",
    subarea = "kab",
    num_trees = 40,
    mse_type = "none",
    seed = 101
  )

  expect_true(fit_nested$is_nested)
  expect_true("subarea" %in% names(fit_nested$estimates))
  expect_true(!is.null(fit_nested$hyperparams$sigma2_area))
  expect_true(!is.null(fit_nested$hyperparams$sigma2_subarea))
  expect_true(!is.null(fit_nested$hyperparams$icc_nested))
  expect_true(fit_nested$hyperparams$icc_nested >= 0 && fit_nested$hyperparams$icc_nested <= 1)

  # Auto-swap test: pass domain = "kab" and subarea = "prov"
  fit_swap <- merf_area(
    formula = y ~ x,
    data = df,
    vardir = "vardir",
    domain = "kab",
    subarea = "prov",
    num_trees = 40,
    mse_type = "none",
    seed = 101
  )
  expect_true(fit_swap$is_nested)
  expect_equal(nrow(fit_swap$estimates), D)
})

test_that("merf_area supports Spatial MERF with adjacency matrix", {
  set.seed(42)
  D <- 12
  x <- stats::rnorm(D)
  W <- matrix(0, D, D)
  for (i in 1:D) {
    left <- if (i == 1) D else i - 1
    right <- if (i == D) 1 else i + 1
    W[i, left] <- 0.5
    W[i, right] <- 0.5
  }
  u <- as.vector(solve(diag(D) - 0.3 * W) %*% stats::rnorm(D, 0, 0.3))
  vardir <- rep(0.02, D)
  y <- 1.5 * x + u + stats::rnorm(D, 0, sqrt(vardir))
  df <- data.frame(domain = paste0("area_", 1:D), y = y, vardir = vardir, x = x)

  fit_sp <- merf_area(
    formula = y ~ x,
    data = df,
    vardir = "vardir",
    spatial = W,
    num_trees = 40,
    mse_type = "none",
    seed = 202
  )

  expect_true(fit_sp$is_spatial)
  expect_true(!is.null(fit_sp$hyperparams$rho_spatial))
  expect_true(fit_sp$hyperparams$rho_spatial > -1 && fit_sp$hyperparams$rho_spatial < 1)
})

test_that("merf_area S3 methods work as expected", {
  set.seed(42)
  D <- 15
  x1 <- stats::rnorm(D)
  df <- data.frame(domain = paste0("d", 1:D), y = 2*x1 + stats::rnorm(D, 0, 0.2), vardir = rep(0.02, D), x1 = x1)
  fit <- merf_area(y ~ x1, data = df, vardir = "vardir", num_trees = 30, mse_type = "none", seed = 303)

  # print & summary
  expect_output(print(fit), "sigma2_u")
  sm <- summary(fit)
  expect_s3_class(sm, "summary.fastsaegpu_merf")
  expect_output(print(sm), "x1")

  # accessors
  expect_equal(length(fitted(fit)), D)
  expect_equal(length(residuals(fit)), D)
  expect_equal(fitted(fit) + residuals(fit), fit$estimates$y)

  # plot methods
  if (requireNamespace("ggplot2", quietly = TRUE)) {
    p1 <- plot(fit, type = "importance")
    expect_s3_class(p1, "ggplot")
    p2 <- plot(fit, type = "estimates")
    expect_s3_class(p2, "ggplot")
  }
})

test_that("merf_area integrates seamlessly with benchmark()", {
  set.seed(42)
  D <- 20
  x1 <- stats::rnorm(D)
  df <- data.frame(
    domain = paste0("d", 1:D),
    y = stats::runif(D, 0.1, 0.4),
    vardir = rep(0.005, D),
    x1 = x1,
    pop = sample(1000:5000, D)
  )

  fit <- merf_area(y ~ x1, data = df, vardir = "vardir", num_trees = 30, mse_type = "none", seed = 404)

  # Self-benchmarking
  bm_self <- benchmark(fit, weight = "pop")
  expect_s3_class(bm_self, "fastsaegpu_benchmark")
  expect_equal(nrow(bm_self), D)

  # External benchmarking
  bm_ext <- benchmark(fit, target = 0.30, weight = "pop")
  expect_s3_class(bm_ext, "fastsaegpu_benchmark")
  expect_equal(sum(bm_ext$weight * bm_ext$benchmarked) / sum(bm_ext$weight), 0.30, tolerance = 1e-6)
})

test_that("merf_area supports smooth_vardir with GVF", {
  set.seed(42)
  D <- 20
  x1 <- stats::rnorm(D)
  df <- data.frame(
    domain = paste0("d", 1:D),
    y = exp(x1) + 2,
    vardir = stats::runif(D, 0.01, 0.08),
    x1 = x1
  )

  fit_gvf <- merf_area(
    formula = y ~ x1,
    data = df,
    vardir = "vardir",
    smooth_vardir = TRUE,
    gvf_method = "log_linear",
    num_trees = 30,
    mse_type = "none",
    seed = 505
  )

  expect_false(is.null(fit_gvf$gvf))
  expect_s3_class(fit_gvf$gvf, "fastsaegpu_gvf")
})

test_that("merf_area errors appropriately on invalid inputs", {
  df <- data.frame(y = c(1, 2, 3), x = c(1, 2, 3), vardir = c(0.1, -0.2, 0.1))
  expect_error(merf_area(y ~ x, data = df, vardir = "vardir"), "strictly positive")
  expect_error(merf_area(y ~ x, data = df, vardir = "missing_col"), "not found")
  expect_error(merf_area(y ~ 1, data = df, vardir = "vardir"), "at least one covariate")
})

test_that("merf_area supports precision weighting and OOB residuals", {
  set.seed(601)
  D <- 20
  df <- data.frame(
    domain = paste0("d", 1:D),
    y = stats::rnorm(D, 5, 1),
    vardir = stats::runif(D, 0.02, 0.20),
    x1 = stats::rnorm(D),
    x2 = stats::runif(D)
  )

  fit_wt <- merf_area(
    formula = y ~ x1 + x2,
    data = df,
    vardir = "vardir",
    weighted = TRUE,
    use_oob = TRUE,
    num_trees = 40,
    mse_type = "none",
    seed = 601
  )

  expect_true(fit_wt$hyperparams$weighted)
  expect_true(fit_wt$hyperparams$use_oob)
  expect_equal(length(fit_wt$estimates$merf), D)
  expect_false(any(is.na(fit_wt$estimates$merf)))
})

test_that("merf_area performs automated feature screening", {
  set.seed(701)
  D <- 25
  x_signal <- stats::rnorm(D)
  x_noise1 <- stats::rnorm(D)
  x_noise2 <- stats::rnorm(D)
  y <- 3 + 2.5 * x_signal + stats::rnorm(D, 0, 0.3)

  df <- data.frame(
    domain = paste0("d", 1:D),
    y = y,
    vardir = rep(0.05, D),
    x_signal = x_signal,
    x_noise1 = x_noise1,
    x_noise2 = x_noise2
  )

  fit_screen <- merf_area(
    formula = y ~ x_signal + x_noise1 + x_noise2,
    data = df,
    vardir = "vardir",
    feature_screening = TRUE,
    importance_threshold = 0.0,
    num_trees = 60,
    mse_type = "none",
    seed = 701
  )

  expect_true(fit_screen$hyperparams$feature_screening)
  expect_true("x_signal" %in% fit_screen$selected_vars)
  expect_true(length(fit_screen$selected_vars) <= 3)
})

test_that("merf_area supports hyperparameter auto-tuning and predict method", {
  set.seed(801)
  D <- 20
  df <- data.frame(
    domain = paste0("d", 1:D),
    y = stats::rnorm(D, 10, 2),
    vardir = rep(0.1, D),
    x1 = stats::rnorm(D),
    x2 = stats::rnorm(D),
    x3 = stats::runif(D)
  )

  fit_tuned <- merf_area(
    formula = y ~ x1 + x2 + x3,
    data = df,
    vardir = "vardir",
    tune_params = TRUE,
    num_trees = 40,
    mse_type = "none",
    seed = 801
  )

  expect_true(fit_tuned$hyperparams$tune_params)

  # Test predict.fastsaegpu_merf in-sample
  p_in <- predict(fit_tuned)
  expect_equal(p_in, fit_tuned$estimates$merf)

  # Test predict.fastsaegpu_merf out-of-sample (newdata)
  df_new <- data.frame(
    x1 = c(0.5, -0.5),
    x2 = c(1.0, -1.0),
    x3 = c(0.2, 0.8)
  )
  p_out <- predict(fit_tuned, newdata = df_new)
  expect_equal(length(p_out), 2)
  expect_false(any(is.na(p_out)))
})


test_that("merf_area rejects missing values instead of misaligning vardir/domain", {
  skip_if_not_installed("ranger")
  set.seed(11)
  df <- data.frame(y = rnorm(20, 5), x = rnorm(20), vardir = runif(20, 0.05, 0.2), dom = paste0("d", 1:20))
  df$y[2] <- NA
  expect_error(
    merf_area(y ~ x, df, vardir = "vardir", domain = "dom", mse_type = "none", num_trees = 20, num_threads = 1),
    "missing values"
  )
})

test_that("nested merf_area analytic MSE uses the total random-effect variance", {
  skip_if_not_installed("ranger")
  set.seed(12)
  df <- data.frame(y = rnorm(24, 5), x = rnorm(24), vardir = rep(0.1, 24),
                   prov = rep(paste0("P", 1:4), each = 6), kab = paste0("K", 1:24))
  fit <- merf_area(y ~ x, df, vardir = "vardir", domain = "prov", subarea = "kab",
                   mse_type = "none", num_trees = 50, num_threads = 1, seed = 1)
  s2 <- fit$hyperparams$sigma2_area + fit$hyperparams$sigma2_subarea
  expect_equal(fit$estimates$mse, s2 * df$vardir / (s2 + df$vardir))
  expect_equal(fit$estimates$mse, fit$estimates$gamma * df$vardir)
})

test_that("nested merf_area does not swap levels when subarea codes restart per area", {
  skip_if_not_installed("ranger")
  set.seed(13)
  prov <- rep(paste0("P", 1:6), each = 4)
  df <- data.frame(y = rnorm(24, 5) + rnorm(6)[as.integer(factor(prov))], x = rnorm(24),
                   vardir = rep(0.1, 24), prov = prov, kab = sprintf("%02d", rep(1:4, 6)))
  df$kab_global <- paste(df$prov, df$kab)
  f_rel <- merf_area(y ~ x, df, vardir = "vardir", domain = "prov", subarea = "kab",
                     mse_type = "none", num_trees = 50, num_threads = 1, seed = 1)
  f_glob <- merf_area(y ~ x, df, vardir = "vardir", domain = "prov", subarea = "kab_global",
                      mse_type = "none", num_trees = 50, num_threads = 1, seed = 1)
  expect_equal(f_rel$estimates$merf, f_glob$estimates$merf)
  expect_equal(f_rel$hyperparams$sigma2_area, f_glob$hyperparams$sigma2_area)
})

test_that("merf_area limits ranger threads under CRAN's _R_CHECK_LIMIT_CORES_", {
  skip_if_not_installed("ranger")
  old <- Sys.getenv("_R_CHECK_LIMIT_CORES_", unset = NA)
  Sys.setenv("_R_CHECK_LIMIT_CORES_" = "TRUE")
  on.exit(if (is.na(old)) Sys.unsetenv("_R_CHECK_LIMIT_CORES_") else Sys.setenv("_R_CHECK_LIMIT_CORES_" = old))
  threads <- c()
  real_ranger <- ranger::ranger
  local_mocked_bindings(ranger = function(...) {
    threads <<- c(threads, list(...)$num.threads %||% NA)
    real_ranger(...)
  }, .package = "ranger")
  set.seed(14)
  df <- data.frame(y = rnorm(15, 5), x = rnorm(15), vardir = rep(0.1, 15))
  merf_area(y ~ x, df, vardir = "vardir", mse_type = "none", num_trees = 20, max_iter = 2, seed = 1)
  expect_true(length(threads) > 0 && all(threads == 2))
})

test_that("merf_area only supports the ranger engine", {
  df <- data.frame(y = rnorm(10, 5), x = rnorm(10), vardir = rep(0.1, 10))
  expect_error(merf_area(y ~ x, df, vardir = "vardir", engine = "randomForest", mse_type = "none"), "should be")
  desc <- read.dcf(system.file("DESCRIPTION", package = "fastsaegpu"))
  expect_false(any(grepl("randomForest", desc[, intersect(colnames(desc), c("Imports", "Suggests"))])))
})

test_that("precision-form SAR likelihood and BLUP match the dense covariance formulas", {
  # Platform-independent check of the math (forests differ across platforms, the algebra does not)
  set.seed(31)
  D <- 12
  W <- matrix(0, D, D)
  for (i in 1:(D - 1)) W[i, i + 1] <- W[i + 1, i] <- 1
  W[1, 5] <- W[5, 1] <- 1
  W <- W / rowSums(W)
  psi <- runif(D, 0.05, 0.3)
  r <- rnorm(D)
  sar <- fastsaegpu:::.sar_likelihood(W, psi)
  for (par in list(c(0.8, 0.15), c(0.3, -0.6), c(2.5, 0.9))) {
    A_inv <- solve(diag(D) - par[2] * W)
    Sigma_u <- par[1] * A_inv %*% t(A_inv)
    V <- Sigma_u + diag(psi)
    nll_dense <- 0.5 * (as.numeric(determinant(V)$modulus) + sum(r * solve(V, r)))
    expect_equal(sar$nll(par, r), nll_dense, tolerance = 1e-10)
    expect_equal(sar$blup(par[1], par[2], r), as.vector(Sigma_u %*% solve(V, r)), tolerance = 1e-10)
  }
  expect_equal(sar$nll(c(1, 0.99), r), 1e10)
})

test_that("merf_area supports transformed formula terms and predict() rebuilds them for newdata", {
  set.seed(15)
  df <- data.frame(x = rnorm(60), x2 = rnorm(60), vardir = runif(60, 0.05, 0.2))
  df$y <- df$x^2 + df$x2 + rnorm(60, 0, 0.3)
  fit <- merf_area(y ~ x + I(x^2) + log(x2 + 5), df, vardir = "vardir", mse_type = "none",
                   num_trees = 50, num_threads = 1, seed = 1)
  expect_equal(fit$selected_vars, c("x", "I(x^2)", "log(x2 + 5)"))
  # predict() takes raw covariates and reproduces the in-sample forest predictions
  expect_equal(predict(fit, newdata = df[, c("x", "x2")]), fit$estimates$rf_pred)
  expect_error(predict(fit, newdata = df[, "x", drop = FALSE]), "Missing required covariates")
})

test_that("nested merf_area is invariant to how the major areas are labelled", {
  set.seed(5)
  g <- rep(1:4, each = 8)
  df <- data.frame(x = rnorm(32), vardir = rep(0.05, 32), kab = paste0("K", 1:32))
  df$y <- 2 + df$x + c(-2, -1, 1, 2)[g] + rnorm(32, 0, 0.2)
  fit_lab <- function(labs) {
    df$prov <- labs[g]
    merf_area(y ~ x, df, vardir = "vardir", domain = "prov", subarea = "kab", mse_type = "bootstrap",
              B = 2, num_trees = 50, num_threads = 1, seed = 1)
  }
  a <- fit_lab(c("A", "B", "C", "D"))
  b <- fit_lab(c("D", "C", "B", "A"))  # first-appearance order differs from alphabetical order
  expect_equal(a$estimates$merf, b$estimates$merf)
  expect_equal(a$estimates$mse, b$estimates$mse)
})

test_that("print shows a long call verbatim (no cli list collapsing of deparse lines)", {
  set.seed(1)
  D <- 20
  df <- data.frame(domain = paste0("d", 1:D), y = rnorm(D), vardir = 0.1,
                   x1 = rnorm(D), x2 = rnorm(D))
  fit <- merf_area(y ~ x1 + x2, data = df, vardir = "vardir", domain = "domain",
                   weighted = TRUE, use_oob = TRUE, num_trees = 20, max_iter = 2,
                   mse_type = "none", seed = 1, verbose = FALSE)
  expect_gt(length(deparse(fit$call)), 1L)
  out <- paste(cli::cli_fmt(print(fit)), collapse = " ")
  expect_false(grepl(", ,", out, fixed = TRUE))
  expect_false(grepl(", and ", out, fixed = TRUE))
})
