if (requireNamespace("reticulate", quietly = TRUE)) {
  try(reticulate::use_virtualenv("r-numpyro-gpu", required = FALSE), silent = TRUE)
}

test_that("check_numpyro_available returns a logical flag", {
  res <- check_numpyro_available()
  expect_type(res, "logical")
})

test_that("Spatial weights conversion, ICAR scaling factor, and validation work", {
  D <- 20
  dom_names <- paste0("area_", 1:D)
  
  # Adjacency matrix for a linear spatial chain
  W <- matrix(0, D, D)
  for (i in 1:(D - 1)) {
    W[i, i + 1] <- 1
    W[i + 1, i] <- 1
  }
  rownames(W) <- colnames(W) <- dom_names
  
  w_conv <- fastsaegpu:::.convert_spatial_weights(W, n_domains = D, domain_names = dom_names)
  expect_equal(dim(w_conv$adj_mat), c(D, D))
  expect_true(w_conv$scale_factor > 0)
  expect_equal(unname(diag(w_conv$adj_mat)), rep(0, D))
  expect_true(isSymmetric(w_conv$adj_mat))

  # Test error when W is NULL
  expect_error(
    fastsaegpu:::.convert_spatial_weights(NULL, n_domains = D),
    "Spatial matrix"
  )

  # Test error when dimension mismatches
  W_bad <- matrix(0, 5, 5)
  expect_error(
    fastsaegpu:::.convert_spatial_weights(W_bad, n_domains = D),
    "must be a"
  )

  # Test error on unsupported object type
  expect_error(
    fastsaegpu:::.convert_spatial_weights("invalid", n_domains = D),
    "Unsupported spatial object type"
  )
})

test_that("Variable extractor helper handles columns, formulas, and errors", {
  df <- data.frame(a = 1:5, b = letters[1:5])
  expect_equal(fastsaegpu:::.get_variable(df, "a"), 1:5)
  expect_equal(fastsaegpu:::.get_variable(df, ~ a), 1:5)
  expect_equal(fastsaegpu:::.get_variable(df, 1:5), 1:5)
  expect_error(fastsaegpu:::.get_variable(df, "c"), "not found in data")
  expect_error(fastsaegpu:::.get_variable(df, ~ c), "does not reference")
  expect_error(fastsaegpu:::.get_variable(df, 1:3), "does not match data")
})

test_that("Temporal operator mathematical properties are valid", {
  T_len <- 5
  
  # 1. AR(1) Toeplitz filter
  rho <- 0.0
  idx <- 0:(T_len - 1)
  diff <- outer(idx, idx, "-")
  mask <- diff > 0
  powers <- ifelse(mask, diff, 1)
  L_ar1 <- ifelse(mask, rho^powers, diag(T_len))
  expect_equal(diag(L_ar1), rep(1, T_len))
  expect_equal(L_ar1[1, 1], 1)
  
  # 2. RW(1) centering operator
  L_cumsum <- lower.tri(matrix(1, T_len, T_len), diag = TRUE) * 1.0
  P_zero <- diag(T_len) - (1 / T_len) * matrix(1, T_len, T_len)
  C_rw1 <- P_zero %*% L_cumsum
  # Sum of each column of centered RW1 operator must be 0
  expect_equal(colSums(C_rw1), rep(0, T_len), tolerance = 1e-10)
})

test_that("S3 methods for fastsae_hb_area class work properly", {
  n <- 10
  df_hb <- data.frame(
    domain = paste0("d", 1:n),
    y = rnorm(n),
    hb = rnorm(n),
    linear_pred = rnorm(n),
    sd = rep(0.1, n),
    mse = rep(0.01, n),
    rse = rep(5.0, n),
    ci_lower = rnorm(n) - 0.2,
    ci_upper = rnorm(n) + 0.2,
    random_effect = rnorm(n),
    stringsAsFactors = FALSE
  )
  estcoef <- data.frame(
    beta = c(1.5, 0.8),
    std.error = c(0.1, 0.05),
    zvalue = c(15, 16),
    pvalue = c(1e-10, 1e-12),
    ci_lower = c(1.3, 0.7),
    ci_upper = c(1.7, 0.9),
    row.names = c("(Intercept)", "x1")
  )
  hyperpar <- data.frame(
    Parameter = c("sigma2_u"),
    Estimate = c(0.25)
  )
  goodness <- c(DIC = 120.5, pD = 4.2, WAIC = 121.0, pWAIC = 4.5)
  
  obj <- structure(
    list(
      df_hb = df_hb,
      hb = df_hb,
      df_eblup = df_hb,
      estcoef = estcoef,
      hyperpar = hyperpar,
      goodness = goodness,
      family = "gaussian",
      spatial = "besag",
      temporal = "none",
      st_interaction = "none",
      device = "metal:0",
      call = quote(hb_area(y ~ x1, data = df))
    ),
    class = c("fastsae_hb_area", "fastsae")
  )
  
  # Test S3 extractors
  cf <- coef(obj)
  expect_equal(length(cf), 2)
  expect_equal(names(cf), c("(Intercept)", "x1"))
  
  fit_vals <- fitted(obj)
  expect_equal(length(fit_vals), n)
  expect_equal(fit_vals, df_hb$hb)
  
  res_vals <- residuals(obj)
  expect_equal(length(res_vals), n)
  expect_equal(res_vals, df_hb$y - df_hb$hb)
  
  sm <- summary(obj)
  expect_s3_class(sm, "summary.fastsae_hb_area")

  # Test print outputs
  expect_output(print(obj), "beta")
  expect_output(print(sm), "beta")
})

test_that("Input validation rejects invalid data across likelihood families", {
  # Non-data.frame
  expect_error(hb_area(y ~ x, data = "not_a_df"), "must be a data frame")

  # Gamma strictly positive
  df_neg <- data.frame(y = c(-1, 2, 3), x = c(1, 2, 3))
  expect_error(
    hb_area(y ~ x, data = df_neg, family = "gamma"),
    "strictly positive"
  )

  # Negative binomial non-negative integers
  expect_error(
    hb_area(y ~ x, data = df_neg, family = "nbinomial"),
    "non-negative integers"
  )

  # Poisson non-negative integers
  expect_error(
    hb_area(y ~ x, data = df_neg, family = "poisson"),
    "non-negative integers"
  )

  # Beta strictly bounded in (0, 1) and vardir requirement
  df_beta_bad <- data.frame(y = c(0.0, 0.5, 1.2), x = c(1, 2, 3), vardir = rep(0.01, 3))
  expect_error(
    hb_area(y ~ x, data = df_beta_bad, vardir = "vardir", family = "beta"),
    "strictly bounded"
  )
  df_beta_novar <- data.frame(y = c(0.2, 0.5, 0.8), x = c(1, 2, 3))
  expect_error(
    hb_area(y ~ x, data = df_beta_novar, family = "beta"),
    "vardir"
  )

  # Gaussian vardir requirement
  expect_error(
    hb_area(y ~ x, data = df_beta_novar, family = "gaussian"),
    "vardir"
  )

  # Binomial trials requirement
  expect_error(
    hb_area(y ~ x, data = df_beta_novar, family = "binomial"),
    "trials"
  )
})

test_that("End-to-end hb_area executes when NumPyro is available", {
  skip_on_cran()
  skip_if_not(fastsaegpu::check_numpyro_available(), "NumPyro/JAX not available in Python environment")
  
  set.seed(123)
  D <- 10
  x <- rnorm(D)
  vardir <- rep(0.1, D)
  y <- 2.0 + 1.2 * x + rnorm(D, sd = sqrt(vardir))
  data_df <- data.frame(y = y, x = x, vardir = vardir)
  
  fit <- hb_area(
    y ~ x,
    data = data_df,
    vardir = "vardir",
    family = "gaussian",
    warmup = 100L,
    samples = 200L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )
  
  expect_s3_class(fit, "fastsae_hb_area")
  expect_equal(nrow(fit$df_hb), D)
  expect_true(fit$estcoef["(Intercept)", "beta"] > 0)
})

test_that("hb_area fits nbinomial and gamma models", {
  skip_on_cran()
  skip_if_not(fastsaegpu::check_numpyro_available(), "NumPyro/JAX not available")

  set.seed(42)
  D <- 12
  x <- rnorm(D)
  exposure <- round(runif(D, 50, 150))
  # Negative binomial counts
  mu_nb <- exp(0.5 + 0.3 * x) * exposure
  y_nb <- rnbinom(D, size = 5, mu = mu_nb)
  data_nb <- data.frame(y = y_nb, x = x, exposure = exposure)

  fit_nb <- hb_area(
    y ~ x,
    data = data_nb,
    exposure = "exposure",
    family = "nbinomial",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )
  expect_s3_class(fit_nb, "fastsae_hb_area")
  expect_true("alpha_dispersion" %in% fit_nb$hyperpar$Parameter)
  expect_true(!is.null(fit_nb$df_hb$estimated_count))

  # Gamma positive continuous data
  mu_gam <- exp(1.0 + 0.4 * x)
  y_gam <- rgamma(D, shape = 10, rate = 10 / mu_gam)
  data_gam <- data.frame(y = y_gam, x = x)

  fit_gam <- hb_area(
    y ~ x,
    data = data_gam,
    family = "gamma",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )
  expect_s3_class(fit_gam, "fastsae_hb_area")
  expect_true("shape_param" %in% fit_gam$hyperpar$Parameter)
})

test_that("hb_area fits BYM and Leroux spatial models", {
  skip_on_cran()
  skip_if_not(fastsaegpu::check_numpyro_available(), "NumPyro/JAX not available")

  set.seed(99)
  D <- 10
  W <- matrix(0, D, D)
  for (i in 1:(D - 1)) {
    W[i, i + 1] <- 1
    W[i + 1, i] <- 1
  }
  dom_names <- paste0("d", 1:D)
  rownames(W) <- colnames(W) <- dom_names
  x <- rnorm(D)
  vardir <- rep(0.1, D)
  y <- 1.5 + 0.5 * x + rnorm(D, sd = sqrt(vardir))
  data_sp <- data.frame(domain = dom_names, y = y, x = x, vardir = vardir)

  # BYM
  fit_bym <- hb_area(
    y ~ x,
    data = data_sp,
    domain = "domain",
    W = W,
    spatial = "bym",
    vardir = "vardir",
    family = "gaussian",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )
  expect_s3_class(fit_bym, "fastsae_hb_area")
  expect_true("sigma2_spatial" %in% fit_bym$hyperpar$Parameter)

  # Leroux
  fit_leroux <- hb_area(
    y ~ x,
    data = data_sp,
    domain = "domain",
    W = W,
    spatial = "leroux",
    vardir = "vardir",
    family = "gaussian",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )
  expect_s3_class(fit_leroux, "fastsae_hb_area")
  expect_true("rho_spatial" %in% fit_leroux$hyperpar$Parameter)
})

test_that("hb_area in-model self-benchmarking and external benchmarking work accurately", {
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")

  set.seed(42)
  D <- 10
  x <- rnorm(D)
  vardir <- rep(0.01, D)
  p_true <- 1 / (1 + exp(-(0.5 + 0.8 * x)))
  y <- pmin(pmax(p_true + rnorm(D, sd = sqrt(vardir)), 0.05), 0.95)
  weights <- runif(D, 50, 150)
  df <- data.frame(y = y, x = x, vardir = vardir, pop_w = weights)

  # 1. In-model Self-Benchmarking
  fit_self <- hb_area(
    y ~ x,
    data = df,
    vardir = "vardir",
    family = "beta",
    benchmark = TRUE,
    benchmark_weights = "pop_w",
    benchmark_method = "logit",
    warmup = 80L,
    samples = 120L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )

  expect_s3_class(fit_self, "fastsae_hb_area")
  expect_true(isTRUE(fit_self$benchmarked))
  expect_equal(fit_self$benchmark_info$type, "self")
  expect_true("hb_unbenchmarked" %in% names(fit_self$df_hb))
  expect_true("sd_unbenchmarked" %in% names(fit_self$df_hb))

  w_norm <- weights / sum(weights)
  direct_agg <- sum(w_norm * y)
  hb_self_agg <- sum(w_norm * fit_self$df_hb$hb)
  expect_equal(hb_self_agg, direct_agg, tolerance = 1e-5)

  # 2. In-model External Benchmarking
  ext_target <- 0.65
  fit_ext <- hb_area(
    y ~ x,
    data = df,
    vardir = "vardir",
    family = "beta",
    benchmark = TRUE,
    benchmark_weights = "pop_w",
    benchmark_target = ext_target,
    benchmark_method = "logit",
    warmup = 80L,
    samples = 120L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )

  expect_true(isTRUE(fit_ext$benchmarked))
  expect_equal(fit_ext$benchmark_info$type, "external")
  hb_ext_agg <- sum(w_norm * fit_ext$df_hb$hb)
  expect_equal(hb_ext_agg, ext_target, tolerance = 1e-5)
})

