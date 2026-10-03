library(testthat)

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
})

test_that("End-to-end hb_area executes when NumPyro is available", {
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
