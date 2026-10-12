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
  skip_on_cran()
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

test_that("hb_area advanced features (Horseshoe, Student-t robust, GVF smoothing) work properly", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")

  set.seed(123)
  D <- 15
  # True signal x1, pure noise x2, x3
  x1 <- rnorm(D)
  x2 <- rnorm(D)
  x3 <- rnorm(D)
  u_outlier <- rnorm(D, sd = 0.2)
  u_outlier[1] <- 3.5 # Severe localized outlier domain
  n_sample <- sample(20:50, D, replace = TRUE)
  vardir_raw <- (0.25 / n_sample) * rlnorm(D, 0, 0.4)
  y <- 1.0 + 1.5 * x1 + 0.0 * x2 + 0.0 * x3 + u_outlier + rnorm(D, sd = sqrt(vardir_raw))

  df <- data.frame(y = y, x1 = x1, x2 = x2, x3 = x3, vardir = vardir_raw, n = n_sample)

  # 1. Test Regularized Horseshoe Prior (Feature B)
  fit_hs <- hb_area(
    y ~ x1 + x2 + x3,
    data = df,
    vardir = "vardir",
    family = "gaussian",
    prior_beta = "horseshoe",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )

  expect_s3_class(fit_hs, "fastsae_hb_area")
  expect_equal(fit_hs$prior_beta, "horseshoe")
  expect_true("tau_horseshoe" %in% fit_hs$hyperpar$Parameter)
  expect_true("shrinkage_factor" %in% names(fit_hs$estcoef))
  expect_equal(nrow(fit_hs$estcoef), 4) # Intercept + 3 slopes
  # Signal x1 should have lower shrinkage (closer to 0) than noise x2/x3
  expect_true(fit_hs$estcoef["x1", "shrinkage_factor"] < fit_hs$estcoef["x2", "shrinkage_factor"] ||
              fit_hs$estcoef["x1", "shrinkage_factor"] < fit_hs$estcoef["x3", "shrinkage_factor"])

  # 2. Test Robust Student-t Random Effects (Feature C)
  fit_rob <- hb_area(
    y ~ x1,
    data = df,
    vardir = "vardir",
    family = "gaussian",
    robust = TRUE,
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )

  expect_s3_class(fit_rob, "fastsae_hb_area")
  expect_true(isTRUE(fit_rob$robust))
  expect_true("nu_degrees_of_freedom" %in% fit_rob$hyperpar$Parameter)
  nu_val <- fit_rob$hyperpar$Estimate[fit_rob$hyperpar$Parameter == "nu_degrees_of_freedom"]
  expect_true(nu_val >= 2.5 && nu_val <= 30.0)

  # 3. Test GVF Smoothing within hb_area (Feature A)
  fit_gvf <- hb_area(
    y ~ x1,
    data = df,
    vardir = "vardir",
    family = "gaussian",
    smooth_vardir = TRUE,
    gvf_method = "log_linear",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )

  expect_s3_class(fit_gvf, "fastsae_hb_area")
  expect_true(isTRUE(fit_gvf$smooth_vardir))
  expect_s3_class(fit_gvf$gvf, "fastsaegpu_gvf")
  expect_true("vardir_raw" %in% names(fit_gvf$df_hb))
  expect_false(all(fit_gvf$df_hb$vardir == fit_gvf$df_hb$vardir_raw))

  # 4. Print method executes cleanly
  expect_no_error(print(fit_hs))
})

test_that("hb_area supports two-level nested sub-area SAE models (Torabi & Rao, 2014)", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")

  set.seed(42)
  # 4 Major areas (e.g., provinces)
  # 5 Sub-areas per major area (total 20 sub-areas, e.g., regencies)
  n_maj <- 4
  n_sub_per_maj <- 5
  N_total <- n_maj * n_sub_per_maj

  prov_id <- paste0("Prov_", rep(1:n_maj, each = n_sub_per_maj))
  kab_id <- paste0("Kab_", 1:N_total)

  # True province random effects u_j ~ N(0, 0.4^2)
  u_prov <- rnorm(n_maj, 0, 0.4)
  # True district random effects v_jk ~ N(0, 0.2^2)
  v_kab <- rnorm(N_total, 0, 0.2)

  x <- rnorm(N_total)
  vardir <- rep(0.04, N_total)

  # Data generating process: y_jk = 1.5 + 0.8 * x_jk + u_j + v_jk + e_jk
  y <- 1.5 + 0.8 * x + u_prov[as.integer(factor(prov_id))] + v_kab + rnorm(N_total, 0, sqrt(vardir))

  df_nested <- data.frame(
    provinsi = prov_id,
    kabupaten = kab_id,
    y = y,
    x = x,
    vardir = vardir
  )

  # 1. Standard specification: domain = "provinsi", subarea = "kabupaten"
  fit_nested <- hb_area(
    y ~ x,
    data = df_nested,
    domain = "provinsi",
    subarea = "kabupaten",
    vardir = "vardir",
    family = "gaussian",
    warmup = 100L,
    samples = 150L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )

  expect_s3_class(fit_nested, "fastsae_hb_area")
  expect_true(isTRUE(fit_nested$is_nested))
  expect_equal(fit_nested$level, "subarea")
  expect_true("domain" %in% names(fit_nested$df_hb))
  expect_true("subarea" %in% names(fit_nested$df_hb))
  expect_equal(nrow(fit_nested$df_hb), N_total)

  # Check hyperparameter estimates: sigma2_u (major area), sigma2_subarea (sub-area), icc_nested
  expect_true("sigma2_u" %in% fit_nested$hyperpar$Parameter)
  expect_true("sigma2_subarea" %in% fit_nested$hyperpar$Parameter)
  expect_true("icc_nested" %in% fit_nested$hyperpar$Parameter)

  icc_val <- fit_nested$hyperpar$Estimate[fit_nested$hyperpar$Parameter == "icc_nested"]
  expect_true(icc_val > 0 && icc_val < 1)

  # 2. Inverted specification test (auto-swap): domain = "kabupaten", subarea = "provinsi"
  fit_swap <- hb_area(
    y ~ x,
    data = df_nested,
    domain = "kabupaten",
    subarea = "provinsi",
    vardir = "vardir",
    family = "gaussian",
    warmup = 80L,
    samples = 100L,
    chains = 1L,
    device = "cpu",
    print_result = FALSE
  )
  expect_true(isTRUE(fit_swap$is_nested))
  expect_equal(length(unique(fit_swap$df_hb$domain)), n_maj)
  expect_equal(length(unique(fit_swap$df_hb$subarea)), N_total)

  # 3. Print method handles nested hierarchy cleanly
  expect_no_error(print(fit_nested))
})




# ---- Regression tests (bug fixes) ------------------------------------------

test_that("hb_area rejects invalid identifiers, auxiliaries and model combinations before sampling", {
  df <- data.frame(y = c(1.2, 2.1, 0.8, 1.5), x = c(0.1, 0.5, -0.3, 0.2),
                   vardir = rep(0.1, 4), dom = paste0("d", 1:4), t = c(1, 1, 2, 2))
  # NA identifiers would become out-of-range indices that JAX silently clamps
  d_na <- df
  d_na$dom[2] <- NA
  expect_error(hb_area(y ~ x, data = d_na, domain = "dom", vardir = "vardir"), "domain.*missing values")
  d_na <- df
  d_na$t[3] <- NA
  expect_error(hb_area(y ~ x, data = d_na, domain = "dom", time = "t", vardir = "vardir"), "time.*missing values")

  # vardir = 0 / NA for a sampled domain
  d_v <- df
  d_v$vardir[1] <- 0
  expect_error(hb_area(y ~ x, data = d_v, vardir = "vardir"), "vardir.*strictly positive")
  d_v$vardir[1] <- NA
  expect_error(hb_area(y ~ x, data = d_v, vardir = "vardir"), "smooth_vardir")

  # binomial: invalid trials and counts above trials
  d_b <- data.frame(y = c(3, 5, 12, 4), x = df$x, n = c(10, 10, 10, 0))
  expect_error(hb_area(y ~ x, data = d_b, trials = "n", family = "binomial"), "trials.*strictly positive")
  d_b$n[4] <- 10
  expect_error(hb_area(y ~ x, data = d_b, trials = "n", family = "binomial"), "\\[0, trials\\]")

  # poisson: non-positive exposure
  d_p <- data.frame(y = c(3, 5, 12, 4), x = df$x, e = c(10, 0, 10, 10))
  expect_error(hb_area(y ~ x, data = d_p, exposure = "e", family = "poisson"), "exposure.*strictly positive")

  # benchmark arguments
  d_w <- transform(df, w = c(1, 2, NA, 1))
  expect_error(hb_area(y ~ x, data = d_w, vardir = "vardir", benchmark = TRUE, benchmark_weights = "w"),
               "benchmark_weights.*strictly positive")
  d_w$w[3] <- 1
  expect_error(hb_area(y ~ x, data = d_w, vardir = "vardir", benchmark = TRUE, benchmark_weights = "w",
                       benchmark_method = "logit"), "requires estimates in")
  expect_error(hb_area(y ~ x, data = d_w, vardir = "vardir", benchmark = TRUE, benchmark_weights = "w",
                       benchmark_target = c(1, 2)), "single finite number")

  # spatio-temporal interactions that would be silently dropped
  expect_error(hb_area(y ~ x, data = df, domain = "dom", time = "t", vardir = "vardir",
                       st_interaction = "type1"), "requires a temporal effect")
  expect_error(hb_area(y ~ x, data = df, domain = "dom", time = "t", vardir = "vardir",
                       temporal = "ar1", st_interaction = "separable"), "requires a structured spatial effect")
})

test_that("W with row names not matching the domains warns about assumed ordering", {
  W <- matrix(c(0, 1, 0, 1, 0, 1, 0, 1, 0), 3, 3, dimnames = list(c("a", "b", "c"), c("a", "b", "c")))
  expect_warning(
    fastsaegpu:::.convert_spatial_weights(W, n_domains = 3, domain_names = c("d1", "d2", "d3")),
    "do not match the domain identifiers"
  )
  expect_no_warning(fastsaegpu:::.convert_spatial_weights(W, n_domains = 3, domain_names = c("c", "a", "b")))
})

test_that(".resolve_nesting keeps subarea codes that restart within each major area distinct", {
  prov <- rep(paste0("P", 1:6), each = 4)
  kab <- sprintf("%02d", rep(1:4, 6))
  nest <- fastsaegpu:::.resolve_nesting(prov, kab)
  expect_false(nest$swapped)
  expect_identical(nest$major, prov)
  expect_equal(length(unique(nest$sub_key)), 24)
  # Genuinely inverted specification is still swapped
  kab_global <- paste0("K", 1:24)
  inv <- fastsaegpu:::.resolve_nesting(kab_global, prov)
  expect_true(inv$swapped)
  expect_identical(inv$major, prov)
})

test_that("Gaussian hb_area is not shrunk towards 0 for large-scale responses", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(1)
  D <- 30
  x <- rnorm(D, 10, 2)
  theta <- 1e6 + 5e4 * x + rnorm(D, 0, 3e4)
  y <- theta + rnorm(D, 0, 1e4)
  df <- data.frame(y = y, x = x, vardir = rep(1e8, D))
  fit <- hb_area(y ~ x, data = df, vardir = "vardir", warmup = 200L, samples = 300L,
                 chains = 1L, device = "cpu", print_result = FALSE)
  expect_lt(mean(abs(fit$df_hb$hb - y) / y), 0.02)
  ols <- coef(lm(y ~ x))
  expect_equal(unname(coef(fit)), unname(ols), tolerance = 0.1)
  expect_gt(fit$random_effect_var, 1e8)
})

test_that("binomial counts: df_hb$y is the direct proportion and self-benchmark targets the proportion scale", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(2)
  D <- 15
  n <- sample(30:80, D, TRUE)
  x <- rnorm(D)
  yc <- rbinom(D, n, plogis(-0.5 + 0.5 * x))
  w <- runif(D, 1, 5)
  df <- data.frame(y = yc, x = x, n = n, w = w)
  fit <- hb_area(y ~ x, data = df, trials = "n", family = "binomial", benchmark = TRUE,
                 benchmark_weights = "w", warmup = 150L, samples = 200L, chains = 1L,
                 device = "cpu", print_result = FALSE)
  expect_equal(fit$df_hb$y, yc / n)
  target <- sum(w * yc / n) / sum(w)
  expect_equal(fit$benchmark_info$target, target, tolerance = 1e-6)
  expect_equal(sum(w * fit$df_hb$hb) / sum(w), target, tolerance = 1e-4)
  expect_true(all(fit$df_hb$hb > 0 & fit$df_hb$hb < 1))
  expect_equal(fit$df_hb$estimated_total, fit$df_hb$hb * n)
  # Post-hoc benchmark() works on the same (proportion) scale
  bm <- benchmark(fit, weight = "w")
  expect_equal(sum(w * bm$benchmarked) / sum(w), target, tolerance = 1e-6)
})

test_that("in-model self-benchmarking with unsampled domains returns finite estimates", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(3)
  D <- 12
  x <- rnorm(D)
  y <- 2 + x + rnorm(D, 0, 0.3)
  y[3] <- NA
  w <- runif(D, 1, 3)
  df <- data.frame(y = y, x = x, vardir = rep(0.09, D), w = w)
  obs <- !is.na(y)
  target <- sum(w[obs] * y[obs]) / sum(w[obs])
  for (m in c("ratio", "optimal")) {
    fit <- hb_area(y ~ x, data = df, vardir = "vardir", benchmark = TRUE, benchmark_weights = "w",
                   benchmark_method = m, warmup = 100L, samples = 150L, chains = 1L,
                   device = "cpu", print_result = FALSE)
    expect_true(all(is.finite(fit$df_hb$hb)))
    expect_equal(sum(w * fit$df_hb$hb) / sum(w), target, tolerance = 1e-4)
  }
})

test_that("Gaussian in-model benchmarking defaults to 'optimal' instead of logit", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(4)
  D <- 12
  x <- rnorm(D)
  y <- 2 + x + rnorm(D, 0, 0.3)
  w <- runif(D, 1, 3)
  df <- data.frame(y = y, x = x, vardir = rep(0.09, D), w = w)
  fit <- hb_area(y ~ x, data = df, vardir = "vardir", benchmark = TRUE, benchmark_weights = "w",
                 warmup = 100L, samples = 150L, chains = 1L, device = "cpu", print_result = FALSE)
  expect_equal(fit$benchmark_info$method, "optimal")
  expect_equal(sum(w * fit$df_hb$hb) / sum(w), sum(w * y) / sum(w), tolerance = 1e-4)
  expect_gt(stats::cor(fit$df_hb$hb, y), 0.9)
})

test_that("nested hb_area: subarea codes restarting per major area give the same fit as unique codes", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(5)
  prov <- rep(paste0("P", 1:6), each = 4)
  kab <- sprintf("%02d", rep(1:4, 6))
  x <- rnorm(24)
  y <- 1 + 0.5 * x + rnorm(6, 0, 0.4)[as.integer(factor(prov))] + rnorm(24, 0, 0.2)
  df <- data.frame(y = y, x = x, vardir = rep(0.04, 24), prov = prov, kab = kab,
                   kab_global = paste(prov, kab))
  args <- list(y ~ x, data = df, domain = "prov", vardir = "vardir", warmup = 80L, samples = 100L,
               chains = 1L, device = "cpu", print_result = FALSE)
  fit_rel <- do.call(hb_area, c(args, subarea = "kab"))
  fit_glob <- do.call(hb_area, c(args, subarea = "kab_global"))
  expect_identical(fit_rel$df_hb$domain, prov)
  expect_equal(fit_rel$df_hb$hb, fit_glob$df_hb$hb)
})

test_that("requesting an unavailable accelerator warns instead of silently using CPU", {
  skip_on_cran()
  skip_if(nzchar(Sys.which("nvidia-smi")), "CUDA GPU present")
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(6)
  df <- data.frame(y = rnorm(8, 2), x = rnorm(8), vardir = rep(0.1, 8))
  expect_warning(
    hb_area(y ~ x, data = df, vardir = "vardir", warmup = 30L, samples = 30L, chains = 1L,
            device = "cuda", print_result = FALSE),
    "not available to JAX"
  )
})

test_that("Python backend detects the jax-metal 'METAL' platform and pins the CPU device when asked", {
  skip_on_cran()
  skip_if_not(Sys.info()[["sysname"]] == "Darwin", "Metal only on macOS")
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  py <- reticulate::py_exe()
  path <- dirname(system.file("python", "hb_area_numpyro.py", package = "fastsaegpu"))
  code <- paste(
    "import sys", sprintf("sys.path.insert(0, %s)", shQuote(path, type = "sh")), "import jax",
    "has = any(d.platform.upper() == 'METAL' for d in jax.devices())",
    "print('HAS_METAL' if has else 'NO_METAL')",
    "import hb_area_numpyro as m",
    "print(m.init_jax_environment('metal')[-1])",
    "import jax.numpy as jnp",
    "print(m.init_jax_environment('cpu')[-1], list(jnp.zeros(1).devices())[0].platform)",
    sep = "\n"
  )
  out <- suppressWarnings(system2(py, c("-c", shQuote(code)), stdout = TRUE, stderr = FALSE,
                                  env = "JAX_PLATFORMS="))
  skip_if(!any(out == "HAS_METAL"), "jax-metal not installed")
  expect_true(any(grepl("^metal:", out)))
  expect_true(any(out == "cpu cpu"))
})

test_that("horseshoe shrinkage factor uses the regularized local scale lambda_tilde", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  be <- fastsaegpu:::.get_numpyro_backend("cpu")
  # Very large lambda: slab caps lambda_tilde at sqrt(c2) / tau, so kappa -> 1 / (1 + c2), not 0
  big <- be$horseshoe_shrinkage(tau = 1, lam = 1e6, c2 = 4)
  expect_equal(as.numeric(big[[1]]), 2, tolerance = 1e-6)
  expect_equal(as.numeric(big[[2]]), 1 / (1 + 4), tolerance = 1e-6)
  # Huge slab: reduces to the plain horseshoe 1 / (1 + tau^2 lambda^2)
  plain <- be$horseshoe_shrinkage(tau = 0.1, lam = 0.5, c2 = 1e12)
  expect_equal(as.numeric(plain[[2]]), 1 / (1 + 0.05^2), tolerance = 1e-6)
})

test_that("horseshoe penalizes every coefficient when the model has no intercept", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(8)
  D <- 15
  df <- data.frame(x1 = rnorm(D), x2 = rnorm(D), vardir = rep(0.1, D))
  df$y <- 1.5 * df$x1 + rnorm(D, 0, 0.4)
  fit <- hb_area(y ~ x1 + x2 - 1, data = df, vardir = "vardir", prior_beta = "horseshoe",
                 warmup = 100L, samples = 150L, chains = 1L, device = "cpu", print_result = FALSE)
  expect_equal(rownames(fit$estcoef), c("x1", "x2"))
  expect_true(all(fit$estcoef$shrinkage_factor > 0))
  expect_output(print(fit), "x1")
})

test_that("Metal NUTS fallback warns with the actual cause", {
  expect_warning(fastsaegpu:::.warn_device_fallback("metal", "cpu (fallback)"), "popcnt")
  expect_warning(fastsaegpu:::.warn_device_fallback("cuda", "cpu"), "not available to JAX")
  expect_no_warning(fastsaegpu:::.warn_device_fallback("auto", "cpu"))
  expect_no_warning(fastsaegpu:::.warn_device_fallback("metal", "metal:METAL(id=0)"))
})

test_that("backend records split random-effect draws only for nested models", {
  skip_on_cran()
  skip_if_not(check_numpyro_available(), "NumPyro/JAX not available")
  be <- fastsaegpu:::.get_numpyro_backend("cpu")
  set.seed(9)
  N <- 12L
  X <- cbind(1, rnorm(N))
  y <- as.numeric(X %*% c(2, 1) + rnorm(N, 0, 0.4))
  res <- be$fit_numpyro_hb(y = y, X = X, domain_idx = 0:(N - 1L), vardir = rep(0.1, N), D = N,
                           num_warmup = 50L, num_samples = 50L, num_chains = 1L, device = "cpu")
  expect_null(res$rand_eff_area_mean)
  expect_null(res$rand_eff_subarea_mean)
  # Gaussian identity link: linear predictor equals the HB estimate
  expect_equal(as.numeric(res$linpred_mean), as.numeric(res$hb_mean))
})
