test_that("hb_twofold validates inputs without MCMC", {
  df <- data.frame(y = c(1, 2, NA), x = c(0.5, -0.5, 0), vardir = c(0.1, 0.1, NA),
                   area = c("A", "A", "B"), w = c(1, 1, 1))
  expect_error(hb_twofold(y ~ x, data = "x", area = "area", vardir = "vardir"), "must be a data frame")
  expect_error(hb_twofold(y ~ x, data = df, area = "missing", vardir = "vardir"), "not found")
  df_bad <- df
  df_bad$vardir <- c(-0.1, 0.1, NA)
  expect_error(hb_twofold(y ~ x, data = df_bad, area = "area", vardir = "vardir"), "strictly positive")
  df_allmiss <- df
  df_allmiss$y <- NA_real_
  expect_error(hb_twofold(y ~ x, data = df_allmiss, area = "area", vardir = "vardir"), "non-NA")
})

test_that("hb_twofold fits Torabi & Rao 2014 model end-to-end", {
  skip_on_cran()
  skip_if_not(fastsaegpu::check_numpyro_available(), "NumPyro/JAX not available")

  set.seed(42)
  m <- 4; k <- 5; n <- m * k
  area <- rep(paste0("A", 1:m), each = k)
  sub <- paste0("S", 1:n)
  u_a <- stats::rnorm(m, 0, 0.4)
  v_s <- stats::rnorm(n, 0, 0.2)
  x <- stats::rnorm(n)
  vardir <- rep(0.04, n)
  y <- 1.5 + 0.8 * x + u_a[as.integer(factor(area))] + v_s + stats::rnorm(n, sd = sqrt(vardir))
  w <- sample(100:500, n, replace = TRUE)
  df <- data.frame(y = y, x = x, vardir = vardir, area = area, sub = sub, w = w)

  fit <- hb_twofold(y ~ x, data = df, area = "area", subarea = "sub",
    vardir = "vardir", weight = "w",
    warmup = 80L, samples = 100L, chains = 1L, device = "cpu", print_result = FALSE)

  expect_s3_class(fit, "fastsae_hb_twofold")
  expect_s3_class(fit, "fastsae_hb_area")
  expect_equal(nrow(fit$Est_sub), n)
  expect_equal(nrow(fit$Est_area), m)
  # Same top-level keys as fastsae::hb_area()/hb_twofold() (+ gpu extras)
  hb_keys <- c("df_hb", "df_subarea", "df_area", "hb", "df_eblup", "estcoef",
    "hyperpar", "random_effect_var", "random_effect_var_time", "phi", "rho",
    "rho_time", "goodness", "family", "spatial", "temporal",
    "st_interaction", "level", "model", "method", "convergence", "fit", "call")
  expect_true(all(hb_keys %in% names(fit)))
  expect_equal(fit$level, "subarea")
  # df_hb column order identical to fastsae::hb_twofold()
  hb_ref_cols <- c("domain", "subarea", "y", "hb", "linear_pred", "vardir",
    "sd", "mse", "rse", "ci_lower", "ci_upper",
    "random_effect_area", "random_effect_subarea")
  expect_equal(names(fit$df_hb)[seq_along(hb_ref_cols)], hb_ref_cols)
  expect_identical(fit$df_subarea, fit$df_hb)
  expect_identical(fit$df_eblup, fit$df_hb)
  expect_identical(fit$hb, fit$df_hb)
  expect_identical(fit$coefficient, fit$estcoef)
  expect_identical(fit$refVar, fit$hyperpar)
  expect_equal(unname(fit$random_effect_var),
    c(fit$sigma2_v, fit$sigma2_subarea))
  # df_area columns identical to fastsae::hb_twofold()
  area_ref_cols <- c("domain", "hb_area", "sd_area", "mse_area", "rse_area",
    "ci_lower_area", "ci_upper_area", "n_subareas")
  expect_equal(names(fit$df_area)[seq_along(area_ref_cols)], area_ref_cols)
  expect_equal(nrow(fit$df_area), m)
  expect_true(all(c("Mean", "SD", "hb", "domain") %in% names(fit$Est_sub)))
  expect_true(all(c("sigma2_u", "sigma2_subarea", "icc_nested") %in% fit$hyperpar$Parameter))
  icc <- fit$hyperpar$Estimate[fit$hyperpar$Parameter == "icc_nested"]
  expect_true(icc > 0 && icc < 1)
  # Area aggregation consistent with subarea weighted means
  for (a in unique(area)) {
    i <- which(area == a)
    expect_equal(fit$Est_area$Mean[fit$Est_area$area == a],
      stats::weighted.mean(fit$Est_sub$Mean[i], w[i]), tolerance = 1e-6)
  }
  # Non-sampled subareas predicted, benchmark + S3 compat
  df2 <- df
  df2$y[c(1, 7)] <- NA
  df2$vardir[c(1, 7)] <- NA
  fit2 <- hb_twofold(y ~ x, data = df2, area = "area", subarea = "sub",
    vardir = "vardir", weight = "w",
    warmup = 80L, samples = 100L, chains = 1L, device = "cpu", print_result = FALSE)
  expect_false(any(is.na(fit2$Est_sub$Mean[c(1, 7)])))
  expect_true(all(!fit2$Est_sub$sampled[c(1, 7)]))
  bm <- benchmark(fit, weight = "w")
  expect_s3_class(bm, "fastsaegpu_benchmark")
  expect_equal(length(coef(fit)), 2)
  expect_equal(length(fitted(fit)), n)
  expect_no_error(print(fit))
})
