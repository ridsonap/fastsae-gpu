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

test_that("hb_twofold rejects missing covariates and subarea ids before sampling", {
  df <- data.frame(y = c(1, 2, 1.5, 2.2), x = c(0.5, NA, 0, 1), vardir = rep(0.1, 4),
                   area = c("A", "A", "B", "B"), sub = c("01", "02", "01", "02"))
  expect_error(hb_twofold(y ~ x, data = df, area = "area", subarea = "sub", vardir = "vardir"),
               "missing or non-finite")
  df$x[2] <- 0.3
  df$sub[3] <- NA
  expect_error(hb_twofold(y ~ x, data = df, area = "area", subarea = "sub", vardir = "vardir"),
               "subarea.*must not contain NA")
})

test_that("hb_twofold keeps subarea codes that restart within each area distinct", {
  skip_on_cran()
  skip_if_not(fastsaegpu::check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(7)
  area <- rep(paste0("A", 1:4), each = 5)
  sub <- sprintf("%02d", rep(1:5, 4))
  x <- stats::rnorm(20)
  y <- 1.5 + 0.8 * x + stats::rnorm(4, 0, 0.4)[as.integer(factor(area))] + stats::rnorm(20, 0, 0.2)
  df <- data.frame(y = y, x = x, vardir = rep(0.04, 20), area = area, sub = sub,
                   sub_global = paste0(area, "-", sub))
  args <- list(y ~ x, data = df, area = "area", vardir = "vardir", warmup = 80L, samples = 100L,
               chains = 1L, device = "cpu", print_result = FALSE)
  fit_rel <- do.call(hb_twofold, c(args, subarea = "sub"))
  fit_glob <- do.call(hb_twofold, c(args, subarea = "sub_global"))
  expect_equal(fit_rel$df_hb$hb, fit_glob$df_hb$hb)
  expect_equal(fit_rel$df_area$hb_area, fit_glob$df_area$hb_area)
})

test_that("hb_twofold area summaries handle areas without sampled subareas", {
  skip_on_cran()
  skip_if_not(fastsaegpu::check_numpyro_available(), "NumPyro/JAX not available")
  set.seed(16)
  area <- rep(c("B", "A", "C"), each = 4)
  x <- stats::rnorm(12)
  y <- 1 + x + stats::rnorm(12, 0, 0.3)
  y[area == "C"] <- NA
  w <- seq_len(12)
  df <- data.frame(y = y, x = x, vardir = ifelse(is.na(y), NA, 0.05), area = area, w = w)
  fit <- hb_twofold(y ~ x, data = df, area = "area", vardir = "vardir", weight = "w",
                    warmup = 60L, samples = 60L, chains = 1L, device = "cpu", print_result = FALSE)
  da <- fit$df_area
  expect_equal(da$domain, c("B", "A", "C"))
  expect_equal(da$n_subareas, c(4L, 4L, 4L))
  expect_equal(da$weight, c(sum(1:4), sum(5:8), sum(9:12)))
  expect_equal(da$direct[1:2], c(weighted.mean(y[1:4], 1:4), weighted.mean(y[5:8], 5:8)))
  expect_true(is.na(da$direct[3]))
  expect_false(anyNA(da$hb_area))
})
