test_that("gvf_smooth works across all estimation methods", {
  set.seed(42)
  D <- 40
  y <- runif(D, 0.15, 0.85)
  n <- sample(20:100, D, replace = TRUE)
  # True sampling variance with noise
  vardir_true <- y * (1 - y) / n
  vardir_noisy <- vardir_true * rlnorm(D, 0, 0.35)
  df <- data.frame(y = y, vardir = vardir_noisy, n = n)

  # 1. Log-linear method (Wolter 2007)
  gvf_ll <- gvf_smooth(y = y, vardir = vardir_noisy, n = n, method = "log_linear")
  expect_s3_class(gvf_ll, "fastsaegpu_gvf")
  expect_length(gvf_ll$vardir_smooth, D)
  expect_true(all(gvf_ll$vardir_smooth > 0))
  expect_false(any(is.na(gvf_ll$vardir_smooth)))
  expect_true(gvf_ll$r_squared > 0)
  expect_output(print(gvf_ll), "Raw_Var")

  # Test with data frame syntax
  gvf_df <- gvf_smooth(y = "y", vardir = "vardir", n = "n", data = df, method = "log_linear")
  expect_equal(gvf_df$vardir_smooth, gvf_ll$vardir_smooth)

  # 2. Power method (Cho et al. 2002)
  gvf_pow <- gvf_smooth(y = y, vardir = vardir_noisy, n = n, method = "power")
  expect_s3_class(gvf_pow, "fastsaegpu_gvf")
  expect_length(gvf_pow$vardir_smooth, D)
  expect_true(all(gvf_pow$vardir_smooth > 0))

  # 3. Ratio / CV2 method (Rivest & Vandal 2003)
  gvf_rat <- gvf_smooth(y = y, vardir = vardir_noisy, method = "ratio")
  expect_s3_class(gvf_rat, "fastsaegpu_gvf")
  expect_length(gvf_rat$vardir_smooth, D)
  expect_true(all(gvf_rat$vardir_smooth > 0))

  # 4. Loess non-parametric method
  gvf_loess <- gvf_smooth(y = y, vardir = vardir_noisy, method = "loess")
  expect_s3_class(gvf_loess, "fastsaegpu_gvf")
  expect_length(gvf_loess$vardir_smooth, D)
  expect_true(all(gvf_loess$vardir_smooth > 0))

  # 5. Plot method
  p <- plot(gvf_ll)
  expect_s3_class(p, "ggplot")
})

test_that("gvf_smooth handles input validation gracefully", {
  expect_error(gvf_smooth(y = c(0.1, 0.2), vardir = c(0.01)), "same length")
  expect_error(gvf_smooth(y = c(0.1, 0.2), vardir = c(0.01, 0.02), n = c(10)), "same length")
  expect_error(gvf_smooth(y = c(0.1, 0.2), vardir = c(-0.01, 0.02)), "minimum 3 required")
})

test_that("gvf_smooth imputes smoothed variances for areas with zero or missing vardir", {
  set.seed(21)
  D <- 25
  y <- runif(D, 0.1, 0.8)
  n <- sample(20:60, D, replace = TRUE)
  v <- y * (1 - y) / n * rlnorm(D, 0, 0.2)
  v[c(3, 7)] <- 0
  v[9] <- NA
  for (m in c("log_linear", "power", "ratio", "loess")) {
    g <- gvf_smooth(y = y, vardir = v, n = n, method = m)
    expect_true(all(is.finite(g$vardir_smooth) & g$vardir_smooth > 0), info = m)
  }
  # areas without a direct estimate stay NA
  y[5] <- NA
  expect_true(is.na(gvf_smooth(y = y, vardir = v, n = n)$vardir_smooth[5]))
})

test_that("gvf_smooth fails clearly when there are too few areas for the model", {
  expect_error(
    gvf_smooth(y = c(0.2, 0.3, 0.4), vardir = c(0.01, 0.02, 0.015), n = c(10, 20, 30)),
    "Not enough valid observations"
  )
})
