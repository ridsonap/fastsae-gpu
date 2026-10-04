test_that("benchmark works for self-benchmarking and external benchmarking", {
  # Mock a fitted fastsae_hb_area object
  df_hb <- data.frame(
    domain = paste0("d", 1:10),
    y = c(0.20, 0.25, 0.30, 0.22, 0.28, 0.35, 0.18, 0.24, 0.32, 0.26),
    hb = c(0.21, 0.24, 0.29, 0.23, 0.27, 0.33, 0.19, 0.25, 0.31, 0.27),
    mse = c(0.002, 0.003, 0.001, 0.004, 0.002, 0.005, 0.002, 0.003, 0.001, 0.004),
    group = rep(c("North", "South"), each = 5),
    N = c(1000, 2000, 1500, 3000, 2500, 1200, 1800, 2200, 1400, 2800),
    stringsAsFactors = FALSE
  )
  
  mock_obj <- structure(
    list(
      df_hb = df_hb,
      data = df_hb,
      family = "beta",
      level = "area",
      model = "Beta SAE"
    ),
    class = c("fastsae_hb_area", "fastsae")
  )

  # 1. Self-Benchmarking (target = NULL) with logit method
  bm_self <- benchmark(mock_obj, weight = "N", method = "logit")
  expect_s3_class(bm_self, "fastsaegpu_benchmark")
  expect_true(all(bm_self$benchmarked > 0 & bm_self$benchmarked < 1))
  
  # Target should match the weighted direct survey estimate
  w_share <- df_hb$N / sum(df_hb$N)
  target_self <- sum(w_share * df_hb$y)
  bench_self <- sum(w_share * bm_self$benchmarked)
  expect_equal(bench_self, target_self, tolerance = 1e-6)

  # 2. External Benchmarking (target = 0.30) with logit method
  bm_ext <- benchmark(mock_obj, target = 0.30, weight = "N", method = "logit")
  expect_equal(sum(w_share * bm_ext$benchmarked), 0.30, tolerance = 1e-6)
  expect_true(all(bm_ext$benchmarked > 0 & bm_ext$benchmarked < 1))

  # 3. External Benchmarking with optimal MSE method
  bm_opt <- benchmark(mock_obj, target = 0.28, weight = "N", method = "optimal")
  expect_equal(sum(w_share * bm_opt$benchmarked), 0.28, tolerance = 1e-6)

  # 4. External Benchmarking with ratio method
  bm_ratio <- benchmark(mock_obj, target = 0.28, weight = "N", method = "ratio")
  expect_equal(sum(w_share * bm_ratio$benchmarked), 0.28, tolerance = 1e-6)

  # 5. External Benchmarking with difference method
  bm_diff <- benchmark(mock_obj, target = 0.28, weight = "N", method = "difference")
  expect_equal(sum(w_share * bm_diff$benchmarked), 0.28, tolerance = 1e-6)

  # 6. Group / Hierarchical Benchmarking
  bm_grp <- benchmark(mock_obj, weight = "N", group = "group", method = "logit")
  expect_true("group" %in% names(bm_grp))
  # Check calibration within North and South
  for (g in c("North", "South")) {
    idx <- which(df_hb$group == g)
    w_g <- df_hb$N[idx] / sum(df_hb$N[idx])
    expect_equal(sum(w_g * bm_grp$benchmarked[idx]), sum(w_g * df_hb$y[idx]), tolerance = 1e-6)
  }

  # 7. S3 Methods
  expect_output(print(bm_self), "Benchmarked Small Area Estimation")
  expect_output(print(bm_self), "\\[OK\\] Target")
  s_bm <- summary(bm_self)
  expect_s3_class(s_bm, "summary.fastsaegpu_benchmark")
  expect_output(print(s_bm), "Summary of Benchmarked SAE Calibration")

  # plot method
  p_bm <- plot(bm_self)
  expect_s3_class(p_bm, "ggplot")
})
