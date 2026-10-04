#!/usr/bin/env Rscript
# ==============================================================================
# Empirical Simulation: Methodological Enhancements for SAE Estimation Quality
# Evaluating Features A (GVF), B (Horseshoe), C (Student-t), and Synergy
# ==============================================================================

suppressPackageStartupMessages({
  library(fastsaegpu)
})

set.seed(2026)

cat("====================================================================\n")
cat("SIMULATION STUDY: ADVANCED SAE METHODOLOGICAL ENHANCEMENTS\n")
cat("Features: A (GVF Smoothing), B (Horseshoe Prior), C (Student-t Robust)\n")
cat("====================================================================\n\n")

# 1. Population & Domain Setup
D <- 50             # 50 Small areas
P_signal <- 3       # 3 true active covariates
P_noise <- 12       # 12 pure noise covariates
P_total <- P_signal + P_noise

# Generate covariates
X_cov <- matrix(rnorm(D * P_total), nrow = D, ncol = P_total)
colnames(X_cov) <- c(paste0("x_sig", 1:P_signal), paste0("x_noise", 1:P_noise))

# True parameter vector
beta_true <- c(2.0, 1.8, -1.2, 0.9, rep(0.0, P_noise))
names(beta_true) <- c("(Intercept)", colnames(X_cov))

# Domain random effects with localized shocks / outliers (4 domains = 8%)
outlier_idx <- c(7, 18, 31, 44)
sigma_u <- 0.25
u_true <- rnorm(D, mean = 0, sd = sigma_u)
u_true[outlier_idx] <- c(1.8, -1.7, 1.9, -1.6) # Severe localized economic/shocks

# True domain means theta_i
X_design <- cbind(1, X_cov)
mu_true <- as.numeric(X_design %*% beta_true + u_true)

# Sample sizes and sampling variance
n_i <- sample(15:65, D, replace = TRUE)
sigma_e0 <- 0.8
vardir_true <- (sigma_e0^2) / n_i

# Direct survey sampling: y_i = mu_i + e_i
y_direct <- mu_true + rnorm(D, mean = 0, sd = sqrt(vardir_true))

# Noisy sampling variance estimate: vardir_obs = vardir_true * exp(N(0, 0.4^2))
vardir_noisy <- vardir_true * exp(rnorm(D, mean = 0, sd = 0.40))

# Population weights for benchmarking
pop_weights <- round(runif(D, 500, 3000))
target_pop_total <- sum(pop_weights * mu_true)
target_pop_mean <- target_pop_total / sum(pop_weights)

# Construct dataset
sim_data <- data.frame(
  domain = paste0("domain_", 1:D),
  y = y_direct,
  vardir = vardir_noisy,
  vardir_true = vardir_true,
  mu_true = mu_true,
  n_sample = n_i,
  pop_weight = pop_weights,
  is_outlier = seq_len(D) %in% outlier_idx
)
sim_data <- cbind(sim_data, as.data.frame(X_cov))

formula_full <- as.formula(paste("y ~", paste(colnames(X_cov), collapse = " + ")))

cat("Dataset generated:\n")
cat(" - Domains (D):", D, "\n")
cat(" - Covariates:", P_total, "(3 Signals, 12 Noise)\n")
cat(" - Outlier Domains:", length(outlier_idx), "(Domains:", paste(outlier_idx, collapse = ", "), ")\n")
cat(" - Sampling Variance Noise: Log-Normal(0, 0.40^2)\n\n")

# Evaluation metric functions
calc_metrics <- function(pred, truth, weights = NULL, target = NULL, outlier_indices = NULL) {
  arb_all <- mean(abs(pred - truth) / abs(truth)) * 100
  rrmse_all <- sqrt(mean(((pred - truth) / abs(truth))^2)) * 100
  mae_all <- mean(abs(pred - truth))
  
  # Outlier domain metrics
  arb_out <- mean(abs(pred[outlier_indices] - truth[outlier_indices]) / abs(truth[outlier_indices])) * 100
  rrmse_out <- sqrt(mean(((pred[outlier_indices] - truth[outlier_indices]) / abs(truth[outlier_indices]))^2)) * 100
  
  # Non-outlier domain metrics
  non_out <- setdiff(seq_along(truth), outlier_indices)
  arb_norm <- mean(abs(pred[non_out] - truth[non_out]) / abs(truth[non_out])) * 100
  rrmse_norm <- sqrt(mean(((pred[non_out] - truth[non_out]) / abs(truth[non_out]))^2)) * 100
  
  # Aggregation consistency error
  agg_err <- NA_real_
  if (!is.null(weights) && !is.null(target)) {
    agg_est <- sum(weights * pred) / sum(weights)
    agg_err <- abs(agg_est - target) / target * 100
  }
  
  data.frame(
    ARB_Overall = arb_all,
    RRMSE_Overall = rrmse_all,
    MAE_Overall = mae_all,
    ARB_Outliers = arb_out,
    RRMSE_Outliers = rrmse_out,
    ARB_Typical = arb_norm,
    RRMSE_Typical = rrmse_norm,
    Agg_Error_Pct = agg_err
  )
}

# ------------------------------------------------------------------------------
# 2. Model Estimation
# ------------------------------------------------------------------------------

cat("Fitting Models...\n")

# Baseline: Direct Survey Estimator
m_direct_res <- calc_metrics(y_direct, mu_true, pop_weights, target_pop_mean, outlier_idx)
m_direct_res$Model <- "1. Direct Survey Estimator"

# M0: Standard Hierarchical Bayes (Normal prior, Gaussian RE, noisy vardir)
cat(" -> M0: Standard HB Area Model (Baseline)...\n")
t0 <- Sys.time()
fit_m0 <- hb_area(
  formula = formula_full,
  data = sim_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "normal",
  robust = FALSE,
  smooth_vardir = FALSE,
  warmup = 300L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
time_m0 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
res_m0 <- calc_metrics(fit_m0$df_hb$hb, mu_true, pop_weights, target_pop_mean, outlier_idx)
res_m0$Model <- "2. Standard HB (Baseline)"

# M1: Feature A - GVF Smoothing (Wolter 2007)
cat(" -> M1: + Feature A (GVF Variance Smoothing)...\n")
t0 <- Sys.time()
fit_m1 <- hb_area(
  formula = formula_full,
  data = sim_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "normal",
  robust = FALSE,
  smooth_vardir = TRUE,
  gvf_method = "log_linear",
  warmup = 300L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
time_m1 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
res_m1 <- calc_metrics(fit_m1$df_hb$hb, mu_true, pop_weights, target_pop_mean, outlier_idx)
res_m1$Model <- "3. + Feature A (GVF Smoothing)"

# M2: Feature B - Regularized Horseshoe Prior (Carvalho et al. 2010)
cat(" -> M2: + Feature B (Regularized Horseshoe Prior)...\n")
t0 <- Sys.time()
fit_m2 <- hb_area(
  formula = formula_full,
  data = sim_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "horseshoe",
  robust = FALSE,
  smooth_vardir = FALSE,
  warmup = 300L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
time_m2 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
res_m2 <- calc_metrics(fit_m2$df_hb$hb, mu_true, pop_weights, target_pop_mean, outlier_idx)
res_m2$Model <- "4. + Feature B (Horseshoe Prior)"

# M3: Feature C - Robust Student-t Random Effects (Bell & Huang 2006)
cat(" -> M3: + Feature C (Robust Student-t Random Effects)...\n")
t0 <- Sys.time()
fit_m3 <- hb_area(
  formula = formula_full,
  data = sim_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "normal",
  robust = TRUE,
  smooth_vardir = FALSE,
  warmup = 300L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
time_m3 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
res_m3 <- calc_metrics(fit_m3$df_hb$hb, mu_true, pop_weights, target_pop_mean, outlier_idx)
res_m3$Model <- "5. + Feature C (Student-t Robust)"

# M4: Full Synergy (Features A + B + C + Benchmarking)
cat(" -> M4: Complete Synergy (A: GVF + B: Horseshoe + C: Robust + Benchmarking)...\n")
t0 <- Sys.time()
fit_m4 <- hb_area(
  formula = formula_full,
  data = sim_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "horseshoe",
  robust = TRUE,
  smooth_vardir = TRUE,
  gvf_method = "log_linear",
  benchmark = TRUE,
  benchmark_weights = "pop_weight",
  benchmark_method = "optimal",
  warmup = 300L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
time_m4 <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
res_m4 <- calc_metrics(fit_m4$df_hb$hb, mu_true, pop_weights, target_pop_mean, outlier_idx)
res_m4$Model <- "6. Full Synergy (A + B + C + Benchmark)"

# Combine all results
results_df <- rbind(m_direct_res, res_m0, res_m1, res_m2, res_m3, res_m4)
results_df <- results_df[, c("Model", "ARB_Overall", "RRMSE_Overall", "MAE_Overall", 
                             "ARB_Outliers", "RRMSE_Outliers", "ARB_Typical", "RRMSE_Typical", "Agg_Error_Pct")]

# Format output
cat("\n====================================================================\n")
cat("SIMULATION STUDY RESULTS SUMMARY (50 Domains, 15 Covariates, 4 Outliers)\n")
cat("====================================================================\n")
print(format(results_df, digits = 3, nsmall = 2))

# Save results
dir.create("benchmarks", showWarnings = FALSE)
write.csv(results_df, "benchmarks/advanced_features_comparison.csv", row.names = FALSE)
cat("\nResults saved to benchmarks/advanced_features_comparison.csv\n")

# Inspection of Horseshoe Shrinkage Weights
if (!is.null(fit_m4$estcoef$shrinkage_factor)) {
  cat("\nHorseshoe Shrinkage Weights (kappa_j) in Synergy Model:\n")
  cat("------------------------------------------------------\n")
  sw_df <- data.frame(
    Variable = rownames(fit_m4$estcoef)[-1],
    True_Beta = beta_true[-1],
    Estimated_Beta = round(fit_m4$estcoef$beta[-1], 3),
    Shrinkage = round(fit_m4$estcoef$shrinkage_factor[-1], 3),
    Status = ifelse(beta_true[-1] != 0, "SIGNAL", "NOISE")
  )
  print(sw_df)
}
