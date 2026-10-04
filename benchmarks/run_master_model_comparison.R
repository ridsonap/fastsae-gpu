#!/usr/bin/env Rscript
# ==============================================================================
# MASTER BENCHMARK: COMPREHENSIVE SMALL AREA ESTIMATION COMPARISON
# Evaluating 8 SAE Models across 4 Major Paradigms on Finite Population (~125,000 units)
#
# Paradigms & Models Evaluated:
#   [PARADIGM 1: SURVEY BASELINE]
#     1. Direct Survey Estimator (SRSWOR Sample Mean)
#   [PARADIGM 2: FREQUENTIST & NUMERICAL INLA]
#     2. fastsae (Frequentist EBLUP via REML)
#     3. fastsae (INLA Laplace Approximation on CPU)
#   [PARADIGM 3: FULL BAYESIAN MCMC (CPU vs GPU)]
#     4. tipsae (Stan NUTS on CPU - De Nicolò & Gardini 2024)
#     5. fastsaegpu (Standard HB Beta on GPU/NumPyro)
#     6. fastsaegpu (Best HB Full Synergy: GVF + Horseshoe + Robust Student-t + Logit Benchmarking)
#   [PARADIGM 4: MACHINE LEARNING SAE]
#     7. fastsaegpu (Standard MERF / FH-RF Machine Learning)
#     8. fastsaegpu (Enhanced MERF: Precision-Weighted + OOB Residuals + Screening + GVF)
#
# Ground Truth:
#   - D = 50 Domains (Kabupaten/Districts)
#   - Total Population N ~ 125,000 units (~2,500 per domain)
#   - Target Indicator: Poverty / Stunting Rate bounded in (0, 1)
#   - 12 Auxiliary Covariates (4 True Signals: linear, binary, sine wave, interaction + 8 Noise)
#   - 4 Outlier / Localized Shock Domains (8% severe anomalies)
# ==============================================================================

suppressPackageStartupMessages({
  library(fastsaegpu)
  library(fastsae)
  library(tipsae)
  library(ranger)
  library(ggplot2)
})

set.seed(2026L)

cat("===================================================================================================\n")
cat(" MASTER BENCHMARK: KOMPARASI LENGKAP 8 MODEL SMALL AREA ESTIMATION\n")
cat(" (Direct vs fastsae EBLUP vs fastsae INLA vs tipsae Stan vs Standard HB vs Best HB vs MERF)\n")
cat("===================================================================================================\n\n")

# ------------------------------------------------------------------------------
# 1. Pembangkitan Populasi Finis Ground Truth (~125.000 Unit Individu)
# ------------------------------------------------------------------------------
D <- 50L
domain_ids <- paste0("domain_", sprintf("%02d", 1:D))
outlier_domains <- c(8L, 19L, 33L, 47L) # 4 wilayah dengan guncangan lokal / outlier

cat(">>> [1/5] Membangkitkan Populasi Finis (~125.000 unit di 50 Wilayah)...\n")

# Efek acak domain laten u_d
sigma_u_true <- 0.35
u_true <- rnorm(D, mean = 0, sd = sigma_u_true)
# Menambahkan shock/anomali ekstrem pada 4 wilayah
u_true[outlier_domains] <- c(1.25, -1.20, 1.30, -1.25)
names(u_true) <- domain_ids

pop_list <- vector("list", D)
true_mean_pop <- numeric(D)
N_d_vec <- integer(D)
names(true_mean_pop) <- names(N_d_vec) <- domain_ids

cov_names <- c(paste0("x_sig", 1:4), paste0("x_noise", 1:8))
pop_X_means <- as.data.frame(matrix(0, nrow = D, ncol = length(cov_names)))
colnames(pop_X_means) <- cov_names
rownames(pop_X_means) <- domain_ids

phi_prec <- 35.0 # Presisi dispersi Beta

for (d in 1:D) {
  dom <- domain_ids[d]
  N_d <- sample(2000:3000, 1) # Rata-rata 2.500 individu per wilayah
  N_d_vec[d] <- N_d
  
  # 4 Kovariat Sinyal:
  x1 <- rnorm(N_d, mean = 1.2, sd = 0.6)
  x2 <- rbinom(N_d, size = 1, prob = 0.45)
  x3 <- runif(N_d, min = 0.5, max = 3.5)
  x4 <- rnorm(N_d, mean = 0.8, sd = 0.5)
  
  # 8 Kovariat Noise murni
  X_noise <- matrix(rnorm(N_d * 8, mean = 0, sd = 1), nrow = N_d, ncol = 8)
  colnames(X_noise) <- paste0("x_noise", 1:8)
  
  # Model Logit Proporsi:
  eta_ind <- -1.50 + 0.45 * x1 - 0.35 * x2 + 0.40 * sin(x3) + 0.30 * (x1 * x4) + u_true[d]
  p_ind <- plogis(eta_ind)
  
  # Sampling variabel respon kontinu bounded (0, 1) dari distribusi Beta
  shape1 <- pmax(1e-4, p_ind * phi_prec)
  shape2 <- pmax(1e-4, (1 - p_ind) * phi_prec)
  y_di <- rbeta(N_d, shape1, shape2)
  y_di <- pmin(pmax(y_di, 1e-4), 1 - 1e-4)
  
  # Ground Truth Populasi Sejati (Exact Finite Population Domain Mean)
  true_mean_pop[d] <- mean(y_di)
  
  # Simpan rata-rata kovariat wilayah
  pop_X_means[d, "x_sig1"] <- mean(x1)
  pop_X_means[d, "x_sig2"] <- mean(x2)
  pop_X_means[d, "x_sig3"] <- mean(x3)
  pop_X_means[d, "x_sig4"] <- mean(x4)
  for (k in 1:8) {
    pop_X_means[d, paste0("x_noise", k)] <- mean(X_noise[, k])
  }
  
  pop_list[[d]] <- data.frame(
    domain = dom,
    y = y_di,
    x_sig1 = x1,
    x_sig2 = x2,
    x_sig3 = x3,
    x_sig4 = x4
  )
}

total_N_pop <- sum(N_d_vec)
cat(sprintf("    Total Populasi: %s individu di %d wilayah.\n", format(total_N_pop, big.mark = "."), D))
cat(sprintf("    Rata-rata Target Sejati Populasi (P_bar_d): %.4f (Min: %.4f, Max: %.4f)\n",
            mean(true_mean_pop), min(true_mean_pop), max(true_mean_pop)))
cat(sprintf("    Wilayah Outlier / Guncangan Lokal: %s\n\n", paste(domain_ids[outlier_domains], collapse = ", ")))

# ------------------------------------------------------------------------------
# 2. Penarikan Sampel Survei (SRSWOR Sampel Kecil)
# ------------------------------------------------------------------------------
cat(">>> [2/5] Menarik Sampel Survei Probabilistik (SRSWOR, n_d = 20 - 45)...\n")

n_sample_vec <- sample(20:45, D, replace = TRUE)
sample_y_dir <- numeric(D)
sample_vardir <- numeric(D)
sample_vardir_true <- numeric(D)

for (d in 1:D) {
  df_d <- pop_list[[d]]
  n_d <- n_sample_vec[d]
  idx <- sample.int(nrow(df_d), size = n_d, replace = FALSE)
  y_samp <- df_d$y[idx]
  
  sample_y_dir[d] <- mean(y_samp)
  fpc <- 1 - (n_d / N_d_vec[d])
  s2 <- var(y_samp)
  sample_vardir_true[d] <- fpc * (s2 / n_d)
  
  # Sampling noise realistis pada varians langsung
  noise_v <- exp(rnorm(1, mean = 0, sd = 0.35))
  sample_vardir[d] <- max(1e-5, sample_vardir_true[d] * noise_v)
}

# Bounds safety for Beta models
sample_y_dir <- pmin(pmax(sample_y_dir, 1e-4), 1 - 1e-4)

# Bobot populasi untuk evaluasi agregasi nasional
pop_weights <- N_d_vec
target_pop_total <- sum(pop_weights * true_mean_pop)
target_pop_mean <- target_pop_total / sum(pop_weights)

survey_data <- data.frame(
  domain = domain_ids,
  y = sample_y_dir,
  vardir = sample_vardir,
  vardir_true = sample_vardir_true,
  n_sample = n_sample_vec,
  pop_weight = pop_weights,
  is_outlier = seq_len(D) %in% outlier_domains
)
survey_data <- cbind(survey_data, pop_X_means)

formula_linear <- as.formula(paste("y ~", paste(cov_names, collapse = " + ")))

cat("    Data survei area berhasil disiapkan:\n")
cat(sprintf("    - Rata-rata sampel: %.1f unit per domain\n", mean(n_sample_vec)))
cat(sprintf("    - Varians sampel langsung rata-rata: %.5f\n\n", mean(sample_vardir)))

# ------------------------------------------------------------------------------
# 3. Fungsi Metrik Evaluasi Komparatif
# ------------------------------------------------------------------------------
calc_eval_metrics <- function(est, truth, name, weights, target_mean, outlier_idx, time_sec = 0) {
  diff <- est - truth
  abs_rel_err <- abs(diff) / abs(truth) * 100
  sq_rel_err <- (diff / abs(truth))^2
  
  arb_all <- mean(abs_rel_err)
  rrmse_all <- sqrt(mean(sq_rel_err)) * 100
  mae_all <- mean(abs(diff))
  corr_all <- cor(est, truth)
  
  arb_out <- mean(abs_rel_err[outlier_idx])
  rrmse_out <- sqrt(mean(sq_rel_err[outlier_idx])) * 100
  
  reg_idx <- setdiff(seq_along(truth), outlier_idx)
  arb_reg <- mean(abs_rel_err[reg_idx])
  rrmse_reg <- sqrt(mean(sq_rel_err[reg_idx])) * 100
  
  mse_all <- mean(diff^2)
  
  agg_val <- sum(weights * est) / sum(weights)
  agg_err_pct <- abs(agg_val - target_mean) / target_mean * 100
  
  data.frame(
    Model = name,
    ARB_All = arb_all,
    RRMSE_All = rrmse_all,
    MAE_All = mae_all,
    Corr = corr_all,
    MSE = mse_all,
    ARB_Outliers = arb_out,
    RRMSE_Outliers = rrmse_out,
    ARB_Regular = arb_reg,
    RRMSE_Regular = rrmse_reg,
    Agg_Error_Pct = agg_err_pct,
    Runtime_Sec = time_sec,
    stringsAsFactors = FALSE
  )
}

# ------------------------------------------------------------------------------
# 4. Eksekusi 8 Model SAE
# ------------------------------------------------------------------------------
cat(">>> [3/5] Mengestimasi 8 Model SAE Lintas Paradigma...\n\n")

eval_list <- list()

# ------------------------------------------------------------------------------
# [PARADIGMA 1: SURVEY BASELINE]
# 1. Direct Survey Estimator
# ------------------------------------------------------------------------------
cat(" [1/8] Direct Survey Estimator (Survey Baseline)...\n")
t0 <- Sys.time()
est_direct <- survey_data$y
t_direct <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
eval_list[[1]] <- calc_eval_metrics(est_direct, true_mean_pop, "1. Direct Survey Estimator",
                                    pop_weights, target_pop_mean, outlier_domains, t_direct)

# ------------------------------------------------------------------------------
# [PARADIGMA 2: FREQUENTIST & INLA]
# 2. fastsae (Frequentist EBLUP via REML)
# ------------------------------------------------------------------------------
cat(" [2/8] fastsae: Frequentist EBLUP (REML C Engine)...\n")
t0 <- Sys.time()
fit_fsae_freq <- fastsae::eblup_fh(
  formula = formula_linear,
  vardir = "vardir",
  data = survey_data,
  method = "REML",
  print_result = FALSE
)
t_fsae_freq <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_fsae_freq <- pmin(pmax(fit_fsae_freq$df_eblup$eblup, 1e-4), 1 - 1e-4)
eval_list[[2]] <- calc_eval_metrics(est_fsae_freq, true_mean_pop, "2. fastsae (Frequentist EBLUP)",
                                    pop_weights, target_pop_mean, outlier_domains, t_fsae_freq)

# ------------------------------------------------------------------------------
# 3. fastsae (INLA Laplace Approximation)
# ------------------------------------------------------------------------------
cat(" [3/8] fastsae: Bayesian Beta SAE (INLA Laplace Approximation CPU)...\n")
t0 <- Sys.time()
fit_fsae_inla <- fastsae::hb_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  family = "beta",
  print_result = FALSE
)
t_fsae_inla <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_fsae_inla <- fit_fsae_inla$df_hb$hb
eval_list[[3]] <- calc_eval_metrics(est_fsae_inla, true_mean_pop, "3. fastsae (INLA Laplace)",
                                    pop_weights, target_pop_mean, outlier_domains, t_fsae_inla)

# ------------------------------------------------------------------------------
# [PARADIGMA 3: FULL MCMC SAMPLING (CPU vs GPU)]
# 4. tipsae (Stan NUTS CPU)
# ------------------------------------------------------------------------------
cat(" [4/8] tipsae: Bayesian Beta SAE (Stan HMC/NUTS on CPU)...\n")
t0 <- Sys.time()
fit_tipsae <- tipsae::fit_sae(
  formula_fixed = formula_linear,
  data = survey_data,
  domains = "domain",
  disp_direct = "vardir",
  type_disp = "var",
  domain_size = "n_sample",
  likelihood = "beta",
  chains = 1L,
  iter = 350L,
  seed = 2026L
)
t_tipsae <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_tipsae <- summary(fit_tipsae)$model_estimates$mean
eval_list[[4]] <- calc_eval_metrics(est_tipsae, true_mean_pop, "4. tipsae (Stan NUTS CPU)",
                                    pop_weights, target_pop_mean, outlier_domains, t_tipsae)

# ------------------------------------------------------------------------------
# 5. fastsaegpu: Standard HB Area (NumPyro NUTS on GPU)
# ------------------------------------------------------------------------------
cat(" [5/8] fastsaegpu: Standard HB Beta (NumPyro NUTS on GPU)...\n")
t0 <- Sys.time()
fit_gpu_std <- fastsaegpu::hb_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  family = "beta",
  prior_beta = "normal",
  robust = FALSE,
  smooth_vardir = FALSE,
  warmup = 150L,
  samples = 250L,
  chains = 1L,
  device = "auto",
  print_result = FALSE
)
t_gpu_std <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_gpu_std <- fit_gpu_std$df_hb$hb
eval_list[[5]] <- calc_eval_metrics(est_gpu_std, true_mean_pop, "5. fastsaegpu (Standard HB GPU)",
                                    pop_weights, target_pop_mean, outlier_domains, t_gpu_std)

# ------------------------------------------------------------------------------
# 6. fastsaegpu: Best HB Area (Full Synergy: GVF + Horseshoe + Robust + Logit Benchmark)
# ------------------------------------------------------------------------------
cat(" [6/8] fastsaegpu: Best HB Synergy (GVF + Horseshoe + Robust + Logit Benchmark)...\n")
t0 <- Sys.time()
fit_gpu_best <- fastsaegpu::hb_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  family = "beta",
  prior_beta = "horseshoe",
  robust = TRUE,
  smooth_vardir = TRUE,
  gvf_method = "log_linear",
  benchmark = TRUE,
  benchmark_weights = "pop_weight",
  benchmark_method = "logit",
  warmup = 200L,
  samples = 350L,
  chains = 1L,
  device = "auto",
  print_result = FALSE
)
t_gpu_best <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_gpu_best <- fit_gpu_best$df_hb$hb
eval_list[[6]] <- calc_eval_metrics(est_gpu_best, true_mean_pop, "6. fastsaegpu (Best HB Synergy)",
                                    pop_weights, target_pop_mean, outlier_domains, t_gpu_best)

# ------------------------------------------------------------------------------
# [PARADIGMA 4: MACHINE LEARNING SAE]
# 7. fastsaegpu: Standard MERF (Baseline ML)
# ------------------------------------------------------------------------------
cat(" [7/8] fastsaegpu: Standard MERF (Baseline Machine Learning SAE)...\n")
t0 <- Sys.time()
fit_merf_std <- fastsaegpu::merf_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  domain = "domain",
  weighted = FALSE,
  use_oob = FALSE,
  feature_screening = FALSE,
  engine = "ranger",
  num_trees = 400,
  max_iter = 25,
  mse_type = "none",
  seed = 2026L
)
t_merf_std <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_merf_std <- pmin(pmax(fit_merf_std$estimates$merf, 1e-4), 1 - 1e-4)
eval_list[[7]] <- calc_eval_metrics(est_merf_std, true_mean_pop, "7. fastsaegpu (Standard MERF)",
                                    pop_weights, target_pop_mean, outlier_domains, t_merf_std)

# ------------------------------------------------------------------------------
# 8. fastsaegpu: Enhanced MERF (Weighted + OOB + Screening + GVF)
# ------------------------------------------------------------------------------
cat(" [8/8] fastsaegpu: Enhanced MERF (Precision-Weighted + OOB + Screening + GVF)...\n")
t0 <- Sys.time()
fit_merf_enh <- fastsaegpu::merf_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  domain = "domain",
  weighted = TRUE,
  use_oob = TRUE,
  feature_screening = TRUE,
  importance_threshold = 0.0,
  smooth_vardir = TRUE,
  gvf_method = "log_linear",
  engine = "ranger",
  num_trees = 400,
  max_iter = 25,
  mse_type = "none",
  seed = 2026L
)
t_merf_enh <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_merf_enh <- pmin(pmax(fit_merf_enh$estimates$merf, 1e-4), 1 - 1e-4)
eval_list[[8]] <- calc_eval_metrics(est_merf_enh, true_mean_pop, "8. fastsaegpu (Enhanced MERF)",
                                    pop_weights, target_pop_mean, outlier_domains, t_merf_enh)

# ------------------------------------------------------------------------------
# 5. Tabulasi Hasil & Visualisasi Komparasi
# ------------------------------------------------------------------------------
cat("\n>>> [4/5] Mengompilasi Tabel Evaluasi Komparatif Lengkap...\n")

df_results <- do.call(rbind, eval_list)
mse_direct <- df_results$MSE[1]
df_results$Relative_Efficiency <- mse_direct / df_results$MSE

summary_table <- data.frame(
  Model = df_results$Model,
  ARB_All = sprintf("%.2f%%", df_results$ARB_All),
  RRMSE_All = sprintf("%.2f%%", df_results$RRMSE_All),
  MAE_All = sprintf("%.4f", df_results$MAE_All),
  Correlation = sprintf("%.4f", df_results$Corr),
  Rel_Efficiency = sprintf("%.2fx", df_results$Relative_Efficiency),
  RRMSE_Outliers = sprintf("%.2f%%", df_results$RRMSE_Outliers),
  RRMSE_Regular = sprintf("%.2f%%", df_results$RRMSE_Regular),
  Agg_Error = sprintf("%.3f%%", df_results$Agg_Error_Pct),
  Runtime = sprintf("%.2fs", df_results$Runtime_Sec)
)

cat("\n=======================================================================================================================\n")
cat(" TABEL MASTER KOMPARASI LENGKAP 8 MODEL SMALL AREA ESTIMATION\n")
cat(" (D = 50 Wilayah, N = ~125.000 Unit Populasi, 12 Kovariat Sinyal/Noise, 4 Outliers, Ground Truth = P_bar_populasi)\n")
cat("=======================================================================================================================\n")
print(summary_table, row.names = FALSE)
cat("=======================================================================================================================\n\n")

# Simpan CSV hasil simulasi
write.csv(df_results, "benchmarks/master_model_comparison.csv", row.names = FALSE)
cat("Tabel hasil komparasi lengkap telah disimpan ke: benchmarks/master_model_comparison.csv\n")

# ------------------------------------------------------------------------------
# 6. Pembuatan Visualisasi Grafik Multi-Model
# ------------------------------------------------------------------------------
cat(">>> [5/5] Membuat Visualisasi Grafik Komparasi Master...\n")

# Plot 1: 6-Panel Scatter Plot Pembanding Utama
key_models <- c("Direct Survey Estimator", "fastsae (INLA Laplace)", "tipsae (Stan NUTS CPU)",
                "fastsaegpu (Standard HB GPU)", "fastsaegpu (Best HB Synergy)", "fastsaegpu (Enhanced MERF)")
key_estimates <- list(est_direct, est_fsae_inla, est_tipsae, est_gpu_std, est_gpu_best, est_merf_enh)

df_plot_domains <- data.frame(
  domain = rep(survey_data$domain, length(key_models)),
  type = factor(rep(key_models, each = D), levels = key_models),
  estimate = unlist(key_estimates),
  true_mean = rep(true_mean_pop, length(key_models)),
  is_outlier = rep(survey_data$is_outlier, length(key_models))
)

p1 <- ggplot(df_plot_domains, aes(x = true_mean, y = estimate, color = is_outlier)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  geom_point(alpha = 0.85, size = 2.2) +
  scale_color_manual(values = c("FALSE" = "#1f77b4", "TRUE" = "#d62728"),
                     labels = c("FALSE" = "Wilayah Reguler", "TRUE" = "Wilayah Outlier / Shock")) +
  facet_wrap(~type, ncol = 3) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 13),
    strip.text = element_text(face = "bold", size = 10)
  ) +
  labs(
    title = "Validasi Master SAE: Estimasi 6 Model Pembanding vs Ground Truth Populasi",
    subtitle = "50 Wilayah (~125.000 Populasi Finis) | Evaluasi Estimasi terhadap Rata-rata Populasi Sejati (P_bar_d)",
    x = "Target Rata-rata Populasi Sejati (Ground Truth P_bar_d)",
    y = "Prediksi Model SAE",
    color = "Status Wilayah:"
  )

# Plot 2: Perbandingan RRMSE & ARB Seluruh 8 Model
plot_bars_df <- data.frame(
  Model = factor(rep(df_results$Model, 2), levels = rev(df_results$Model)),
  Metric = rep(c("RRMSE (%)", "ARB (%)"), each = nrow(df_results)),
  Value = c(df_results$RRMSE_All, df_results$ARB_All)
)

p2 <- ggplot(plot_bars_df, aes(x = Value, y = Model, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = c("RRMSE (%)" = "#2b5c8f", "ARB (%)" = "#e26d5c")) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 9)
  ) +
  labs(
    title = "Perbandingan Akurasi 8 Model SAE Lintas Paradigma (RRMSE & ARB)",
    subtitle = "Evaluasi Komparatif terhadap Ground Truth Populasi Finis (Semakin kecil persentase, semakin akurat)",
    x = "Persentase Kesalahan (%)",
    y = ""
  )

ggsave("benchmarks/master_model_scatter_comparison.png", plot = p1, width = 11.5, height = 7.5, dpi = 300)
ggsave("benchmarks/master_model_accuracy_bars.png", plot = p2, width = 10.0, height = 6.5, dpi = 300)

cat("Grafik visualisasi master telah disimpan:\n")
cat(" - benchmarks/master_model_scatter_comparison.png\n")
cat(" - benchmarks/master_model_accuracy_bars.png\n\n")

cat("===================================================================================================\n")
cat(" MASTER BENCHMARK SELESAI DENGAN SUKSES!\n")
cat("===================================================================================================\n")
