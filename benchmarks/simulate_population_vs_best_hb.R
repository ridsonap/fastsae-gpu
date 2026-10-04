#!/usr/bin/env Rscript
# ==============================================================================
# Monte Carlo Finite Population Simulation Study
# Comparison: Direct vs Standard HB vs Best HB Synergy vs MERF (Machine Learning)
#
# Context:
#   - D = 50 Small Areas (Domains)
#   - Total Population: ~125,000 individuals (~2,500 per domain)
#   - 12 Covariates: 4 Signal (Linear, Non-Linear Sine, Interaction), 8 Noise
#   - Domain Outliers / Shocks: 4 domains (8%) with localized severe anomalies
#   - Sampling Design: SRSWOR with small sample sizes (n_d = 20 to 45 per domain)
#   - Ground Truth: Exact population domain mean \bar{Y}_d calculated from all N_d units
#
# Models Evaluated:
#   1. Direct Survey Estimator
#   2. Standard HB Area Model (Normal prior, Gaussian RE, noisy direct variance)
#   3. HB Area + Regularized Horseshoe Prior (Sparse covariate shrinkage)
#   4. HB Area + Robust Student-t Random Effects (Outlier protection)
#   5. Best HB Area Model (Full Synergy: GVF Smoothing + Horseshoe + Robust Student-t + Benchmarking)
#   6. MERF / FH-RF (Mixed Effects Random Forest Machine Learning SAE)
#   7. MERF + GVF Variance Smoothing (Synergy ML SAE)
# ==============================================================================

suppressPackageStartupMessages({
  library(fastsaegpu)
  library(ranger)
  library(ggplot2)
})

set.seed(2026L)

cat("===============================================================================\n")
cat(" STUDI SIMULASI POPULASI FINIS & EVALUASI KOMPARATIF MODEL TERBAIK HB_AREA\n")
cat("===============================================================================\n\n")

# ------------------------------------------------------------------------------
# 1. Pembangkitan Populasi Finis Ground Truth (~125.000 Unit Individu)
# ------------------------------------------------------------------------------
D <- 50L
domain_ids <- paste0("domain_", sprintf("%02d", 1:D))
outlier_domains <- c(8L, 19L, 33L, 47L) # 4 wilayah dengan guncangan lokal / outlier

cat(">>> [1/5] Membangkitkan Populasi Finis Tetap (~125.000 unit di 50 Wilayah)...\n")

# Efek acak area sejati u_d
sigma_u_true <- 0.35
u_true <- rnorm(D, mean = 0, sd = sigma_u_true)
# Menambahkan shock/anomali ekstrem pada 4 wilayah
u_true[outlier_domains] <- c(1.85, -1.75, 1.95, -1.80)
names(u_true) <- domain_ids

pop_list <- vector("list", D)
true_mean_pop <- numeric(D)
N_d_vec <- integer(D)
names(true_mean_pop) <- names(N_d_vec) <- domain_ids

# Matriks penyimpanan nilai rata-rata kovariat populasi (diketahui dari sensus/administrasi)
cov_names <- c(paste0("x_sig", 1:4), paste0("x_noise", 1:8))
pop_X_means <- as.data.frame(matrix(0, nrow = D, ncol = length(cov_names)))
colnames(pop_X_means) <- cov_names
rownames(pop_X_means) <- domain_ids

for (d in 1:D) {
  dom <- domain_ids[d]
  N_d <- sample(2000:3000, 1) # Rata-rata 2.500 individu per wilayah
  N_d_vec[d] <- N_d
  
  # 4 Kovariat Sinyal:
  # x1: Linear continue
  # x2: Biner (status sosial/demografi)
  # x3: Non-linear (gelombang siklis/kelembapan/ketinggian)
  # x4: Berinteraksi dengan x1
  x1 <- rnorm(N_d, mean = 2.0, sd = 1.0)
  x2 <- rbinom(N_d, size = 1, prob = 0.45)
  x3 <- runif(N_d, min = 0.2, max = 3.8)
  x4 <- rnorm(N_d, mean = 1.0, sd = 0.8)
  
  # 8 Kovariat Noise murni (tidak ada efek sejati pada y)
  X_noise <- matrix(rnorm(N_d * 8, mean = 0, sd = 1), nrow = N_d, ncol = 8)
  colnames(X_noise) <- paste0("x_noise", 1:8)
  
  # Fungsi respon individu di tingkat populasi:
  # y_di = Intercept + beta1*x1 + beta2*x2 + f_nonlin(x3) + f_interact(x1, x4) + u_d + e_di
  mu_ind <- 5.0 + 1.6 * x1 - 1.2 * x2 + 1.5 * sin(x3) + 0.8 * (x1 * x4) + u_true[d]
  e_di <- rnorm(N_d, mean = 0, sd = 2.2)
  y_di <- mu_ind + e_di
  
  # Hitung Ground Truth Populasi Finis Eksak
  true_mean_pop[d] <- mean(y_di)
  
  # Simpan rata-rata kovariat tingkat wilayah (Auxiliary Census Data)
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
cat(sprintf("    Rata-rata Target Populasi Sejati (Y_bar_d): %.4f (Rentang: %.4f - %.4f)\n",
            mean(true_mean_pop), min(true_mean_pop), max(true_mean_pop)))
cat(sprintf("    Wilayah Outlier / Shock Lokal (4 domain): %s\n\n", paste(domain_ids[outlier_domains], collapse = ", ")))

# ------------------------------------------------------------------------------
# 2. Penarikan Sampel Survei (SRSWOR Desain Sampel Kecil)
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
  
  # Penduga Langsung (Direct Estimator)
  sample_y_dir[d] <- mean(y_samp)
  
  # Varians Sampling Langsung dengan FPC (Finite Population Correction)
  fpc <- 1 - (n_d / N_d_vec[d])
  s2 <- var(y_samp)
  sample_vardir_true[d] <- fpc * (s2 / n_d)
  
  # Tambahkan sampling variability realistis pada direct variance
  noise_v <- exp(rnorm(1, mean = 0, sd = 0.35))
  sample_vardir[d] <- max(1e-4, sample_vardir_true[d] * noise_v)
}

# Bobot populasi untuk evaluasi konsistensi agregasi
pop_weights <- N_d_vec
target_pop_total <- sum(pop_weights * true_mean_pop)
target_pop_mean <- target_pop_total / sum(pop_weights)

# Buat Data Frame Survei Area-Level
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

cat("    Data survei berhasil disiapkan:\n")
cat(sprintf("    - Rata-rata sampel per wilayah: %.1f individu\n", mean(n_sample_vec)))
cat(sprintf("    - Varians sampel langsung rata-rata: %.4f (SE rata-rata: %.4f)\n\n",
            mean(sample_vardir), mean(sqrt(sample_vardir))))

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
  
  # Evaluasi Domain Outlier vs Non-Outlier
  arb_out <- mean(abs_rel_err[outlier_idx])
  rrmse_out <- sqrt(mean(sq_rel_err[outlier_idx])) * 100
  
  reg_idx <- setdiff(seq_along(truth), outlier_idx)
  arb_reg <- mean(abs_rel_err[reg_idx])
  rrmse_reg <- sqrt(mean(sq_rel_err[reg_idx])) * 100
  
  # Efisiensi Relatif (MSE Ratio vs Direct)
  mse_all <- mean(diff^2)
  
  # Agregasi
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
# 4. Estimasi Model-Model SAE
# ------------------------------------------------------------------------------
cat(">>> [3/5] Mengestimasi Model-Model SAE...\n\n")

eval_list <- list()

# ------------------------------------------------------------------------------
# Model 1: Direct Survey Estimator (Baseline)
# ------------------------------------------------------------------------------
cat(" [1/7] Direct Survey Estimator (Survey Baseline)...\n")
t0 <- Sys.time()
est_direct <- survey_data$y
t_direct <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
eval_list[[1]] <- calc_eval_metrics(est_direct, true_mean_pop, "1. Direct Survey Estimator",
                                    pop_weights, target_pop_mean, outlier_domains, t_direct)

# ------------------------------------------------------------------------------
# Model 2: Standard HB Area (Normal Prior, Gaussian RE, No GVF)
# ------------------------------------------------------------------------------
cat(" [2/7] Standard HB Area (Normal prior, Gaussian RE, noisy vardir)...\n")
t0 <- Sys.time()
fit_hb_std <- hb_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "normal",
  robust = FALSE,
  smooth_vardir = FALSE,
  warmup = 250L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
t_hb_std <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_hb_std <- fit_hb_std$df_hb$hb
eval_list[[2]] <- calc_eval_metrics(est_hb_std, true_mean_pop, "2. Standard HB Area",
                                    pop_weights, target_pop_mean, outlier_domains, t_hb_std)

# ------------------------------------------------------------------------------
# Model 3: HB Area + Horseshoe Prior (Sparse Shrinkage untuk 8 Noise Covariates)
# ------------------------------------------------------------------------------
cat(" [3/7] HB Area + Regularized Horseshoe Prior (Carvalho et al. 2010)...\n")
t0 <- Sys.time()
fit_hb_hs <- hb_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "horseshoe",
  robust = FALSE,
  smooth_vardir = FALSE,
  warmup = 250L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
t_hb_hs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_hb_hs <- fit_hb_hs$df_hb$hb
eval_list[[3]] <- calc_eval_metrics(est_hb_hs, true_mean_pop, "3. HB Area + Horseshoe Prior",
                                    pop_weights, target_pop_mean, outlier_domains, t_hb_hs)

# ------------------------------------------------------------------------------
# Model 4: HB Area + Robust Student-t Random Effects (Bell & Huang 2006)
# ------------------------------------------------------------------------------
cat(" [4/7] HB Area + Robust Student-t Random Effects (Proteksi Outlier)...\n")
t0 <- Sys.time()
fit_hb_rob <- hb_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  family = "gaussian",
  prior_beta = "normal",
  robust = TRUE,
  smooth_vardir = FALSE,
  warmup = 250L,
  samples = 500L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
t_hb_rob <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_hb_rob <- fit_hb_rob$df_hb$hb
eval_list[[4]] <- calc_eval_metrics(est_hb_rob, true_mean_pop, "4. HB Area + Student-t Robust",
                                    pop_weights, target_pop_mean, outlier_domains, t_hb_rob)

# ------------------------------------------------------------------------------
# Model 5: Best HB Area Model (Full Synergy: GVF + Horseshoe + Robust + Benchmarking)
# ------------------------------------------------------------------------------
cat(" [5/7] Model Terbaik HB Area: Full Synergy (GVF + Horseshoe + Robust + Benchmark)...\n")
t0 <- Sys.time()
fit_hb_best <- hb_area(
  formula = formula_linear,
  data = survey_data,
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
  samples = 600L,
  chains = 1L,
  device = "cpu",
  print_result = FALSE
)
t_hb_best <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_hb_best <- fit_hb_best$df_hb$hb
eval_list[[5]] <- calc_eval_metrics(est_hb_best, true_mean_pop, "5. Best HB Area (Full Synergy)",
                                    pop_weights, target_pop_mean, outlier_domains, t_hb_best)

# ------------------------------------------------------------------------------
# Model 6: MERF / FH-RF (Mixed Effects Random Forest)
# ------------------------------------------------------------------------------
cat(" [6/7] MERF / FH-RF Machine Learning SAE (ranger C++ Multithreaded)...\n")
t0 <- Sys.time()
fit_merf <- merf_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  domain = "domain",
  engine = "ranger",
  num_trees = 500,
  max_iter = 30,
  mse_type = "bootstrap",
  B = 30,
  seed = 2026L
)
t_merf <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_merf <- fit_merf$estimates$merf
eval_list[[6]] <- calc_eval_metrics(est_merf, true_mean_pop, "6. MERF (Mixed Effects Random Forest)",
                                    pop_weights, target_pop_mean, outlier_domains, t_merf)

# ------------------------------------------------------------------------------
# Model 7: MERF + GVF Variance Smoothing (Synergy ML SAE)
# ------------------------------------------------------------------------------
cat(" [7/7] MERF + GVF Variance Smoothing (Synergy ML SAE)...\n")
t0 <- Sys.time()
fit_merf_gvf <- merf_area(
  formula = formula_linear,
  data = survey_data,
  vardir = "vardir",
  domain = "domain",
  smooth_vardir = TRUE,
  gvf_method = "log_linear",
  engine = "ranger",
  num_trees = 500,
  max_iter = 30,
  mse_type = "bootstrap",
  B = 30,
  seed = 2026L
)
t_merf_gvf <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
est_merf_gvf <- fit_merf_gvf$estimates$merf
eval_list[[7]] <- calc_eval_metrics(est_merf_gvf, true_mean_pop, "7. MERF + GVF Smoothing",
                                    pop_weights, target_pop_mean, outlier_domains, t_merf_gvf)

# ------------------------------------------------------------------------------
# 5. Tabulasi & Visualisasi Hasil Komparatif
# ------------------------------------------------------------------------------
cat("\n>>> [4/5] Mengompilasi Metrik Evaluasi & Menyimpan Hasil...\n")

df_results <- do.call(rbind, eval_list)
# Hitung Relative Efficiency (RE) terhadap Direct Estimator: MSE_direct / MSE_model
mse_direct <- df_results$MSE[1]
df_results$Relative_Efficiency <- mse_direct / df_results$MSE

# Tampilkan ringkasan utama
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

cat("\n===================================================================================================\n")
cat(" TABEL HASIL SIMULASI POPULASI FINIS: KOMPARASI PERFORMA MODEL SAE\n")
cat(" (D = 50 Wilayah, 12 Kovariat Sinyal/Noise, 4 Outlier Shocks, Ground Truth = Y_bar_populasi)\n")
cat("===================================================================================================\n")
print(summary_table, row.names = FALSE)
cat("===================================================================================================\n\n")

# Simpan CSV hasil simulasi
write.csv(df_results, "benchmarks/population_simulation_model_comparison.csv", row.names = FALSE)
cat("Hasil tabel lengkap telah disimpan ke: benchmarks/population_simulation_model_comparison.csv\n")

# Buat Data Frame Prediksi per Domain untuk Visualisasi
df_plot_domains <- data.frame(
  domain = rep(survey_data$domain, 4),
  type = factor(rep(c("Direct Survey", "Standard HB", "Best HB (Synergy)", "MERF (ML)"), each = D),
                levels = c("Direct Survey", "Standard HB", "Best HB (Synergy)", "MERF (ML)")),
  estimate = c(est_direct, est_hb_std, est_hb_best, est_merf),
  true_mean = rep(true_mean_pop, 4),
  is_outlier = rep(survey_data$is_outlier, 4)
)

# ------------------------------------------------------------------------------
# Visualisasi Grafik Komparasi Model
# ------------------------------------------------------------------------------
cat(">>> [5/5] Membuat Visualisasi Grafik Komparasi...\n")

# Plot 1: Scatter plot Prediksi vs Ground Truth Populasi Sejati
p1 <- ggplot(df_plot_domains, aes(x = true_mean, y = estimate, color = is_outlier)) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray40", linewidth = 0.8) +
  geom_point(alpha = 0.85, size = 2.2) +
  scale_color_manual(values = c("FALSE" = "#1f77b4", "TRUE" = "#d62728"),
                     labels = c("FALSE" = "Wilayah Reguler", "TRUE" = "Wilayah Outlier / Shock")) +
  facet_wrap(~type, ncol = 2) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "bottom",
    plot.title = element_text(face = "bold", size = 13),
    strip.text = element_text(face = "bold", size = 11)
  ) +
  labs(
    title = "Validasi Simulasi Populasi Finis: Estimasi SAE vs Target Populasi Sejati",
    subtitle = "50 Wilayah (~125.000 Unit Populasi) | Perbandingan Akurasi Titik Estimasi terhadap Ground Truth",
    x = "Target Rata-rata Populasi Sejati (Ground Truth Y_bar_d)",
    y = "Prediksi Model SAE",
    color = "Status Wilayah:"
  )

# Plot 2: Perbandingan RRMSE & ARB
plot_metrics_df <- data.frame(
  Model = factor(rep(df_results$Model, 2), levels = rev(df_results$Model)),
  Metric = rep(c("RRMSE (%)", "ARB (%)"), each = nrow(df_results)),
  Value = c(df_results$RRMSE_All, df_results$ARB_All)
)

p2 <- ggplot(plot_metrics_df, aes(x = Value, y = Model, fill = Metric)) +
  geom_col(position = position_dodge(width = 0.7), width = 0.6) +
  scale_fill_manual(values = c("RRMSE (%)" = "#2b5c8f", "ARB (%)" = "#e26d5c")) +
  theme_minimal(base_size = 11) +
  theme(
    legend.position = "top",
    plot.title = element_text(face = "bold", size = 12),
    axis.text.y = element_text(size = 9)
  ) +
  labs(
    title = "Perbandingan Akurasi Model SAE pada Populasi Finis (RRMSE & ARB)",
    subtitle = "Semakin kecil nilai persentase error, semakin tinggi presisi dan akurasi model",
    x = "Persentase Kesalahan (%)",
    y = ""
  )

# Simpan grafik
ggsave("benchmarks/population_sim_scatter_comparison.png", plot = p1, width = 9.5, height = 7.5, dpi = 300)
ggsave("benchmarks/population_sim_accuracy_bars.png", plot = p2, width = 9.0, height = 5.5, dpi = 300)
cat("Grafik visualisasi telah disimpan:\n")
cat(" - benchmarks/population_sim_scatter_comparison.png\n")
cat(" - benchmarks/population_sim_accuracy_bars.png\n\n")

cat("===============================================================================\n")
cat(" SIMULASI SELESAI DENGAN SUKSES!\n")
cat("===============================================================================\n")
