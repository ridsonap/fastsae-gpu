#!/usr/bin/env Rscript
# ==============================================================================
# Monte Carlo Finite Population Simulation Study (D = 50 Domains, x = 3 Covariates)
# Comparison: fastsaegpu vs tipsae vs fastsae (Frequentist & Bayesian)
# Models Evaluated:
#   1. Non-Spatial Beta SAE
#   2. Spatial Beta SAE (Besag ICAR)
# Population Size: Large (~150,000 individuals, ~3,000 per domain)
# Sampling Design: SRSWOR (n_d = 30 per domain)
# ==============================================================================

suppressPackageStartupMessages({
  library(sf)
  library(spdep)
  library(tipsae)
  library(fastsae)
  library(fastsaegpu)
  library(ggplot2)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)
R_reps <- 30L
if ("--reps" %in% args) {
  idx <- which(args == "--reps") + 1L
  if (idx <= length(args)) R_reps <- as.integer(args[idx])
}

cat("===============================================================================\n")
cat(" STUDI SIMULASI MONTE CARLO POPULASI FINIS\n")
cat(sprintf(" D = 50 Wilayah, x = 3 Kovariat | %d Replikasi Monte Carlo\n", R_reps))
cat(" Model: Non-Spasial & Spasial (Besag ICAR)\n")
cat(" Pembanding: Direct, fastsae (Frequentist), fastsae (Bayesian), tipsae, fastsaegpu\n")
cat("===============================================================================\n\n")

set.seed(2026L)
D <- 50L
domain_ids <- paste0("domain_", sprintf("%02d", 1:D))

# ------------------------------------------------------------------------------
# 1. Bangun Kisi Spasial & Matriks Ketetanggaan W (10 x 5 Regular Grid)
# ------------------------------------------------------------------------------
cat(">>> [1/5] Membangun Kisi Spasial (10 x 5) dan Matriks Ketetanggaan W...\n")
grid_poly <- st_polygon(list(matrix(c(0, 0, 10, 0, 10, 5, 0, 5, 0, 0), ncol = 2, byrow = TRUE)))
grid <- st_make_grid(grid_poly, n = c(10, 5))
grid_sf <- st_sf(domain = domain_ids, geometry = grid)

nb <- poly2nb(grid_sf, queen = TRUE)
W <- nb2mat(nb, style = "B", zero.policy = TRUE)
rownames(W) <- colnames(W) <- domain_ids

# Efek acak spasial laten (ICAR Besag)
Q_s <- diag(rowSums(W)) - W
eig_s <- eigen(Q_s, symmetric = TRUE)
pos_idx <- which(eig_s$values > 1e-6)
V_s <- eig_s$vectors[, pos_idx]
lambda_s <- eig_s$values[pos_idx]
v_raw <- as.vector(V_s %*% (rnorm(length(pos_idx)) / sqrt(lambda_s)))
v_spatial_true <- (v_raw - mean(v_raw)) / sd(v_raw) * 0.35
names(v_spatial_true) <- domain_ids

# Efek acak non-spasial laten (IID)
u_nonspatial_true <- rnorm(D, mean = 0, sd = 0.30)
names(u_nonspatial_true) <- domain_ids

# ------------------------------------------------------------------------------
# 2. Pembangkitan Populasi Finis Tetap (~150.000 Unit Individu)
# ------------------------------------------------------------------------------
cat(">>> [2/5] Membangkitkan Populasi Finis Tetap (~150.000 Unit Individu)...\n")
beta_true <- c(beta0 = -0.50, beta1 = 0.35, beta2 = -0.25, beta3 = 0.40)
phi_prec <- 30.0

pop_list_ns <- vector("list", D)
pop_list_sp <- vector("list", D)
true_P_ns <- numeric(D)
true_P_sp <- numeric(D)
N_d_vec <- integer(D)
names(true_P_ns) <- names(true_P_sp) <- names(N_d_vec) <- domain_ids

# Population mean auxiliary covariates (known from census/registry)
pop_X_means <- data.frame(
  domain = domain_ids,
  x1 = numeric(D),
  x2 = numeric(D),
  x3 = numeric(D),
  stringsAsFactors = FALSE
)

for (d in 1:D) {
  dom <- domain_ids[d]
  N_d <- sample(2500:3500, 1) # Rata-rata 3.000 unit per domain
  N_d_vec[d] <- N_d
  
  # 3 Kovariat level individu
  x1_ind <- rnorm(N_d, mean = 1.5, sd = 0.5)
  x2_ind <- rbinom(N_d, size = 1, prob = 0.4)
  x3_ind <- runif(N_d, min = 0.5, max = 2.5)
  
  pop_X_means$x1[d] <- mean(x1_ind)
  pop_X_means$x2[d] <- mean(x2_ind)
  pop_X_means$x3[d] <- mean(x3_ind)
  
  # Linear predictor
  eta_fixed <- beta_true["beta0"] + beta_true["beta1"] * x1_ind + 
    beta_true["beta2"] * x2_ind + beta_true["beta3"] * x3_ind
  
  # Model Non-Spasial
  eta_ns <- eta_fixed + u_nonspatial_true[dom]
  p_ns <- plogis(eta_ns)
  y_ns <- rbeta(N_d, pmax(p_ns * phi_prec, 1e-4), pmax((1 - p_ns) * phi_prec, 1e-4))
  y_ns <- pmin(pmax(y_ns, 1e-5), 1 - 1e-5)
  true_P_ns[d] <- mean(y_ns)
  pop_list_ns[[d]] <- data.frame(domain = dom, y = y_ns, x1 = x1_ind, x2 = x2_ind, x3 = x3_ind)
  
  # Model Spasial
  eta_sp <- eta_fixed + v_spatial_true[dom]
  p_sp <- plogis(eta_sp)
  y_sp <- rbeta(N_d, pmax(p_sp * phi_prec, 1e-4), pmax((1 - p_sp) * phi_prec, 1e-4))
  y_sp <- pmin(pmax(y_sp, 1e-5), 1 - 1e-5)
  true_P_sp[d] <- mean(y_sp)
  pop_list_sp[[d]] <- data.frame(domain = dom, y = y_sp, x1 = x1_ind, x2 = x2_ind, x3 = x3_ind)
}

pop_df_ns <- do.call(rbind, pop_list_ns)
pop_df_sp <- do.call(rbind, pop_list_sp)
total_N_pop <- sum(N_d_vec)

cat(sprintf("    Total Populasi: %s individu di %d wilayah (Min: %d, Max: %d).\n",
            format(total_N_pop, big.mark = "."), D, min(N_d_vec), max(N_d_vec)))
cat(sprintf("    Rata-rata True P_d (Non-Spasial): %.4f (Min: %.4f, Max: %.4f)\n",
            mean(true_P_ns), min(true_P_ns), max(true_P_ns)))
cat(sprintf("    Rata-rata True P_d (Spasial)    : %.4f (Min: %.4f, Max: %.4f)\n\n",
            mean(true_P_sp), min(true_P_sp), max(true_P_sp)))

# ------------------------------------------------------------------------------
# 3. Setup Struktur Penyimpanan Monte Carlo
# ------------------------------------------------------------------------------
n_sample_d <- 30L # Sample size per domain (f ~ 1%)

estimators <- c("Direct", "fastsae_freq", "fastsae_bayes", "tipsae", "fastsaegpu")

# Matriks penyimpanan estimasi (D x R) untuk tiap estimator & tipe model
est_ns <- setNames(lapply(estimators, function(x) matrix(NA_real_, D, R_reps)), estimators)
est_sp <- setNames(lapply(estimators, function(x) matrix(NA_real_, D, R_reps)), estimators)

# Matriks cakupan interval kepercayaan 95% (D x R)
ci_ns <- setNames(lapply(estimators, function(x) matrix(0L, D, R_reps)), estimators)
ci_sp <- setNames(lapply(estimators, function(x) matrix(0L, D, R_reps)), estimators)

# Waktu komputasi kumulatif
times_ns <- setNames(numeric(length(estimators)), estimators)
times_sp <- setNames(numeric(length(estimators)), estimators)

# ------------------------------------------------------------------------------
# 4. Loop Replikasi Monte Carlo (R Replikasi)
# ------------------------------------------------------------------------------
cat(sprintf(">>> [3/5] Memulai Eksekusi %d Replikasi Monte Carlo...\n", R_reps))
t_mc_start <- Sys.time()

for (r in 1:R_reps) {
  cat(sprintf("\n--- Replikasi [%2d / %2d] ---\n", r, R_reps))
  
  # A. Penarikan sampel SRSWOR n_d = 30 dari tiap domain
  samp_ns_list <- vector("list", D)
  samp_sp_list <- vector("list", D)
  
  df_sample_ns <- data.frame(
    domain = domain_ids,
    y = numeric(D),
    vardir = numeric(D),
    x1 = pop_X_means$x1,
    x2 = pop_X_means$x2,
    x3 = pop_X_means$x3,
    n = n_sample_d,
    stringsAsFactors = FALSE
  )
  df_sample_sp <- df_sample_ns
  
  for (d in 1:D) {
    dom <- domain_ids[d]
    sub_ns <- pop_list_ns[[d]]
    sub_sp <- pop_list_sp[[d]]
    idx_s <- sample.int(nrow(sub_ns), size = n_sample_d, replace = FALSE)
    
    # Non-spasial sample
    s_y_ns <- sub_ns$y[idx_s]
    df_sample_ns$y[d] <- mean(s_y_ns)
    v_s_ns <- var(s_y_ns)
    fpc_ns <- (1 - n_sample_d / N_d_vec[d])
    df_sample_ns$vardir[d] <- max(1e-6, fpc_ns * v_s_ns / n_sample_d)
    
    # Spasial sample
    s_y_sp <- sub_sp$y[idx_s]
    df_sample_sp$y[d] <- mean(s_y_sp)
    v_s_sp <- var(s_y_sp)
    fpc_sp <- (1 - n_sample_d / N_d_vec[d])
    df_sample_sp$vardir[d] <- max(1e-6, fpc_sp * v_s_sp / n_sample_d)
  }
  
  # Ensure bounds in (1e-4, 1 - 1e-4) for Beta modeling
  df_sample_ns$y <- pmin(pmax(df_sample_ns$y, 1e-4), 1 - 1e-4)
  df_sample_sp$y <- pmin(pmax(df_sample_sp$y, 1e-4), 1 - 1e-4)
  
  # ============================================================================
  # [MODEL 1: NON-SPATIAL]
  # ============================================================================
  # 1. Direct Estimator
  t0 <- Sys.time()
  est_ns$Direct[, r] <- df_sample_ns$y
  se_dir_ns <- sqrt(df_sample_ns$vardir)
  ci_low_dir_ns <- df_sample_ns$y - 1.96 * se_dir_ns
  ci_upp_dir_ns <- df_sample_ns$y + 1.96 * se_dir_ns
  ci_ns$Direct[, r] <- as.integer(true_P_ns >= ci_low_dir_ns & true_P_ns <= ci_upp_dir_ns)
  times_ns["Direct"] <- times_ns["Direct"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 2. fastsae Frequentist (EBLUP Fay-Herriot)
  t0 <- Sys.time()
  fit_fsae_freq_ns <- fastsae::eblup_fh(
    formula = y ~ x1 + x2 + x3,
    vardir = "vardir",
    data = df_sample_ns,
    method = "REML",
    print_result = FALSE
  )
  est_fsae_freq_ns <- fit_fsae_freq_ns$df_eblup$eblup
  est_ns$fastsae_freq[, r] <- est_fsae_freq_ns
  se_fsae_freq_ns <- sqrt(fit_fsae_freq_ns$df_eblup$mse)
  ci_ns$fastsae_freq[, r] <- as.integer(
    true_P_ns >= (est_fsae_freq_ns - 1.96 * se_fsae_freq_ns) &
    true_P_ns <= (est_fsae_freq_ns + 1.96 * se_fsae_freq_ns)
  )
  times_ns["fastsae_freq"] <- times_ns["fastsae_freq"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 3. fastsae Bayesian (INLA Laplace)
  t0 <- Sys.time()
  fit_fsae_bayes_ns <- fastsae::hb_area(
    formula = y ~ x1 + x2 + x3,
    data = df_sample_ns,
    domain = "domain",
    vardir = "vardir",
    family = "beta",
    spatial = "none",
    temporal = "none",
    print_result = FALSE
  )
  est_fsae_bayes_ns <- fit_fsae_bayes_ns$df_hb$hb
  est_ns$fastsae_bayes[, r] <- est_fsae_bayes_ns
  ci_ns$fastsae_bayes[, r] <- as.integer(
    true_P_ns >= fit_fsae_bayes_ns$df_hb$ci_lower &
    true_P_ns <= fit_fsae_bayes_ns$df_hb$ci_upper
  )
  times_ns["fastsae_bayes"] <- times_ns["fastsae_bayes"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 4. tipsae (Stan NUTS CPU)
  t0 <- Sys.time()
  fit_tip_ns <- tipsae::fit_sae(
    formula_fixed = y ~ x1 + x2 + x3,
    data = df_sample_ns,
    domains = "domain",
    disp_direct = "vardir",
    type_disp = "var",
    domain_size = "n",
    spatial_error = FALSE,
    temporal_error = FALSE,
    chains = 1L,
    iter = 350L,
    seed = 42L + r
  )
  s_tip_ns <- summary(fit_tip_ns)$model_estimates
  est_tip_ns <- s_tip_ns$mean
  est_ns$tipsae[, r] <- est_tip_ns
  ci_ns$tipsae[, r] <- as.integer(true_P_ns >= s_tip_ns$`2.5%` & true_P_ns <= s_tip_ns$`97.5%`)
  times_ns["tipsae"] <- times_ns["tipsae"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 5. fastsaegpu (NumPyro NUTS JAX)
  t0 <- Sys.time()
  fit_gpu_ns <- fastsaegpu::hb_area(
    formula = y ~ x1 + x2 + x3,
    data = df_sample_ns,
    domain = "domain",
    vardir = "vardir",
    family = "beta",
    spatial = "none",
    temporal = "none",
    warmup = 150L,
    samples = 200L,
    chains = 1L,
    device = "auto",
    seed = 42L + r,
    print_result = FALSE
  )
  est_gpu_ns <- fit_gpu_ns$df_hb$hb
  est_ns$fastsaegpu[, r] <- est_gpu_ns
  ci_ns$fastsaegpu[, r] <- as.integer(
    true_P_ns >= fit_gpu_ns$df_hb$ci_lower &
    true_P_ns <= fit_gpu_ns$df_hb$ci_upper
  )
  times_ns["fastsaegpu"] <- times_ns["fastsaegpu"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # ============================================================================
  # [MODEL 2: SPATIAL BESAG ICAR]
  # ============================================================================
  # 1. Direct Estimator (Spatial)
  t0 <- Sys.time()
  est_sp$Direct[, r] <- df_sample_sp$y
  se_dir_sp <- sqrt(df_sample_sp$vardir)
  ci_low_dir_sp <- df_sample_sp$y - 1.96 * se_dir_sp
  ci_upp_dir_sp <- df_sample_sp$y + 1.96 * se_dir_sp
  ci_sp$Direct[, r] <- as.integer(true_P_sp >= ci_low_dir_sp & true_P_sp <= ci_upp_dir_sp)
  times_sp["Direct"] <- times_sp["Direct"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 2. fastsae Frequentist (Spatial Fay-Herriot SFH)
  t0 <- Sys.time()
  fit_fsae_freq_sp <- fastsae::eblup_sfh(
    formula = y ~ x1 + x2 + x3,
    vardir = "vardir",
    W = W,
    data = df_sample_sp,
    method = "REML",
    print_result = FALSE
  )
  est_fsae_freq_sp <- fit_fsae_freq_sp$df_eblup$eblup
  est_sp$fastsae_freq[, r] <- est_fsae_freq_sp
  se_fsae_freq_sp <- sqrt(fit_fsae_freq_sp$df_eblup$mse)
  ci_sp$fastsae_freq[, r] <- as.integer(
    true_P_sp >= (est_fsae_freq_sp - 1.96 * se_fsae_freq_sp) &
    true_P_sp <= (est_fsae_freq_sp + 1.96 * se_fsae_freq_sp)
  )
  times_sp["fastsae_freq"] <- times_sp["fastsae_freq"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 3. fastsae Bayesian (Spatial Besag ICAR via INLA)
  t0 <- Sys.time()
  fit_fsae_bayes_sp <- fastsae::hb_area(
    formula = y ~ x1 + x2 + x3,
    data = df_sample_sp,
    domain = "domain",
    vardir = "vardir",
    family = "beta",
    spatial = "besag",
    W = W,
    print_result = FALSE
  )
  est_fsae_bayes_sp <- fit_fsae_bayes_sp$df_hb$hb
  est_sp$fastsae_bayes[, r] <- est_fsae_bayes_sp
  ci_sp$fastsae_bayes[, r] <- as.integer(
    true_P_sp >= fit_fsae_bayes_sp$df_hb$ci_lower &
    true_P_sp <= fit_fsae_bayes_sp$df_hb$ci_upper
  )
  times_sp["fastsae_bayes"] <- times_sp["fastsae_bayes"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 4. tipsae (Spatial Stan NUTS CPU)
  t0 <- Sys.time()
  fit_tip_sp <- tipsae::fit_sae(
    formula_fixed = y ~ x1 + x2 + x3,
    data = df_sample_sp,
    domains = "domain",
    disp_direct = "vardir",
    type_disp = "var",
    domain_size = "n",
    spatial_error = TRUE,
    spatial_df = grid_sf,
    domains_spatial_df = "domain",
    chains = 1L,
    iter = 350L,
    seed = 42L + r
  )
  s_tip_sp <- summary(fit_tip_sp)$model_estimates
  est_tip_sp <- s_tip_sp$mean
  est_sp$tipsae[, r] <- est_tip_sp
  ci_sp$tipsae[, r] <- as.integer(true_P_sp >= s_tip_sp$`2.5%` & true_P_sp <= s_tip_sp$`97.5%`)
  times_sp["tipsae"] <- times_sp["tipsae"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  # 5. fastsaegpu (Spatial Besag ICAR via NumPyro NUTS)
  t0 <- Sys.time()
  fit_gpu_sp <- fastsaegpu::hb_area(
    formula = y ~ x1 + x2 + x3,
    data = df_sample_sp,
    domain = "domain",
    vardir = "vardir",
    family = "beta",
    spatial = "besag",
    W = W,
    warmup = 150L,
    samples = 200L,
    chains = 1L,
    device = "auto",
    seed = 42L + r,
    print_result = FALSE
  )
  est_gpu_sp <- fit_gpu_sp$df_hb$hb
  est_sp$fastsaegpu[, r] <- est_gpu_sp
  ci_sp$fastsaegpu[, r] <- as.integer(
    true_P_sp >= fit_gpu_sp$df_hb$ci_lower &
    true_P_sp <= fit_gpu_sp$df_hb$ci_upper
  )
  times_sp["fastsaegpu"] <- times_sp["fastsaegpu"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  
  cat(sprintf("  >> Replikasi [%2d / %2d] selesai.\n", r, R_reps))
}

t_mc_total <- as.numeric(difftime(Sys.time(), t_mc_start, units = "mins"))
cat(sprintf("\n>>> Simulasi Monte Carlo selesai dalam %.2f menit.\n\n", t_mc_total))

# ------------------------------------------------------------------------------
# 5. Hitung Metrik Evaluasi Statistik Akurasi & Waktu
# ------------------------------------------------------------------------------
cat(">>> [4/5] Mengkalkulasi Metrik Akurasi (ARB, RRMSE, RMSE, MAE, CP95, r) & Waktu...\n")

calc_metrics <- function(est_mat, ci_mat, true_P, time_total, time_baseline_tipsae, model_label) {
  res_list <- vector("list", length(estimators))
  
  for (i in seq_along(estimators)) {
    est_name <- estimators[i]
    M <- est_mat[[est_name]]
    CI <- ci_mat[[est_name]]
    t_tot <- time_total[est_name]
    t_avg <- t_tot / R_reps
    speedup <- if (est_name == "Direct") NA_real_ else time_baseline_tipsae / t_tot
    
    # 1. Domain-specific Bias and Relative Bias
    bias_d <- rowMeans(M) - true_P
    rel_bias_d <- (bias_d / true_P) * 100
    arb <- mean(abs(rel_bias_d))
    
    # 2. Domain-specific MSE and Relative RMSE
    mse_d <- rowMeans((M - true_P)^2)
    rmse <- sqrt(mean(mse_d))
    rrmse_d <- (sqrt(mse_d) / true_P) * 100
    rrmse <- mean(rrmse_d)
    
    # 3. Overall MAE
    mae <- mean(abs(as.vector(M) - rep(true_P, R_reps)))
    
    # 4. Coverage Probability (95%)
    cp95 <- mean(CI) * 100
    
    # 5. Average Pearson correlation with true P_d across replications
    cor_vec <- sapply(1:R_reps, function(col) {
      cv <- suppressWarnings(cor(M[, col], true_P, use = "complete.obs"))
      if (is.na(cv)) 0 else cv
    })
    mean_cor <- mean(cor_vec, na.rm = TRUE)
    
    label_pkg <- switch(est_name,
      "Direct" = "Direct Estimator",
      "fastsae_freq" = "fastsae (Frequentist EBLUP)",
      "fastsae_bayes" = "fastsae (Bayesian INLA)",
      "tipsae" = "tipsae (Bayesian Stan NUTS)",
      "fastsaegpu" = "fastsaegpu (Bayesian NumPyro GPU)"
    )
    
    backend_pkg <- switch(est_name,
      "Direct" = "Survei Sampel Langsung",
      "fastsae_freq" = "REML Estimator (C)",
      "fastsae_bayes" = "INLA Laplace (C/Fortran)",
      "tipsae" = "Stan NUTS (C++)",
      "fastsaegpu" = "NumPyro NUTS (JAX XLA)"
    )
    
    hardware_pkg <- switch(est_name,
      "Direct" = "CPU",
      "fastsae_freq" = "CPU Single-core",
      "fastsae_bayes" = "CPU Multithread",
      "tipsae" = "CPU Single-core",
      "fastsaegpu" = "GPU / JAX Vectorized"
    )
    
    res_list[[i]] <- data.frame(
      Model = model_label,
      Method = label_pkg,
      Estimator_ID = est_name,
      Backend = backend_pkg,
      Hardware = hardware_pkg,
      ARB_pct = round(arb, 2),
      RRMSE_pct = round(rrmse, 2),
      RMSE = round(rmse, 4),
      MAE = round(mae, 4),
      CP95_pct = round(cp95, 2),
      Correlation_r = round(mean_cor, 4),
      Avg_Time_Sec = round(t_avg, 3),
      Total_Time_Sec = round(t_tot, 2),
      Speedup_vs_tipsae = round(speedup, 2),
      stringsAsFactors = FALSE
    )
  }
  do.call(rbind, res_list)
}

summary_ns <- calc_metrics(est_ns, ci_ns, true_P_ns, times_ns, times_ns["tipsae"], "Non-Spasial (Beta SAE)")
summary_sp <- calc_metrics(est_sp, ci_sp, true_P_sp, times_sp, times_sp["tipsae"], "Spasial (Besag ICAR)")
final_summary <- rbind(summary_ns, summary_sp)

cat("\n===============================================================================\n")
cat(" RINGKASAN HASIL SIMULASI MONTE CARLO\n")
cat("===============================================================================\n")
print(final_summary[, c("Model", "Method", "ARB_pct", "RRMSE_pct", "CP95_pct", "Correlation_r", "Avg_Time_Sec", "Speedup_vs_tipsae")])
cat("===============================================================================\n\n")

# Simpan data ringkasan ke CSV
write.csv(final_summary, "benchmarks/simulation_summary.csv", row.names = FALSE)
cat(">>> Dataset ringkasan tersimpan di 'benchmarks/simulation_summary.csv'.\n")

# Simpan estimasi domain rerata Monte Carlo
domain_est_list <- list()
for (m in estimators) {
  domain_est_list[[paste0("ns_", m)]] <- data.frame(
    domain = domain_ids,
    model = "Non-Spasial",
    estimator = m,
    true_P = true_P_ns,
    mean_estimate = rowMeans(est_ns[[m]]),
    sd_estimate = apply(est_ns[[m]], 1, sd),
    stringsAsFactors = FALSE
  )
  domain_est_list[[paste0("sp_", m)]] <- data.frame(
    domain = domain_ids,
    model = "Spasial",
    estimator = m,
    true_P = true_P_sp,
    mean_estimate = rowMeans(est_sp[[m]]),
    sd_estimate = apply(est_sp[[m]], 1, sd),
    stringsAsFactors = FALSE
  )
}
domain_est_df <- do.call(rbind, domain_est_list)
write.csv(domain_est_df, "benchmarks/simulation_domain_estimates.csv", row.names = FALSE)
cat(">>> Dataset estimasi domain tersimpan di 'benchmarks/simulation_domain_estimates.csv'.\n")

# ------------------------------------------------------------------------------
# 6. Render Visualisasi Grafik Publikasi (300 DPI)
# ------------------------------------------------------------------------------
cat(">>> [5/5] Merender Visualisasi Grafis Komparasi Akurasi dan Waktu...\n")

theme_set(theme_minimal(base_size = 13))

# Plot 1: Akurasi (RRMSE % vs ARB %)
p_acc <- ggplot(final_summary, aes(x = ARB_pct, y = RRMSE_pct, color = Method, shape = Method)) +
  geom_point(size = 4.5, stroke = 1.2) +
  facet_wrap(~ Model, scales = "free") +
  geom_text(aes(label = paste0(RRMSE_pct, "%")), vjust = -1.0, size = 3.6, show.legend = FALSE) +
  scale_color_manual(values = c(
    "Direct Estimator" = "#888888",
    "fastsae (Frequentist EBLUP)" = "#2ca02c",
    "fastsae (Bayesian INLA)" = "#1f77b4",
    "tipsae (Bayesian Stan NUTS)" = "#ff7f0e",
    "fastsaegpu (Bayesian NumPyro GPU)" = "#d62728"
  )) +
  labs(
    title = "Perbandingan Akurasi Estimasi: Studi Simulasi Populasi Finis (D = 50, x = 3)",
    subtitle = "Nilai lebih rendah menunjukkan akurasi lebih tinggi (Mendekati titik 0,0)",
    x = "Average Absolute Relative Bias / ARB (%)",
    y = "Relative Root Mean Squared Error / RRMSE (%)",
    color = "Metode / Paket",
    shape = "Metode / Paket"
  ) +
  theme(
    legend.position = "bottom",
    legend.box = "horizontal",
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 12)
  )

ggsave("benchmarks/simulation_accuracy_comparison.png", p_acc, width = 11, height = 6, dpi = 300)
cat(">>> Grafik akurasi tersimpan di 'benchmarks/simulation_accuracy_comparison.png'.\n")

# Plot 2: Waktu Komputasi per Replikasi & Rasio Speedup
df_time <- final_summary[final_summary$Estimator_ID != "Direct", ]
p_time <- ggplot(df_time, aes(x = Method, y = Avg_Time_Sec, fill = Method)) +
  geom_col(width = 0.65, color = "black", alpha = 0.85) +
  facet_wrap(~ Model, scales = "free_y") +
  geom_text(aes(label = sprintf("%.2fs\n(%.1fx)", Avg_Time_Sec, Speedup_vs_tipsae)), 
            vjust = -0.3, size = 3.4, fontface = "bold") +
  scale_fill_manual(values = c(
    "fastsae (Frequentist EBLUP)" = "#2ca02c",
    "fastsae (Bayesian INLA)" = "#1f77b4",
    "tipsae (Bayesian Stan NUTS)" = "#ff7f0e",
    "fastsaegpu (Bayesian NumPyro GPU)" = "#d62728"
  )) +
  labs(
    title = "Perbandingan Efisiensi Waktu Komputasi per Replikasi (Detik)",
    subtitle = "Label mencantumkan rata-rata detik per run dan rasio percepatan (speedup) vs tipsae",
    x = NULL,
    y = "Rata-rata Waktu Eksekusi (Detik)",
    fill = "Metode / Paket"
  ) +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 25, hjust = 1, face = "bold"),
    panel.grid.minor = element_blank(),
    plot.title = element_text(face = "bold", size = 14),
    strip.text = element_text(face = "bold", size = 12)
  )

ggsave("benchmarks/simulation_runtime_comparison.png", p_time, width = 11, height = 6, dpi = 300)
cat(">>> Grafik runtime tersimpan di 'benchmarks/simulation_runtime_comparison.png'.\n")

cat("\n===============================================================================\n")
cat(" SELURUH TAHAPAN SIMULASI POPULASI FINIS SELESAI DENGAN SUKSES!\n")
cat("===============================================================================\n")
