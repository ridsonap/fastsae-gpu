#!/usr/bin/env Rscript
# ==============================================================================
# Eksperimen Evaluasi Seleksi Kovariat: 2 Kovariat Utama vs 5 Kovariat
# Kasus: Spatio-Temporal Beta SAE (D = 50, T = 5, N = 250)
# Tujuan: Membuktikan secara empiris apakah model parsimonis (2 kovariat sinyal)
#         menghasilkan WAIC, DIC, ARB, MRRMSE, dan CP95 yang lebih baik
#         dibandingkan model overfitted (5 kovariat dengan noise & multikolinearitas).
# ==============================================================================

suppressPackageStartupMessages({
  library(fastsae)
  library(fastsaegpu)
  reticulate::use_virtualenv("/Users/ridsonap/.virtualenvs/r-numpyro-gpu", required = TRUE)
})

cat("==============================================================================\n")
cat(" EKSPERIMEN EMPIRIS: 2 KOVARIAT UTAMA VS 5 KOVARIAT (BETA SPATIO-TEMPORAL)\n")
cat(" D = 50 Wilayah, T = 5 Periode Waktu (N = 250 Domain) | R = 10 Replikasi\n")
cat("==============================================================================\n\n")

set.seed(2026L)
R_reps <- 10L
D <- 50L
T_periods <- 5L
N_cells <- D * T_periods

domain_ids <- paste0("domain_", sprintf("%02d", 1:D))
time_points <- 1:T_periods

# 1. Matriks Ketetanggaan Spasial W (50 x 50)
W <- matrix(0, D, D)
rownames(W) <- colnames(W) <- domain_ids
for (i in 1:D) {
  if (i > 1) W[i, i - 1] <- 1
  if (i < D) W[i, i + 1] <- 1
  if (i + 5 <= D) {
    W[i, i + 5] <- 1
    W[i + 5, i] <- 1
  }
}

# 2. Efek Laten Spasial (ICAR) & Temporal (AR1)
Q_s <- diag(rowSums(W)) - W
eig_s <- eigen(Q_s, symmetric = TRUE)
pos_idx <- which(eig_s$values > 1e-6)
V_s <- eig_s$vectors[, pos_idx]
lambda_s <- eig_s$values[pos_idx]
u_spatial_raw <- as.vector(V_s %*% (rnorm(length(pos_idx)) / sqrt(lambda_s)))
u_spatial_true <- (u_spatial_raw - mean(u_spatial_raw)) / sd(u_spatial_raw) * 0.35
names(u_spatial_true) <- domain_ids

rho_true <- 0.70
v_temporal_true <- as.numeric(filter(rnorm(T_periods, sd = 0.25), filter = rho_true, method = "recursive"))
st_matrix <- matrix(rnorm(N_cells, sd = 0.15), nrow = D, ncol = T_periods)

# 3. Pembangkitan Populasi Finis (~75.000 Unit)
# KUNCI DESAIN: Sinyal populasi HANYA dibentuk oleh x1 dan x2 (True beta3 = beta4 = beta5 = 0)
cat(">>> [1/3] Membangkitkan Populasi Finis Tetap (~75.000 unit)...\n")
cat("    Sinyal Asli (Ground Truth): Hanya x1 dan x2 yang memiliki efek nyata.\n")
cat("    x3 (noise murni), x4 (kolinear dengan x1), x5 (noise biner acak).\n\n")

beta_true_signal <- c(beta0 = -0.50, beta1 = 0.60, beta2 = -0.45)

pop_list <- vector("list", N_cells)
true_P_dt <- numeric(N_cells)
true_N_dt <- integer(N_cells)

cell_idx <- 1L
for (d in 1:D) {
  dom_id <- domain_ids[d]
  for (t in 1:T_periods) {
    N_units <- sample(250:350, 1)
    true_N_dt[cell_idx] <- N_units
    
    # 2 Kovariat Sinyal Asli
    x1 <- rnorm(N_units, mean = 1.5, sd = 0.6)
    x2 <- rbinom(N_units, size = 1, prob = 0.45)
    
    # 3 Kovariat Tambahan (Noise & Multikolinear)
    x3 <- rnorm(N_units, mean = 0, sd = 1.0)                 # Noise independen
    x4 <- 0.85 * x1 + rnorm(N_units, mean = 0, sd = 0.3)     # Multikolinear kuat dengan x1
    x5 <- rbinom(N_units, size = 1, prob = 0.5)              # Noise biner
    
    # Nilai riil HANYA bergantung pada x1 dan x2
    eta_unit <- beta_true_signal["beta0"] +
      beta_true_signal["beta1"] * x1 +
      beta_true_signal["beta2"] * x2 +
      u_spatial_true[dom_id] +
      v_temporal_true[t] +
      st_matrix[d, t]
    
    p_unit <- plogis(eta_unit)
    phi_ind <- 25.0
    y_unit <- rbeta(N_units, pmax(p_unit * phi_ind, 1e-4), pmax((1 - p_unit) * phi_ind, 1e-4))
    y_unit <- pmin(pmax(y_unit, 1e-5), 1 - 1e-5)
    
    true_P_dt[cell_idx] <- mean(y_unit)
    
    pop_list[[cell_idx]] <- data.frame(
      cell_id = cell_idx, domain = dom_id, time = t,
      y = y_unit, x1 = x1, x2 = x2, x3 = x3, x4 = x4, x5 = x5,
      stringsAsFactors = FALSE
    )
    cell_idx <- cell_idx + 1L
  }
}
pop_df <- do.call(rbind, pop_list)

# 4. Looping 10 Replikasi Monte Carlo
cat(sprintf(">>> [2/3] Menjalankan Penarikan Sampel Berulang (R = %d Replikasi)...\n", R_reps))

# Matriks penyimpanan estimasi
est_direct <- matrix(NA_real_, N_cells, R_reps)
est_m2_gpu <- matrix(NA_real_, N_cells, R_reps)
est_m5_gpu <- matrix(NA_real_, N_cells, R_reps)
est_m2_inla <- matrix(NA_real_, N_cells, R_reps)
est_m5_inla <- matrix(NA_real_, N_cells, R_reps)

cp_direct <- matrix(0L, N_cells, R_reps)
cp_m2_gpu <- matrix(0L, N_cells, R_reps)
cp_m5_gpu <- matrix(0L, N_cells, R_reps)
cp_m2_inla <- matrix(0L, N_cells, R_reps)
cp_m5_inla <- matrix(0L, N_cells, R_reps)

waic_m2_list <- numeric(R_reps)
waic_m5_list <- numeric(R_reps)
dic_m2_list  <- numeric(R_reps)
dic_m5_list  <- numeric(R_reps)

time_m2_total <- 0
time_m5_total <- 0

# Simpan standard error beta1 dan beta2 untuk melihat inflasi varians
se_beta1_m2 <- numeric(R_reps)
se_beta1_m5 <- numeric(R_reps)
se_beta2_m2 <- numeric(R_reps)
se_beta2_m5 <- numeric(R_reps)

t_start_all <- Sys.time()

for (r in 1:R_reps) {
  cat(sprintf("--- Replikasi [%2d / %2d] ---\n", r, R_reps))
  
  # Tarik sampel SRSWOR n_dt = 20 unit
  sample_rows <- unlist(lapply(split(seq_len(nrow(pop_df)), pop_df$cell_id), function(idx) {
    sample(idx, size = 20L, replace = FALSE)
  }), use.names = FALSE)
  sample_df <- pop_df[sample_rows, ]
  
  # Agregasi area
  agg_list <- lapply(split(sample_df, sample_df$cell_id), function(sub) {
    cid <- sub$cell_id[1]
    dom <- sub$domain[1]
    tm  <- sub$time[1]
    n_s <- nrow(sub)
    N_p <- true_N_dt[cid]
    
    y_bar <- mean(sub$y)
    y_bar_clip <- pmin(pmax(y_bar, 1e-4), 1 - 1e-4)
    s2 <- var(sub$y)
    fpc <- 1 - (n_s / N_p)
    v_dir <- max(fpc * (s2 / n_s), 1e-5)
    
    data.frame(
      cell_id = cid, domain = dom, time = tm,
      y = y_bar_clip, vardir = v_dir,
      x1 = mean(sub$x1), x2 = mean(sub$x2),
      x3 = mean(sub$x3), x4 = mean(sub$x4), x5 = mean(sub$x5),
      stringsAsFactors = FALSE
    )
  })
  area_df <- do.call(rbind, agg_list)
  area_df <- area_df[order(area_df$cell_id), ]
  rownames(area_df) <- NULL
  
  # Direct estimator
  est_direct[, r] <- area_df$y
  se_dir <- sqrt(area_df$vardir)
  cp_direct[, r] <- as.integer(true_P_dt >= (area_df$y - 1.96 * se_dir) & true_P_dt <= (area_df$y + 1.96 * se_dir))
  
  # --- Model A: 2 Kovariat Utama (y ~ x1 + x2) ---
  t0 <- Sys.time()
  fit_m2_gpu <- fastsaegpu::hb_area(
    y ~ x1 + x2, data = area_df, domain = "domain", time = "time",
    vardir = "vardir", family = "beta", spatial = "besag", temporal = "ar1",
    st_interaction = "separable", W = W, warmup = 150L, samples = 250L,
    chains = 1L, device = "auto", print_result = FALSE
  )
  time_m2_total <- time_m2_total + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_m2_gpu[, r] <- fit_m2_gpu$df_hb$hb
  cp_m2_gpu[, r] <- as.integer(true_P_dt >= fit_m2_gpu$df_hb$ci_lower & true_P_dt <= fit_m2_gpu$df_hb$ci_upper)
  waic_m2_list[r] <- fit_m2_gpu$goodness["WAIC"]
  dic_m2_list[r]  <- fit_m2_gpu$goodness["DIC"]
  se_beta1_m2[r]  <- fit_m2_gpu$estcoef["x1", "std.error"]
  se_beta2_m2[r]  <- fit_m2_gpu$estcoef["x2", "std.error"]
  
  # Model A INLA
  fit_m2_inla <- fastsae::hb_area(
    y ~ x1 + x2, data = area_df, domain = "domain", time = "time",
    vardir = "vardir", family = "beta", spatial = "besag", temporal = "ar1",
    st_interaction = "separable", W = W, print_result = FALSE
  )
  est_m2_inla[, r] <- fit_m2_inla$df_hb$hb
  cp_m2_inla[, r] <- as.integer(true_P_dt >= fit_m2_inla$df_hb$ci_lower & true_P_dt <= fit_m2_inla$df_hb$ci_upper)

  # --- Model B: 5 Kovariat (y ~ x1 + x2 + x3 + x4 + x5) ---
  t0 <- Sys.time()
  fit_m5_gpu <- fastsaegpu::hb_area(
    y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain", time = "time",
    vardir = "vardir", family = "beta", spatial = "besag", temporal = "ar1",
    st_interaction = "separable", W = W, warmup = 150L, samples = 250L,
    chains = 1L, device = "auto", print_result = FALSE
  )
  time_m5_total <- time_m5_total + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_m5_gpu[, r] <- fit_m5_gpu$df_hb$hb
  cp_m5_gpu[, r] <- as.integer(true_P_dt >= fit_m5_gpu$df_hb$ci_lower & true_P_dt <= fit_m5_gpu$df_hb$ci_upper)
  waic_m5_list[r] <- fit_m5_gpu$goodness["WAIC"]
  dic_m5_list[r]  <- fit_m5_gpu$goodness["DIC"]
  se_beta1_m5[r]  <- fit_m5_gpu$estcoef["x1", "std.error"]
  se_beta2_m5[r]  <- fit_m5_gpu$estcoef["x2", "std.error"]
  
  # Model B INLA
  fit_m5_inla <- fastsae::hb_area(
    y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain", time = "time",
    vardir = "vardir", family = "beta", spatial = "besag", temporal = "ar1",
    st_interaction = "separable", W = W, print_result = FALSE
  )
  est_m5_inla[, r] <- fit_m5_inla$df_hb$hb
  cp_m5_inla[, r] <- as.integer(true_P_dt >= fit_m5_inla$df_hb$ci_lower & true_P_dt <= fit_m5_inla$df_hb$ci_upper)
}

t_total_all <- as.numeric(difftime(Sys.time(), t_start_all, units = "secs"))
cat(sprintf("\n>>> Eksperimen selesai dalam %.2f detik (%.2f menit) <<<\n\n", t_total_all, t_total_all / 60))

# 5. Evaluasi Kinerja Empiris
calc_metrics <- function(est_matrix, cp_matrix, true_vals) {
  rb_cell <- rowMeans((est_matrix - true_vals) / true_vals) * 100
  arb <- mean(abs(rb_cell))
  rrmse_cell <- sqrt(rowMeans((est_matrix - true_vals)^2)) / true_vals * 100
  mrrmse <- mean(rrmse_cell)
  cp_mean <- mean(rowMeans(cp_matrix)) * 100
  c(ARB = arb, MRRMSE = mrrmse, CP95 = cp_mean)
}

cat("==============================================================================\n")
cat(" HASIL KOMPARASI: 2 KOVARIAT (PARSIMONIOUS) VS 5 KOVARIAT (OVERFITTED)\n")
cat("==============================================================================\n\n")

met_dir <- calc_metrics(est_direct, cp_direct, true_P_dt)
met_m2_gpu <- calc_metrics(est_m2_gpu, cp_m2_gpu, true_P_dt)
met_m5_gpu <- calc_metrics(est_m5_gpu, cp_m5_gpu, true_P_dt)
met_m2_inla <- calc_metrics(est_m2_inla, cp_m2_inla, true_P_dt)
met_m5_inla <- calc_metrics(est_m5_inla, cp_m5_inla, true_P_dt)

# 1. Tabel Kinerja Akurasi
cat("--- 1. AKURASI ESTIMASI TERHADAP TRUE POPULATION PROPORTION ---\n")
tab_acc <- data.frame(
  Model = c(
    "Direct Estimator",
    "NumPyro (2 Kovariat Utama)",
    "NumPyro (5 Kovariat + Noise)",
    "INLA (2 Kovariat Utama)",
    "INLA (5 Kovariat + Noise)"
  ),
  ARB_Persen = c(met_dir["ARB"], met_m2_gpu["ARB"], met_m5_gpu["ARB"], met_m2_inla["ARB"], met_m5_inla["ARB"]),
  MRRMSE_Persen = c(met_dir["MRRMSE"], met_m2_gpu["MRRMSE"], met_m5_gpu["MRRMSE"], met_m2_inla["MRRMSE"], met_m5_inla["MRRMSE"]),
  CP95_Persen = c(met_dir["CP95"], met_m2_gpu["CP95"], met_m5_gpu["CP95"], met_m2_inla["CP95"], met_m5_inla["CP95"])
)
print(tab_acc, row.names = FALSE)
cat("\n")

# 2. Tabel Kriteria Informasi Bayes (WAIC & DIC)
cat("--- 2. KRITERIA INFORMASI MODEL SELEKSI (Rata-rata atas 10 Replikasi) ---\n")
tab_ic <- data.frame(
  Kriteria = c("WAIC (NumPyro)", "DIC (NumPyro)"),
  Model_2_Kovariat = c(mean(waic_m2_list), mean(dic_m2_list)),
  Model_5_Kovariat = c(mean(waic_m5_list), mean(dic_m5_list)),
  Pemenang = c(
    ifelse(mean(waic_m2_list) < mean(waic_m5_list), "2 Kovariat (Lebih Rendah)", "5 Kovariat"),
    ifelse(mean(dic_m2_list) < mean(dic_m5_list), "2 Kovariat (Lebih Rendah)", "5 Kovariat")
  )
)
print(tab_ic, row.names = FALSE)
cat("\n")

# 3. Inflasi Standard Error Koefisien
cat("--- 3. INFLASI VARIANS KOEFISIEN (Variance Inflation karena Kolinearitas) ---\n")
tab_se <- data.frame(
  Parameter = c("SE(beta1) untuk x1", "SE(beta2) untuk x2"),
  SE_pada_2_Kovariat = c(mean(se_beta1_m2), mean(se_beta2_m2)),
  SE_pada_5_Kovariat = c(mean(se_beta1_m5), mean(se_beta2_m5)),
  Peningkatan_SE_Persen = c(
    (mean(se_beta1_m5) - mean(se_beta1_m2)) / mean(se_beta1_m2) * 100,
    (mean(se_beta2_m5) - mean(se_beta2_m2)) / mean(se_beta2_m2) * 100
  )
)
print(tab_se, row.names = FALSE)
cat("\n")

# 4. Waktu Komputasi
cat("--- 4. WAKTU KOMPUTASI NUMPYRO ---\n")
cat(sprintf("Model 2 Kovariat: Total %.2f detik (Rata-rata %.2f detik/rep)\n", time_m2_total, time_m2_total / R_reps))
cat(sprintf("Model 5 Kovariat: Total %.2f detik (Rata-rata %.2f detik/rep)\n", time_m5_total, time_m5_total / R_reps))
cat("==============================================================================\n")
