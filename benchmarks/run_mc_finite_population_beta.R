#!/usr/bin/env Rscript
# ==============================================================================
# Monte Carlo Finite Population Simulation Study (R = 10 Replications)
# Likelihood: Beta (Unit interval response y in (0, 1))
# Dimensions: D = 50 Domains, T = 5 Time Periods (N = 250 Area-Time cells)
# Covariates: 5 Covariates (x1, x2, x3, x4, x5)
# Models Evaluated:
#   1. Non-Spatial Beta (Direct vs INLA vs NumPyro)
#   2. Spatial Beta (INLA vs NumPyro)
#   3. Spatio-Temporal Beta (INLA vs NumPyro)
# ==============================================================================

suppressPackageStartupMessages({
  library(fastsae)
  library(fastsaegpu)
  reticulate::use_virtualenv("/Users/ridsonap/.virtualenvs/r-numpyro-gpu", required = TRUE)
})

cat("==============================================================================\n")
cat(" STUDI SIMULASI MONTE CARLO POPULASI FINIS (R = 10 REPLIKASI)\n")
cat(" Distribusi: BETA | D = 50 Wilayah, T = 5 Waktu (Total N = 250) | 5 Kovariat\n")
cat("==============================================================================\n\n")

set.seed(2026L)
R_reps <- 10L
D <- 50L
T_periods <- 5L
N_cells <- D * T_periods

domain_ids <- paste0("domain_", sprintf("%02d", 1:D))
time_points <- 1:T_periods

# 1. Bangun Matriks Ketetanggaan Spasial W (50 x 50)
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

# 2. Bangun Efek Spasio-Temporal Laten Sebenarnya
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

st_grid <- expand.grid(time = time_points, domain = domain_ids, stringsAsFactors = FALSE)
st_grid <- st_grid[order(st_grid$domain, st_grid$time), ]
rownames(st_grid) <- NULL
st_matrix <- matrix(rnorm(N_cells, sd = 0.15), nrow = D, ncol = T_periods)

# 3. BANGKITKAN POPULASI FINIS TETAP (N_pop ~ 75.000 Unit Individu)
cat(">>> [Langkah 1/3] Membangkitkan Populasi Finis Tetap (~75.000 Unit Individu)...\n")
pop_list <- vector("list", N_cells)
true_P_dt <- numeric(N_cells)
true_N_dt <- integer(N_cells)

beta_true <- c(beta0 = -0.40, beta1 = 0.25, beta2 = -0.30, beta3 = 0.15, beta4 = -0.20, beta5 = 0.35)

cell_idx <- 1L
for (d in 1:D) {
  dom_id <- domain_ids[d]
  for (t in 1:T_periods) {
    N_units <- sample(250:350, 1) # ~300 unit per domain-waktu
    true_N_dt[cell_idx] <- N_units
    
    # Kovariat unit
    x1 <- rnorm(N_units, mean = 1.5, sd = 0.5)
    x2 <- rbinom(N_units, size = 1, prob = 0.4)
    x3 <- rnorm(N_units, mean = 0.0, sd = 1.0)
    x4 <- runif(N_units, min = 0.5, max = 2.5)
    x5 <- rbinom(N_units, size = 1, prob = 0.6)
    
    eta_unit <- beta_true["beta0"] +
      beta_true["beta1"] * x1 +
      beta_true["beta2"] * x2 +
      beta_true["beta3"] * x3 +
      beta_true["beta4"] * x4 +
      beta_true["beta5"] * x5 +
      u_spatial_true[dom_id] +
      v_temporal_true[t] +
      st_matrix[d, t]
    
    p_unit <- plogis(eta_unit)
    # Nilai riil unit interval (0, 1)
    phi_ind <- 25.0
    y_unit <- rbeta(N_units, pmax(p_unit * phi_ind, 1e-4), pmax((1 - p_unit) * phi_ind, 1e-4))
    y_unit <- pmin(pmax(y_unit, 1e-5), 1 - 1e-5)
    
    true_P_dt[cell_idx] <- mean(y_unit) # True Finite Population Parameter
    
    pop_list[[cell_idx]] <- data.frame(
      cell_id = cell_idx,
      domain = dom_id,
      time = t,
      y = y_unit,
      x1 = x1,
      x2 = x2,
      x3 = x3,
      x4 = x4,
      x5 = x5,
      stringsAsFactors = FALSE
    )
    cell_idx <- cell_idx + 1L
  }
}
pop_df <- do.call(rbind, pop_list)
cat(sprintf("    Total Populasi Finis: %s unit individu di %d sel domain-waktu.\n",
            format(nrow(pop_df), big.mark = "."), N_cells))
cat(sprintf("    Rata-rata True P_dt Populasi: %.4f (Min: %.4f, Max: %.4f)\n\n",
            mean(true_P_dt), min(true_P_dt), max(true_P_dt)))

# 4. LOOPING REPLIKASI PENARIKAN SAMPEL MONTE CARLO (R = 10)
cat(sprintf(">>> [Langkah 2/3] Memulai Loop Penarikan Sampel Berulang (R = %d Replikasi)...\n", R_reps))

# Matriks penyimpanan estimasi untuk tiap replikasi (N_cells x R)
est_direct   <- matrix(NA_real_, N_cells, R_reps)
est_beta_inla   <- matrix(NA_real_, N_cells, R_reps)
est_beta_numpyro <- matrix(NA_real_, N_cells, R_reps)
est_sp_inla     <- matrix(NA_real_, N_cells, R_reps)
est_sp_numpyro   <- matrix(NA_real_, N_cells, R_reps)
est_st_inla     <- matrix(NA_real_, N_cells, R_reps)
est_st_numpyro   <- matrix(NA_real_, N_cells, R_reps)

# Coverage tracking (0 atau 1)
cp_direct   <- matrix(0L, N_cells, R_reps)
cp_beta_inla   <- matrix(0L, N_cells, R_reps)
cp_beta_numpyro <- matrix(0L, N_cells, R_reps)
cp_sp_inla     <- matrix(0L, N_cells, R_reps)
cp_sp_numpyro   <- matrix(0L, N_cells, R_reps)
cp_st_inla     <- matrix(0L, N_cells, R_reps)
cp_st_numpyro   <- matrix(0L, N_cells, R_reps)

# Waktu komputasi total per model
time_totals <- c(
  "Beta INLA" = 0, "Beta NumPyro" = 0,
  "Spatial Beta INLA" = 0, "Spatial Beta NumPyro" = 0,
  "Spatio-Temporal Beta INLA" = 0, "Spatio-Temporal Beta NumPyro" = 0
)

get_rss_mb <- function() {
  pid <- Sys.getpid()
  res <- tryCatch(system2("ps", c("-o", "rss=", "-p", pid), stdout = TRUE), error = function(e) "0")
  as.numeric(trimws(res[1])) / 1024
}

mem_peak <- get_rss_mb()

t_sim_start <- Sys.time()

for (r in 1:R_reps) {
  cat(sprintf("--- Replikasi [%2d / %2d] ---\n", r, R_reps))
  
  # A. Penarikan Sampel SRSWOR berukuran kecil n_dt = 20 unit
  sample_rows <- unlist(lapply(split(seq_len(nrow(pop_df)), pop_df$cell_id), function(idx) {
    sample(idx, size = 20L, replace = FALSE)
  }), use.names = FALSE)
  sample_df <- pop_df[sample_rows, ]
  
  # B. Agregasi ke level area (Direct Estimator & Varians Sampling)
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
      cell_id = cid,
      domain = dom,
      time = tm,
      y = y_bar_clip,
      vardir = v_dir,
      x1 = mean(sub$x1),
      x2 = mean(sub$x2),
      x3 = mean(sub$x3),
      x4 = mean(sub$x4),
      x5 = mean(sub$x5),
      stringsAsFactors = FALSE
    )
  })
  area_df <- do.call(rbind, agg_list)
  area_df <- area_df[order(area_df$cell_id), ]
  rownames(area_df) <- NULL
  
  # Direct estimator stats
  est_direct[, r] <- area_df$y
  se_dir <- sqrt(area_df$vardir)
  cp_direct[, r] <- as.integer(true_P_dt >= (area_df$y - 1.96 * se_dir) & true_P_dt <= (area_df$y + 1.96 * se_dir))
  
  # --- Model 1: Non-Spatial Beta ---
  t0 <- Sys.time()
  m1_inla <- fastsae::hb_area(y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain",
                              vardir = "vardir", family = "beta", print_result = FALSE)
  time_totals["Beta INLA"] <- time_totals["Beta INLA"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_beta_inla[, r] <- m1_inla$df_hb$hb
  cp_beta_inla[, r] <- as.integer(true_P_dt >= m1_inla$df_hb$ci_lower & true_P_dt <= m1_inla$df_hb$ci_upper)
  
  t0 <- Sys.time()
  m1_gpu <- fastsaegpu::hb_area(y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain",
                               vardir = "vardir", family = "beta", warmup = 150L, samples = 250L,
                               chains = 1L, device = "auto", print_result = FALSE)
  time_totals["Beta NumPyro"] <- time_totals["Beta NumPyro"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_beta_numpyro[, r] <- m1_gpu$df_hb$hb
  cp_beta_numpyro[, r] <- as.integer(true_P_dt >= m1_gpu$df_hb$ci_lower & true_P_dt <= m1_gpu$df_hb$ci_upper)
  
  # --- Model 2: Spatial Beta ---
  t0 <- Sys.time()
  m2_inla <- fastsae::hb_area(y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain",
                              vardir = "vardir", family = "beta", spatial = "besag", W = W, print_result = FALSE)
  time_totals["Spatial Beta INLA"] <- time_totals["Spatial Beta INLA"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_sp_inla[, r] <- m2_inla$df_hb$hb
  cp_sp_inla[, r] <- as.integer(true_P_dt >= m2_inla$df_hb$ci_lower & true_P_dt <= m2_inla$df_hb$ci_upper)
  
  t0 <- Sys.time()
  m2_gpu <- fastsaegpu::hb_area(y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain",
                               vardir = "vardir", family = "beta", spatial = "besag", W = W,
                               warmup = 150L, samples = 250L, chains = 1L, device = "auto", print_result = FALSE)
  time_totals["Spatial Beta NumPyro"] <- time_totals["Spatial Beta NumPyro"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_sp_numpyro[, r] <- m2_gpu$df_hb$hb
  cp_sp_numpyro[, r] <- as.integer(true_P_dt >= m2_gpu$df_hb$ci_lower & true_P_dt <= m2_gpu$df_hb$ci_upper)
  
  # --- Model 3: Spatio-Temporal Beta ---
  t0 <- Sys.time()
  m3_inla <- fastsae::hb_area(y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain", time = "time",
                              vardir = "vardir", family = "beta", spatial = "besag", temporal = "ar1",
                              st_interaction = "separable", W = W, print_result = FALSE)
  time_totals["Spatio-Temporal Beta INLA"] <- time_totals["Spatio-Temporal Beta INLA"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_st_inla[, r] <- m3_inla$df_hb$hb
  cp_st_inla[, r] <- as.integer(true_P_dt >= m3_inla$df_hb$ci_lower & true_P_dt <= m3_inla$df_hb$ci_upper)
  
  t0 <- Sys.time()
  m3_gpu <- fastsaegpu::hb_area(y ~ x1 + x2 + x3 + x4 + x5, data = area_df, domain = "domain", time = "time",
                               vardir = "vardir", family = "beta", spatial = "besag", temporal = "ar1",
                               st_interaction = "separable", W = W, warmup = 150L, samples = 250L,
                               chains = 1L, device = "auto", print_result = FALSE)
  time_totals["Spatio-Temporal Beta NumPyro"] <- time_totals["Spatio-Temporal Beta NumPyro"] + as.numeric(difftime(Sys.time(), t0, units = "secs"))
  est_st_numpyro[, r] <- m3_gpu$df_hb$hb
  cp_st_numpyro[, r] <- as.integer(true_P_dt >= m3_gpu$df_hb$ci_lower & true_P_dt <= m3_gpu$df_hb$ci_upper)
  
  current_mem <- get_rss_mb()
  mem_peak <- max(mem_peak, current_mem)
  cat(sprintf("    Rep %d selesai. Memory RSS: %.1f MB\n", r, current_mem))
}

t_sim_total <- as.numeric(difftime(Sys.time(), t_sim_start, units = "secs"))
cat(sprintf("\n>>> Simulasi %d replikasi selesai dalam %.2f detik (%.2f menit) <<<\n\n",
            R_reps, t_sim_total, t_sim_total / 60))

# 5. PERHITUNGAN METRIK KINERJA EMPIRIS
cat("==============================================================================\n")
cat(" HASIL EVALUASI STUDI SIMULASI POPULASI FINIS (R = 10 REPLIKASI)\n")
cat("==============================================================================\n\n")

calc_metrics <- function(est_matrix, cp_matrix, true_vals) {
  # est_matrix: N_cells x R
  # Relative Bias (%) per cell: mean((est - true) / true) * 100
  rb_cell <- rowMeans((est_matrix - true_vals) / true_vals) * 100
  arb <- mean(abs(rb_cell))
  
  # Relative RMSE (%) per cell: sqrt(mean((est - true)^2)) / true * 100
  rrmse_cell <- sqrt(rowMeans((est_matrix - true_vals)^2)) / true_vals * 100
  mrrmse <- mean(rrmse_cell)
  
  # Coverage Probability 95%
  cp_mean <- mean(rowMeans(cp_matrix)) * 100
  
  c(ARB = arb, MRRMSE = mrrmse, CP95 = cp_mean)
}

models_list <- list(
  "Direct Estimator" = list(est = est_direct, cp = cp_direct),
  "Beta (INLA)" = list(est = est_beta_inla, cp = cp_beta_inla),
  "Beta (NumPyro)" = list(est = est_beta_numpyro, cp = cp_beta_numpyro),
  "Spatial Beta (INLA)" = list(est = est_sp_inla, cp = cp_sp_inla),
  "Spatial Beta (NumPyro)" = list(est = est_sp_numpyro, cp = cp_sp_numpyro),
  "Spatio-Temporal Beta (INLA)" = list(est = est_st_inla, cp = cp_st_inla),
  "Spatio-Temporal Beta (NumPyro)" = list(est = est_st_numpyro, cp = cp_st_numpyro)
)

res_rows <- lapply(names(models_list), function(mname) {
  met <- calc_metrics(models_list[[mname]]$est, models_list[[mname]]$cp, true_P_dt)
  data.frame(
    Model = mname,
    ARB_Persen = round(met["ARB"], 3),
    MRRMSE_Persen = round(met["MRRMSE"], 3),
    CP95_Persen = round(met["CP95"], 2),
    stringsAsFactors = FALSE
  )
})
res_table <- do.call(rbind, res_rows)

cat("--- 1. METRIK AKURASI EMPIRIS (Berdasarkan Ground Truth Populasi Finis) ---\n")
print(res_table, row.names = FALSE)
cat("\n")

cat("--- 2. KOMPARASI WAKTU KOMPUTASI ATAS 10 REPLIKASI ---\n")
perf_df <- data.frame(
  Model_Backend = names(time_totals),
  Total_Waktu_Detik = round(as.numeric(time_totals), 2),
  Rata2_Detik_Per_Rep = round(as.numeric(time_totals) / R_reps, 2)
)
print(perf_df, row.names = FALSE)
cat(sprintf("\nPeak Memory RSS Selama Simulasi: %.2f MB\n", mem_peak))
cat("==============================================================================\n")
