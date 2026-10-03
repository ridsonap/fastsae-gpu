#!/usr/bin/env Rscript
# ==============================================================================
# Comprehensive Benchmark: fastsaegpu vs tipsae vs fastsae
# Models:
#   1. Beta SAE (Non-spatial)
#   2. Spatial Beta SAE (Besag ICAR)
#   3. Spatio-Temporal Beta SAE (Besag ICAR x AR(1) Separable)
# Datasets:
#   - Real-world: Emilia-Romagna Poverty Indicators (emilia_cs, emilia)
# ==============================================================================

suppressPackageStartupMessages({
  library(tipsae)
  library(fastsae)
  library(fastsaegpu)
  library(spdep)
  library(ggplot2)
  library(patchwork)
})

cat("===============================================================================\n")
cat(" BENCHMARK KOMPARASI EMPIRIS & FITUR: fastsaegpu vs tipsae vs fastsae\n")
cat("===============================================================================\n\n")

# Utility function to measure memory (RSS in MB)
get_rss_mb <- function() {
  pid <- Sys.getpid()
  res <- tryCatch(system2("ps", c("-o", "rss=", "-p", pid), stdout = TRUE), error = function(e) "0")
  as.numeric(trimws(res[1])) / 1024
}

# ------------------------------------------------------------------------------
# 1. DATA PREPARATION (Emilia-Romagna Poverty Datasets from CRAN tipsae)
# ------------------------------------------------------------------------------
cat(">>> [1/5] Mempersiapkan data survei Emilia-Romagna (tipsae)...\n")
data("emilia_cs", package = "tipsae")
data("emilia", package = "tipsae")
data("emilia_shp", package = "tipsae")

# Build Spatial Adjacency Matrix W from shapefile
nb <- poly2nb(emilia_shp, row.names = emilia_shp$NAME_DISTRICT)
W <- nb2mat(nb, style = "B", zero.policy = TRUE)
rownames(W) <- colnames(W) <- emilia_shp$NAME_DISTRICT

cat(sprintf("    Cross-sectional areas: %d districts\n", nrow(emilia_cs)))
cat(sprintf("    Panel data: %d observations (%d districts x %d time periods)\n\n",
            nrow(emilia), length(unique(emilia$id)), length(unique(emilia$year))))

# Data frames to store benchmark metrics
summary_records <- list()
domain_records <- list()

# ==============================================================================
# MODEL 1: BETA SAE (NON-SPATIAL)
# ==============================================================================
cat("===============================================================================\n")
cat(" MODEL 1: BETA SAE (NON-SPATIAL, N = 38)\n")
cat("===============================================================================\n\n")

# 1A. tipsae (Stan MCMC CPU)
cat(">>> [1/3] Menjalankan tipsae (Stan NUTS, 2 chains, 1000 iter)...\n")
gc(reset = TRUE)
mem_start_tip1 <- get_rss_mb()
t0_tip1 <- Sys.time()

fit_tip1 <- tipsae::fit_sae(
  formula_fixed = hcr ~ x,
  data = emilia_cs,
  domains = "id",
  disp_direct = "vars",
  type_disp = "var",
  domain_size = "n",
  spatial_error = FALSE,
  temporal_error = FALSE,
  chains = 2L,
  iter = 1000L,
  seed = 42L
)

t1_tip1 <- Sys.time()
time_tip1 <- as.numeric(difftime(t1_tip1, t0_tip1, units = "secs"))
mem_end_tip1 <- get_rss_mb()
delta_mem_tip1 <- max(0, mem_end_tip1 - mem_start_tip1)
s_tip1 <- summary(fit_tip1)

rhats_tip1 <- tryCatch(rstan::summary(fit_tip1$stanfit)$summary[, "Rhat"], error = function(e) NA)
max_rhat_tip1 <- max(rhats_tip1, na.rm = TRUE)
looic_tip1 <- tryCatch(as.numeric(s_tip1$loo$estimates["looic", "Estimate"]), error = function(e) NA_real_)

cat(sprintf("    tipsae selesai: %.2f detik | Max R-hat: %.3f | LOOIC: %.2f\n\n",
            time_tip1, max_rhat_tip1, looic_tip1))

# 1B. fastsae (INLA Laplace CPU)
cat(">>> [2/3] Menjalankan fastsae (INLA Laplace Baseline)...\n")
gc(reset = TRUE)
mem_start_fsae1 <- get_rss_mb()
t0_fsae1 <- Sys.time()

fit_fsae1 <- fastsae::hb_area(
  formula = hcr ~ x,
  data = emilia_cs,
  domain = "id",
  vardir = "vars",
  family = "beta",
  spatial = "none",
  temporal = "none",
  print_result = FALSE
)

t1_fsae1 <- Sys.time()
time_fsae1 <- as.numeric(difftime(t1_fsae1, t0_fsae1, units = "secs"))
mem_end_fsae1 <- get_rss_mb()
delta_mem_fsae1 <- max(0, mem_end_fsae1 - mem_start_fsae1)
dic_fsae1 <- as.numeric(fit_fsae1$goodness["DIC"])
waic_fsae1 <- as.numeric(fit_fsae1$goodness["WAIC"])

cat(sprintf("    fastsae selesai: %.2f detik | DIC: %.2f | WAIC: %.2f\n\n",
            time_fsae1, dic_fsae1, waic_fsae1))

# 1C. fastsaegpu (NumPyro JAX NUTS)
cat(">>> [3/3] Menjalankan fastsaegpu (NumPyro NUTS, 2 chains, 500 samples)...\n")
gc(reset = TRUE)
mem_start_gpu1 <- get_rss_mb()
t0_gpu1 <- Sys.time()

fit_gpu1 <- fastsaegpu::hb_area(
  formula = hcr ~ x,
  data = emilia_cs,
  domain = "id",
  vardir = "vars",
  family = "beta",
  spatial = "none",
  temporal = "none",
  warmup = 500L,
  samples = 500L,
  chains = 2L,
  device = "auto",
  seed = 42L,
  print_result = FALSE
)

t1_gpu1 <- Sys.time()
time_gpu1 <- as.numeric(difftime(t1_gpu1, t0_gpu1, units = "secs"))
mem_end_gpu1 <- get_rss_mb()
delta_mem_gpu1 <- max(0, mem_end_gpu1 - mem_start_gpu1)
dic_gpu1 <- as.numeric(fit_gpu1$goodness["DIC"])
waic_gpu1 <- as.numeric(fit_gpu1$goodness["WAIC"])

cat(sprintf("    fastsaegpu selesai: %.2f detik | DIC: %.2f | WAIC: %.2f\n\n",
            time_gpu1, dic_gpu1, waic_gpu1))

# Merge Model 1 Domain Estimates
est_df_tip1 <- data.frame(
  domain = as.character(s_tip1$model_estimates$Domains),
  est_tipsae = s_tip1$model_estimates$mean,
  sd_tipsae = s_tip1$model_estimates$sd,
  stringsAsFactors = FALSE
)
est_df_fsae1 <- data.frame(
  domain = as.character(fit_fsae1$df_hb$domain),
  direct = fit_fsae1$df_hb$y,
  est_fastsae = fit_fsae1$df_hb$hb,
  sd_fastsae = fit_fsae1$df_hb$sd,
  stringsAsFactors = FALSE
)
est_df_gpu1 <- data.frame(
  domain = as.character(fit_gpu1$df_hb$domain),
  est_fastsaegpu = fit_gpu1$df_hb$hb,
  sd_fastsaegpu = fit_gpu1$df_hb$sd,
  stringsAsFactors = FALSE
)

m1_estimates <- merge(est_df_fsae1, est_df_tip1, by = "domain")
m1_estimates <- merge(m1_estimates, est_df_gpu1, by = "domain")
m1_estimates$model <- "Beta SAE (Non-spatial)"
m1_estimates$time <- NA_integer_
domain_records[[1]] <- m1_estimates

cor_tip_gpu1 <- cor(m1_estimates$est_tipsae, m1_estimates$est_fastsaegpu)
cor_fsae_gpu1 <- cor(m1_estimates$est_fastsae, m1_estimates$est_fastsaegpu)
mae_tip_gpu1 <- mean(abs(m1_estimates$est_tipsae - m1_estimates$est_fastsaegpu))
mae_fsae_gpu1 <- mean(abs(m1_estimates$est_fastsae - m1_estimates$est_fastsaegpu))

cat(sprintf("    Korelasi Prediksi Domain: tipsae vs fastsaegpu = %.5f | fastsae vs fastsaegpu = %.5f\n",
            cor_tip_gpu1, cor_fsae_gpu1))
cat(sprintf("    MAE Selisih Prediksi: tipsae vs fastsaegpu = %.5f | fastsae vs fastsaegpu = %.5f\n\n",
            mae_tip_gpu1, mae_fsae_gpu1))

# Store Model 1 Summaries
summary_records[[1]] <- data.frame(
  Model = "Beta SAE (Non-spatial)",
  Package = "tipsae",
  Backend = "Stan NUTS (C++)",
  Hardware = "CPU (Single-core)",
  Observations = nrow(emilia_cs),
  Time_Sec = round(time_tip1, 3),
  Speedup_vs_tipsae = 1.0,
  Speedup_vs_fastsae = round(time_fsae1 / time_tip1, 2),
  Delta_RAM_MB = round(delta_mem_tip1, 1),
  Total_RAM_MB = round(mem_end_tip1, 1),
  Beta_Intercept = round(s_tip1$fixed_coeff["(Intercept)", "mean"], 4),
  SE_Intercept = round(s_tip1$fixed_coeff["(Intercept)", "sd"], 4),
  Beta_Slope = round(s_tip1$fixed_coeff["x", "mean"], 4),
  SE_Slope = round(s_tip1$fixed_coeff["x", "sd"], 4),
  Fit_Criterion = paste0("LOOIC: ", round(looic_tip1, 1)),
  Cor_with_GPU = round(cor_tip_gpu1, 5),
  MAE_with_GPU = round(mae_tip_gpu1, 5),
  stringsAsFactors = FALSE
)

summary_records[[2]] <- data.frame(
  Model = "Beta SAE (Non-spatial)",
  Package = "fastsae",
  Backend = "INLA Laplace",
  Hardware = "CPU (Multi-thread)",
  Observations = nrow(emilia_cs),
  Time_Sec = round(time_fsae1, 3),
  Speedup_vs_tipsae = round(time_tip1 / time_fsae1, 2),
  Speedup_vs_fastsae = 1.0,
  Delta_RAM_MB = round(delta_mem_fsae1, 1),
  Total_RAM_MB = round(mem_end_fsae1, 1),
  Beta_Intercept = round(fit_fsae1$estcoef["(Intercept)", "beta"], 4),
  SE_Intercept = round(fit_fsae1$estcoef["(Intercept)", "std.error"], 4),
  Beta_Slope = round(fit_fsae1$estcoef["x", "beta"], 4),
  SE_Slope = round(fit_fsae1$estcoef["x", "std.error"], 4),
  Fit_Criterion = paste0("DIC: ", round(dic_fsae1, 1), " / WAIC: ", round(waic_fsae1, 1)),
  Cor_with_GPU = round(cor_fsae_gpu1, 5),
  MAE_with_GPU = round(mae_fsae_gpu1, 5),
  stringsAsFactors = FALSE
)

summary_records[[3]] <- data.frame(
  Model = "Beta SAE (Non-spatial)",
  Package = "fastsaegpu",
  Backend = paste0("NumPyro NUTS (", fit_gpu1$device, ")"),
  Hardware = "GPU / Vectorized JAX",
  Observations = nrow(emilia_cs),
  Time_Sec = round(time_gpu1, 3),
  Speedup_vs_tipsae = round(time_tip1 / time_gpu1, 2),
  Speedup_vs_fastsae = round(time_fsae1 / time_gpu1, 2),
  Delta_RAM_MB = round(delta_mem_gpu1, 1),
  Total_RAM_MB = round(mem_end_gpu1, 1),
  Beta_Intercept = round(fit_gpu1$estcoef["(Intercept)", "beta"], 4),
  SE_Intercept = round(fit_gpu1$estcoef["(Intercept)", "std.error"], 4),
  Beta_Slope = round(fit_gpu1$estcoef["x", "beta"], 4),
  SE_Slope = round(fit_gpu1$estcoef["x", "std.error"], 4),
  Fit_Criterion = paste0("DIC: ", round(dic_gpu1, 1), " / WAIC: ", round(waic_gpu1, 1)),
  Cor_with_GPU = 1.00000,
  MAE_with_GPU = 0.00000,
  stringsAsFactors = FALSE
)

# ==============================================================================
# MODEL 2: SPATIAL BETA SAE (BESAG ICAR)
# ==============================================================================
cat("===============================================================================\n")
cat(" MODEL 2: SPATIAL BETA SAE (BESAG ICAR, N = 38)\n")
cat("===============================================================================\n\n")

# 2A. tipsae (Spatial Besag ICAR via Shapefile)
cat(">>> [1/3] Menjalankan tipsae (Spatial Stan NUTS, 2 chains, 1000 iter)...\n")
gc(reset = TRUE)
mem_start_tip2 <- get_rss_mb()
t0_tip2 <- Sys.time()

fit_tip2 <- tipsae::fit_sae(
  formula_fixed = hcr ~ x,
  data = emilia_cs,
  domains = "id",
  disp_direct = "vars",
  type_disp = "var",
  domain_size = "n",
  spatial_error = TRUE,
  spatial_df = emilia_shp,
  domains_spatial_df = "NAME_DISTRICT",
  temporal_error = FALSE,
  chains = 2L,
  iter = 1000L,
  seed = 42L
)

t1_tip2 <- Sys.time()
time_tip2 <- as.numeric(difftime(t1_tip2, t0_tip2, units = "secs"))
mem_end_tip2 <- get_rss_mb()
delta_mem_tip2 <- max(0, mem_end_tip2 - mem_start_tip2)
s_tip2 <- summary(fit_tip2)

rhats_tip2 <- tryCatch(rstan::summary(fit_tip2$stanfit)$summary[, "Rhat"], error = function(e) NA)
max_rhat_tip2 <- max(rhats_tip2, na.rm = TRUE)
looic_tip2 <- tryCatch(as.numeric(s_tip2$loo$estimates["looic", "Estimate"]), error = function(e) NA_real_)

cat(sprintf("    tipsae spatial selesai: %.2f detik | Max R-hat: %.3f | LOOIC: %.2f\n\n",
            time_tip2, max_rhat_tip2, looic_tip2))

# 2B. fastsae (Spatial Besag ICAR via INLA)
cat(">>> [2/3] Menjalankan fastsae (Spatial Besag ICAR INLA)...\n")
gc(reset = TRUE)
mem_start_fsae2 <- get_rss_mb()
t0_fsae2 <- Sys.time()

fit_fsae2 <- fastsae::hb_area(
  formula = hcr ~ x,
  data = emilia_cs,
  domain = "id",
  vardir = "vars",
  family = "beta",
  spatial = "besag",
  W = W,
  print_result = FALSE
)

t1_fsae2 <- Sys.time()
time_fsae2 <- as.numeric(difftime(t1_fsae2, t0_fsae2, units = "secs"))
mem_end_fsae2 <- get_rss_mb()
delta_mem_fsae2 <- max(0, mem_end_fsae2 - mem_start_fsae2)
dic_fsae2 <- as.numeric(fit_fsae2$goodness["DIC"])
waic_fsae2 <- as.numeric(fit_fsae2$goodness["WAIC"])

cat(sprintf("    fastsae spatial selesai: %.2f detik | DIC: %.2f | WAIC: %.2f\n\n",
            time_fsae2, dic_fsae2, waic_fsae2))

# 2C. fastsaegpu (Spatial Besag ICAR via NumPyro JAX NUTS)
cat(">>> [3/3] Menjalankan fastsaegpu (Spatial Besag ICAR NumPyro NUTS)...\n")
gc(reset = TRUE)
mem_start_gpu2 <- get_rss_mb()
t0_gpu2 <- Sys.time()

fit_gpu2 <- fastsaegpu::hb_area(
  formula = hcr ~ x,
  data = emilia_cs,
  domain = "id",
  vardir = "vars",
  family = "beta",
  spatial = "besag",
  W = W,
  warmup = 500L,
  samples = 500L,
  chains = 2L,
  device = "auto",
  seed = 42L,
  print_result = FALSE
)

t1_gpu2 <- Sys.time()
time_gpu2 <- as.numeric(difftime(t1_gpu2, t0_gpu2, units = "secs"))
mem_end_gpu2 <- get_rss_mb()
delta_mem_gpu2 <- max(0, mem_end_gpu2 - mem_start_gpu2)
dic_gpu2 <- as.numeric(fit_gpu2$goodness["DIC"])
waic_gpu2 <- as.numeric(fit_gpu2$goodness["WAIC"])

cat(sprintf("    fastsaegpu spatial selesai: %.2f detik | DIC: %.2f | WAIC: %.2f\n\n",
            time_gpu2, dic_gpu2, waic_gpu2))

# Merge Model 2 Domain Estimates
est_df_tip2 <- data.frame(
  domain = as.character(s_tip2$model_estimates$Domains),
  est_tipsae = s_tip2$model_estimates$mean,
  sd_tipsae = s_tip2$model_estimates$sd,
  stringsAsFactors = FALSE
)
est_df_fsae2 <- data.frame(
  domain = as.character(fit_fsae2$df_hb$domain),
  direct = fit_fsae2$df_hb$y,
  est_fastsae = fit_fsae2$df_hb$hb,
  sd_fastsae = fit_fsae2$df_hb$sd,
  stringsAsFactors = FALSE
)
est_df_gpu2 <- data.frame(
  domain = as.character(fit_gpu2$df_hb$domain),
  est_fastsaegpu = fit_gpu2$df_hb$hb,
  sd_fastsaegpu = fit_gpu2$df_hb$sd,
  stringsAsFactors = FALSE
)

m2_estimates <- merge(est_df_fsae2, est_df_tip2, by = "domain")
m2_estimates <- merge(m2_estimates, est_df_gpu2, by = "domain")
m2_estimates$model <- "Spatial Beta SAE"
m2_estimates$time <- NA_integer_
domain_records[[2]] <- m2_estimates

cor_tip_gpu2 <- cor(m2_estimates$est_tipsae, m2_estimates$est_fastsaegpu)
cor_fsae_gpu2 <- cor(m2_estimates$est_fastsae, m2_estimates$est_fastsaegpu)
mae_tip_gpu2 <- mean(abs(m2_estimates$est_tipsae - m2_estimates$est_fastsaegpu))
mae_fsae_gpu2 <- mean(abs(m2_estimates$est_fastsae - m2_estimates$est_fastsaegpu))

cat(sprintf("    Korelasi Prediksi Domain Spatial: tipsae vs fastsaegpu = %.5f | fastsae vs fastsaegpu = %.5f\n",
            cor_tip_gpu2, cor_fsae_gpu2))
cat(sprintf("    MAE Selisih Prediksi Spatial: tipsae vs fastsaegpu = %.5f | fastsae vs fastsaegpu = %.5f\n\n",
            mae_tip_gpu2, mae_fsae_gpu2))

# Store Model 2 Summaries
summary_records[[4]] <- data.frame(
  Model = "Spatial Beta SAE",
  Package = "tipsae",
  Backend = "Stan NUTS (C++)",
  Hardware = "CPU (Single-core)",
  Observations = nrow(emilia_cs),
  Time_Sec = round(time_tip2, 3),
  Speedup_vs_tipsae = 1.0,
  Speedup_vs_fastsae = round(time_fsae2 / time_tip2, 2),
  Delta_RAM_MB = round(delta_mem_tip2, 1),
  Total_RAM_MB = round(mem_end_tip2, 1),
  Beta_Intercept = round(s_tip2$fixed_coeff["(Intercept)", "mean"], 4),
  SE_Intercept = round(s_tip2$fixed_coeff["(Intercept)", "sd"], 4),
  Beta_Slope = round(s_tip2$fixed_coeff["x", "mean"], 4),
  SE_Slope = round(s_tip2$fixed_coeff["x", "sd"], 4),
  Fit_Criterion = paste0("LOOIC: ", round(looic_tip2, 1)),
  Cor_with_GPU = round(cor_tip_gpu2, 5),
  MAE_with_GPU = round(mae_tip_gpu2, 5),
  stringsAsFactors = FALSE
)

summary_records[[5]] <- data.frame(
  Model = "Spatial Beta SAE",
  Package = "fastsae",
  Backend = "INLA Laplace",
  Hardware = "CPU (Multi-thread)",
  Observations = nrow(emilia_cs),
  Time_Sec = round(time_fsae2, 3),
  Speedup_vs_tipsae = round(time_tip2 / time_fsae2, 2),
  Speedup_vs_fastsae = 1.0,
  Delta_RAM_MB = round(delta_mem_fsae2, 1),
  Total_RAM_MB = round(mem_end_fsae2, 1),
  Beta_Intercept = round(fit_fsae2$estcoef["(Intercept)", "beta"], 4),
  SE_Intercept = round(fit_fsae2$estcoef["(Intercept)", "std.error"], 4),
  Beta_Slope = round(fit_fsae2$estcoef["x", "beta"], 4),
  SE_Slope = round(fit_fsae2$estcoef["x", "std.error"], 4),
  Fit_Criterion = paste0("DIC: ", round(dic_fsae2, 1), " / WAIC: ", round(waic_fsae2, 1)),
  Cor_with_GPU = round(cor_fsae_gpu2, 5),
  MAE_with_GPU = round(mae_fsae_gpu2, 5),
  stringsAsFactors = FALSE
)

summary_records[[6]] <- data.frame(
  Model = "Spatial Beta SAE",
  Package = "fastsaegpu",
  Backend = paste0("NumPyro NUTS (", fit_gpu2$device, ")"),
  Hardware = "GPU / Vectorized JAX",
  Observations = nrow(emilia_cs),
  Time_Sec = round(time_gpu2, 3),
  Speedup_vs_tipsae = round(time_tip2 / time_gpu2, 2),
  Speedup_vs_fastsae = round(time_fsae2 / time_gpu2, 2),
  Delta_RAM_MB = round(delta_mem_gpu2, 1),
  Total_RAM_MB = round(mem_end_gpu2, 1),
  Beta_Intercept = round(fit_gpu2$estcoef["(Intercept)", "beta"], 4),
  SE_Intercept = round(fit_gpu2$estcoef["(Intercept)", "std.error"], 4),
  Beta_Slope = round(fit_gpu2$estcoef["x", "beta"], 4),
  SE_Slope = round(fit_gpu2$estcoef["x", "std.error"], 4),
  Fit_Criterion = paste0("DIC: ", round(dic_gpu2, 1), " / WAIC: ", round(waic_gpu2, 1)),
  Cor_with_GPU = 1.00000,
  MAE_with_GPU = 0.00000,
  stringsAsFactors = FALSE
)

# ==============================================================================
# MODEL 3: SPATIO-TEMPORAL BETA SAE (N = 190)
# ==============================================================================
cat("===============================================================================\n")
cat(" MODEL 3: SPATIO-TEMPORAL BETA SAE (BESAG ICAR x AR1, N = 190)\n")
cat("===============================================================================\n\n")

# 3A. tipsae (Spatial + Temporal Stan NUTS, 2 chains, 500 iter)
cat(">>> [1/3] Menjalankan tipsae (Spatio-Temporal Stan NUTS, 2 chains, 500 iter)...\n")
gc(reset = TRUE)
mem_start_tip3 <- get_rss_mb()
t0_tip3 <- Sys.time()

fit_tip3 <- tipsae::fit_sae(
  formula_fixed = hcr ~ x,
  data = emilia,
  domains = "id",
  disp_direct = "vars",
  type_disp = "var",
  domain_size = "n",
  spatial_error = TRUE,
  spatial_df = emilia_shp,
  domains_spatial_df = "NAME_DISTRICT",
  temporal_error = TRUE,
  temporal_variable = "year",
  chains = 2L,
  iter = 500L,
  seed = 42L
)

t1_tip3 <- Sys.time()
time_tip3 <- as.numeric(difftime(t1_tip3, t0_tip3, units = "secs"))
mem_end_tip3 <- get_rss_mb()
delta_mem_tip3 <- max(0, mem_end_tip3 - mem_start_tip3)
s_tip3 <- summary(fit_tip3)

rhats_tip3 <- tryCatch(rstan::summary(fit_tip3$stanfit)$summary[, "Rhat"], error = function(e) NA)
max_rhat_tip3 <- max(rhats_tip3, na.rm = TRUE)
looic_tip3 <- tryCatch(as.numeric(s_tip3$loo$estimates["looic", "Estimate"]), error = function(e) NA_real_)

cat(sprintf("    tipsae spatio-temporal selesai: %.2f detik | Max R-hat: %.3f | LOOIC: %.2f\n\n",
            time_tip3, max_rhat_tip3, looic_tip3))

# 3B. fastsae (Spatio-Temporal Besag ICAR x AR1 INLA)
cat(">>> [2/3] Menjalankan fastsae (Spatio-Temporal Besag ICAR x AR1 INLA)...\n")
gc(reset = TRUE)
mem_start_fsae3 <- get_rss_mb()
t0_fsae3 <- Sys.time()

fit_fsae3 <- fastsae::hb_area(
  formula = hcr ~ x,
  data = emilia,
  domain = "id",
  time = "year",
  vardir = "vars",
  family = "beta",
  spatial = "besag",
  temporal = "ar1",
  st_interaction = "separable",
  W = W,
  print_result = FALSE
)

t1_fsae3 <- Sys.time()
time_fsae3 <- as.numeric(difftime(t1_fsae3, t0_fsae3, units = "secs"))
mem_end_fsae3 <- get_rss_mb()
delta_mem_fsae3 <- max(0, mem_end_fsae3 - mem_start_fsae3)
dic_fsae3 <- as.numeric(fit_fsae3$goodness["DIC"])
waic_fsae3 <- as.numeric(fit_fsae3$goodness["WAIC"])

cat(sprintf("    fastsae spatio-temporal selesai: %.2f detik | DIC: %.2f | WAIC: %.2f\n\n",
            time_fsae3, dic_fsae3, waic_fsae3))

# 3C. fastsaegpu (Spatio-Temporal Besag ICAR x AR1 NumPyro JAX NUTS)
cat(">>> [3/3] Menjalankan fastsaegpu (Spatio-Temporal Besag ICAR x AR1 NumPyro NUTS)...\n")
gc(reset = TRUE)
mem_start_gpu3 <- get_rss_mb()
t0_gpu3 <- Sys.time()

fit_gpu3 <- fastsaegpu::hb_area(
  formula = hcr ~ x,
  data = emilia,
  domain = "id",
  time = "year",
  vardir = "vars",
  family = "beta",
  spatial = "besag",
  temporal = "ar1",
  st_interaction = "separable",
  W = W,
  warmup = 300L,
  samples = 300L,
  chains = 2L,
  device = "auto",
  seed = 42L,
  print_result = FALSE
)

t1_gpu3 <- Sys.time()
time_gpu3 <- as.numeric(difftime(t1_gpu3, t0_gpu3, units = "secs"))
mem_end_gpu3 <- get_rss_mb()
delta_mem_gpu3 <- max(0, mem_end_gpu3 - mem_start_gpu3)
dic_gpu3 <- as.numeric(fit_gpu3$goodness["DIC"])
waic_gpu3 <- as.numeric(fit_gpu3$goodness["WAIC"])

cat(sprintf("    fastsaegpu spatio-temporal selesai: %.2f detik | DIC: %.2f | WAIC: %.2f\n\n",
            time_gpu3, dic_gpu3, waic_gpu3))

# Merge Model 3 Domain Estimates
est_df_tip3 <- data.frame(
  domain = as.character(s_tip3$model_estimates$Domains),
  time = as.integer(as.character(s_tip3$model_estimates$Times)),
  est_tipsae = s_tip3$model_estimates$mean,
  sd_tipsae = s_tip3$model_estimates$sd,
  stringsAsFactors = FALSE
)
est_df_fsae3 <- data.frame(
  domain = as.character(fit_fsae3$df_hb$domain),
  time = as.integer(fit_fsae3$df_hb$time),
  direct = fit_fsae3$df_hb$y,
  est_fastsae = fit_fsae3$df_hb$hb,
  sd_fastsae = fit_fsae3$df_hb$sd,
  stringsAsFactors = FALSE
)
est_df_gpu3 <- data.frame(
  domain = as.character(fit_gpu3$df_hb$domain),
  time = as.integer(fit_gpu3$df_hb$time),
  est_fastsaegpu = fit_gpu3$df_hb$hb,
  sd_fastsaegpu = fit_gpu3$df_hb$sd,
  stringsAsFactors = FALSE
)

m3_estimates <- merge(est_df_fsae3, est_df_tip3, by = c("domain", "time"))
m3_estimates <- merge(m3_estimates, est_df_gpu3, by = c("domain", "time"))
m3_estimates$model <- "Spatio-Temporal Beta SAE"
domain_records[[3]] <- m3_estimates

cor_tip_gpu3 <- cor(m3_estimates$est_tipsae, m3_estimates$est_fastsaegpu)
cor_fsae_gpu3 <- cor(m3_estimates$est_fastsae, m3_estimates$est_fastsaegpu)
mae_tip_gpu3 <- mean(abs(m3_estimates$est_tipsae - m3_estimates$est_fastsaegpu))
mae_fsae_gpu3 <- mean(abs(m3_estimates$est_fastsae - m3_estimates$est_fastsaegpu))

cat(sprintf("    Korelasi Prediksi Domain Spatio-Temporal: tipsae vs fastsaegpu = %.5f | fastsae vs fastsaegpu = %.5f\n",
            cor_tip_gpu3, cor_fsae_gpu3))
cat(sprintf("    MAE Selisih Prediksi Spatio-Temporal: tipsae vs fastsaegpu = %.5f | fastsae vs fastsaegpu = %.5f\n\n",
            mae_tip_gpu3, mae_fsae_gpu3))

# Store Model 3 Summaries
summary_records[[7]] <- data.frame(
  Model = "Spatio-Temporal Beta SAE",
  Package = "tipsae",
  Backend = "Stan NUTS (C++)",
  Hardware = "CPU (Single-core)",
  Observations = nrow(emilia),
  Time_Sec = round(time_tip3, 3),
  Speedup_vs_tipsae = 1.0,
  Speedup_vs_fastsae = round(time_fsae3 / time_tip3, 2),
  Delta_RAM_MB = round(delta_mem_tip3, 1),
  Total_RAM_MB = round(mem_end_tip3, 1),
  Beta_Intercept = round(s_tip3$fixed_coeff["(Intercept)", "mean"], 4),
  SE_Intercept = round(s_tip3$fixed_coeff["(Intercept)", "sd"], 4),
  Beta_Slope = round(s_tip3$fixed_coeff["x", "mean"], 4),
  SE_Slope = round(s_tip3$fixed_coeff["x", "sd"], 4),
  Fit_Criterion = paste0("LOOIC: ", round(looic_tip3, 1)),
  Cor_with_GPU = round(cor_tip_gpu3, 5),
  MAE_with_GPU = round(mae_tip_gpu3, 5),
  stringsAsFactors = FALSE
)

summary_records[[8]] <- data.frame(
  Model = "Spatio-Temporal Beta SAE",
  Package = "fastsae",
  Backend = "INLA Laplace",
  Hardware = "CPU (Multi-thread)",
  Observations = nrow(emilia),
  Time_Sec = round(time_fsae3, 3),
  Speedup_vs_tipsae = round(time_tip3 / time_fsae3, 2),
  Speedup_vs_fastsae = 1.0,
  Delta_RAM_MB = round(delta_mem_fsae3, 1),
  Total_RAM_MB = round(mem_end_fsae3, 1),
  Beta_Intercept = round(fit_fsae3$estcoef["(Intercept)", "beta"], 4),
  SE_Intercept = round(fit_fsae3$estcoef["(Intercept)", "std.error"], 4),
  Beta_Slope = round(fit_fsae3$estcoef["x", "beta"], 4),
  SE_Slope = round(fit_fsae3$estcoef["x", "std.error"], 4),
  Fit_Criterion = paste0("DIC: ", round(dic_fsae3, 1), " / WAIC: ", round(waic_fsae3, 1)),
  Cor_with_GPU = round(cor_fsae_gpu3, 5),
  MAE_with_GPU = round(mae_fsae_gpu3, 5),
  stringsAsFactors = FALSE
)

summary_records[[9]] <- data.frame(
  Model = "Spatio-Temporal Beta SAE",
  Package = "fastsaegpu",
  Backend = paste0("NumPyro NUTS (", fit_gpu3$device, ")"),
  Hardware = "GPU / Vectorized JAX",
  Observations = nrow(emilia),
  Time_Sec = round(time_gpu3, 3),
  Speedup_vs_tipsae = round(time_tip3 / time_gpu3, 2),
  Speedup_vs_fastsae = round(time_fsae3 / time_gpu3, 2),
  Delta_RAM_MB = round(delta_mem_gpu3, 1),
  Total_RAM_MB = round(mem_end_gpu3, 1),
  Beta_Intercept = round(fit_gpu3$estcoef["(Intercept)", "beta"], 4),
  SE_Intercept = round(fit_gpu3$estcoef["(Intercept)", "std.error"], 4),
  Beta_Slope = round(fit_gpu3$estcoef["x", "beta"], 4),
  SE_Slope = round(fit_gpu3$estcoef["x", "std.error"], 4),
  Fit_Criterion = paste0("DIC: ", round(dic_gpu3, 1), " / WAIC: ", round(waic_gpu3, 1)),
  Cor_with_GPU = 1.00000,
  MAE_with_GPU = 0.00000,
  stringsAsFactors = FALSE
)

# ------------------------------------------------------------------------------
# SAVE OUTPUTS (CSV & PLOTS)
# ------------------------------------------------------------------------------
cat(">>> [4/5] Menyimpan dataset benchmark terstruktur ke benchmarks/...\n")

df_summary <- do.call(rbind, summary_records)
write.csv(df_summary, "benchmarks/benchmark_summary.csv", row.names = FALSE)
cat("    Tersimpan: benchmarks/benchmark_summary.csv\n")

df_domain_all <- do.call(rbind, domain_records)
write.csv(df_domain_all, "benchmarks/benchmark_domain_estimates.csv", row.names = FALSE)
cat("    Tersimpan: benchmarks/benchmark_domain_estimates.csv\n")

# Feature Comparison Matrix Table
feature_records <- data.frame(
  Feature_Dimension = c(
    "Inference Engine & Backend",
    "Hardware Acceleration",
    "Likelihood Families Supported",
    "Spatial Covariance Priors",
    "Temporal Dynamic Priors",
    "Spatio-Temporal Interaction",
    "MCMC Sampling Method",
    "Compilation / Acceleration",
    "Vectorized Parallel Chains",
    "Computational Scaling (N > 500)",
    "Small Area Indicator Output",
    "Direct Sampling Variance Handling",
    "Output S3 Methods"
  ),
  fastsaegpu = c(
    "NumPyro / JAX (Python C-API via Reticulate)",
    "Apple Silicon Metal (MPS), NVIDIA CUDA, Multithread CPU",
    "Gaussian, Binomial, Poisson, Beta, Neg-Binomial, Gamma",
    "Besag (ICAR), BYM, BYM2 (scaled), Leroux CAR",
    "AR(1), RW(1), IID temporal dynamics",
    "Separable, Type I-IV, Domain-Specific (Kronecker Contraction)",
    "Hamiltonian Monte Carlo / No-U-Turn Sampler (NUTS)",
    "XLA Ahead-Of-Time / Just-In-Time GPU Kernel Fusion",
    "Yes (SIMD Vectorized across chains)",
    "High (O(D x T) Kronecker tensor operations)",
    "Point estimate, posterior SE, MSE, CV/RSE%, Credible Intervals",
    "Direct vardir vector parameterization",
    "print, summary, coef, fitted, residuals"
  ),
  tipsae = c(
    "Stan / rstan (C++ via Rcpp)",
    "CPU Only (Single-threaded per chain)",
    "Beta, Flexible Beta, Zero/One-Inflated Beta",
    "Besag York Mollié (BYM) via SpatialPolygonsDataFrame",
    "Random Walk 1 (RW1) domain-specific temporal error",
    "Additive spatial + temporal (No Kronecker interaction)",
    "No-U-Turn Sampler (NUTS)",
    "C++ compilation via gcc/clang",
    "No (Sequential or CPU multi-process fork)",
    "Moderate to Slow (Treedepth saturations on large N)",
    "Point estimate, posterior SD, quantiles (2.5% - 97.5%)",
    "Direct variance or effective sample size (neff)",
    "print, summary, extract, map, export, benchmark"
  ),
  fastsae = c(
    "INLA / Laplace Approximation (C / Fortran binaries)",
    "CPU Multithread (OpenMP / Pthreads)",
    "Gaussian, Binomial, Poisson, Neg-Binomial, Beta, Gamma",
    "Besag, BYM, BYM2, Generic1, SLM",
    "AR(1), RW(1), RW(2), IID",
    "Separable, Domain-Specific, Type I-IV",
    "Deterministic Integrated Nested Laplace Approximation (INLA)",
    "Compiled sparse Cholesky / Pardiso solver",
    "N/A (Deterministic Laplace Integration)",
    "Very High (Fast sparse matrix factorizations)",
    "HB EBLUP, posterior SD, MSE, RSE%, Credible Intervals",
    "Direct vardir vector parameterization",
    "print, summary, coef, fitted, residuals, autoplot, diagnose"
  ),
  Practical_Significance = c(
    "Full Bayesian MCMC with modern tensor backend vs Classic Stan / INLA",
    "Unlocks workstation GPU and unified memory for large Bayesian SAE surveys",
    "fastsaegpu covers diverse survey outcomes beyond proportions",
    "Leroux and BYM2 prevent variance confounding; fastsaegpu avoids matrix inversion",
    "Captures seasonal and multi-year trend persistence in survey panels",
    "Enables modeling localized space-time shocks without O(N^2) memory explosion",
    "Exact posterior distributions for complex non-Gaussian hierarchies",
    "Dramatically reduces per-iteration leapfrog evaluation latency",
    "Runs 2-8 MCMC chains simultaneously on unified GPU memory",
    "Prevents Stan memory overflow and treedepth saturation in large national surveys",
    "Ready for official statistics reporting standards (BPS, Eurostat, US Census)",
    "Handles heterogeneous survey sampling errors directly",
    "Familiar R modeling ergonomics matching stats::lm and lme4"
  ),
  stringsAsFactors = FALSE
)

write.csv(feature_records, "benchmarks/benchmark_feature_comparison.csv", row.names = FALSE)
cat("    Tersimpan: benchmarks/benchmark_feature_comparison.csv\n\n")

# ------------------------------------------------------------------------------
# 5. VISUALIZATION: GENERATE COMPARISON CHARTS
# ------------------------------------------------------------------------------
cat(">>> [5/5] Menghasilkan grafik komparasi performa (benchmarks/*.png)...\n")

# Plot 1: Runtime & Speedup Dual-Panel Chart
p1_data <- df_summary
p1_data$Package <- factor(p1_data$Package, levels = c("tipsae", "fastsaegpu", "fastsae"))
p1_data$Model <- factor(p1_data$Model, levels = c("Beta SAE (Non-spatial)", "Spatial Beta SAE", "Spatio-Temporal Beta SAE"))

pkg_colors <- c("tipsae" = "#E64B35", "fastsaegpu" = "#00A087", "fastsae" = "#3C5488")

p1_time <- ggplot(p1_data, aes(x = Model, y = Time_Sec, fill = Package)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.25) +
  geom_text(aes(label = sprintf("%.1fs", Time_Sec)),
            position = position_dodge(width = 0.8), vjust = -0.4, size = 3.3, fontface = "bold") +
  scale_fill_manual(values = pkg_colors) +
  scale_y_continuous(expand = expansion(mult = c(0, 0.15))) +
  labs(
    title = "(A) Waktu Komputasi Nyata (Detik)",
    subtitle = "Skala linier (lebih rendah lebih cepat)",
    x = NULL,
    y = "Waktu Eksekusi (Detik)",
    fill = "Paket"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray30", size = 9.5),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 12, hjust = 1, face = "bold")
  )

p1_speedup <- ggplot(p1_data, aes(x = Model, y = Speedup_vs_tipsae, fill = Package)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.8), width = 0.7, color = "black", linewidth = 0.25) +
  geom_hline(yintercept = 1.0, linetype = "dashed", color = "#C0392B", linewidth = 0.7) +
  geom_text(aes(label = sprintf("%.2fx", Speedup_vs_tipsae)),
            position = position_dodge(width = 0.8), vjust = -0.35, size = 3.3, fontface = "bold") +
  scale_fill_manual(values = pkg_colors) +
  scale_y_log10(breaks = c(0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0),
                labels = c("0.1x", "0.2x", "0.5x", "1.0x", "2.0x", "5.0x", "10x", "20x", "50x"),
                expand = expansion(mult = c(0.05, 0.2))) +
  labs(
    title = "(B) Rasio Percepatan (Speedup vs Stan tipsae Baseline)",
    subtitle = "Skala log10 (Garis merah putus-putus = Baseline tipsae 1.0x)",
    x = NULL,
    y = "Faktor Speedup (Skala Log)",
    fill = "Paket"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 12),
    plot.subtitle = element_text(color = "gray30", size = 9.5),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    panel.grid.minor = element_blank(),
    panel.grid.major.x = element_blank(),
    axis.text.x = element_text(angle = 12, hjust = 1, face = "bold")
  )

p1_combined <- (p1_time | p1_speedup) +
  plot_layout(guides = "collect") &
  theme(legend.position = "top")

ggsave("benchmarks/benchmark_runtime_comparison.png", plot = p1_combined, width = 11, height = 5.5, dpi = 300)
cat("    Tersimpan: benchmarks/benchmark_runtime_comparison.png\n")

# Plot 2: Correlation & Consistency Scatter Plot for BOTH tipsae and fastsae
df_tip <- data.frame(
  domain = df_domain_all$domain,
  model = factor(df_domain_all$model, levels = c("Beta SAE (Non-spatial)", "Spatial Beta SAE", "Spatio-Temporal Beta SAE")),
  comparator = "vs tipsae (Stan MCMC)",
  est_comp = df_domain_all$est_tipsae,
  est_gpu = df_domain_all$est_fastsaegpu,
  stringsAsFactors = FALSE
)

df_fsae <- data.frame(
  domain = df_domain_all$domain,
  model = factor(df_domain_all$model, levels = c("Beta SAE (Non-spatial)", "Spatial Beta SAE", "Spatio-Temporal Beta SAE")),
  comparator = "vs fastsae (INLA Laplace)",
  est_comp = df_domain_all$est_fastsae,
  est_gpu = df_domain_all$est_fastsaegpu,
  stringsAsFactors = FALSE
)

df_scatter <- rbind(df_tip, df_fsae)
df_scatter$comparator <- factor(df_scatter$comparator, levels = c("vs tipsae (Stan MCMC)", "vs fastsae (INLA Laplace)"))

ann_list <- list()
for (m in levels(df_scatter$model)) {
  for (comp in levels(df_scatter$comparator)) {
    sub_df <- subset(df_scatter, model == m & comparator == comp)
    r_val <- cor(sub_df$est_comp, sub_df$est_gpu)
    mae_val <- mean(abs(sub_df$est_comp - sub_df$est_gpu))
    ann_list[[paste(m, comp)]] <- data.frame(
      model = factor(m, levels = levels(df_scatter$model)),
      comparator = factor(comp, levels = levels(df_scatter$comparator)),
      x_pos = min(sub_df$est_comp) + 0.05 * diff(range(sub_df$est_comp)),
      y_pos = max(sub_df$est_gpu) - 0.05 * diff(range(sub_df$est_gpu)),
      label = sprintf("r = %.4f\nMAE = %.4f", r_val, mae_val)
    )
  }
}
ann_df <- do.call(rbind, ann_list)

p2 <- ggplot(df_scatter, aes(x = est_comp, y = est_gpu, color = comparator)) +
  geom_point(alpha = 0.75, size = 2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "black", linewidth = 0.5) +
  geom_text(data = ann_df, aes(x = x_pos, y = y_pos, label = label),
            inherit.aes = FALSE, hjust = 0, vjust = 1, size = 3.3, fontface = "italic", color = "gray20") +
  facet_grid(comparator ~ model, scales = "free") +
  scale_color_manual(values = c("vs tipsae (Stan MCMC)" = "#E64B35", "vs fastsae (INLA Laplace)" = "#3C5488")) +
  labs(
    title = "Konsistensi Estimasi Domain Area: fastsaegpu vs tipsae dan fastsae",
    subtitle = "Garis putus-putus menunjukkan garis identitas sempurna (y = x). Evaluasi pada 3 model SAE.",
    x = "Estimasi Indikator Area Paket Pembanding (tipsae / fastsae)",
    y = "Estimasi Indikator Area fastsaegpu (NumPyro NUTS)",
    color = "Pembanding"
  ) +
  theme_minimal(base_size = 11) +
  theme(
    plot.title = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(color = "gray30", size = 10),
    legend.position = "top",
    legend.title = element_text(face = "bold", size = 10),
    strip.text = element_text(face = "bold", size = 10.5),
    panel.grid.minor = element_blank()
  )

ggsave("benchmarks/benchmark_estimates_scatter.png", plot = p2, width = 10, height = 6.2, dpi = 300)
cat("    Tersimpan: benchmarks/benchmark_estimates_scatter.png\n")

cat("\n===============================================================================\n")
cat(" BENCHMARK SELESAI DENGAN SUKSES!\n")
cat("===============================================================================\n")
print(df_summary[, c("Model", "Package", "Backend", "Time_Sec", "Speedup_vs_tipsae", "Cor_with_GPU", "MAE_with_GPU")])
cat("\n")
