suppressPackageStartupMessages({
  library(fastsae)
  library(fastsaegpu)
  reticulate::use_virtualenv("/Users/ridsonap/.virtualenvs/r-numpyro-gpu", required = TRUE)
})

cat("=======================================================================\n")
cat(" BENCHMARK KOMPARASI: fastsae (INLA) vs fastsaegpu (NumPyro)\n")
cat(" Model: Spatio-Temporal Fay-Herriot (D=50, T=10, N=500)\n")
cat("=======================================================================\n\n")

# 1. Generate Realistic Spatio-Temporal Data
set.seed(2026)
D <- 50L
T_periods <- 10L
N <- D * T_periods

domains <- paste0("domain_", sprintf("%02d", 1:D))
time_points <- 1:T_periods

df <- expand.grid(time = time_points, domain = domains, stringsAsFactors = FALSE)
df <- df[order(df[["domain"]], df[["time"]]), ]
rownames(df) <- NULL

# Adjacency matrix W (nearest neighbors lattice)
W <- matrix(0, D, D)
rownames(W) <- colnames(W) <- domains
for (i in 1:D) {
  if (i > 1) W[i, i - 1] <- 1
  if (i < D) W[i, i + 1] <- 1
  if (i + 5 <= D) {
    W[i, i + 5] <- 1
    W[i + 5, i] <- 1
  }
}

# Generate synthetic covariates & true signals
x1 <- rnorm(N, mean = 2, sd = 1)
x2 <- rbinom(N, size = 1, prob = 0.4)
beta_true <- c(3.0, 1.5, -0.8)
vardir <- runif(N, min = 0.05, max = 0.20)

u_spatial_true <- rnorm(D, sd = 0.4)
v_temp_true <- filter(rnorm(T_periods, sd = 0.25), filter = 0.65, method = "recursive")
st_true <- rnorm(N, sd = 0.15)

theta_true <- beta_true[1] + beta_true[2] * x1 + beta_true[3] * x2 +
  u_spatial_true[as.integer(factor(df$domain, levels = domains))] +
  v_temp_true[df$time] + st_true

y <- theta_true + rnorm(N, sd = sqrt(vardir))

df$y <- y
df$x1 <- x1
df$x2 <- x2
df$vardir <- vardir

get_rss_mb <- function() {
  pid <- Sys.getpid()
  res <- tryCatch(system2("ps", c("-o", "rss=", "-p", pid), stdout = TRUE), error = function(e) "0")
  as.numeric(trimws(res[1])) / 1024
}

# ---------------------------------------------------------------------
# BENCHMARK 1: fastsae (INLA)
# ---------------------------------------------------------------------
cat(">>> [1/2] Menjalankan fastsae::hb_area (Backend: INLA)...\n")
gc(reset = TRUE)
mem_start_inla <- get_rss_mb()
t0_inla <- Sys.time()

fit_inla <- fastsae::hb_area(
  formula = y ~ x1 + x2,
  data = df,
  domain = "domain",
  time = "time",
  vardir = "vardir",
  family = "gaussian",
  spatial = "besag",
  temporal = "ar1",
  st_interaction = "separable",
  W = W,
  print_result = FALSE
)

t1_inla <- Sys.time()
time_inla <- as.numeric(difftime(t1_inla, t0_inla, units = "secs"))
mem_end_inla <- get_rss_mb()
delta_mem_inla <- max(0, mem_end_inla - mem_start_inla)
cat(sprintf("    Selesai dalam: %.2f detik | Alokasi Memori: +%.2f MB (Total RSS: %.1f MB)\n\n",
            time_inla, delta_mem_inla, mem_end_inla))

# ---------------------------------------------------------------------
# BENCHMARK 2: fastsaegpu (NumPyro / JAX NUTS)
# ---------------------------------------------------------------------
cat(">>> [2/2] Menjalankan fastsaegpu::hb_area (Backend: NumPyro JAX NUTS)...\n")
gc(reset = TRUE)
mem_start_gpu <- get_rss_mb()
t0_gpu <- Sys.time()

fit_gpu <- fastsaegpu::hb_area(
  formula = y ~ x1 + x2,
  data = df,
  domain = "domain",
  time = "time",
  vardir = "vardir",
  family = "gaussian",
  spatial = "besag",
  temporal = "ar1",
  st_interaction = "separable",
  W = W,
  warmup = 300L,
  samples = 500L,
  chains = 2L,
  device = "auto",
  print_result = FALSE
)

t1_gpu <- Sys.time()
time_gpu <- as.numeric(difftime(t1_gpu, t0_gpu, units = "secs"))
mem_end_gpu <- get_rss_mb()
delta_mem_gpu <- max(0, mem_end_gpu - mem_start_gpu)
cat(sprintf("    Selesai dalam: %.2f detik | Alokasi Memori: +%.2f MB (Total RSS: %.1f MB)\n\n",
            time_gpu, delta_mem_gpu, mem_end_gpu))

# ---------------------------------------------------------------------
# EVALUASI & KOMPARASI HASIL ESTIMASI
# ---------------------------------------------------------------------
cat("=======================================================================\n")
cat(" HASIL KOMPARASI LENGKAP: ESTIMASI, WAKTU & MEMORI\n")
cat("=======================================================================\n\n")

# 1. Tabel Waktu & Memori
cat("--- 1. KOMPUTASI (Waktu & Penggunaan Memori) ---\n")
tab_perf <- data.frame(
  Metode = c("fastsae (INLA)", "fastsaegpu (NumPyro NUTS)"),
  Backend = c("INLA (Laplace approx)", paste0("NumPyro / JAX (", fit_gpu$device, ")")),
  Waktu_Detik = c(round(time_inla, 2), round(time_gpu, 2)),
  Percepatan = c("1.00x (Baseline)", sprintf("%.2fx", time_inla / time_gpu)),
  Delta_Mem_MB = c(round(delta_mem_inla, 2), round(delta_mem_gpu, 2)),
  Total_RSS_MB = c(round(mem_end_inla, 2), round(mem_end_gpu, 2))
)
print(tab_perf, row.names = FALSE)
cat("\n")

# 2. Tabel Estimasi Parameter Regresi
cat("--- 2. ESTIMASI PARAMETER REGRESI (Fixed Effects) ---\n")
coef_inla <- fit_inla$estcoef
coef_gpu <- fit_gpu$estcoef

tab_coef <- data.frame(
  Parameter = c("(Intercept)", "x1", "x2"),
  True_Value = beta_true,
  INLA_Est = round(coef_inla[c("(Intercept)", "x1", "x2"), "beta"], 4),
  INLA_SE = round(coef_inla[c("(Intercept)", "x1", "x2"), "std.error"], 4),
  NumPyro_Est = round(coef_gpu[c("(Intercept)", "x1", "x2"), "beta"], 4),
  NumPyro_SE = round(coef_gpu[c("(Intercept)", "x1", "x2"), "std.error"], 4)
)
print(tab_coef, row.names = FALSE)
cat("\n")

# 3. Akurasi Estimasi Domain (SAE Indicator)
cat("--- 3. AKURASI & KONSISTENSI ESTIMASI DOMAIN (N = 500) ---\n")
hb_inla <- fit_inla$df_hb$hb
hb_gpu <- fit_gpu$df_hb$hb

cor_pred <- cor(hb_inla, hb_gpu)
mae_diff <- mean(abs(hb_inla - hb_gpu))
rmse_inla_true <- sqrt(mean((hb_inla - theta_true)^2))
rmse_gpu_true <- sqrt(mean((hb_gpu - theta_true)^2))

tab_acc <- data.frame(
  Metrik = c(
    "Korelasi Prediksi (INLA vs NumPyro)",
    "Rata-rata Selisih Absolut (MAE INLA vs NumPyro)",
    "RMSE Prediksi INLA ke True Signal",
    "RMSE Prediksi NumPyro ke True Signal"
  ),
  Nilai = c(
    round(cor_pred, 5),
    round(mae_diff, 5),
    round(rmse_inla_true, 5),
    round(rmse_gpu_true, 5)
  )
)
print(tab_acc, row.names = FALSE)
cat("\n=======================================================================\n")
