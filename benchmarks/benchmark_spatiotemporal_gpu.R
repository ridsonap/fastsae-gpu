#!/usr/bin/env Rscript
# Benchmark Script: Spatio-Temporal SAE (n = 500) using NumPyro GPU
# D = 50 areas, T = 10 time periods, total N = 500

suppressPackageStartupMessages({
  if (requireNamespace("fastsaegpu", quietly = TRUE)) {
    library(fastsaegpu)
  } else {
    # If running from source directory
    devtools::load_all(".")
  }
})

cat("=================================================================\n")
cat(" Benchmark Spatio-Temporal SAE with NumPyro GPU (n = 500)\n")
cat("=================================================================\n\n")

# 1. Simulate Spatio-Temporal Data (D = 50, T = 10 -> N = 500)
set.seed(2026)
D <- 50L
T_periods <- 10L
N <- D * T_periods

cat(sprintf("Simulating data: %d domains across %d time periods (Total N = %d)...\n", D, T_periods, N))

domains <- paste0("domain_", sprintf("%02d", 1:D))
time_points <- 1:T_periods

df <- expand.grid(time = time_points, domain = domains, stringsAsFactors = FALSE)
# Reorder by domain then time
df <- df[order(df$domain, df$time), ]
rownames(df) <- NULL

# Create synthetic spatial adjacency matrix (nearest-neighbor lattice)
W <- matrix(0, D, D)
rownames(W) <- colnames(W) <- domains
for (i in 1:D) {
  # connect to adjacent neighbors
  if (i > 1) W[i, i - 1] <- 1
  if (i < D) W[i, i + 1] <- 1
  if (i + 5 <= D) {
    W[i, i + 5] <- 1
    W[i + 5, i] <- 1
  }
}

# Covariates and true parameters
x1 <- rnorm(N, mean = 2, sd = 1)
x2 <- rbinom(N, size = 1, prob = 0.4)
beta_true <- c(3.0, 1.5, -0.8)
vardir <- runif(N, min = 0.05, max = 0.25)

# Simulate latent random effects
u_spatial_true <- rnorm(D, sd = 0.5)
v_temp_true <- filter(rnorm(T_periods, sd = 0.3), filter = 0.7, method = "recursive")
st_true <- rnorm(N, sd = 0.2)

linpred_true <- beta_true[1] + beta_true[2] * x1 + beta_true[3] * x2 +
  u_spatial_true[as.integer(factor(df$domain, levels = domains))] +
  v_temp_true[df$time] + st_true

y_dir <- linpred_true + rnorm(N, sd = sqrt(vardir))

df$y <- y_dir
df$x1 <- x1
df$x2 <- x2
df$vardir <- vardir

cat("Data simulation completed.\n\n")

# 2. Check if GPU NumPyro is ready
if (!check_numpyro_available()) {
  cat("Notice: NumPyro or JAX is not configured in the active Python environment.\n")
  cat("To set up the GPU environment, run:\n")
  cat("  fastsaegpu::setup_numpyro_env(device = 'auto')\n\n")
  cat("Exiting benchmark demo.\n")
  quit(status = 0)
}

# 3. Fit Spatio-Temporal Model using NumPyro GPU
cat("Fitting Spatio-Temporal Model with NumPyro GPU (NUTS)...\n")
t_start <- Sys.time()

fit_gpu <- hb_area(
  formula = y ~ x1 + x2,
  data = df,
  domain = "domain",
  time = "time",
  vardir = "vardir",
  family = "gaussian",
  spatial = "besag",
  temporal = "ar1",
  st_interaction = "type4",
  W = W,
  warmup = 500L,
  samples = 1000L,
  chains = 2L,
  device = "auto",
  print_result = TRUE
)

t_end <- Sys.time()
elapsed <- as.numeric(difftime(t_end, t_start, units = "secs"))

cat("\n=================================================================\n")
cat(sprintf(" Total Execution Time: %.2f seconds\n", elapsed))
cat(sprintf(" Device Used: %s\n", fit_gpu$device))
cat("=================================================================\n")
