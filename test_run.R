Sys.setenv(JAX_PLATFORMS = "cpu")
library(reticulate)
use_virtualenv("/Users/ridsonap/.virtualenvs/r-numpyro-gpu", required = TRUE)
library(fastsaegpu)

set.seed(2026)
D <- 50L
T_periods <- 10L
N <- D * T_periods

domains <- paste0("domain_", sprintf("%02d", 1:D))
time_points <- 1:T_periods

df <- expand.grid(time = time_points, domain = domains, stringsAsFactors = FALSE)
df <- df[order(df[["domain"]], df[["time"]]), ]
rownames(df) <- NULL

# Adjacency matrix (lattice structure)
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

x1 <- rnorm(N, mean = 2, sd = 1)
x2 <- rbinom(N, size = 1, prob = 0.4)
vardir <- runif(N, min = 0.05, max = 0.25)

df$y <- 3.0 + 1.5 * x1 - 0.8 * x2 + rnorm(N, sd = sqrt(vardir))
df$x1 <- x1
df$x2 <- x2
df$vardir <- vardir

cat(sprintf("=== Menjalankan hb_area Spatio-Temporal SAE (D=%d, T=%d, N=%d) ===\n", D, T_periods, N))
t0 <- Sys.time()

fit <- hb_area(
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
  warmup = 300L,
  samples = 500L,
  chains = 2L,
  device = "auto",
  print_result = TRUE
)

t1 <- Sys.time()
elapsed <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("\n>>> STATUS: BERHASIL TANPA ERROR <<<\n"))
cat(sprintf(">>> TOTAL WAKTU EKSEKUSI: %.2f detik <<<\n", elapsed))
