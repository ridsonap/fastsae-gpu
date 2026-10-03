library(fastsaegpu)
reticulate::use_virtualenv("/Users/ridsonap/.virtualenvs/r-numpyro-gpu", required = TRUE)

set.seed(42)
D <- 50L
T_periods <- 10L
N <- D * T_periods
domains <- paste0("domain_", sprintf("%02d", 1:D))
df <- expand.grid(time = 1:T_periods, domain = domains, stringsAsFactors = FALSE)
df <- df[order(df[["domain"]], df[["time"]]), ]
rownames(df) <- NULL

W <- matrix(0, D, D)
rownames(W) <- colnames(W) <- domains
for (i in 1:D) {
  if (i > 1) W[i, i - 1] <- 1
  if (i < D) W[i, i + 1] <- 1
}

df$x <- rnorm(N)
df$vardir <- runif(N, 0.05, 0.2)
df$y <- 2.0 + 1.0 * df$x + rnorm(N, sd = sqrt(df$vardir))

cat("Menjalankan hb_area Spatio-Temporal SAE (D=50, T=10, N=500)...\n")
t0 <- Sys.time()
fit <- hb_area(
  y ~ x,
  data = df,
  domain = "domain",
  time = "time",
  vardir = "vardir",
  spatial = "besag",
  temporal = "ar1",
  st_interaction = "type4",
  W = W,
  warmup = 300L,
  samples = 500L,
  chains = 2L,
  device = "auto"
)
t1 <- Sys.time()
elapsed <- as.numeric(difftime(t1, t0, units = "secs"))
cat(sprintf("\n>>> UJI SELESAI TANPA ERROR! WAKTU: %.2f DETIK <<<\n", elapsed))
