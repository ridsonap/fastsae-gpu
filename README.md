# fastsaegpu: GPU-Accelerated Hierarchical Bayesian Small Area Estimation (SAE)

`fastsaegpu` adalah implementasi paket R untuk fungsi **`hb_area`** (Hierarchical Bayesian Area-Level Small Area Estimation) yang ditenagai oleh mesin MCMC berkinerja tinggi **NumPyro (JAX)** pada **GPU** (Apple Silicon Metal & NVIDIA CUDA).

---

## 🚀 Fitur Utama

- **Akselerasi GPU Penuh:** Menggunakan *No-U-Turn Sampler* (NUTS) yang dikompilasi secara JIT via XLA langsung di GPU (Apple Silicon M-Series via `jax-metal` dan NVIDIA GPU via `jax[cuda]`), dengan fallback otomatis ke CPU.
- **Rantai MCMC Tervektorisasi (SIMD):** Menjalankan 2 hingga 8 rantai paralel secara serentak di GPU (`chain_method = "vectorized"`).
- **Efisiensi Spatio-Temporal ($n \approx 500$):**
  - **Non-Centered Whitening:** Mengeliminasi korelasi posterior ekstrem (*Neal's funnel*) antara varians hiperparameter dan efek acak laten.
  - **Trik Kontraksi Tensor Kronecker:** Menghitung interaksi spatio-temporal $Q_{st} = Q_s \otimes Q_t$ dalam hitungan detik tanpa operasi matriks $n \times n$ langsung.
  - **Stabilisasi Gradien AR(1):** Mencegah singularitas gradien $0^0 = \text{NaN}$ pada $\rho = 0$ dalam reverse-mode autodiff JAX.
  - **Identifiabilitas RW(1):** Proyeksi *sum-to-zero* terpusat agar efek acak temporal tidak kolinear dengan intersep fixed-effects.
- **Output Kompatibel S3 Penuh:** Mengembalikan objek kelas `c("fastsae_hb_area", "fastsae")` lengkap dengan metode `print()`, `summary()`, `coef()`, `fitted()`, dan `residuals()`.

---

## 📦 Struktur Paket

```
/Volumes/work/_Projects/fastsaegpu/
├── DESCRIPTION                   # Metadata paket R
├── NAMESPACE                     # Ekspor fungsi dan metode S3
├── R/
│   ├── hb_area.R                 # Fungsi utama hb_area()
│   ├── methods.R                 # S3 methods: print, summary, coef, fitted, residuals
│   ├── python_env.R              # Helper instalasi & deteksi lingkungan NumPyro/JAX
│   └── utils.R                   # Validasi variabel dan bobot spasial ICAR/Leroux
├── inst/
│   └── python/
│       └── hb_area_numpyro.py    # Python NumPyro JAX engine (GPU-accelerated)
├── tests/
│   ├── testthat.R
│   └── testthat/test-hb_area.R   # Test suite lengkap (28 unit tests)
└── benchmarks/
    └── benchmark_spatiotemporal_gpu.R  # Skrip benchmark Spatio-Temporal n=500
```

---

## 🛠️ Instalasi & Konfigurasi Lingkungan

### 1. Instalasi Paket R di RStudio / R Console
```R
# Pasang paket langsung dari folder ini:
devtools::install("/Volumes/work/_Projects/fastsaegpu")
library(fastsaegpu)
```

### 2. Konfigurasi Otomatis GPU Environment (NumPyro & JAX)
Di konsol R, jalankan perintah:
```R
# Otomatis mendeteksi Apple Silicon M-Series atau NVIDIA CUDA:
setup_numpyro_env(device = "auto")
```
Fungsi ini akan menyiapkan virtual environment Python terisolasi dengan dependensi:
- **Apple Silicon Mac:** `numpy`, `scipy`, `numpyro`, `jax==0.5.0`, `jaxlib==0.5.0`, `jax-metal==0.1.1`
- **Linux NVIDIA:** `numpy`, `scipy`, `numpyro`, `jax[cuda12]`
- **CPU (Fallback):** `numpy`, `scipy`, `numpyro`, `jax`, `jaxlib`

---

## 💡 Contoh Penggunaan: Spatio-Temporal SAE ($n = 500$)

```R
library(fastsaegpu)

# 1. Model Fay-Herriot Spatio-Temporal pada GPU (Gaussian)
fit <- hb_area(
  formula = y ~ x1 + x2,
  data = data_spatiotemporal,
  domain = "domain",
  time = "year",
  vardir = "vardir",
  family = "gaussian",
  spatial = "leroux",           # Leroux CAR spatial model
  temporal = "ar1",             # Autoregressive AR(1) temporal dynamics
  st_interaction = "separable", # Space x Time interaction
  W = W_adjacency_matrix,       # Matriks ketetanggaan wilayah
  warmup = 500L,
  samples = 1000L,
  chains = 2L,
  device = "auto"               # "metal", "cuda", atau "cpu"
)

# 2. Model Negative Binomial untuk Data Cacahan (Overdispersed Counts)
fit_nb <- hb_area(
  formula = y_count ~ x1 + x2,
  data = data_area,
  exposure = "population",
  family = "nbinomial",
  spatial = "bym",              # Besag-York-Mollié (ICAR + IID)
  W = W_adjacency_matrix
)

# 3. Model Gamma untuk Data Kontinu Positif Menjulur (Skewed Positive)
fit_gamma <- hb_area(
  formula = expenditure ~ income,
  data = data_area,
  family = "gamma",
  spatial = "besag",
  W = W_adjacency_matrix
)

# Ringkasan hasil estimasi
summary(fit)

# Akses hasil estimasi domain
head(fit$df_hb)

# Ekstraksi koefisien regresi & residual
coef(fit)
res <- residuals(fit)
```

---

## 🔬 Spesifikasi Model yang Didukung

| Komponen | Opsi yang Didukung | Keterangan |
| :--- | :--- | :--- |
| **Distribusi (Family)** | `gaussian` | Fay-Herriot standar dengan varians sampling direct `vardir` |
| | `binomial` | Model logit proporsi / binomial dengan `trials` |
| | `poisson` | Model log-linear Poisson counts dengan offset `exposure` |
| | `beta` | Model proporsi kontinu $y \in (0, 1)$ |
| | `nbinomial` | Negative Binomial 2 dengan parameter dispersi $\alpha$ & `exposure` |
| | `gamma` | Model log link data kontinu positif menjulur ($y > 0$) |
| **Efek Spasial** | `none` | Tanpa efek spasial |
| | `besag` | Intrinsic Autoregressive (ICAR) standar |
| | `bym2` | Scaled Besag-York-Mollié (Riebler et al., 2016) |
| | `bym` | Besag-York-Mollié klasik (ICAR $\sigma_s^2$ + IID $\sigma_{\text{iid}}^2$) |
| | `leroux` | Leroux CAR ($Q(\rho) = \rho(D - W) + (1-\rho)I$) bebas inversi matriks |
| **Efek Temporal** | `none`, `ar1`, `rw1`, `iid` | Dinamika deret waktu terpusat |
| **Interaksi Spatio-Temporal** | `none`, `separable`, `domain-specific`, `type1`, `type2`, `type3`, `type4` | Formulasi kontraksi Kronecker cepat GPU |
| **Hardware** | Apple Silicon (`metal`), NVIDIA GPU (`cuda`), CPU Multithread | Kompilasi NUTS via JAX XLA SIMD |
