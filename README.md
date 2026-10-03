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
/Volumes/work/_Projects/mcmc/
├── DESCRIPTION                   # Metadata paket R
├── NAMESPACE                     # Ekspor fungsi dan metode S3
├── R/
│   ├── hb_area.R                 # Fungsi utama hb_area()
│   ├── methods.R                 # S3 methods: print, summary, coef, fitted, residuals
│   ├── python_env.R              # Helper instalasi & deteksi lingkungan NumPyro/JAX
│   └── utils.R                   # Validasi variabel dan bobot spasial ICAR
├── inst/
│   └── python/
│       └── hb_area_numpyro.py    # Python NumPyro JAX engine (GPU-accelerated)
├── tests/
│   ├── testthat.R
│   └── testthat/test-hb_area.R   # Test suite lengkap
└── benchmarks/
    └── benchmark_spatiotemporal_gpu.R  # Skrip benchmark Spatio-Temporal n=500
```

---

## 🛠️ Instalasi & Konfigurasi Lingkungan

### 1. Instalasi Paket R di RStudio / R Console
```R
# Pasang paket langsung dari folder ini:
devtools::install("/Volumes/work/_Projects/mcmc")
library(fastsaegpu)
```

### 2. Konfigurasi Otomatis GPU Environment (NumPyro & JAX)
Di konsol R, jalankan perintah:
```R
# Otomatis mendeteksi Apple Silicon M-Series atau NVIDIA CUDA:
setup_numpyro_env(device = "auto")
```
Fungsi ini akan menyiapkan virtual environment Python terisolasi dengan dependensi:
- **Apple Silicon Mac:** `numpy`, `scipy`, `numpyro`, `jax`, `jax-metal`
- **Linux NVIDIA:** `numpy`, `scipy`, `numpyro`, `jax[cuda12]`
- **CPU (Fallback):** `numpy`, `scipy`, `numpyro`, `jax`, `jaxlib`

---

## 💡 Contoh Penggunaan: Spatio-Temporal SAE ($n = 500$)

```R
library(fastsaegpu)

# Jalankan model Fay-Herriot Spatio-Temporal pada GPU:
fit <- hb_area(
  formula = y ~ x1 + x2,
  data = data_spatiotemporal,
  domain = "domain",
  time = "year",
  vardir = "vardir",
  family = "gaussian",
  spatial = "besag",            # ICAR Spatial random effect
  temporal = "ar1",             # Autoregressive AR(1) temporal dynamics
  st_interaction = "type4",     # Knorr-Held Type IV (Space x Time)
  W = W_adjacency_matrix,       # Matriks ketetanggaan wilayah
  warmup = 500L,
  samples = 1000L,
  chains = 2L,
  device = "auto"               # "metal", "cuda", atau "cpu"
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

| Komponen | Opsi yang Didukung |
| :--- | :--- |
| **Distribusi (Family)** | `gaussian` (Fay-Herriot), `binomial` (Logit), `poisson` (Log-linear), `beta` |
| **Efek Spasial** | `none`, `besag` (ICAR), `bym2` (Scaled Besag-York-Mollié) |
| **Efek Temporal** | `none`, `ar1`, `rw1`, `iid` |
| **Interaksi Spatio-Temporal** | `none`, `separable`, `domain-specific`, `type1`, `type2`, `type3`, `type4` |
| **Hardware** | Apple Silicon (`metal`), NVIDIA GPU (`cuda`), CPU Multithread |
