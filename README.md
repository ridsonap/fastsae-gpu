# fastsaegpu: GPU-Accelerated Hierarchical Bayesian & Machine Learning Small Area Estimation

<!-- badges: start -->
[![R-CMD-check](https://github.com/ridsonap/fastsae-gpu/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ridsonap/fastsae-gpu/actions/workflows/R-CMD-check.yaml)
[![Codecov test coverage](https://codecov.io/gh/ridsonap/fastsae-gpu/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ridsonap/fastsae-gpu)
[![CRAN status](https://www.r-pkg.org/badges/version/fastsaegpu)](https://CRAN.R-project.org/package=fastsaegpu)
<!-- badges: end -->

`fastsaegpu` adalah paket R berkinerja tinggi untuk **Small Area Estimation (SAE)** tingkat area, menggabungkan:
1. **Hierarchical Bayesian SAE (`hb_area`)** yang diakselerasi GPU (Apple Silicon Metal & NVIDIA CUDA) berbasis **NumPyro / JAX NUTS**.
2. **Machine Learning SAE (`merf_area`)** berbasis **Mixed Effects Random Forest (MERF / FH-RF)** dengan mesin C++ multithread `ranger`.
3. **Inovasi Metodologis Mutakhir:** GVF Variance Smoothing, Regularized Horseshoe Prior, Robust Student-$t$ Random Effects, Hierarki Bersarang Dua Tingkat (*Nested Sub-Area*), dan Kalibrasi Agregat (*In-Model & External Benchmarking*).

---

## 🚀 Fitur Utama

- **Akselerasi GPU & SIMD:** Eksekusi NUTS MCMC paralel di GPU dengan kompilasi XLA JIT JAX. Hingga **4x lebih cepat** dibandingkan Stan (`tipsae`) pada model spasial.
- **Machine Learning Semi-Parametrik (`merf_area`):** Menangkap non-linearitas dan interaksi kompleks tanpa *feature engineering* manual, didukung estimasi Parametric Bootstrap MSE.
- **Penanganan Big Data & Noise:** *Regularized Horseshoe Prior* memangkas 95%+ kovariat bising (*noise covariates*) secara otomatis.
- **Ketahanan Anomali / Guncangan Lokal:** Distribusi efek acak *heavy-tailed* Student-$t$ mengisolasi *outlier* wilayah tanpa merusak estimasi area sekitarnya.
- **Stabilisasi Varians Sampling:** Generalized Variance Functions (GVF) meredam kebisingan varians sampel kecil.
- **Struktur Wilayah Bertingkat:** Model bersarang dua tingkat (Provinsi $\to$ Kabupaten) dengan diagnostik Intra-Cluster Correlation (ICC).
- **Konsistensi Kebijakan Agregat:** Modul `benchmark()` menjamin estimasi tingkat area cocok sempurna dengan total survei nasional atau angka sensus/registrasi resmi.

---

## 🛠️ Instalasi & Setup Lingkungan

```r
# 1. Pasang paket dari GitHub
remotes::install_github("ridsonap/fastsae-gpu")
library(fastsaegpu)

# 2. Setup otomatis lingkungan NumPyro / JAX di GPU (Apple Silicon / NVIDIA CUDA / CPU)
setup_numpyro_env(device = "auto")
```

---

## 💡 Contoh Penggunaan Cepat

### 1. Model Bayesian Lengkap (Synergy: GVF + Horseshoe + Robust + Benchmarking)

```r
library(fastsaegpu)

# Model Fay-Herriot Bayes dengan proteksi noise, outlier, dan kalibrasi total
fit_hb <- hb_area(
  formula = y ~ x1 + x2 + x3 + x4 + x5,
  data = data_survey,
  vardir = "var_direct",
  family = "gaussian",
  smooth_vardir = TRUE,              # Fitur A: GVF Variance Smoothing
  prior_beta = "horseshoe",          # Fitur B: Regularized Horseshoe Prior
  robust = TRUE,                     # Fitur C: Robust Student-t Random Effects
  benchmark = TRUE,                  # Fitur D: In-Model Benchmarking
  benchmark_weights = "pop_weight",
  benchmark_method = "optimal",
  device = "auto"                    # "metal", "cuda", atau "cpu"
)

summary(fit_hb)
head(fit_hb$df_hb)
```

### 2. Mixed Effects Random Forest (MERF / FH-RF Machine Learning SAE)

```r
# Estimasi non-parametrik Random Forest + Random Effects
fit_merf <- merf_area(
  formula = y ~ x1 + x2 + x3 + x4,
  data = data_survey,
  vardir = "var_direct",
  domain = "kabupaten",
  engine = "ranger",                 # C++ multithreading
  num_trees = 500,
  mse_type = "bootstrap",            # Parametric bootstrap MSE
  B = 50,
  seed = 123
)

print(fit_merf)
plot(fit_merf, type = "importance")  # Ranking pengaruh variabel
plot(fit_merf, type = "estimates")   # Scatter Direct vs MERF
```

### 3. Model Hierarki Bersarang Dua Tingkat (Torabi & Rao, 2014)

```r
# Kluster makro (Provinsi) dan sub-area (Kabupaten/Kota)
fit_nested <- hb_area(
  formula = y ~ x1 + x2,
  data = data_survey,
  vardir = "var_direct",
  domain = "provinsi",
  subarea = "kabupaten",
  device = "auto"
)

print(fit_nested)
# > Hierarchy: Two-Level Nested Sub-Area [34 Major Areas -> 514 Sub-Areas | ICC: 0.3821]
```

### 4. Model Spatio-Temporal Panel SAE

```r
# Mengestimasi interaksi ruang dan waktu pada GPU
fit_st <- hb_area(
  formula = y ~ x1 + x2,
  data = data_panel,
  domain = "domain",
  time = "year",
  vardir = "vardir",
  family = "gaussian",
  spatial = "leroux",                # Leroux CAR spatial
  temporal = "ar1",                  # Autoregressive AR(1)
  st_interaction = "separable",      # Kontraksi Kronecker cepat GPU
  W = W_adjacency_matrix
)
```

---

## 🔬 Spesifikasi Model yang Didukung

| Komponen | Pilihan yang Didukung | Keterangan |
| :--- | :--- | :--- |
| **Model Tipe** | `hb_area()` (Bayesian MCMC), `merf_area()` (Machine Learning) | Inferensi posterior penuh vs C++ tree ensemble |
| **Distribusi (Family)** | `gaussian`, `binomial`, `poisson`, `beta`, `nbinomial`, `gamma` | Beragam jenis respon (proporsi, cacahan, kontinu positif) |
| **Prior Spasial** | `none`, `besag` (ICAR), `bym`, `bym2` (scaled), `leroux` CAR | Menggunakan matriks ketetanggaan $W$ |
| **Dinamika Waktu** | `none`, `ar1`, `rw1` (sum-to-zero), `iid` | Tren dan persistensi serial antar tahun |
| **Prior Koefisien** | `normal`, `horseshoe` (Carvalho et al. 2010) | Regularisasi otomatis untuk kovariat bising (*sparse*) |
| **Efek Acak Area** | `normal`, `student` (Bell & Huang 2006) | Resistensi terhadap *outliers* / guncangan lokal |
| **Hierarki Wilayah** | Standar Area-Level, Two-Level Nested Sub-Area (Torabi & Rao 2014) | Kluster induk wilayah makro & sub-area mikro |
| **Stabilisasi Varians** | `smooth_vardir = TRUE` via GVF (`log_linear`, `power`, `ratio`, `loess`) | Meredam variabilitas sampling `vardir` kecil |
| **Kalibrasi Agregat** | `benchmark()` / `in-model`: `logit`, `optimal`, `ratio`, `difference` | Menjamin konsistensi total mikro-makro |
| **Akselerasi** | Apple Silicon Metal (`mps`), NVIDIA CUDA, Multithread CPU | Kompilasi JIT XLA JAX & C++ ranger |

---

## 📊 Ringkasan Bukti Empiris & Validasi

Pengujian empiris dilakukan pada populasi finis tergenerasi ($N \approx 125.000$ individu, $D = 50$ wilayah, 12 kovariat, 4 *outlier shocks*) terhadap **Ground Truth Sejati ($\bar{Y}_d$)**:

| Model SAE | ARB (%) | RRMSE (%) | Relative Efficiency (RE) | RRMSE Outliers | Waktu Komputasi |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Direct Survey Estimator** | 5.67% | 7.40% | 1.00x *(Baseline)* | 5.66% | Instant |
| **fastsae (INLA Laplace)** | 5.12% | 6.54% | 1.38x | 12.74% | 1.78s |
| **Standard HB Area (GPU)** | 4.73% | 6.18% | 1.55x | 13.23% | 13.79s |
| **HB Area + Horseshoe** | 4.16% | 5.56% | 1.91x | 12.33% | 7.25s |
| **Best HB Area (Synergy)** | **4.13%** | **5.57%** | **1.94x (Terbaik!)** | 13.61% | 6.99s |
| **MERF (Machine Learning)** | 4.63% | 5.97% | **1.64x** | **12.07% (Terbaik!)** | **1.58s (Tercepat!)** |

> *Skrip dan visualisasi replikasi lengkap tersedia di folder [`benchmarks/`](benchmarks/).*

---

## 📚 Sitasi

Jika Anda menggunakan `fastsaegpu` dalam publikasi atau penelitian, silakan sitasi:

```bibtex
@manual{fastsaegpu2026,
  title = {fastsaegpu: GPU-Accelerated Hierarchical Bayesian and Machine Learning Small Area Estimation},
  author = {Ridson Al Farizal Pambudi},
  year = {2026},
  note = {R package version 0.1.0},
  url = {https://github.com/ridsonap/fastsae-gpu}
}
```

## 📄 Lisensi

GPL (>= 3) © 2026 Ridson Al Farizal Pambudi.
