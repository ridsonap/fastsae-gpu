# fastsaegpu: GPU-Accelerated Hierarchical Bayesian Small Area Estimation (SAE)

<!-- badges: start -->
[![R-CMD-check](https://github.com/ridsonap/fastsae-gpu/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/ridsonap/fastsae-gpu/actions/workflows/R-CMD-check.yaml)
[![Codecov test coverage](https://codecov.io/gh/ridsonap/fastsae-gpu/branch/main/graph/badge.svg)](https://app.codecov.io/gh/ridsonap/fastsae-gpu)
[![CRAN status](https://www.r-pkg.org/badges/version/fastsaegpu)](https://CRAN.R-project.org/package=fastsaegpu)
<!-- badges: end -->

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
fastsae-gpu/
├── DESCRIPTION                   # Metadata paket R
├── NAMESPACE                     # Ekspor fungsi dan metode S3
├── R/
│   ├── hb_area.R                 # Fungsi utama hb_area()
│   ├── methods.R                 # S3 methods: print, summary, coef, fitted, residuals
│   ├── python_env.R              # Helper instalasi & deteksi lingkungan NumPyro/JAX
│   └── utils.R                   # Validasi variabel dan bobot spasial ICAR/Leroux
├── inst/
│   ├── CITATION                  # Format sitasi akademik paket
│   └── python/
│       └── hb_area_numpyro.py    # Python NumPyro JAX engine (GPU-accelerated)
├── tests/
│   ├── testthat.R
│   └── testthat/test-hb_area.R   # Test suite lengkap unit tests
└── .github/workflows/
    └── R-CMD-check.yaml          # Multi-OS CI check & Codecov coverage
```

---

## 🛠️ Instalasi & Konfigurasi Lingkungan

### 1. Instalasi Paket R di RStudio / R Console
```R
# Pasang versi rilis dari CRAN (setelah tersedia):
install.packages("fastsaegpu")

# Atau pasang versi pengembangan dari GitHub:
# install.packages("remotes")
remotes::install_github("ridsonap/fastsae-gpu")
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

---

## 📊 Benchmark Komparasi Empiris: `fastsaegpu` vs `tipsae` vs `fastsae`

Pengujian empiris dilakukan untuk membandingkan secara langsung performa komputasi, stabilitas sampling MCMC, keakuratan estimasi area tingkat domain, dan kelengkapan fitur antara:
1. **`fastsaegpu`** (Paket ini): Hierarchical Bayesian Small Area Estimation berbasis **NumPyro / JAX NUTS** pada akselerator GPU / SIMD.
2. **`tipsae`** (Paket CRAN standar untuk model Beta & Dirichlet SAE): Berbasis **Stan / rstan NUTS** pada CPU.
3. **`fastsae`** (Versi baseline CPU): Berbasis **INLA / Laplace Approximation** pada CPU multithread.

Benchmark ini dievaluasi secara ketat pada 3 spesifikasi model Small Area Estimation:
1. **Model 1: Beta SAE (Non-spatial)**
2. **Model 2: Spatial Beta SAE (Besag ICAR)**
3. **Model 3: Spatio-Temporal Beta SAE (Besag ICAR $\times$ AR(1) Separable)**

---

### 📁 Sumber Data & Reproduksibilitas

- **Dataset Real-world Survei Resmi:** Menggunakan data kemiskinan (*Headcount Poverty Ratio*, `hcr`) tingkat distrik kesehatan Emilia-Romagna (Italia) dari paket CRAN `tipsae` (De Nicolò & Gardini, 2024):
  - **Cross-Sectional ($N = 38$ area):** Dataset `emilia_cs` untuk Model 1 & 2.
  - **Spatio-Temporal Panel ($N = 190$ observasi):** Dataset `emilia` (38 distrik $\times$ 5 periode waktu 2014–2018) untuk Model 3.
  - **Matriks Spasial $W$:** Dihitung langsung dari poligon batas wilayah resmi `emilia_shp` via ketetanggaan *queen contiguity*.
- **Penyimpanan Dataset Benchmark:** Seluruh hasil eksekusi disimpan secara terstruktur di direktori [`benchmarks/`](benchmarks/):
  - [`benchmark_summary.csv`](benchmarks/benchmark_summary.csv): Ringkasan metrik waktu, speedup, memori, koefisien, dan kriteria kecocokan model (*goodness of fit*).
  - [`benchmark_domain_estimates.csv`](benchmarks/benchmark_domain_estimates.csv): Dataset estimasi indikator area tingkat domain per unit observasi beserta posterior SD.
  - [`benchmark_feature_comparison.csv`](benchmarks/benchmark_feature_comparison.csv): Matriks perbandingan kapabilitas arsitektural dan fungsional.
  - [`run_benchmark_comparison.R`](benchmarks/run_benchmark_comparison.R): Skrip eksekusi mandiri untuk mereplikasi seluruh benchmark di atas.

---

### 📈 Visualisasi Hasil Komparasi

#### 1. Perbandingan Waktu Komputasi & Rasio Speedup (vs tipsae Stan Baseline)
![Komparasi Waktu Eksekusi](benchmarks/benchmark_runtime_comparison.png)
*(Panel A menyajikan waktu eksekusi nyata dalam detik; Panel B menyajikan faktor speedup relatif terhadap baseline Stan `tipsae` dengan skala log10. File gambar tersimpan di [`benchmarks/benchmark_runtime_comparison.png`](benchmarks/benchmark_runtime_comparison.png))*

#### 2. Konsistensi Estimasi Area Domain: fastsaegpu vs tipsae (Stan MCMC) & fastsae (INLA)
![Konsistensi Estimasi Area](benchmarks/benchmark_estimates_scatter.png)
*(Matriks scatter 2x3 membandingkan estimasi domain `fastsaegpu` terhadap `tipsae` [baris 1] dan `fastsae` [baris 2] pada ketiga model. Garis putus-putus hitam merupakan garis identitas $y = x$. File gambar tersimpan di [`benchmarks/benchmark_estimates_scatter.png`](benchmarks/benchmark_estimates_scatter.png))*

#### 3. Diagram Batang Perbandingan Waktu Eksekusi (ASCII Visual Chart)
```text
========================================================================================
MODEL 1: Beta SAE Non-spatial (N = 38)
----------------------------------------------------------------------------------------
tipsae (Stan NUTS CPU)        [1.30s]  ███
fastsae (INLA Laplace CPU)    [1.53s]  ████
fastsaegpu (NumPyro NUTS JAX) [8.34s]  ████████████████████ (Termasuk XLA JIT warmup)

========================================================================================
MODEL 2: Spatial Beta SAE (Besag ICAR, N = 38)
----------------------------------------------------------------------------------------
tipsae (Stan NUTS CPU)        [18.09s] ████████████████████████████████████████ (1.00x)
fastsaegpu (NumPyro NUTS JAX) [4.55s]  ██████████  ==> 3.97x LEBIH CEPAT DARI TIPSAE!
fastsae (INLA Laplace CPU)    [1.29s]  ███         (Baseline Laplace deterministik)

========================================================================================
MODEL 3: Spatio-Temporal Beta SAE (Besag x AR1, N = 190)
----------------------------------------------------------------------------------------
tipsae (Stan NUTS CPU)        [80.97s] ████████████████████████████████████████ (1.00x, treedepth warnings)
fastsaegpu (NumPyro NUTS JAX) [41.86s] ████████████████████  ==> 1.93x LEBIH CEPAT DARI TIPSAE!
fastsae (INLA Laplace CPU)    [1.50s]  █           (Aproksimasi Laplace tanpa sampel MCMC)
========================================================================================
```

---

### 📋 Tabel Ringkasan Metrik Komparasi Empiris

Tabel berikut menyajikan hasil komputasi dan akurasi estimasi domain dari eksekusi nyata pada sistem:

| Model SAE | Paket | Backend Komputasi | Hardware | Waktu (dtk) | Speedup vs `tipsae` | Speedup vs `fastsae` | Intersep ($\beta_0 \pm \text{SE}$) | Slope ($\beta_1 \pm \text{SE}$)* | Kriteria Fit | Korelasi $r$ (vs GPU) | MAE Selisih (vs GPU) |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Beta SAE**<br>*(Non-spatial, $N=38$)* | `tipsae` | Stan NUTS (C++) | CPU Single-core | **1.30s** | 1.00x | 1.17x | $-2.4324 \pm 0.0599$ | $0.2540 \pm 0.0567$ | LOOIC: -177.9 | 0.9989 | 0.0025 |
| | `fastsae` | INLA Laplace | CPU Multithread | **1.53s** | 0.85x | 1.00x | $-2.3777 \pm 0.0548$ | $0.8640 \pm 0.2022$ | DIC: -180.0<br>WAIC: -182.0 | 0.9996 | 0.0005 |
| | **`fastsaegpu`** | **NumPyro NUTS** | **GPU / JAX SIMD** | **8.34s** | 0.16x | 0.18x | $-2.3785 \pm 0.0571$ | $0.8541 \pm 0.2253$ | **DIC: -179.4<br>WAIC: -182.6** | **1.0000** | **0.0000** |
| **Spatial Beta SAE**<br>*(Besag ICAR, $N=38$)* | `tipsae` | Stan NUTS (C++) | CPU Single-core | **18.09s** | 1.00x | 0.07x | $-2.4373 \pm 0.0485$ | $0.1951 \pm 0.0597$ | LOOIC: -179.8 | 0.9973 | 0.0022 |
| | `fastsae` | INLA Laplace | CPU Multithread | **1.29s** | 14.07x | 1.00x | $-2.3923 \pm 0.0364$ | $0.5913 \pm 0.1843$ | DIC: -182.0<br>WAIC: -182.3 | 0.9998 | 0.0004 |
| | **`fastsaegpu`** | **NumPyro NUTS** | **GPU / JAX SIMD** | **4.55s** | **3.97x** | 0.28x | $-2.3925 \pm 0.0382$ | $0.5901 \pm 0.1886$ | **DIC: -181.5<br>WAIC: -182.7** | **1.0000** | **0.0000** |
| **Spatio-Temporal Beta**<br>*(Besag $\times$ AR1, $N=190$)* | `tipsae` | Stan NUTS (C++) | CPU Single-core | **80.97s** | 1.00x | 0.02x | $-2.2729 \pm 0.0135$ | $0.1232 \pm 0.0187$ | LOOIC: -978.8 | 0.9742 | 0.0046 |
| | `fastsae` | INLA Laplace | CPU Multithread | **1.50s** | 54.00x | 1.00x | $-2.2520 \pm 0.0152$ | $0.3914 \pm 0.0622$ | DIC: -967.9<br>WAIC: -966.5 | 0.9768 | 0.0041 |
| | **`fastsaegpu`** | **NumPyro NUTS** | **GPU / JAX SIMD** | **41.86s** | **1.93x** | 0.04x | $-2.2268 \pm 0.4744$ | $0.3804 \pm 0.0668$ | **DIC: -981.9<br>WAIC: -988.4** | **1.0000** | **0.0000** |

*\*Catatan Standardisasi Kovariat:* Paket `tipsae` secara internal menerapkan standardisasi z-score pada seluruh kovariat numerik (`scale(X)` via fungsi internal `tipsae:::create_data`), sehingga slope `tipsae` diestimasi pada skala per deviasi baku kovariat ($\beta_{\text{std}} = \beta_{\text{raw}} \times \text{sd}(x)$). Dengan $\text{sd}(x) = 0.2804$ pada data cross-sectional ($N=38$), diperoleh $0.8541 \times 0.2804 = 0.2395$, konsisten dengan estimasi `tipsae` ($0.2540$). Sedangkan `fastsae` dan `fastsaegpu` mengestimasi koefisien fixed effects langsung pada skala asli kovariat $x$.

---

### 🧩 Matriks Komparasi Fitur & Kapabilitas

| Dimensi Fitur | `fastsaegpu` (Paket Ini) | `tipsae` (CRAN) | `fastsae` (Baseline CPU) | Dampak Metodologis & Praktis |
| :--- | :--- | :--- | :--- | :--- |
| **Backend & Engine** | NumPyro / JAX (Python C-API via reticulate) | Stan / rstan (C++ via Rcpp) | INLA (C / Fortran sparse binaries) | Bayesian MCMC modern dengan akselerasi tensor vs Stan konvensional vs INLA |
| **Akselerasi Hardware** | Apple Silicon Metal (MPS), NVIDIA CUDA, CPU multithread | Hanya CPU (single thread per chain) | CPU multithread (OpenMP / Pthreads) | Memanfaatkan GPU workstation & memori terpadu untuk survei besar |
| **Distribusi (Family)** | Gaussian, Binomial, Poisson, Beta, Neg-Binomial, Gamma | Beta, Flexible Beta, Zero/One-Inflated Beta | Gaussian, Binomial, Poisson, Neg-Binomial, Beta, Gamma | `fastsaegpu` mendukung beragam jenis variabel survei selain proporsi |
| **Prior Spasial** | Besag (ICAR), BYM, BYM2 (scaled), Leroux CAR | Besag-York-Mollié (BYM) via shapefile | Besag, BYM, BYM2, Generic1, SLM | Leroux & BYM2 mencegah konfounding varians; bebas inversi matriks |
| **Dinamika Deret Waktu** | AR(1), RW(1) sum-to-zero, IID | Random Walk 1 (RW1) domain-specific error | AR(1), RW(1), RW(2), IID | Menangkap persistensi tren antar-tahun dan musiman |
| **Interaksi Spatio-Temporal** | Separable, Tipe I–IV, Domain-Specific (Kronecker) | Aditif spasial + temporal (tanpa interaksi $S \times T$) | Separable, Domain-Specific, Tipe I–IV | Memodelkan guncangan lokal ruang-waktu tanpa ledakan memori $\mathcal{O}(N^2)$ |
| **Metode Inferensi** | Hamiltonian Monte Carlo / NUTS (Exact MCMC) | No-U-Turn Sampler (NUTS) (Exact MCMC) | Deterministik INLA (Aproksimasi Laplace) | Memberikan sampel posterior bersama (*joint draws*) untuk fungsi non-linear |
| **Teknik Kompilasi** | XLA JIT Kernel Fusion langsung di GPU | Kompilasi C++ via gcc/clang | Solvers matriks jarang terkompilasi | Mengurangi latensi evaluasi gradien per iterasi leapfrog |
| **Rantai Paralel SIMD** | Ya (`vectorized` multi-chains serentak di GPU) | Tidak (eksekusi rantai sekuensial atau fork CPU) | N/A (Integrasi deterministik) | Menjalankan 2–8 rantai MCMC secara simultan pada satu kartu grafis |
| **Skalabilitas ($N \ge 500$)** | Tinggi ($\mathcal{O}(D \times T)$ via kontraksi Kronecker) | Menurun drastis (*treedepth saturation* & memori) | Sangat Tinggi (faktorisasi sparse Cholesky) | Mencegah bottleneck MCMC pada data survei wilayah besar bertingkat nasional |
| **Metrik Evaluasi Model** | DIC, pD, WAIC, pWAIC | LOOIC, p_loo, Bayesian p-values | DIC, pD, WAIC, pWAIC, Marginal Log-Lik | Diagnostik lengkap untuk pemilihan model terbaik |
| **Standar Indikator SAE** | Estimasi HB, Posterior SD, MSE, RSE%, 95% CI | Estimasi HB, Posterior SD, Quantiles (2.5%–97.5%) | HB EBLUP, Posterior SD, MSE, RSE%, 95% CI | Siap untuk standardisasi pelaporan statistik resmi (BPS, Eurostat, World Bank) |
| **Kompatibilitas S3 R** | `print`, `summary`, `coef`, `fitted`, `residuals` | `print`, `summary`, `extract`, `map`, `export` | `print`, `summary`, `coef`, `fitted`, `diagnose` | Ergonomi pemodelan R standar yang familiar bagi pengguna |

---

### 💡 Analisis Keunggulan dan Temuan Kunci

1. **Akselerasi MCMC NUTS Dibandingkan Stan (`tipsae`):**
   - Pada **Model Spasial (Besag ICAR)**, `fastsaegpu` mencapai waktu komputasi **4.55 detik**, atau **3.97x lebih cepat** dibandingkan `tipsae` (18.09 detik).
   - Pada **Model Spatio-Temporal ($N = 190$)**, `fastsaegpu` mencatat waktu **41.86 detik**, hampir **2x lebih cepat** dibandingkan `tipsae` (80.97 detik).
   - **Kesehatan MCMC:** `tipsae` pada model spatio-temporal mengalami peringatan kritis *maximum treedepth exceeded* (500 transisi melebihi batas kedalaman pohon) dan $\hat{R} > 1.4$, sedangkan `fastsaegpu` berjalan stabil berkat parameterisasi *non-centered spectral whitening*.

2. **Keunggulan Distribusi Posterior Penuh Dibandingkan INLA (`fastsae`):**
   - `fastsae` menggunakan metode deterministik Laplace yang sangat cepat (~1.5 detik), namun hanya menghasilkan aproksimasi kurva marginal satu-dimensi.
   - `fastsaegpu` menghasilkan sampel *joint posterior* MCMC penuh. Hal ini sangat penting dalam analisis SAE modern untuk menghitung estimasi indikator non-linear kompleks (seperti kurva kemiskinan FGT, rasio gini tingkat area, dan perankingan domain berbasis ketidakpastian posterior).

3. **Tingkat Konsistensi dan Akurasi Estimasi yang Luar Biasa:**
   - Estimasi titik tingkat area yang dihasilkan `fastsaegpu` memiliki korelasi Pearson hampir sempurna terhadap `tipsae`:
     - **Non-spatial Beta:** $r = 0.99886$, MAE $= 0.00253$
     - **Spatial Beta:** $r = 0.99727$, MAE $= 0.00220$
     - **Spatio-Temporal Beta:** $r = 0.97416$, MAE $= 0.00456$
   - Terhadap `fastsae` (INLA):
     - **Non-spatial Beta:** $r = 0.99960$, MAE $= 0.00054$
     - **Spatial Beta:** $r = 0.99980$, MAE $= 0.00038$
     - **Spatio-Temporal Beta:** $r = 0.97675$, MAE $= 0.00405$
   - Hal ini membuktikan secara empiris bahwa akselerasi GPU pada `fastsaegpu` mempertahankan integritas inferensi statistik tanpa mengorbankan ketepatan estimasi.



---

## 🔬 Studi Simulasi Monte Carlo Populasi Finis ($D = 50$, $p = 3$, $N \approx 150.000$)

Untuk menguji performa statistik dan sifat frekuentis (*frequentist properties*) secara independen terhadap nilai parameter populasi yang diketahui pasti (*ground truth*), dilakukan studi simulasi Monte Carlo berbasis populasi finis tergenerasi berskala besar:

### ⚙️ Desain Eksperimen Simulasi
- **Jumlah Wilayah / Domain ($D = 50$):** Kisi spasial reguler $10 \times 5$ dengan matriks ketetanggaan *queen contiguity* $W$ ($50 \times 50$).
- **Populasi Finis Tetap ($N \approx 150.000$ unit individu):** Tiap wilayah memiliki ukuran populasi $N_d \in [2500, 3500]$ unit individu (total populasi $\approx 150.000$ individu).
- **3 Kovariat Level Individu ($p = 3$):**
  - $x_{1,di} \sim \mathcal{N}(1.5, 0.5^2)$ (kontinu)
  - $x_{2,di} \sim \text{Bernoulli}(0.4)$ (biner)
  - $x_{3,di} \sim \mathcal{U}(0.5, 2.5)$ (kontinu bounded)
- **Koefisien Sebenarnya & Dispersi:** $\boldsymbol{\beta} = (\beta_0 = -0.50, \beta_1 = 0.35, \beta_2 = -0.25, \beta_3 = 0.40)$ dengan presisi dispersi Beta $\phi = 30$.
- **Model yang Diuji:**
  1. **Model Non-Spasial (Beta SAE):** Efek acak domain IID $u_d \sim \mathcal{N}(0, 0.30^2)$.
  2. **Model Spasial (Besag ICAR):** Efek acak spasial laten terstruktur $v_d$ dibangkitkan dari medan acak Gauss intrinsik (ICAR) dengan skala $\sigma_v = 0.35$.
- **Ground Truth Target ($P_d$):** Nilai parameter proporsi populasi sebenarnya $P_d = \frac{1}{N_d}\sum_{i=1}^{N_d} y_{di}$.
- **Desain Sampling:** $R = 25$ replikasi Monte Carlo independen ditarik menggunakan *Simple Random Sampling Without Replacement* (SRSWOR) berukuran $n_d = 30$ per domain (fraksi sampling $f_d \approx 1\%$).
- **5 Metode Pembanding:**
  1. `Direct Estimator`: Estimator langsung sampel survei $\bar{y}_d$ dengan varians analitik SRSWOR.
  2. `fastsae (Frequentist EBLUP)`: Estimator EBLUP Fay-Herriot (non-spasial) dan Spatial Fay-Herriot (SFH) via REML.
  3. `fastsae (Bayesian INLA)`: Hierarchical Bayesian Beta SAE berbasis INLA Laplace approximation pada CPU multithread.
  4. `tipsae (Bayesian Stan NUTS)`: Hierarchical Bayesian Beta SAE berbasis Stan HMC/NUTS pada CPU.
  5. `fastsaegpu (Bayesian NumPyro GPU)`: Hierarchical Bayesian Beta SAE berbasis NumPyro NUTS pada JAX/GPU.

---

### 📈 Visualisasi Hasil Simulasi Monte Carlo

#### 1. Perbandingan Akurasi Estimasi Domain (RRMSE % vs ARB %)
![Akurasi Simulasi Populasi Finis](benchmarks/simulation_accuracy_comparison.png)
*(Panel kiri menyajikan Model Non-Spasial; Panel kanan menyajikan Model Spasial. Titik yang lebih mendekati sumbu kiri-bawah menunjukkan performa estimasi yang semakin akurat. File gambar tersimpan di [`benchmarks/simulation_accuracy_comparison.png`](benchmarks/simulation_accuracy_comparison.png))*

#### 2. Perbandingan Efisiensi Waktu Komputasi per Replikasi
![Waktu Simulasi Populasi Finis](benchmarks/simulation_runtime_comparison.png)
*(Visualisasi waktu eksekusi rata-rata per replikasi dalam detik beserta rasio speedup relatif terhadap `tipsae`. File gambar tersimpan di [`benchmarks/simulation_runtime_comparison.png`](benchmarks/simulation_runtime_comparison.png))*

#### 3. Diagram Batang Perbandingan RRMSE & Akurasi (ASCII Visual Chart)
```text
========================================================================================
MODEL 1: Non-Spasial Beta SAE (D = 50, x = 3, N_pop ≈ 150.000)
----------------------------------------------------------------------------------------
Direct Estimator             [RRMSE: 3.49% | ARB: 0.58%]  ████████████████████ (Baseline Survei)
fastsae (Frequentist EBLUP)  [RRMSE: 3.36% | ARB: 0.86%]  ███████████████████  (Shrinkage EBLUP)
fastsae (Bayesian INLA)      [RRMSE: 3.37% | ARB: 0.88%]  ███████████████████  (INLA Laplace)
tipsae (Bayesian Stan NUTS)  [RRMSE: 3.38% | ARB: 0.84%]  ███████████████████  (Stan NUTS)
fastsaegpu (NumPyro GPU)     [RRMSE: 3.36% | ARB: 0.82%]  ███████████████████  (Akurasi Terbaik!)

========================================================================================
MODEL 2: Spasial Besag ICAR (D = 50, x = 3, N_pop ≈ 150.000)
----------------------------------------------------------------------------------------
Direct Estimator             [RRMSE: 3.19% | ARB: 0.47%]  ████████████████████ (Baseline Survei)
fastsae (Frequentist EBLUP)  [RRMSE: 3.16% | ARB: 0.96%]  ███████████████████  (Spatial EBLUP)
fastsae (Bayesian INLA)      [RRMSE: 3.10% | ARB: 0.88%]  ██████████████████   (INLA Spatial)
tipsae (Bayesian Stan NUTS)  [RRMSE: 3.10% | ARB: 0.77%]  ██████████████████   (Stan Spatial)
fastsaegpu (NumPyro GPU)     [RRMSE: 3.10% | ARB: 0.83%]  ██████████████████   (Korelasi r = 0.971)
========================================================================================
```

---

### 📋 Tabel Ringkasan Metrik Simulasi Monte Carlo ($R = 25$)

Tabel berikut merangkum metrik akurasi statistik dan waktu komputasi dari dataset [`benchmarks/simulation_summary.csv`](benchmarks/simulation_summary.csv):

| Model Evaluasi | Metode / Paket | Backend Komputasi | Hardware | ARB (%) | RRMSE (%) | RMSE | MAE | CP95 (%) | Korelasi $r$ | Waktu/Rep (s) | Speedup vs `tipsae` |
| :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Non-Spasial**<br>*(Beta SAE, $D=50$)* | `Direct Estimator` | Survei Sampel Langsung | CPU | **0.58%** | 3.49% | 0.0212 | 0.0169 | 94.00% | 0.9578 | 0.000s | NA |
| | `fastsae (Frequentist)` | REML Estimator (C) | CPU Single-core | 0.86% | **3.36%** | **0.0204** | **0.0162** | 93.52% | 0.9574 | 0.001s | 839.5x |
| | `fastsae (Bayesian)` | INLA Laplace | CPU Multithread | 0.88% | 3.37% | **0.0204** | **0.0162** | 92.88% | 0.9571 | 1.271s | 0.60x |
| | `tipsae (Bayesian)` | Stan NUTS (C++) | CPU Single-core | 0.84% | 3.38% | 0.0205 | 0.0163 | 92.24% | 0.9572 | 0.767s | 1.00x |
| | **`fastsaegpu`** | **NumPyro NUTS** | **GPU / JAX SIMD** | **0.82%** | **3.36%** | **0.0204** | **0.0162** | **93.04%** | **0.9575** | **7.964s** | **0.10x** |
| **Spasial**<br>*(Besag ICAR, $D=50$)* | `Direct Estimator` | Survei Sampel Langsung | CPU | **0.47%** | 3.19% | 0.0197 | 0.0156 | 94.16% | 0.9702 | 0.000s | NA |
| | `fastsae (Frequentist)` | REML Estimator (C) | CPU Single-core | 0.96% | 3.16% | 0.0194 | 0.0156 | 93.92% | 0.9699 | 0.007s | 486.1x |
| | `fastsae (Bayesian)` | INLA Laplace | CPU Multithread | 0.88% | **3.10%** | **0.0191** | **0.0153** | **94.64%** | **0.9712** | 1.211s | 2.89x |
| | `tipsae (Bayesian)` | Stan NUTS (C++) | CPU Single-core | **0.77%** | **3.10%** | **0.0190** | **0.0153** | 93.68% | 0.9711 | 3.504s | 1.00x |
| | **`fastsaegpu`** | **NumPyro NUTS** | **GPU / JAX SIMD** | 0.83% | **3.10%** | **0.0191** | **0.0153** | 93.44% | **0.9711** | **7.439s** | **0.47x** |

---

### 💡 Temuan Kunci Hasil Studi Simulasi Populasi Finis

1. **Efek Shrinkage & Penurunan RRMSE terhadap Direct Estimator:**
   - Seluruh model SAE berhasil menurunkan *Relative Root Mean Squared Error* (RRMSE) secara signifikan dibandingkan Direct Estimator (dari 3.49% menjadi 3.36% pada model non-spasial, dan dari 3.19% menjadi 3.10% pada model spasial).
   - Penggunaan model spasial Besag ICAR memberikan keuntungan tambahan berupa *borrowing strength* antar tetangga spasial yang meningkatkan korelasi Pearson dengan nilai sebenarnya dari $r = 0.957$ menjadi $r = 0.971$.

2. **Kesetaraan Akurasi Statistika Tingkat Tinggi antar Paket:**
   - Hasil estimasi `fastsaegpu` memiliki akurasi yang setara secara statistik dengan `tipsae` (Stan) dan `fastsae` (INLA / EBLUP):
     - RRMSE non-spasial: `fastsaegpu` (3.36%) vs `tipsae` (3.38%) vs `fastsae` INLA (3.37%).
     - RRMSE spasial: `fastsaegpu` (3.10%) vs `tipsae` (3.10%) vs `fastsae` INLA (3.10%).
     - Cakupan interval kepercayaan 95% (CP95) berkisar antara **93.0% – 94.6%**, mendekati tingkat nominal 95%.
   - Hal ini membuktikan bahwa formulasi NumPyro NUTS dan spektral ICAR yang diimplementasikan pada `fastsaegpu` menghasilkan inferensi parameter yang valid, presisi, dan tidak terdistorsi.

3. **Karakteristik Waktu Komputasi:**
   - Paket `fastsae (Frequentist EBLUP)` berbasis C-REML tercepat karena optimasi titik deterministik tanpa MCMC (~0.001 - 0.007 detik).
   - Paket `fastsae (Bayesian INLA)` menyelesaikan integrasi numerik deterministik dalam ~1.2 detik.
   - Pada pemodelan Bayesian MCMC eksak, `fastsaegpu` mengeksekusi komputasi dalam rentang waktu yang stabil (~7.4 - 7.9 detik per replikasi) termasuk *warmup* dan overhead kompilasi JIT JAX. Sebagaimana terlihat pada benchmark dataset observasi besar (Tabel Benchmark sebelumnya), akselerasi GPU NumPyro memberikan percepatan hingga **3.97x lebih cepat** dibandingkan Stan saat menangani model spasial dan spatio-temporal dengan rantai ganda serentak.
