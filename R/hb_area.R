#' Hierarchical Bayes for Area-Level Small Area Estimation with GPU NumPyro Backend
#'
#' @param formula Fixed effects model formula (e.g., y ~ x1 + x2).
#' @param data Data frame containing area/domain observations.
#' @param domain Domain identifier column or vector (major area level when nested).
#' @param subarea Optional sub-area identifier column or vector for two-level nested sub-area SAE models (Torabi & Rao, 2014).
#'   When specified alongside \code{domain}, the model fits nested random effects:
#'   \eqn{\theta_{jk} = \mathbf{x}_{jk}^\top \boldsymbol{\beta} + u_j + v_{jk}} where \eqn{u_j} is the major area
#'   (e.g., province) effect and \eqn{v_{jk}} is the sub-area (e.g., district) effect.
#' @param time Time period identifier (required if temporal != "none").
#' @param family Likelihood family: "gaussian", "binomial", "poisson", "beta", "nbinomial", or "gamma".
#' @param spatial Spatial effect structure: "none", "besag", "bym2", "bym", or "leroux".
#' @param temporal Temporal effect structure: "none", "ar1", "rw1", or "iid".
#' @param st_interaction Spatio-temporal interaction: "none", "separable", "domain-specific",
#'   "type1", "type2", "type3", or "type4".
#' @param W Spatial adjacency matrix (required if spatial != "none").
#' @param vardir Known direct sampling variances (for Gaussian and Beta).
#' @param trials Total trials / sample size per area (for Binomial).
#' @param exposure Expected exposure / offsets (for Poisson and Negative Binomial).
#' @param benchmark Logical: whether to enable in-model self-benchmarking or calibration (default FALSE).
#' @param benchmark_weights Survey or population weights vector, or column name in \code{data}.
#' @param benchmark_target Target benchmark value. If NULL (default), performs self-benchmarking
#'   using the survey-weighted direct total.
#' @param benchmark_method Calibration method: "logit", "optimal", "ratio", or "difference".
#' @param prior_beta Regression prior distribution: \code{"normal"} (default) or \code{"horseshoe"}
#'   for sparse high-dimensional shrinkage (Carvalho et al. 2010; Piironen & Vehtari 2017).
#' @param robust Logical: whether to use heavy-tailed Student-t random effects to protect against
#'   outlier domains (Bell & Huang 2006; Gershunskaya & Lahiri 2018) (default FALSE).
#' @param smooth_vardir Logical: whether to automatically smooth direct sampling variances using
#'   Generalized Variance Functions (GVF) before estimation (default FALSE).
#' @param gvf_method GVF smoothing method if \code{smooth_vardir = TRUE}: \code{"log_linear"} (default),
#'   \code{"power"}, \code{"ratio"}, or \code{"loess"}.
#' @param warmup Number of MCMC warmup iterations (default 500).
#' @param samples Number of MCMC post-warmup samples (default 1000).
#' @param chains Number of parallel MCMC chains on GPU (default 2).
#' @param device Target hardware: "auto" (CUDA on Linux/Windows, CPU on macOS for stability; use "metal" explicitly for Apple Silicon GPU), "metal" (Apple Silicon), "cuda" (NVIDIA), or "cpu". Note: \code{device="metal"} on macOS may fall back to CPU if JAX Metal shaders fail (see Details).
#' @param seed Random seed for MCMC reproducibility (default 42).
#' @param print_result Logical: print summary of results upon completion (default TRUE).
#' @param ... Additional arguments.
#' @return An object of class \code{c("fastsae_hb_area", "fastsae")}.
#' @examples
#' \donttest{
#' if (check_numpyro_available()) {
#'   set.seed(42)
#'   D <- 15
#'   x <- rnorm(D)
#'   vardir <- rep(0.1, D)
#'   y <- 2.0 + 1.2 * x + rnorm(D, sd = sqrt(vardir))
#'   df <- data.frame(y = y, x = x, vardir = vardir)
#'
#'   fit <- hb_area(
#'     y ~ x,
#'     data = df,
#'     vardir = "vardir",
#'     family = "gaussian",
#'     warmup = 100L,
#'     samples = 200L,
#'     chains = 1L,
#'     device = "cpu",
#'     print_result = FALSE
#'   )
#'   print(fit)
#' }
#' }
#' @export
hb_area <- function(
  formula,
  data,
  domain = NULL,
  subarea = NULL,
  time = NULL,
  family = c("gaussian", "binomial", "poisson", "beta", "nbinomial", "gamma"),
  spatial = c("none", "besag", "bym2", "bym", "leroux"),
  temporal = c("none", "ar1", "rw1", "iid"),
  st_interaction = c("none", "separable", "domain-specific", "type1", "type2", "type3", "type4"),
  W = NULL,
  vardir = NULL,
  trials = NULL,
  exposure = NULL,
  benchmark = FALSE,
  benchmark_weights = NULL,
  benchmark_target = NULL,
  benchmark_method = c("logit", "optimal", "ratio", "difference"),
  prior_beta = c("normal", "horseshoe"),
  robust = FALSE,
  smooth_vardir = FALSE,
  gvf_method = c("log_linear", "power", "ratio", "loess"),
  warmup = 500L,
  samples = 1000L,
  chains = 2L,
  device = c("auto", "metal", "cuda", "cpu"),
  seed = 42L,
  print_result = TRUE,
  ...
) {
  call_matched <- match.call()
  family <- match.arg(tolower(family), choices = c("gaussian", "binomial", "poisson", "beta", "nbinomial", "gamma"))
  spatial <- match.arg(tolower(spatial), choices = c("none", "besag", "bym2", "bym", "leroux"))
  temporal <- match.arg(tolower(temporal), choices = c("none", "ar1", "rw1", "iid"))
  st_interaction <- match.arg(tolower(st_interaction), choices = c("none", "separable", "domain-specific", "type1", "type2", "type3", "type4"))
  device <- match.arg(tolower(device), choices = c("auto", "metal", "cuda", "cpu"))
  benchmark_method <- match.arg(tolower(benchmark_method), choices = c("logit", "optimal", "ratio", "difference"))
  prior_beta <- match.arg(tolower(prior_beta), choices = c("normal", "horseshoe"))
  gvf_method <- match.arg(tolower(gvf_method), choices = c("log_linear", "power", "ratio", "loess"))

  # Automatically configure JAX backend platform before python initializes
  old_plat <- Sys.getenv("JAX_PLATFORMS", unset = NA)
  on.exit({
    if (is.na(old_plat) || !nzchar(old_plat)) {
      Sys.unsetenv("JAX_PLATFORMS")
    } else {
      Sys.setenv(JAX_PLATFORMS = old_plat)
    }
  }, add = TRUE)

  if (!nzchar(Sys.getenv("JAX_PLATFORMS"))) {
    if (device == "cpu" || (device == "auto" && Sys.info()["sysname"] == "Darwin")) {
      Sys.setenv(JAX_PLATFORMS = "cpu")
    } else if (device == "cuda") {
      Sys.setenv(JAX_PLATFORMS = "cuda,cpu")
    }
  }

  if (!is.data.frame(data)) {
    cli::cli_abort("{.arg data} must be a data frame or tibble.")
  }
  n_obs <- nrow(data)

  # 1. Domain, Sub-Area, and Time parsing
  subarea_raw <- if (!is.null(subarea)) .get_variable(data, subarea) else NULL
  if (is.null(domain)) {
    if (!is.null(subarea_raw)) {
      domain_raw <- rep("Area_1", n_obs)
    } else if (is.null(time)) {
      domain_raw <- seq_len(n_obs)
    } else {
      cli::cli_abort("When {.arg time} is specified, {.arg domain} must also be specified.")
    }
  } else {
    domain_raw <- .get_variable(data, domain)
  }

  is_nested <- !is.null(subarea_raw)
  if (is_nested) {
    u_dom <- length(unique(domain_raw))
    u_sub <- length(unique(subarea_raw))
    # Intelligent hierarchy auto-detection: if domain has more levels than subarea, user inverted them
    if (u_dom > u_sub) {
      cli::cli_alert_info("Hierarchical subarea detected: auto-mapping {.val {u_sub}} major areas and {.val {u_dom}} subareas.")
      tmp <- domain_raw
      domain_raw <- subarea_raw
      subarea_raw <- tmp
    }
    unique_domains <- unique(domain_raw)
    n_domains <- length(unique_domains)
    domain_idx <- as.integer(factor(domain_raw, levels = unique_domains)) - 1L

    unique_subareas <- unique(subarea_raw)
    n_subareas <- length(unique_subareas)
    subarea_idx <- as.integer(factor(subarea_raw, levels = unique_subareas)) - 1L

    domain_vec <- domain_raw
    subarea_vec <- subarea_raw
  } else {
    unique_domains <- unique(domain_raw)
    n_domains <- length(unique_domains)
    domain_idx <- as.integer(factor(domain_raw, levels = unique_domains)) - 1L

    domain_vec <- domain_raw
    subarea_vec <- NULL
    subarea_idx <- NULL
    n_subareas <- 1L
  }

  time_vec <- if (!is.null(time)) .get_variable(data, time) else NULL
  if (temporal != "none" && is.null(time_vec)) {
    cli::cli_abort("When {.code temporal != 'none'}, {.arg time} column must be specified.")
  }
  unique_times <- if (!is.null(time_vec)) sort(unique(time_vec)) else NULL
  n_times <- if (!is.null(unique_times)) length(unique_times) else 1L
  time_idx <- if (!is.null(time_vec)) as.integer(factor(time_vec, levels = unique_times)) - 1L else NULL

  # 2. Design matrix X and Response y
  mf <- stats::model.frame(formula, data, na.action = stats::na.pass)
  y_raw <- as.numeric(stats::model.response(mf))
  X_mat <- stats::model.matrix(formula, data = mf)
  if (anyNA(X_mat)) {
    cli::cli_abort("Covariate matrix {.code X} contains missing values (NA). Please impute or remove rows with NA covariates before fitting.")
  }
  if (any(!is.finite(X_mat))) {
    cli::cli_abort("Covariate matrix {.code X} contains non-finite values (Inf/-Inf).")
  }
  coef_names <- colnames(X_mat)

  # Validate optional variance / trial / exposure arguments
  vardir_vec <- if (!is.null(vardir)) as.numeric(.get_variable(data, vardir)) else NULL
  trials_vec <- if (!is.null(trials)) as.numeric(.get_variable(data, trials)) else NULL
  exposure_vec <- if (!is.null(exposure)) as.numeric(.get_variable(data, exposure)) else NULL

  # 2b. GVF Smoothing for sampling variances (Feature A: Wolter 2007, Otto & Bell 1995)
  gvf_obj <- NULL
  if (isTRUE(smooth_vardir)) {
    if (is.null(vardir_vec)) {
      cli::cli_abort("When {.code smooth_vardir = TRUE}, {.arg vardir} must be provided.")
    }
    n_for_gvf <- NULL
    if (!is.null(trials_vec)) {
      n_for_gvf <- trials_vec
    } else {
      n_candidates <- c("n", "samp_size", "sample_size", "size", "N_sample")
      found_n <- intersect(n_candidates, names(data))
      if (length(found_n) > 0) {
        n_for_gvf <- as.numeric(data[[found_n[1]]])
      }
    }
    cli::cli_alert_info("Applying Generalized Variance Function (GVF) smoothing ({gvf_method})...")
    gvf_obj <- gvf_smooth(y = y_raw, vardir = vardir_vec, n = n_for_gvf, method = gvf_method)
    vardir_vec <- gvf_obj$vardir_smooth
  }

  # Validate response data by likelihood family
  y <- y_raw
  if (family == "gaussian") {
    if (is.null(vardir_vec)) {
      cli::cli_abort("For {.code family = 'gaussian'}, {.arg vardir} must be specified.")
    }
  }

  if (family == "binomial") {
    if (is.null(trials_vec)) {
      cli::cli_abort("For {.code family = 'binomial'}, {.arg trials} must be specified.")
    }
    y_valid <- y[!is.na(y)]
    if (length(y_valid) > 0 && all(y_valid >= 0 & y_valid <= 1) && any(y_valid %% 1 != 0)) {
      cli::cli_alert_info("Response appears to be proportions; converting to integer counts: {.code round(y * trials)}.")
      y <- as.numeric(round(y * trials_vec))
    }
  }

  if (family == "beta") {
    if (is.null(vardir_vec)) {
      cli::cli_abort("For {.code family = 'beta'}, {.arg vardir} must be specified.")
    }
    y_valid <- y[!is.na(y)]
    if (any(y_valid <= 0 | y_valid >= 1)) {
      cli::cli_abort("For {.code family = 'beta'}, response variable must be strictly bounded in (0, 1).")
    }
  }

  if (family == "poisson") {
    y_valid <- y[!is.na(y)]
    if (any(y_valid < 0) || any(y_valid %% 1 != 0)) {
      cli::cli_abort("For {.code family = 'poisson'}, response variable must be non-negative integers.")
    }
  }

  if (family == "gamma") {
    y_valid <- y[!is.na(y)]
    if (any(y_valid <= 0)) {
      cli::cli_abort("For {.code family = 'gamma'}, response variable must be strictly positive (y > 0).")
    }
  }

  if (family == "nbinomial") {
    y_valid <- y[!is.na(y)]
    if (any(y_valid < 0) || any(y_valid %% 1 != 0)) {
      cli::cli_abort("For {.code family = 'nbinomial'}, response variable must be non-negative integers.")
    }
  }

  # 3. Spatial adjacency matrix
  W_obj <- NULL
  if (spatial != "none") {
    W_obj <- .convert_spatial_weights(W, n_domains = n_domains, domain_names = unique_domains)
  }

  # 3b. Benchmark weights parsing
  bm_weights_vec <- NULL
  if (isTRUE(benchmark)) {
    if (!is.null(benchmark_weights)) {
      bm_weights_vec <- as.numeric(.get_variable(data, benchmark_weights))
    } else {
      candidates <- c("weights", "weight", "w", "pop", "population", "pop_weights")
      found <- intersect(candidates, names(data))
      if (length(found) > 0) {
        cli::cli_alert_info("Using column {.val {found[1]}} as benchmark weights.")
        bm_weights_vec <- as.numeric(data[[found[1]]])
      } else {
        cli::cli_abort("When {.code benchmark = TRUE}, {.arg benchmark_weights} must be provided or present in {.arg data}.")
      }
    }
  }

  # 4. Invoke NumPyro Python Backend
  backend <- .get_numpyro_backend(device = device)
  
  cli::cli_alert_info("Running NumPyro NUTS MCMC on GPU/accelerator ({samples} samples, {warmup} warmup, {chains} chains)...")
  t0 <- Sys.time()
  
  fit_py <- backend$fit_numpyro_hb(
    y = y,
    X = X_mat,
    domain_idx = domain_idx,
    time_idx = time_idx,
    subarea_idx = subarea_idx,
    num_subareas = as.integer(n_subareas),
    vardir = vardir_vec,
    trials = trials_vec,
    exposure = exposure_vec,
    D = as.integer(n_domains),
    T = as.integer(n_times),
    family = family,
    spatial = spatial,
    temporal = temporal,
    st_interaction = st_interaction,
    W_adj = if (!is.null(W_obj)) W_obj$adj_mat else NULL,
    scale_factor = if (!is.null(W_obj)) as.numeric(W_obj$scale_factor) else 1.0,
    eig_values = if (!is.null(W_obj)) W_obj$eig_values else NULL,
    eig_vectors = if (!is.null(W_obj)) W_obj$eig_vectors else NULL,
    num_warmup = as.integer(warmup),
    num_samples = as.integer(samples),
    num_chains = as.integer(chains),
    device = device,
    seed = as.integer(seed),
    benchmark = isTRUE(benchmark),
    benchmark_weights = bm_weights_vec,
    benchmark_target = if (!is.null(benchmark_target)) as.numeric(benchmark_target) else NULL,
    benchmark_method = benchmark_method,
    prior_beta = prior_beta,
    robust = isTRUE(robust)
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cli::cli_alert_success("Sampling completed in {round(elapsed, 2)} seconds.")

  # 5. Extract and format outputs
  beta_mean <- as.numeric(fit_py$beta_mean)
  beta_sd <- as.numeric(fit_py$beta_sd)
  beta_ci_lower <- as.numeric(fit_py$beta_ci_lower)
  beta_ci_upper <- as.numeric(fit_py$beta_ci_upper)
  
  estcoef <- data.frame(
    beta = beta_mean,
    std.error = beta_sd,
    zvalue = beta_mean / pmax(beta_sd, 1e-8),
    pvalue = 2 * stats::pnorm(abs(beta_mean / pmax(beta_sd, 1e-8)), lower.tail = FALSE),
    ci_lower = beta_ci_lower,
    ci_upper = beta_ci_upper,
    row.names = coef_names
  )

  if (prior_beta == "horseshoe" && !is.null(fit_py$shrinkage_weights)) {
    sw <- as.numeric(fit_py$shrinkage_weights)
    if (length(sw) == (length(coef_names) - 1)) {
      estcoef$shrinkage_factor <- c(0, round(sw, 4))
    } else if (length(sw) == length(coef_names)) {
      estcoef$shrinkage_factor <- round(sw, 4)
    }
  }

  hb_pred <- as.numeric(fit_py$hb_mean)
  hb_sd <- as.numeric(fit_py$hb_sd)
  hb_ci_lower <- as.numeric(fit_py$hb_ci_lower)
  hb_ci_upper <- as.numeric(fit_py$hb_ci_upper)
  linpred <- as.numeric(fit_py$linpred_mean)
  rand_eff <- as.numeric(fit_py$rand_eff_mean)

  is_benchmarked <- isTRUE(fit_py$benchmarked)
  if (is_benchmarked && !is.null(fit_py$hb_bench_mean)) {
    hb_final <- as.numeric(fit_py$hb_bench_mean)
    sd_final <- as.numeric(fit_py$hb_bench_sd)
    ci_lower_final <- as.numeric(fit_py$hb_ci_lower_bench)
    ci_upper_final <- as.numeric(fit_py$hb_ci_upper_bench)
  } else {
    hb_final <- hb_pred
    sd_final <- hb_sd
    ci_lower_final <- hb_ci_lower
    ci_upper_final <- hb_ci_upper
  }

  df_hb <- data.frame(
    domain = domain_vec,
    y = y_raw,
    hb = hb_final,
    linear_pred = linpred,
    sd = sd_final,
    mse = sd_final^2,
    rse = ifelse(abs(hb_final) < 1e-8, NA_real_, (sd_final / abs(hb_final)) * 100),
    ci_lower = ci_lower_final,
    ci_upper = ci_upper_final,
    random_effect = rand_eff,
    stringsAsFactors = FALSE
  )
  if (is_nested && !is.null(subarea_vec)) {
    df_hb$subarea <- subarea_vec
  }
  if (is_benchmarked) {
    df_hb$hb_unbenchmarked <- hb_pred
    df_hb$sd_unbenchmarked <- hb_sd
  }
  if (!is.null(time_vec)) {
    df_hb$time <- time_vec
  }
  if (!is.null(vardir_vec)) {
    df_hb$vardir <- vardir_vec
  }
  if (!is.null(gvf_obj)) {
    df_hb$vardir_raw <- gvf_obj$vardir_raw
  }
  if (!is.null(trials_vec) && family == "binomial") {
    df_hb$trials <- trials_vec
    df_hb$estimated_total <- hb_pred * trials_vec
  }
  if (!is.null(exposure_vec) && family %in% c("poisson", "nbinomial")) {
    df_hb$exposure <- exposure_vec
    df_hb$estimated_count <- hb_pred * exposure_vec
  }

  hyper_names <- names(fit_py$hyperparameters)
  hyper_vals <- as.numeric(fit_py$hyperparameters)
  hyper_df <- if (length(hyper_names) > 0) {
    data.frame(
      Parameter = hyper_names,
      Estimate = hyper_vals,
      row.names = NULL
    )
  } else {
    data.frame(Parameter = character(0), Estimate = numeric(0))
  }

  goodness <- c(
    DIC = as.numeric(fit_py$dic),
    pD = as.numeric(fit_py$p_dic),
    WAIC = as.numeric(fit_py$waic),
    pWAIC = as.numeric(fit_py$p_waic)
  )

  # Format model string label
  model_label <- paste0("HB-", toupper(family), " (NumPyro GPU)")
  enhancements <- c()
  if (is_nested) enhancements <- c(enhancements, "Nested Sub-Area")
  if (spatial != "none") enhancements <- c(enhancements, toupper(spatial))
  if (temporal != "none") enhancements <- c(enhancements, toupper(temporal))
  if (st_interaction != "none") enhancements <- c(enhancements, paste0("ST:", toupper(st_interaction)))
  if (prior_beta == "horseshoe") enhancements <- c(enhancements, "Horseshoe")
  if (isTRUE(robust)) enhancements <- c(enhancements, "Student-t")
  if (isTRUE(smooth_vardir)) enhancements <- c(enhancements, paste0("GVF:", toupper(gvf_method)))
  if (length(enhancements) > 0) {
    model_label <- paste0(model_label, " [", paste(enhancements, collapse = " + "), "]")
  }

  res <- list(
    df_hb = df_hb,
    hb = df_hb,
    df_eblup = df_hb,
    estcoef = estcoef,
    hyperpar = hyper_df,
    random_effect_var = fit_py$hyperparameters$sigma2_u %||% NULL,
    random_effect_var_time = fit_py$hyperparameters$sigma2_t %||% NULL,
    sigma2_spatial = fit_py$hyperparameters$sigma2_spatial %||% NULL,
    sigma2_iid = fit_py$hyperparameters$sigma2_iid %||% NULL,
    sigma2_subarea = fit_py$hyperparameters$sigma2_subarea %||% NULL,
    icc_nested = fit_py$hyperparameters$icc_nested %||% NULL,
    phi = fit_py$hyperparameters$phi %||% NULL,
    rho_spatial = fit_py$hyperparameters$rho_spatial %||% NULL,
    rho_time = fit_py$hyperparameters$rho_t %||% NULL,
    alpha_dispersion = fit_py$hyperparameters$alpha_dispersion %||% NULL,
    shape_param = fit_py$hyperparameters$shape_param %||% NULL,
    goodness = goodness,
    family = family,
    spatial = spatial,
    temporal = temporal,
    st_interaction = st_interaction,
    level = if (is_nested) "subarea" else "area",
    model = model_label,
    device = fit_py$device_used,
    elapsed_seconds = elapsed,
    convergence = TRUE,
    benchmarked = is_benchmarked,
    benchmark_info = if (is_benchmarked) list(
      target = as.numeric(fit_py$benchmark_target),
      method = as.character(fit_py$benchmark_method),
      weights = bm_weights_vec,
      type = if (is.null(benchmark_target)) "self" else "external"
    ) else NULL,
    prior_beta = prior_beta,
    robust = isTRUE(robust),
    smooth_vardir = isTRUE(smooth_vardir),
    gvf = gvf_obj,
    shrinkage_weights = fit_py$shrinkage_weights,
    subarea = if (is_nested) subarea_vec else NULL,
    is_nested = is_nested,
    data = data,
    call = call_matched
  )
  class(res) <- c("fastsae_hb_area", "fastsae")

  if (print_result) {
    print(res)
  }
  res
}
