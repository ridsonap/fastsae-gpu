#' Hierarchical Bayes for Area-Level Small Area Estimation with GPU NumPyro Backend
#'
#' @param formula Fixed effects model formula (e.g., y ~ x1 + x2).
#' @param data Data frame containing area/domain observations.
#' @param domain Domain identifier column or vector.
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
#' @param warmup Number of MCMC warmup iterations (default 500).
#' @param samples Number of MCMC post-warmup samples (default 1000).
#' @param chains Number of parallel MCMC chains on GPU (default 2).
#' @param device Target hardware: "auto", "metal" (Apple Silicon), "cuda" (NVIDIA), or "cpu".
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
  time = NULL,
  family = c("gaussian", "binomial", "poisson", "beta", "nbinomial", "gamma"),
  spatial = c("none", "besag", "bym2", "bym", "leroux"),
  temporal = c("none", "ar1", "rw1", "iid"),
  st_interaction = c("none", "separable", "domain-specific", "type1", "type2", "type3", "type4"),
  W = NULL,
  vardir = NULL,
  trials = NULL,
  exposure = NULL,
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

  # Automatically configure JAX backend platform before python initializes
  old_plat <- Sys.getenv("JAX_PLATFORMS", unset = NA)
  on.exit({
    if (is.na(old_plat)) {
      Sys.unsetenv("JAX_PLATFORMS")
    } else {
      Sys.setenv(JAX_PLATFORMS = old_plat)
    }
  }, add = TRUE)

  if (device == "cpu" || (device == "auto" && Sys.info()["sysname"] == "Darwin")) {
    if (Sys.getenv("JAX_PLATFORMS") == "") {
      Sys.setenv(JAX_PLATFORMS = "cpu")
    }
  } else if (device == "cuda") {
    if (Sys.getenv("JAX_PLATFORMS") == "") {
      Sys.setenv(JAX_PLATFORMS = "cuda,cpu")
    }
  }

  if (!is.data.frame(data)) {
    cli::cli_abort("{.arg data} must be a data frame or tibble.")
  }
  n_obs <- nrow(data)

  # 1. Domain and Time parsing
  if (is.null(domain)) {
    if (is.null(time)) {
      domain_vec <- seq_len(n_obs)
    } else {
      cli::cli_abort("When {.arg time} is specified, {.arg domain} must also be specified.")
    }
  } else {
    domain_vec <- .get_variable(data, domain)
  }
  unique_domains <- unique(domain_vec)
  n_domains <- length(unique_domains)
  domain_idx <- as.integer(factor(domain_vec, levels = unique_domains)) - 1L

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
  coef_names <- colnames(X_mat)

  # Validate optional variance / trial / exposure arguments
  vardir_vec <- if (!is.null(vardir)) as.numeric(.get_variable(data, vardir)) else NULL
  trials_vec <- if (!is.null(trials)) as.numeric(.get_variable(data, trials)) else NULL
  exposure_vec <- if (!is.null(exposure)) as.numeric(.get_variable(data, exposure)) else NULL

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

  # 4. Invoke NumPyro Python Backend
  backend <- .get_numpyro_backend(device = device)
  
  cli::cli_alert_info("Running NumPyro NUTS MCMC on GPU/accelerator ({samples} samples, {warmup} warmup, {chains} chains)...")
  t0 <- Sys.time()
  
  fit_py <- backend$fit_numpyro_hb(
    y = y,
    X = X_mat,
    domain_idx = domain_idx,
    time_idx = time_idx,
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
    num_warmup = as.integer(warmup),
    num_samples = as.integer(samples),
    num_chains = as.integer(chains),
    device = device,
    seed = as.integer(seed)
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

  hb_pred <- as.numeric(fit_py$hb_mean)
  hb_sd <- as.numeric(fit_py$hb_sd)
  hb_ci_lower <- as.numeric(fit_py$hb_ci_lower)
  hb_ci_upper <- as.numeric(fit_py$hb_ci_upper)
  linpred <- as.numeric(fit_py$linpred_mean)
  rand_eff <- as.numeric(fit_py$rand_eff_mean)

  df_hb <- data.frame(
    domain = domain_vec,
    y = y_raw,
    hb = hb_pred,
    linear_pred = linpred,
    sd = hb_sd,
    mse = hb_sd^2,
    rse = ifelse(abs(hb_pred) < 1e-8, NA_real_, (hb_sd / abs(hb_pred)) * 100),
    ci_lower = hb_ci_lower,
    ci_upper = hb_ci_upper,
    random_effect = rand_eff,
    stringsAsFactors = FALSE
  )
  if (!is.null(time_vec)) {
    df_hb$time <- time_vec
  }
  if (!is.null(vardir_vec)) {
    df_hb$vardir <- vardir_vec
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
  if (spatial != "none" || temporal != "none") {
    comps <- c()
    if (spatial != "none") comps <- c(comps, toupper(spatial))
    if (temporal != "none") comps <- c(comps, toupper(temporal))
    if (st_interaction != "none") comps <- c(comps, paste0("ST:", toupper(st_interaction)))
    model_label <- paste0(model_label, " [", paste(comps, collapse = " + "), "]")
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
    level = "area",
    model = model_label,
    device = fit_py$device_used,
    elapsed_seconds = elapsed,
    convergence = TRUE,
    data = data,
    call = call_matched
  )
  class(res) <- c("fastsae_hb_area", "fastsae")

  if (print_result) {
    print(res)
  }
  res
}
