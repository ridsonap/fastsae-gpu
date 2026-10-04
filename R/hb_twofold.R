#' Hierarchical Bayes Twofold Subarea Small Area Estimation (GPU NumPyro)
#'
#' Fits the twofold subarea-level model of Torabi & Rao (2014) with
#' Hierarchical Bayes using the GPU NumPyro backend:
#' \deqn{y_{jk} \mid \theta_{jk} \sim N(\theta_{jk}, \psi_{jk})}{y_jk | theta_jk ~ N(theta_jk, psi_jk)}
#' \deqn{\theta_{jk} = \mathbf{x}_{jk}^\top \boldsymbol{\beta} + v_j + u_{jk}}{theta_jk = x_jk' beta + v_j + u_jk}
#' where \eqn{v_j \sim N(0, \sigma_v^2)} is the area effect (e.g. province) and
#' \eqn{u_{jk} \sim N(0, \sigma_u^2)} is the subarea effect (e.g. district).
#' Area means are aggregated simultaneously as
#' \eqn{\theta_{j.} = \sum_k W_{jk} \theta_{jk}} with normalized weights
#' \eqn{W_{jk} = w_{jk} / \sum_k w_{jk}} (Rao & Molina 2015, Ch. 8;
#' Mohadjer et al. 2007; Erciulescu et al. 2019).
#'
#' Non-sampled subareas (\code{y = NA}) are predicted from the linking model
#' only (Torabi & Rao 2014, Sec. 4), matching \code{saeHB.twofold::NormalTF}.
#'
#' @param formula Fixed effects formula (e.g. \code{y ~ x1 + x2}).
#' @param data Data frame with one row per subarea.
#' @param area Major-area identifier column or vector (m areas, e.g. province code).
#' @param subarea Optional subarea identifier column or vector. If \code{NULL},
#'   row indices are used (each row is one subarea).
#' @param vardir Known subarea sampling variances \eqn{\psi_{jk}} (column or vector).
#'   May be \code{NA} for non-sampled rows where \code{y} is \code{NA}.
#' @param weight Aggregation weights \eqn{w_{jk}} (column or vector, e.g. population
#'   shares). If \code{NULL} (default), equal weights per subarea are used.
#' @param warmup Number of MCMC warmup iterations (default 500).
#' @param samples Number of MCMC post-warmup samples (default 1000).
#' @param chains Number of parallel MCMC chains (default 2).
#' @param device Target hardware: "auto", "metal", "cuda", or "cpu".
#' @param seed Random seed (default 42).
#' @param print_result Logical: print summary upon completion (default TRUE).
#' @param ... Additional arguments (currently unused).
#' @return An object of class \code{c("fastsae_hb_twofold", "fastsae_hb_area", "fastsae_hb", "fastsae")}
#'   with the same structure as \code{fastsae::hb_area()} /
#'   \code{fastsae::hb_twofold()}:
#'   \code{df_hb} (subarea level: \code{domain} = major area, \code{subarea},
#'   \code{y}, \code{hb}, \code{linear_pred}, \code{vardir}, \code{sd},
#'   \code{mse}, \code{rse}, \code{ci_lower}, \code{ci_upper},
#'   \code{random_effect_area}, \code{random_effect_subarea}, plus
#'   \code{random_effect}/\code{eblup}/\code{weight}/\code{sampled} extras for
#'   \code{benchmark()}/\code{fitted()} interop),
#'   \code{df_subarea} (alias of \code{df_hb}),
#'   \code{df_area} (area level: \code{domain}, \code{hb_area},
#'   \code{sd_area}, \code{mse_area}, \code{rse_area}, \code{ci_lower_area},
#'   \code{ci_upper_area}, \code{n_subareas}, plus \code{y}/\code{direct} and
#'   \code{hb}/\code{Mean} aliases),
#'   \code{estcoef}, \code{hyperpar},
#'   \code{random_effect_var} (named \code{c(sigma2_v, sigma2_u)}),
#'   \code{phi}, \code{goodness}, \code{family}, \code{spatial},
#'   \code{temporal}, \code{st_interaction}, \code{level} (\code{"subarea"}),
#'   \code{model}, \code{method}, \code{convergence}, \code{fit}, \code{call},
#'   plus \code{hb}/\code{df_eblup} aliases (as in \code{hb_area()}),
#'   \code{Est_sub}/\code{Est_area} (as in \code{saeHB.twofold::NormalTF}:
#'   \code{Mean}/\code{SD}/\code{CV}/\code{MSE} aliases),
#'   \code{coefficient} (alias of \code{estcoef}) and \code{refVar} (alias of
#'   \code{hyperpar}), and GPU extras (\code{device}, \code{elapsed_seconds},
#'   \code{icc_nested}, \code{sigma2_subarea}, ...).
#' @references
#'   Torabi, M., & Rao, J. N. K. (2014). On small area estimation under a sub-area
#'   level model. \emph{Journal of Multivariate Analysis}, 127, 36-55.
#'   \doi{10.1016/j.jmva.2014.02.001}
#'
#'   Rao, J. N. K., & Molina, I. (2015). \emph{Small Area Estimation} (2nd ed.).
#'   Wiley, Chapter 8.
#'
#'   Mohadjer, L. K., Rao, J. N. K., Liu, B., Krenzke, T., & Van de Kerckhove, W. (2007).
#'   Hierarchical Bayes small area estimates of adult literacy using unmatched sampling
#'   and linking models. \emph{Proc. ASA Survey Research Methods Section}.
#'
#'   Erciulescu, A. L., Cruze, N. B., & Nandram, B. (2019). Model-based county level crop
#'   estimates incorporating auxiliary sources of information. \emph{J. R. Stat. Soc. A},
#'   182, 283-303. \doi{10.1111/rssa.12390}
#' @examples
#' \donttest{
#' if (check_numpyro_available()) {
#'   set.seed(42)
#'   m <- 4; k <- 5; n <- m * k
#'   area <- rep(paste0("A", 1:m), each = k)
#'   x <- rnorm(n); vardir <- rep(0.04, n)
#'   y <- 1.5 + 0.8 * x + rnorm(n, sd = sqrt(vardir))
#'   df <- data.frame(y = y, x = x, vardir = vardir, area = area)
#'   fit <- hb_twofold(y ~ x, data = df, area = "area", vardir = "vardir",
#'     warmup = 100L, samples = 200L, chains = 1L, device = "cpu",
#'     print_result = FALSE)
#'   print(fit)
#' }
#' }
#' @export
hb_twofold <- function(
  formula,
  data,
  area,
  subarea = NULL,
  vardir,
  weight = NULL,
  warmup = 500L,
  samples = 1000L,
  chains = 2L,
  device = c("auto", "metal", "cuda", "cpu"),
  seed = 42L,
  print_result = TRUE,
  ...
) {
  call_matched <- match.call()
  device <- match.arg(tolower(device), choices = c("auto", "metal", "cuda", "cpu"))

  old_plat <- Sys.getenv("JAX_PLATFORMS", unset = NA)
  on.exit({
    if (is.na(old_plat)) Sys.unsetenv("JAX_PLATFORMS") else Sys.setenv(JAX_PLATFORMS = old_plat)
  }, add = TRUE)
  if (device == "cpu" || (device == "auto" && Sys.info()["sysname"] == "Darwin")) {
    if (Sys.getenv("JAX_PLATFORMS") == "") Sys.setenv(JAX_PLATFORMS = "cpu")
  } else if (device == "cuda") {
    if (Sys.getenv("JAX_PLATFORMS") == "") Sys.setenv(JAX_PLATFORMS = "cuda,cpu")
  }

  if (!is.data.frame(data)) cli::cli_abort("{.arg data} must be a data frame.")
  n_obs <- nrow(data)

  area_raw <- .get_variable(data, area)
  sub_raw <- if (!is.null(subarea)) .get_variable(data, subarea) else paste0("sub_", seq_len(n_obs))
  # ponytail: gaussian-only twofold per Torabi & Rao 2014; add family= when non-normal twofold needed.
  if (length(area_raw) != n_obs || length(sub_raw) != n_obs) {
    cli::cli_abort("{.arg area} and {.arg subarea} must match rows of {.arg data}.")
  }
  if (any(is.na(area_raw))) cli::cli_abort("{.arg area} must not contain NA.")

  mf <- stats::model.frame(formula, data, na.action = stats::na.pass)
  y_raw <- as.numeric(stats::model.response(mf))
  X_mat <- stats::model.matrix(formula, data = mf)
  coef_names <- colnames(X_mat)

  psi <- as.numeric(.get_variable(data, vardir))
  if (length(psi) != n_obs) cli::cli_abort("{.arg vardir} must match rows of {.arg data}.")
  sampled <- !is.na(y_raw)
  if (!any(sampled)) cli::cli_abort("At least one sampled subarea (non-NA {.arg y}) is required.")
  if (any(!is.na(psi[sampled]) & psi[sampled] <= 0)) cli::cli_abort("{.arg vardir} must be strictly positive for sampled subareas.")
  if (any(sampled & is.na(psi))) cli::cli_abort("Sampled rows must not have NA {.arg vardir}.")

  w_raw <- if (!is.null(weight)) as.numeric(.get_variable(data, weight)) else rep(1.0, n_obs)
  if (any(is.na(w_raw)) || any(w_raw <= 0)) cli::cli_abort("{.arg weight} must be strictly positive and non-missing.")

  uniq_area <- unique(area_raw)
  D <- length(uniq_area)
  domain_idx <- as.integer(factor(area_raw, levels = uniq_area)) - 1L
  uniq_sub <- unique(sub_raw)
  sub_idx <- as.integer(factor(sub_raw, levels = uniq_sub)) - 1L

  backend <- .get_numpyro_backend(device = device)
  cli::cli_alert_info("Running twofold HB (Torabi & Rao 2014) on GPU/accelerator ({samples} samples, {warmup} warmup, {chains} chains)...")
  t0 <- Sys.time()
  fit_py <- backend$fit_numpyro_hb(
    y = y_raw,
    X = X_mat,
    domain_idx = domain_idx,
    time_idx = NULL,
    subarea_idx = sub_idx,
    num_subareas = as.integer(length(uniq_sub)),
    vardir = psi,
    trials = NULL,
    exposure = NULL,
    D = as.integer(D),
    T = 1L,
    family = "gaussian",
    spatial = "none",
    temporal = "none",
    st_interaction = "none",
    W_adj = NULL,
    scale_factor = 1.0,
    num_warmup = as.integer(warmup),
    num_samples = as.integer(samples),
    num_chains = as.integer(chains),
    device = device,
    seed = as.integer(seed),
    benchmark = FALSE,
    benchmark_weights = NULL,
    benchmark_target = NULL,
    benchmark_method = "optimal",
    prior_beta = "normal",
    robust = FALSE,
    twofold_weights = w_raw
  )
  elapsed <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
  cli::cli_alert_success("Sampling completed in {round(elapsed, 2)} seconds.")

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

  hb <- as.numeric(fit_py$hb_mean)
  sdv <- as.numeric(fit_py$hb_sd)
  linpred <- as.numeric(fit_py$linpred_mean)
  rand_eff <- as.numeric(fit_py$rand_eff_mean)
  re_area <- tryCatch(as.numeric(fit_py$rand_eff_area_mean),
    error = function(e) NULL)
  re_sub <- tryCatch(as.numeric(fit_py$rand_eff_subarea_mean),
    error = function(e) NULL)
  if (is.null(re_area) || length(re_area) != n_obs) re_area <- rep(NA_real_, n_obs)
  if (is.null(re_sub) || length(re_sub) != n_obs) re_sub <- rand_eff - re_area
  # Same columns as fastsae::hb_twofold df_hb (domain = major area, subarea,
  # y, hb, linear_pred, vardir, sd, mse, rse, ci_lower, ci_upper,
  # random_effect_area, random_effect_subarea); hb_area extras appended after.
  df_hb <- data.frame(
    domain = area_raw,
    subarea = sub_raw,
    y = y_raw,
    hb = hb,
    linear_pred = linpred,
    vardir = psi,
    sd = sdv,
    mse = sdv^2,
    rse = ifelse(abs(hb) < 1e-8, NA_real_, sdv / abs(hb) * 100),
    ci_lower = as.numeric(fit_py$hb_ci_lower),
    ci_upper = as.numeric(fit_py$hb_ci_upper),
    random_effect_area = re_area,
    random_effect_subarea = re_sub,
    stringsAsFactors = FALSE
  )
  df_hb$random_effect <- rand_eff
  df_hb$eblup <- df_hb$hb
  df_hb$weight <- w_raw
  df_hb$sampled <- sampled

  # Area-level aggregation from full posterior draws (Python); fallback to weighted mean.
  # Same columns as fastsae::hb_twofold df_area.
  a_mean <- tryCatch(as.numeric(fit_py$area_mean), error = function(e) NULL)
  a_sd <- tryCatch(as.numeric(fit_py$area_sd), error = function(e) NULL)
  a_lo <- tryCatch(as.numeric(fit_py$area_ci_lower), error = function(e) NULL)
  a_hi <- tryCatch(as.numeric(fit_py$area_ci_upper), error = function(e) NULL)
  if (is.null(a_mean) || length(a_mean) != D) {
    a_mean <- vapply(uniq_area, function(a) {
      i <- which(area_raw == a)
      stats::weighted.mean(hb[i], w_raw[i])
    }, numeric(1))
    a_sd <- vapply(uniq_area, function(a) {
      i <- which(area_raw == a)
      sqrt(sum((w_raw[i] / sum(w_raw[i]))^2 * sdv[i]^2))
    }, numeric(1))
    a_lo <- a_mean - 1.96 * a_sd
    a_hi <- a_mean + 1.96 * a_sd
  }
  a_dir <- vapply(uniq_area, function(a) {
    i <- which(area_raw == a & sampled)
    if (!length(i)) NA_real_ else stats::weighted.mean(y_raw[i], w_raw[i])
  }, numeric(1))
  # Same columns as fastsae::hb_twofold df_area (y = NA: no direct area mean).
  Est_area <- data.frame(
    domain = uniq_area,
    hb_area = as.numeric(a_mean),
    sd_area = as.numeric(a_sd),
    mse_area = as.numeric(a_sd)^2,
    rse_area = ifelse(abs(a_mean) < 1e-8, NA_real_, as.numeric(a_sd) / abs(as.numeric(a_mean)) * 100),
    ci_lower_area = as.numeric(a_lo),
    ci_upper_area = as.numeric(a_hi),
    n_subareas = as.integer(table(factor(area_raw, levels = uniq_area))),
    stringsAsFactors = FALSE
  )
  Est_area$y <- Est_area$direct <- as.numeric(a_dir)
  Est_area$hb <- Est_area$Mean <- Est_area$hb_area
  Est_area$sd <- Est_area$SD <- Est_area$sd_area
  Est_area$mse <- Est_area$MSE <- Est_area$mse_area
  Est_area$rse <- Est_area$CV <- Est_area$rse_area
  Est_area$ci_lower <- Est_area$ci_lower_area
  Est_area$ci_upper <- Est_area$ci_upper_area
  Est_area$eblup <- Est_area$hb
  Est_area$weight <- vapply(uniq_area, function(a) sum(w_raw[area_raw == a]), numeric(1))
  Est_area$area <- uniq_area

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

  # saeHB.twofold::NormalTF aliases (Mean/SD/CV/MSE) layered on top of df_hb columns
  Est_sub <- df_hb
  Est_sub$Mean <- Est_sub$hb
  Est_sub$SD <- Est_sub$sd
  Est_sub$MSE <- Est_sub$mse
  Est_sub$CV <- Est_sub$rse

  s2_area <- fit_py$hyperparameters$sigma2_u %||% NA_real_
  s2_sub <- fit_py$hyperparameters$sigma2_subarea %||% NA_real_
  # ponytail: named vector like fastsae::hb_twofold (sigma2_v=area, sigma2_u=subarea)
  rev_named <- c(sigma2_v = as.numeric(s2_area), sigma2_u = as.numeric(s2_sub))

  # Same top-level order as fastsae::hb_twofold()/hb_area(): df_hb, df_subarea,
  # df_area, hb/df_eblup compat, estcoef, hyperpar, random_effect_var, phi,
  # rho, goodness, family, spatial, temporal, st_interaction, level, model,
  # method, convergence, fit, call — gpu/NormalTF extras appended after.
  res <- list(
    df_hb = df_hb,
    df_subarea = df_hb,
    df_area = Est_area,
    hb = df_hb,
    df_eblup = df_hb,
    estcoef = estcoef,
    hyperpar = hyper_df,
    random_effect_var = rev_named,
    random_effect_var_time = NULL,
    phi = fit_py$hyperparameters$phi %||% NULL,
    rho = NULL,
    rho_time = NULL,
    goodness = goodness,
    family = "gaussian",
    spatial = "none",
    temporal = "none",
    st_interaction = "none",
    level = "subarea",
    model = "HB-TWOFOLD-GAUSSIAN (Non-spatial) [NumPyro GPU]",
    method = "NumPyro NUTS (GPU)",
    convergence = TRUE,
    fit = fit_py,
    call = call_matched,
    Est_sub = Est_sub,
    Est_area = Est_area,
    coefficient = estcoef,
    refVar = hyper_df,
    self_benchmark = FALSE,
    benchmark_summary = NULL,
    device = fit_py$device_used,
    elapsed_seconds = elapsed,
    benchmarked = FALSE,
    benchmark_info = NULL,
    prior_beta = "normal",
    robust = FALSE,
    smooth_vardir = FALSE,
    gvf = NULL,
    shrinkage_weights = NULL,
    sigma2_spatial = NULL,
    sigma2_iid = NULL,
    sigma2_subarea = fit_py$hyperparameters$sigma2_subarea %||% NULL,
    sigma2_v = fit_py$hyperparameters$sigma2_u %||% NULL,
    icc_nested = fit_py$hyperparameters$icc_nested %||% NULL,
    rho_spatial = NULL,
    alpha_dispersion = NULL,
    shape_param = NULL,
    subarea = sub_raw,
    is_nested = TRUE,
    area = area_raw,
    weight = w_raw,
    W = NULL,
    data = data,
    formula = formula
  )
  class(res) <- c("fastsae_hb_twofold", "fastsae_hb_area", "fastsae_hb", "fastsae")
  if (print_result) print(res)
  res
}

#' @rdname hb_twofold
#' @param x An object of class \code{fastsae_hb_twofold}.
#' @export
print.fastsae_hb_twofold <- function(x, ...) {
  cli::cli_h1("Hierarchical Bayes Twofold Subarea SAE (Torabi & Rao 2014, GPU)")
  if (!is.null(x$call)) cli::cli_text("{.strong Call}: {deparse(x$call)}")
  cli::cli_text("{.strong Backend}: NumPyro / JAX on {.val {x$device}}")
  cli::cli_text("{.strong Areas}: {nrow(x$Est_area)} | {.strong Subareas}: {nrow(x$Est_sub)} ({sum(!x$Est_sub$sampled)} non-sampled)")
  if (!is.null(x$icc_nested)) cli::cli_text("{.strong ICC}: {round(x$icc_nested, 4)}")
  if (!is.null(x$estcoef)) {
    cli::cli_h2("Regression Coefficients:")
    cols <- intersect(c("beta", "std.error", "zvalue", "pvalue"), names(x$estcoef))
    stats::printCoefmat(as.matrix(x$estcoef[, cols, drop = FALSE]), signif.stars = TRUE, ...)
  }
  if (!is.null(x$hyperpar) && nrow(x$hyperpar) > 0) {
    cli::cli_h2("Variance Components (refVar):")
    print(x$hyperpar, row.names = FALSE)
  }
  if (!is.null(x$Est_area)) {
    cli::cli_h2("Area Estimates (first 6):")
    print(utils::head(x$Est_area, 6), ...)
  }
  if (!is.null(x$Est_sub)) {
    cli::cli_h2("Subarea Estimates (first 6):")
    print(utils::head(x$Est_sub, 6), ...)
  }
  invisible(x)
}
