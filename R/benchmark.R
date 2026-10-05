#' Benchmark Small Area Estimation Models
#'
#' Calibrates or benchmarks small area estimates to aggregate targets.
#' Supports both \strong{Self-Benchmarking} (internal calibration where the
#' benchmark target is derived directly from the survey sample direct estimates)
#' and \strong{External Benchmarking} (where external official totals from a
#' census, registry, or published national release are provided).
#'
#' @param object A fitted model object of class \code{fastsae_hb_area} or \code{fastsaegpu_merf}.
#' @param ... Additional arguments passed to specific methods.
#'
#' @export
benchmark <- function(object, ...) {
  UseMethod("benchmark")
}

#' @rdname benchmark
#' @export
benchmark_sae <- function(object, ...) {
  benchmark(object, ...)
}

#' @rdname benchmark
#' @export
benchmark.fastsaegpu_merf <- function(object,
                                      target = NULL,
                                      weight = NULL,
                                      method = NULL,
                                      group = NULL,
                                      type = c("mean", "total"),
                                      type_target = c("auto", "internal", "external"),
                                      ...) {
  benchmark.fastsae_hb_area(
    object = object,
    target = target,
    weight = weight,
    method = method,
    group = group,
    type = type,
    type_target = type_target,
    ...
  )
}

#' @rdname benchmark
#' @param target Optional numeric target value (scalar) or named numeric vector
#'   for group-specific targets. If \code{NULL} (the default),
#'   \strong{Self-Benchmarking} is performed using the weighted direct sample
#'   survey estimates.
#' @param weight Numeric vector of domain weights (e.g. population sizes \eqn{N_d}
#'   or domain sample sizes \eqn{n_d}), or character string specifying a column name
#'   in the original data. If \code{NULL}, equal weights (\eqn{w_d = 1}) are used.
#' @param method Character string specifying the benchmarking calibration method:
#'   \itemize{
#'     \item \code{"logit"}: Logit-scale calibration via 1D root-finding. Ensures
#'       all benchmarked estimates are strictly bounded in \eqn{(0, 1)}.
#'       Default for \code{family = "beta"} and \code{"binomial"}.
#'     \item \code{"optimal"}: Constrained quadratic loss minimization weighted by
#'       domain estimation error variance (MSE). Areas with higher uncertainty receive
#'       proportionally larger adjustments. Default for other likelihood families.
#'     \item \code{"ratio"}: Multiplicative proportional adjustment (raking ratio).
#'     \item \code{"difference"}: Additive uniform difference adjustment.
#'   }
#' @param group Optional character string or vector specifying grouping / higher-level
#'   geographic domains (e.g. provinces) for hierarchical benchmarking.
#' @param type Character: \code{"mean"} (target is a rate / proportion / mean) or
#'   \code{"total"} (target is an aggregate population count / sum). Default \code{"mean"}.
#' @param type_target Character: \code{"auto"}, \code{"internal"} (Self-Benchmarking),
#'   or \code{"external"} (External Benchmarking). Default \code{"auto"}.
#'
#' @return A data frame of class \code{c("fastsaegpu_benchmark", "data.frame")} containing:
#'   \item{domain}{Domain identifier.}
#'   \item{group}{Higher-level group identifier (if specified).}
#'   \item{weight}{Domain weight used in aggregation.}
#'   \item{direct}{Original direct sample estimate.}
#'   \item{original}{Original model-based estimate before benchmarking.}
#'   \item{benchmarked}{Calibrated benchmarked estimate.}
#'   \item{adjustment}{Absolute change (\code{benchmarked - original}).}
#'   \item{rel_adjustment_pct}{Percentage relative adjustment.}
#'   \item{target}{Benchmark target value.}
#'
#' @references
#' Battese, G. E., Harter, R. M., & Fuller, W. A. (1988). An error-components model for prediction of county crop areas using survey and satellite data. Journal of the American Statistical Association, 83(401), 28-36.
#'
#' Pfeffermann, D. (2013). New important developments in small area estimation. Statistical Science, 28(1), 40-68.
#'
#' Rao, J. N. K., & Molina, I. (2015). Small Area Estimation (2nd ed.). John Wiley & Sons.
#'
#' You, Y., & Rao, J. N. K. (2002). A pseudo-EBLUP approach to small area estimation with self-benchmarking. Canadian Journal of Statistics, 30(3), 431-446.
#'
#' @examples
#' \donttest{
#' # Fit a Beta SAE model
#' set.seed(123)
#' D <- 20
#' df <- data.frame(
#'   domain = paste0("d", 1:D),
#'   y = stats::rbeta(D, 2, 8),
#'   vardir = rep(0.005, D),
#'   x1 = stats::rnorm(D),
#'   N = sample(1000:5000, D)
#' )
#' fit <- hb_area(y ~ x1, data = df, domain = "domain", vardir = "vardir",
#'                family = "beta", warmup = 50, samples = 50, chains = 1)
#'
#' # 1. Self-Benchmarking (target derived automatically from sample survey)
#' bm_self <- benchmark(fit, weight = "N")
#' print(bm_self)
#'
#' # 2. External Benchmarking (calibrated to official published total 0.22)
#' bm_ext <- benchmark(fit, target = 0.22, weight = "N", method = "logit")
#' print(bm_ext)
#' }
#' @export
benchmark.fastsae_hb_area <- function(object,
                                      target = NULL,
                                      weight = NULL,
                                      method = NULL,
                                      group = NULL,
                                      type = c("mean", "total"),
                                      type_target = c("auto", "internal", "external"),
                                      ...) {
  type <- match.arg(type)
  type_target <- match.arg(type_target)
  call_matched <- match.call()

  df_est <- object$df_hb
  if (is.null(df_est)) {
    cli::cli_abort("The fitted model object does not contain domain estimates ({.code df_hb}).")
  }

  n_domains <- nrow(df_est)
  domain_vec <- df_est$domain %||% paste0("domain_", seq_len(n_domains))
  y_mod <- df_est$hb
  y_dir <- df_est$y %||% y_mod
  mse_vec <- df_est$mse %||% (if (!is.null(df_est$sd)) df_est$sd^2 else rep(1.0, n_domains))
  family <- tolower(object$family %||% "gaussian")

  # Default method selection based on family
  if (is.null(method)) {
    if (family %in% c("beta", "binomial")) {
      method <- "logit"
    } else {
      method <- "optimal"
    }
  } else {
    method <- match.arg(tolower(method), c("logit", "optimal", "ratio", "difference"))
  }

  # Parse weights
  w_vec <- NULL
  if (is.character(weight) && length(weight) == 1) {
    if (!is.null(object$data) && weight %in% names(object$data)) {
      w_vec <- as.numeric(object$data[[weight]])
    } else if (weight %in% names(df_est)) {
      w_vec <- as.numeric(df_est[[weight]])
    } else {
      cli::cli_abort("Weight column {.val {weight}} not found in model data.")
    }
  } else if (is.numeric(weight)) {
    if (length(weight) != n_domains) {
      cli::cli_abort("Length of {.arg weight} ({length(weight)}) does not match number of domains ({n_domains}).")
    }
    w_vec <- as.numeric(weight)
  } else if (is.null(weight)) {
    w_vec <- rep(1.0, n_domains)
  }

  if (any(is.na(w_vec)) || any(w_vec <= 0)) {
    cli::cli_abort("All {.arg weight} values must be strictly positive numeric values.")
  }

  # Parse grouping variable
  group_vec <- NULL
  if (is.character(group) && length(group) == 1) {
    if (!is.null(object$data) && group %in% names(object$data)) {
      group_vec <- as.character(object$data[[group]])
    } else if (group %in% names(df_est)) {
      group_vec <- as.character(df_est[[group]])
    } else {
      cli::cli_abort("Grouping column {.val {group}} not found in model data.")
    }
  } else if (!is.null(group)) {
    if (length(group) != n_domains) {
      cli::cli_abort("Length of {.arg group} ({length(group)}) does not match number of domains ({n_domains}).")
    }
    group_vec <- as.character(group)
  } else {
    group_vec <- rep("All", n_domains)
  }

  unique_groups <- unique(group_vec)
  is_hierarchical <- length(unique_groups) > 1

  # Determine target mode: self-benchmarking vs external
  if (is.null(target) || type_target == "internal") {
    target_mode <- "Self-Benchmarking (Internal Direct Survey Total)"
    # Derive target internally from direct survey estimate (skip non-sampled NA)
    target_map <- stats::setNames(numeric(length(unique_groups)), unique_groups)
    for (g in unique_groups) {
      idx_g <- which(group_vec == g)
      valid_g <- !is.na(y_dir[idx_g])
      if (!any(valid_g)) {
        cli::cli_abort("No sampled domains (non-NA direct estimates) in group {.val {g}} for self-benchmarking.")
      }
      w_g <- w_vec[idx_g][valid_g]
      y_dir_g <- y_dir[idx_g][valid_g]
      if (type == "mean") {
        target_map[g] <- sum(w_g * y_dir_g) / sum(w_g)
      } else {
        target_map[g] <- sum(w_g * y_dir_g)
      }
    }
  } else {
    target_mode <- "External Benchmarking (Known/Official Target)"
    if (is.numeric(target)) {
      if (length(target) == 1) {
        if (is_hierarchical) {
          # Two-stage national target calibration across groups
          # Step 1: Calculate initial group direct totals
          g_direct <- vapply(unique_groups, function(g) {
            idx <- which(group_vec == g)
            if (type == "mean") sum(w_vec[idx] * y_dir[idx]) / sum(w_vec[idx])
            else sum(w_vec[idx] * y_dir[idx])
          }, numeric(1))
          g_weights <- vapply(unique_groups, function(g) sum(w_vec[group_vec == g]), numeric(1))
          w_g_share <- g_weights / sum(g_weights)
          current_nat <- if (type == "mean") sum(w_g_share * g_direct) else sum(g_direct)
          ratio_nat <- as.numeric(target) / current_nat
          target_map <- g_direct * ratio_nat
        } else {
          target_map <- stats::setNames(as.numeric(target), unique_groups)
        }
      } else if (length(target) == length(unique_groups)) {
        if (!is.null(names(target)) && all(unique_groups %in% names(target))) {
          target_map <- target[unique_groups]
        } else {
          target_map <- stats::setNames(as.numeric(target), unique_groups)
        }
      } else {
        cli::cli_abort("Length of {.arg target} ({length(target)}) does not match number of groups ({length(unique_groups)}).")
      }
    } else {
      cli::cli_abort("{.arg target} must be a numeric scalar or named vector.")
    }
  }

  # Calibration execution per group
  y_bench <- numeric(n_domains)

  for (g in unique_groups) {
    idx_g <- which(group_vec == g)
    y_g <- y_mod[idx_g]
    w_g <- w_vec[idx_g]
    mse_g <- mse_vec[idx_g]
    T_g <- target_map[g]

    w_share <- if (type == "mean") w_g / sum(w_g) else w_g
    agg_initial <- sum(w_share * y_g)
    delta <- T_g - agg_initial

    if (method == "logit") {
      if (any(y_g <= 0 | y_g >= 1)) {
        cli::cli_abort("For {.code method = 'logit'}, estimates must be strictly within (0, 1).")
      }
      if (type == "mean" && (T_g <= 0 || T_g >= 1)) {
        cli::cli_abort("For logit benchmarking with {.code type = 'mean'}, target must be strictly in (0, 1).")
      }
      logit_y <- stats::qlogis(pmin(pmax(y_g, 1e-6), 1 - 1e-6))
      f_obj <- function(alpha) {
        sum(w_share * stats::plogis(logit_y + alpha)) - T_g
      }
      # Robust bracket search for root finding
      low_b <- -25.0
      upp_b <- 25.0
      f_low <- f_obj(low_b)
      f_upp <- f_obj(upp_b)
      if (f_low * f_upp > 0) {
        low_b <- -100.0
        upp_b <- 100.0
      }
      sol <- stats::uniroot(f_obj, interval = c(low_b, upp_b), tol = 1e-10)
      y_bench[idx_g] <- stats::plogis(logit_y + sol$root)

    } else if (method == "optimal") {
      # Analytical solution to quadratic loss constrained by domain MSE
      inv_prec <- pmax(mse_g, 1e-8)
      denom <- sum(w_share * inv_prec)
      if (abs(denom) < 1e-12) {
        y_bench[idx_g] <- y_g + delta
      } else {
        y_bench[idx_g] <- y_g + (inv_prec / denom) * delta
      }
      if (family %in% c("beta", "binomial")) {
        y_bench[idx_g] <- pmin(pmax(y_bench[idx_g], 1e-5), 1 - 1e-5)
      }

    } else if (method == "ratio") {
      if (abs(agg_initial) < .Machine$double.eps) {
        cli::cli_abort("Initial weighted aggregation for group {.val {g}} is zero; ratio adjustment undefined.")
      }
      ratio_factor <- T_g / agg_initial
      y_bench[idx_g] <- y_g * ratio_factor
      if (family %in% c("beta", "binomial")) {
        y_bench[idx_g] <- pmin(pmax(y_bench[idx_g], 1e-5), 1 - 1e-5)
      }

    } else if (method == "difference") {
      y_bench[idx_g] <- y_g + delta
      if (family %in% c("beta", "binomial")) {
        y_bench[idx_g] <- pmin(pmax(y_bench[idx_g], 1e-5), 1 - 1e-5)
      }
    }
  }

  # Build result data frame
  res_df <- data.frame(
    domain = domain_vec,
    stringsAsFactors = FALSE
  )
  if (is_hierarchical) {
    res_df$group <- group_vec
  }
  res_df$weight <- w_vec
  res_df$direct <- y_dir
  res_df$original <- y_mod
  res_df$benchmarked <- y_bench
  res_df$adjustment <- y_bench - y_mod
  res_df$rel_adjustment_pct <- ifelse(abs(y_mod) > 1e-9, (res_df$adjustment / y_mod) * 100, NA_real_)
  res_df$target <- target_map[group_vec]

  # Verification of calibration accuracy
  verification <- list()
  for (g in unique_groups) {
    idx_g <- which(group_vec == g)
    w_calc <- if (type == "mean") w_vec[idx_g] / sum(w_vec[idx_g]) else w_vec[idx_g]
    orig_agg <- sum(w_calc * y_mod[idx_g])
    bench_agg <- sum(w_calc * y_bench[idx_g])
    t_val <- target_map[g]
    verification[[g]] <- list(
      group = g,
      target = t_val,
      original_sum = orig_agg,
      benchmarked_sum = bench_agg,
      discrepancy = abs(bench_agg - t_val)
    )
  }

  attr(res_df, "target_mode") <- target_mode
  attr(res_df, "method") <- method
  attr(res_df, "type") <- type
  attr(res_df, "hierarchical") <- is_hierarchical
  attr(res_df, "verification") <- verification
  attr(res_df, "family") <- family
  attr(res_df, "call") <- call_matched
  class(res_df) <- c("fastsaegpu_benchmark", "data.frame")

  res_df
}
