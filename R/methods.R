`%||%` <- function(x, y) if (is.null(x)) y else x

#' Methods for fastsae_hb_area objects
#'
#' Methods for extracting components and summarizing fitted \code{fastsae_hb_area} objects.
#'
#' @param x,object An object of class \code{fastsae_hb_area} or \code{summary.fastsae_hb_area}.
#' @param ... Additional arguments passed to specific methods.
#' @return
#' \itemize{
#'   \item \code{print}: Invisibly returns the object \code{x}.
#'   \item \code{summary}: An object of class \code{summary.fastsae_hb_area}.
#'   \item \code{coef}: Named vector of regression coefficients.
#'   \item \code{fitted}: Numeric vector of fitted values.
#'   \item \code{residuals}: Numeric vector of residuals.
#' }
#' @examples
#' df_hb <- data.frame(
#'   domain = paste0("d", 1:5),
#'   y = c(1.2, 2.3, 1.8, 3.1, 2.5),
#'   hb = c(1.1, 2.1, 1.9, 2.9, 2.6),
#'   stringsAsFactors = FALSE
#' )
#' estcoef <- data.frame(
#'   beta = c(1.0, 0.5),
#'   std.error = c(0.1, 0.05),
#'   zvalue = c(10, 10),
#'   pvalue = c(1e-4, 1e-4),
#'   row.names = c("(Intercept)", "x")
#' )
#' obj <- structure(
#'   list(
#'     df_hb = df_hb,
#'     estcoef = estcoef,
#'     family = "gaussian",
#'     spatial = "none",
#'     temporal = "none",
#'     st_interaction = "none",
#'     device = "cpu"
#'   ),
#'   class = c("fastsae_hb_area", "fastsae")
#' )
#' print(obj)
#' summary(obj)
#' coef(obj)
#' fitted(obj)
#' residuals(obj)
#' @name fastsae_hb_area-methods
NULL

#' @rdname fastsae_hb_area-methods
#' @export
print.fastsae_hb_area <- function(x, ...) {
  cli::cli_h1("GPU-Accelerated Hierarchical Bayesian Small Area Estimation")
  if (!is.null(x$call)) {
    cli::cli_text("{.strong Call}: {deparse(x$call)}")
  }
  cli::cli_text("{.strong Backend}: NumPyro / JAX on {.val {x$device}}")
  cli::cli_text("{.strong Family}: {toupper(x$family)}")
  cli::cli_text("{.strong Spatial}: {toupper(x$spatial)} | {.strong Temporal}: {toupper(x$temporal)} | {.strong Interaction}: {toupper(x$st_interaction)}")
  if (isTRUE(x$is_nested) && !is.null(x$subarea)) {
    n_maj <- length(unique(x$df_hb$domain))
    n_sub <- length(unique(x$df_hb$subarea))
    icc_str <- if (!is.null(x$icc_nested)) paste0(" | ICC: ", round(x$icc_nested, 4)) else ""
    cli::cli_text("{.strong Hierarchy}: Two-Level Nested Sub-Area [{n_maj} Major Areas -> {n_sub} Sub-Areas{icc_str}]")
  }
  if (identical(x$prior_beta, "horseshoe")) {
    cli::cli_text("{.strong Prior Beta}: Horseshoe (Finnish Regularized Sparse Shrinkage)")
  }
  if (isTRUE(x$robust)) {
    cli::cli_text("{.strong Random Effects}: Robust Student-t (Heavy-Tailed Outlier Resistance)")
  }
  if (isTRUE(x$smooth_vardir) && !is.null(x$gvf)) {
    cli::cli_text("{.strong GVF Smoothing}: ACTIVE [{toupper(x$gvf$method)} | R2: {round(x$gvf$r_squared, 4)}]")
  }
  if (isTRUE(x$benchmarked) && !is.null(x$benchmark_info)) {
    type_str <- if (x$benchmark_info$type == "self") "Self-Benchmarking (Direct Survey)" else "External Benchmarking"
    cli::cli_text("{.strong Benchmarking}: ACTIVE [{type_str} | Method: {toupper(x$benchmark_info$method)} | Target: {round(x$benchmark_info$target, 5)}]")
  }
  
  if (!is.null(x$estcoef)) {
    cli::cli_h2("Regression Coefficients:")
    # Print standard 4 columns to avoid printCoefmat distortion
    cols_to_print <- intersect(c("beta", "std.error", "zvalue", "pvalue"), names(x$estcoef))
    stats::printCoefmat(as.matrix(x$estcoef[, cols_to_print, drop = FALSE]), signif.stars = TRUE, ...)
    if ("shrinkage_factor" %in% names(x$estcoef) && nrow(x$estcoef) > 1) {
      cli::cli_alert_info("Horseshoe shrinkage weights kappa_j (1 = noise pruned, 0 = signal retained):")
      sw_df <- data.frame(
        Variable = rownames(x$estcoef)[-1],
        Shrinkage = x$estcoef$shrinkage_factor[-1],
        Signal_Retained = paste0(round((1 - x$estcoef$shrinkage_factor[-1]) * 100, 1), "%")
      )
      print(sw_df, row.names = FALSE)
    }
  }
  
  if (!is.null(x$hyperpar) && nrow(x$hyperpar) > 0) {
    cli::cli_h2("Hyperparameters:")
    print(x$hyperpar, row.names = FALSE)
  }
  
  if (!is.null(x$goodness)) {
    cli::cli_h2("Model Diagnostics:")
    print(round(x$goodness, 4))
  }
  
  if (!is.null(x$df_hb)) {
    cli::cli_h2("Estimates (First 6 domains):")
    print(utils::head(x$df_hb, 6), ...)
    if (nrow(x$df_hb) > 6) {
      cli::cli_text("... and {nrow(x$df_hb) - 6} more rows.")
    }
  }
  invisible(x)
}

#' @rdname fastsae_hb_area-methods
#' @export
summary.fastsae_hb_area <- function(object, ...) {
  structure(
    list(
      call = object$call,
      model = object$model,
      device = object$device,
      family = object$family,
      spatial = object$spatial,
      temporal = object$temporal,
      st_interaction = object$st_interaction,
      estcoef = object$estcoef,
      hyperpar = object$hyperpar,
      goodness = object$goodness,
      df_hb = object$df_hb
    ),
    class = "summary.fastsae_hb_area"
  )
}

#' @rdname fastsae_hb_area-methods
#' @export
print.summary.fastsae_hb_area <- function(x, ...) {
  print.fastsae_hb_area(x, ...)
}

#' @rdname fastsae_hb_area-methods
#' @export
coef.fastsae_hb_area <- function(object, ...) {
  if (!is.null(object$estcoef) && "beta" %in% names(object$estcoef)) {
    return(stats::setNames(object$estcoef$beta, rownames(object$estcoef)))
  }
  NULL
}

#' @rdname fastsae_hb_area-methods
#' @export
fitted.fastsae_hb_area <- function(object, ...) {
  if (!is.null(object$df_hb) && "hb" %in% names(object$df_hb)) {
    return(object$df_hb$hb)
  }
  NULL
}

#' @rdname fastsae_hb_area-methods
#' @export
residuals.fastsae_hb_area <- function(object, ...) {
  if (!is.null(object$df_hb) && all(c("y", "hb") %in% names(object$df_hb))) {
    return(object$df_hb$y - object$df_hb$hb)
  }
  NULL
}


#' Methods for fastsaegpu_benchmark objects
#'
#' Print, summarize, and plot benchmarked small area estimates.
#'
#' @param x,object An object of class \code{fastsaegpu_benchmark}.
#' @param ... Additional arguments passed to methods.
#' @return
#' \itemize{
#'   \item \code{print}: Invisibly returns \code{x}.
#'   \item \code{summary}: A list containing calibration summary statistics.
#'   \item \code{plot}: A \code{ggplot2} object visualizing the benchmark adjustment.
#' }
#' @name fastsaegpu_benchmark-methods
NULL

#' @rdname fastsaegpu_benchmark-methods
#' @export
print.fastsaegpu_benchmark <- function(x, ...) {
  target_mode <- attr(x, "target_mode") %||% "Benchmarking"
  method <- attr(x, "method") %||% "optimal"
  type <- attr(x, "type") %||% "mean"
  
  cat("== Benchmarked Small Area Estimation ==========================================\n")
  cat(sprintf("Mode: %s\n", target_mode))
  cat(sprintf("Method: %s (%s)\n\n", toupper(method), type))
  
  verif <- attr(x, "verification")
  if (!is.null(verif) && length(verif) > 0) {
    cat("-- Calibration Verification: --\n")
    for (v in verif) {
      status_icon <- if (v$discrepancy < 1e-6) "[OK]" else "[MISMATCH]"
      cat(sprintf("%s Target: %.5f | Benchmarked Sum: %.5f (Discrepancy: %s)\n",
                  status_icon, v$target, v$benchmarked_sum, format(v$discrepancy, scientific = TRUE)))
    }
    cat("\n")
  }
  
  cat("-- Adjustment Statistics: --\n")
  adj <- x$adjustment
  rel_adj <- x$rel_adjustment_pct
  cat(sprintf("Absolute Adjustment: Min = %.5f, Mean = %.5f, Max = %.5f\n", min(adj), mean(adj), max(adj)))
  cat(sprintf("Relative Adjustment (%%): Min = %.2f%%, Mean = %.2f%%, Max = %.2f%%\n\n",
              min(rel_adj, na.rm = TRUE), mean(rel_adj, na.rm = TRUE), max(rel_adj, na.rm = TRUE)))
  
  cat("-- Estimates (First 6 domains): --\n")
  print(utils::head(as.data.frame(x), 6), ...)
  if (nrow(x) > 6) {
    cat(sprintf("... and %d more rows.\n", nrow(x) - 6))
  }
  invisible(x)
}

#' @rdname fastsaegpu_benchmark-methods
#' @export
summary.fastsaegpu_benchmark <- function(object, ...) {
  structure(
    list(
      target_mode = attr(object, "target_mode"),
      method = attr(object, "method"),
      type = attr(object, "type"),
      verification = attr(object, "verification"),
      adj_summary = summary(object$adjustment),
      rel_adj_summary = summary(object$rel_adjustment_pct),
      n_domains = nrow(object)
    ),
    class = "summary.fastsaegpu_benchmark"
  )
}

#' @rdname fastsaegpu_benchmark-methods
#' @export
print.summary.fastsaegpu_benchmark <- function(x, ...) {
  cat("== Summary of Benchmarked SAE Calibration =====================================\n")
  cat(sprintf("Total Domains: %d\n", x$n_domains))
  cat(sprintf("Calibration Method: %s\n\n", toupper(x$method)))
  cat("-- Absolute Adjustment: --\n")
  print(x$adj_summary)
  cat("\n-- Relative Adjustment (%): --\n")
  print(x$rel_adj_summary)
  invisible(x)
}

#' @rdname fastsaegpu_benchmark-methods
#' @export
plot.fastsaegpu_benchmark <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    cli::cli_abort("Package {.pkg ggplot2} is required to plot benchmarked objects.")
  }
  df <- as.data.frame(x)
  p <- ggplot2::ggplot(df, ggplot2::aes(x = original, y = benchmarked)) +
    ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
    ggplot2::geom_point(color = "#1f77b4", size = 2.5, alpha = 0.8) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Small Area Estimation: Original vs Benchmarked",
      subtitle = paste0("Method: ", toupper(attr(x, "method")), " (", attr(x, "target_mode"), ")"),
      x = "Original Model Estimate (hb)",
      y = "Benchmarked / Calibrated Estimate"
    )
  p
}

# ==============================================================================
# S3 Methods for fastsaegpu_merf (Mixed Effects Random Forest)
# ==============================================================================

#' @rdname fastsae_hb_area-methods
#' @export
print.fastsaegpu_merf <- function(x, ...) {
  cli::cli_h1("Mixed Effects Random Forest Small Area Estimation (MERF / FH-RF)")
  if (!is.null(x$call)) {
    cli::cli_text("{.strong Call}: {deparse(x$call)}")
  }
  cli::cli_text("{.strong Engine}: {x$engine} (Random Forest)")
  if (isTRUE(x$is_nested)) {
    cli::cli_text("{.strong Hierarchy}: Two-Level Nested Sub-Area [ICC: {round(x$hyperparams$icc_nested, 4)}]")
  } else if (isTRUE(x$is_spatial)) {
    cli::cli_text("{.strong Spatial}: SAR Autoregressive (rho = {round(x$hyperparams$rho_spatial, 4)})")
  } else {
    cli::cli_text("{.strong Structure}: Standard Area-Level (Fay-Herriot RF)")
  }
  cli::cli_text("{.strong Iterations}: {x$hyperparams$iterations} (Converged: {x$hyperparams$converged})")

  cli::cli_h2("Variance Components & Hyperparameters:")
  hp_df <- data.frame(
    Parameter = "sigma2_u",
    Estimate = as.numeric(x$hyperparams$sigma2_u),
    stringsAsFactors = FALSE
  )
  if (isTRUE(x$is_nested)) {
    hp_df <- rbind(
      hp_df,
      data.frame(Parameter = "sigma2_subarea", Estimate = as.numeric(x$hyperparams$sigma2_subarea)),
      data.frame(Parameter = "icc_nested", Estimate = as.numeric(x$hyperparams$icc_nested))
    )
  }
  if (isTRUE(x$is_spatial)) {
    hp_df <- rbind(
      hp_df,
      data.frame(Parameter = "rho_spatial", Estimate = as.numeric(x$hyperparams$rho_spatial))
    )
  }
  print(hp_df, row.names = FALSE)

  cli::cli_h2("Top Variable Importance:")
  top_n <- min(5, length(x$importance))
  vimp_top <- data.frame(
    Variable = names(x$importance)[seq_len(top_n)],
    Importance = round(as.numeric(x$importance)[seq_len(top_n)], 4),
    stringsAsFactors = FALSE
  )
  print(vimp_top, row.names = FALSE)

  cli::cli_h2("Estimates (First 6 domains):")
  disp_cols <- intersect(c("domain", "y", "merf", "rf_pred", "random_effect", "gamma", "sd", "mse", "rse", "subarea"), names(x$estimates))
  print(utils::head(x$estimates[, disp_cols, drop = FALSE], 6))
  if (nrow(x$estimates) > 6) {
    cli::cli_text("... and {nrow(x$estimates) - 6} more rows.")
  }
  invisible(x)
}

#' @rdname fastsae_hb_area-methods
#' @export
summary.fastsaegpu_merf <- function(object, ...) {
  structure(
    list(
      call = object$call,
      engine = object$engine,
      is_nested = object$is_nested,
      is_spatial = object$is_spatial,
      hyperparams = object$hyperparams,
      importance = object$importance,
      n_domains = nrow(object$estimates),
      estimates_summary = summary(object$estimates$merf),
      gamma_summary = summary(object$estimates$gamma),
      rse_summary = summary(object$estimates$rse)
    ),
    class = "summary.fastsaegpu_merf"
  )
}

#' @rdname fastsae_hb_area-methods
#' @export
print.summary.fastsaegpu_merf <- function(x, ...) {
  cli::cli_h1("Summary of MERF Small Area Estimation (FH-RF)")
  if (!is.null(x$call)) {
    cli::cli_text("{.strong Call}: {deparse(x$call)}")
  }
  cli::cli_text("Domains: {x$n_domains} | Engine: {x$engine}")
  cli::cli_text("Iterations: {x$hyperparams$iterations} (Converged: {x$hyperparams$converged})")
  cli::cli_text("sigma2_u: {round(x$hyperparams$sigma2_u, 5)}")

  cli::cli_h2("Shrinkage Factors (gamma):")
  print(x$gamma_summary)

  cli::cli_h2("RSE (%) Distribution:")
  print(x$rse_summary)

  cli::cli_h2("Variable Importance:")
  print(x$importance)
  invisible(x)
}

#' @rdname fastsae_hb_area-methods
#' @param type Character specifying plot type: \code{"importance"} (variable importance bar plot) or \code{"estimates"} (scatter of direct vs MERF estimates).
#' @export
plot.fastsaegpu_merf <- function(x, type = c("importance", "estimates"), ...) {
  type <- match.arg(type)
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    cli::cli_abort("Package {.pkg ggplot2} is required to plot {.cls fastsaegpu_merf} objects.")
  }
  if (type == "importance") {
    vimp <- x$importance
    df_vimp <- data.frame(
      Variable = factor(names(vimp), levels = rev(names(vimp))),
      Importance = as.numeric(vimp)
    )
    p <- ggplot2::ggplot(df_vimp, ggplot2::aes(x = Importance, y = Variable)) +
      ggplot2::geom_col(fill = "#2ca02c", alpha = 0.85, width = 0.6) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = "Random Forest Variable Importance (MERF / FH-RF)",
        subtitle = paste0("Engine: ", x$engine, " | Trees: ", x$forest$num.trees %||% 500),
        x = "Importance Score",
        y = "Covariate"
      )
    return(p)
  } else {
    df <- x$estimates
    p <- ggplot2::ggplot(df, ggplot2::aes(x = y, y = merf)) +
      ggplot2::geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
      ggplot2::geom_point(color = "#1f77b4", size = 2.5, alpha = 0.8) +
      ggplot2::theme_minimal() +
      ggplot2::labs(
        title = "Small Area Estimation: Direct vs MERF",
        subtitle = paste0("Convergence in ", x$hyperparams$iterations, " iterations | sigma2_u: ", round(x$hyperparams$sigma2_u, 4)),
        x = "Direct Survey Estimate (y)",
        y = "MERF Small Area Predictor"
      )
    return(p)
  }
}

#' @rdname fastsae_hb_area-methods
#' @export
coef.fastsaegpu_merf <- function(object, ...) {
  object$importance
}

#' @rdname fastsae_hb_area-methods
#' @export
fitted.fastsaegpu_merf <- function(object, ...) {
  object$estimates$merf
}

#' @rdname fastsae_hb_area-methods
#' @export
residuals.fastsaegpu_merf <- function(object, ...) {
  object$estimates$y - object$estimates$merf
}

#' @rdname fastsae_hb_area-methods
#' @param newdata Optional data frame containing new covariates for out-of-sample synthetic prediction.
#' @export
predict.fastsaegpu_merf <- function(object, newdata = NULL, ...) {
  if (is.null(newdata)) {
    return(object$estimates$merf)
  }
  newdata_df <- as.data.frame(newdata)
  if (object$engine == "ranger") {
    preds <- stats::predict(object$forest, data = newdata_df)$predictions
  } else {
    preds <- stats::predict(object$forest, newdata = newdata_df)
  }
  return(preds)
}
