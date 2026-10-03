`%||%` <- function(x, y) if (is.null(x)) y else x

#' @export
print.fastsae_hb_area <- function(x, ...) {
  cli::cli_h1("GPU-Accelerated Hierarchical Bayesian Small Area Estimation")
  if (!is.null(x$call)) {
    cli::cli_text("{.strong Call}: {deparse(x$call)}")
  }
  cli::cli_text("{.strong Backend}: NumPyro / JAX on {.val {x$device}}")
  cli::cli_text("{.strong Family}: {toupper(x$family)}")
  cli::cli_text("{.strong Spatial}: {toupper(x$spatial)} | {.strong Temporal}: {toupper(x$temporal)} | {.strong Interaction}: {toupper(x$st_interaction)}")
  
  if (!is.null(x$estcoef)) {
    cli::cli_h2("Regression Coefficients:")
    # Print standard 4 columns to avoid printCoefmat distortion
    cols_to_print <- intersect(c("beta", "std.error", "zvalue", "pvalue"), names(x$estcoef))
    stats::printCoefmat(as.matrix(x$estcoef[, cols_to_print, drop = FALSE]), signif.stars = TRUE, ...)
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

#' @export
print.summary.fastsae_hb_area <- function(x, ...) {
  print.fastsae_hb_area(x, ...)
}

#' @export
coef.fastsae_hb_area <- function(object, ...) {
  if (!is.null(object$estcoef) && "beta" %in% names(object$estcoef)) {
    return(stats::setNames(object$estcoef$beta, rownames(object$estcoef)))
  }
  NULL
}

#' @export
fitted.fastsae_hb_area <- function(object, ...) {
  if (!is.null(object$df_hb) && "hb" %in% names(object$df_hb)) {
    return(object$df_hb$hb)
  }
  NULL
}

#' @export
residuals.fastsae_hb_area <- function(object, ...) {
  if (!is.null(object$df_hb) && all(c("y", "hb") %in% names(object$df_hb))) {
    return(object$df_hb$y - object$df_hb$hb)
  }
  NULL
}
