#' Generalized Variance Functions (GVF) for Smoothing Direct Sampling Variances
#'
#' Implements Generalized Variance Function (GVF) smoothing models for direct sampling
#' variances in Small Area Estimation (Wolter, 2007; Otto & Bell, 1995; Rivest & Vandal, 2003).
#' Raw sampling variances from small domain sample sizes often suffer from high sampling variability;
#' GVF smoothing stabilizes sampling variance estimates, prevents artificial over-shrinkage,
#' and improves model stability and MSE accuracy.
#'
#' @param y Numeric vector or formula term of direct area estimates.
#' @param vardir Numeric vector or formula term of direct sampling variances.
#' @param n Optional numeric vector of area sample sizes.
#' @param method Smoothing method: \code{"log_linear"} (default, Wolter 2007),
#'   \code{"power"} (Cho et al., 2002), \code{"ratio"} (CV-squared model),
#'   or \code{"loess"} (non-parametric local regression).
#' @param data Optional data frame containing variables.
#' @param ... Additional arguments passed to modeling functions.
#' @return An object of class \code{c("fastsaegpu_gvf", "fastsae_gvf")} containing:
#'   \itemize{
#'     \item \code{vardir_smooth}: Numeric vector of smoothed sampling variances.
#'     \item \code{vardir_raw}: Original direct sampling variances.
#'     \item \code{r_squared}: R-squared metric of the smoothing model.
#'     \item \code{method}: The GVF model method used.
#'     \item \code{data}: Data frame with estimates, original variances, and smoothed variances.
#'     \item \code{model}: The underlying fitted model object.
#'   }
#' @references
#'   Wolter, K. M. (2007). \emph{Introduction to Variance Estimation}. Springer.
#'
#'   Otto, M. C., & Bell, W. R. (1995). Sampling error modeling of poverty and
#'   income statistics for states. \emph{Proceedings of the Government Statistics Section},
#'   American Statistical Association, 160-165.
#'
#'   Rivest, L. P., & Vandal, N. (2003). Smooth variance estimators for small
#'   areas. \emph{Proceedings of the Annual Meeting of the American Statistical Association}.
#' @examples
#' set.seed(42)
#' D <- 30
#' y <- runif(D, 0.1, 0.8)
#' n <- sample(15:50, D, replace = TRUE)
#' vardir_raw <- (y * (1 - y) / n) * rlnorm(D, 0, 0.3)
#'
#' gvf_fit <- gvf_smooth(y = y, vardir = vardir_raw, n = n, method = "log_linear")
#' print(gvf_fit)
#' @export
gvf_smooth <- function(
  y,
  vardir,
  n = NULL,
  method = c("log_linear", "power", "ratio", "loess"),
  data = NULL,
  ...
) {
  method <- match.arg(tolower(method), choices = c("log_linear", "power", "ratio", "loess"))

  # Extract variables from data if provided
  if (!is.null(data)) {
    if (is.character(y) && length(y) == 1) y <- data[[y]]
    if (is.character(vardir) && length(vardir) == 1) vardir <- data[[vardir]]
    if (!is.null(n) && is.character(n) && length(n) == 1) n <- data[[n]]
  }

  y_vec <- as.numeric(y)
  v_vec <- as.numeric(vardir)
  n_vec <- if (!is.null(n)) as.numeric(n) else NULL

  if (length(y_vec) != length(v_vec)) {
    cli::cli_abort("{.arg y} and {.arg vardir} must have the same length.")
  }
  if (!is.null(n_vec) && length(n_vec) != length(y_vec)) {
    cli::cli_abort("{.arg n} must have the same length as {.arg y}.")
  }

  valid_idx <- which(!is.na(y_vec) & !is.na(v_vec) & v_vec > 0)
  if (length(valid_idx) < 3) {
    cli::cli_abort("Insufficient valid positive observations for GVF smoothing (minimum 3 required).")
  }

  y_val <- y_vec[valid_idx]
  v_val <- v_vec[valid_idx]
  n_val <- if (!is.null(n_vec)) n_vec[valid_idx] else NULL

  r_sq <- NA_real_
  model_obj <- NULL
  smooth_val <- v_val

  if (method == "log_linear") {
    # Wolter (2007) / Otto & Bell (1995)
    # log(vardir) = a0 + a1 * log(y) + a2 * log(1-y) + [a3 * log(n)]
    log_v <- log(v_val)
    # Clip y safely for logs
    y_clip <- pmin(pmax(y_val, 1e-4), 1 - 1e-4)
    log_y <- log(y_clip)
    log_1my <- log(1 - y_clip)

    df_mod <- data.frame(log_v = log_v, log_y = log_y, log_1my = log_1my)
    form <- log_v ~ log_y + log_1my
    if (!is.null(n_val) && all(n_val > 0)) {
      df_mod$log_n <- log(n_val)
      form <- log_v ~ log_y + log_1my + log_n
    }

    fit_lm <- stats::lm(form, data = df_mod)
    pred_log <- stats::predict(fit_lm)
    # Log-normal mean correction: E[V] = exp(mu + sigma^2 / 2)
    s2_res <- stats::deviance(fit_lm) / stats::df.residual(fit_lm)
    smooth_val <- exp(pred_log + s2_res / 2)
    r_sq <- summary(fit_lm)$r.squared
    model_obj <- fit_lm

  } else if (method == "power") {
    # Cho et al. (2002): log(vardir) = a0 + a1 * log(y) + a2 / n
    log_v <- log(v_val)
    y_clip <- pmax(y_val, 1e-4)
    log_y <- log(y_clip)

    df_mod <- data.frame(log_v = log_v, log_y = log_y)
    form <- log_v ~ log_y
    if (!is.null(n_val) && all(n_val > 0)) {
      df_mod$inv_n <- 1 / n_val
      form <- log_v ~ log_y + inv_n
    }

    fit_lm <- stats::lm(form, data = df_mod)
    pred_log <- stats::predict(fit_lm)
    s2_res <- stats::deviance(fit_lm) / stats::df.residual(fit_lm)
    smooth_val <- exp(pred_log + s2_res / 2)
    r_sq <- summary(fit_lm)$r.squared
    model_obj <- fit_lm

  } else if (method == "ratio") {
    # Rivest & Vandal (2003): CV^2 = vardir / y^2 = a0 + a1 / y
    cv2 <- v_val / (pmax(y_val, 1e-4)^2)
    inv_y <- 1 / pmax(y_val, 1e-4)
    df_mod <- data.frame(cv2 = cv2, inv_y = inv_y)
    fit_lm <- stats::lm(cv2 ~ inv_y, data = df_mod)
    pred_cv2 <- pmax(stats::predict(fit_lm), 1e-6)
    smooth_val <- pred_cv2 * (y_val^2)
    r_sq <- summary(fit_lm)$r.squared
    model_obj <- fit_lm

  } else if (method == "loess") {
    # Non-parametric local polynomial regression
    log_v <- log(v_val)
    fit_loess <- stats::loess(log_v ~ y_val, span = 0.75, ...)
    pred_log <- stats::predict(fit_loess)
    # Fallback to linear if NA
    if (any(is.na(pred_log))) {
      pred_log[is.na(pred_log)] <- mean(log_v)
    }
    s2_res <- mean((log_v - pred_log)^2)
    smooth_val <- exp(pred_log + s2_res / 2)
    r_sq <- 1 - sum((log_v - pred_log)^2) / sum((log_v - mean(log_v))^2)
    model_obj <- fit_loess
  }

  # Ensure positivity
  smooth_val <- pmax(smooth_val, 1e-8)

  # Reconstruct full vector
  vardir_smooth <- rep(NA_real_, length(y_vec))
  vardir_smooth[valid_idx] <- smooth_val

  res_df <- data.frame(
    y = y_vec,
    vardir_raw = v_vec,
    vardir_smooth = vardir_smooth,
    sd_raw = sqrt(v_vec),
    sd_smooth = sqrt(vardir_smooth),
    ratio = vardir_smooth / pmax(v_vec, 1e-8),
    stringsAsFactors = FALSE
  )
  if (!is.null(n_vec)) {
    res_df$n <- n_vec
  }

  structure(
    list(
      vardir_smooth = vardir_smooth,
      vardir_raw = v_vec,
      r_squared = r_sq,
      method = method,
      data = res_df,
      model = model_obj
    ),
    class = c("fastsaegpu_gvf", "fastsae_gvf")
  )
}

#' @export
print.fastsaegpu_gvf <- function(x, ...) {
  cli::cli_h1("Generalized Variance Function (GVF) Smoothing")
  cli::cli_text("{.strong Method}: {toupper(x$method)}")
  if (!is.na(x$r_squared)) {
    cli::cli_text("{.strong Model R-Squared}: {round(x$r_squared, 4)}")
  }
  cli::cli_h2("Variance Summary (Raw vs Smoothed):")
  smry <- data.frame(
    Metric = c("Mean", "Median", "Min", "Max", "SD"),
    Raw_Var = c(mean(x$vardir_raw, na.rm = TRUE), stats::median(x$vardir_raw, na.rm = TRUE),
                min(x$vardir_raw, na.rm = TRUE), max(x$vardir_raw, na.rm = TRUE),
                stats::sd(x$vardir_raw, na.rm = TRUE)),
    Smoothed_Var = c(mean(x$vardir_smooth, na.rm = TRUE), stats::median(x$vardir_smooth, na.rm = TRUE),
                     min(x$vardir_smooth, na.rm = TRUE), max(x$vardir_smooth, na.rm = TRUE),
                     stats::sd(x$vardir_smooth, na.rm = TRUE))
  )
  print(smry, row.names = FALSE)
  cli::cli_text("Ratio Smoothed / Raw Variance Range: [{round(min(x$data$ratio, na.rm = TRUE), 3)}, {round(max(x$data$ratio, na.rm = TRUE), 3)}]")
  invisible(x)
}

#' @export
plot.fastsaegpu_gvf <- function(x, ...) {
  if (!requireNamespace("ggplot2", quietly = TRUE)) {
    graphics::plot(x$data$y, x$data$vardir_raw, pch = 16, col = "gray40",
                   xlab = "Estimate (y)", ylab = "Sampling Variance",
                   main = paste0("GVF Smoothing: ", toupper(x$method)))
    graphics::points(x$data$y, x$data$vardir_smooth, pch = 17, col = "blue")
    graphics::legend("topright", legend = c("Raw", "Smoothed"), col = c("gray40", "blue"), pch = c(16, 17))
    return(invisible(x))
  }

  df_plot <- data.frame(
    y = rep(x$data$y, 2),
    variance = c(x$data$vardir_raw, x$data$vardir_smooth),
    Type = rep(c("Raw Variance", "GVF Smoothed"), each = nrow(x$data))
  )

  p <- ggplot2::ggplot(df_plot, ggplot2::aes(x = .data$y, y = .data$variance, color = .data$Type)) +
    ggplot2::geom_point(alpha = 0.7, size = 2) +
    ggplot2::scale_color_manual(values = c("Raw Variance" = "gray50", "GVF Smoothed" = "#1f77b4")) +
    ggplot2::theme_minimal() +
    ggplot2::labs(
      title = "Generalized Variance Function (GVF) Smoothing",
      subtitle = paste0("Method: ", toupper(x$method), if (!is.na(x$r_squared)) paste0(" | R-sq = ", round(x$r_squared, 4)) else ""),
      x = "Direct Estimate (y)",
      y = "Sampling Variance"
    )
  p
}
