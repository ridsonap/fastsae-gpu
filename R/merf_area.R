#' Mixed Effects Random Forest Small Area Estimation (MERF / FH-RF)
#'
#' Fits a Mixed Effects Random Forest (MERF) model for area-level Small Area
#' Estimation based on Krennmair & Schmid (2022), Hajjem et al. (2014), and
#' Bukhari et al. (2025). The linear fixed-effects component of the classical
#' Fay-Herriot model is replaced by a flexible non-parametric Random Forest,
#' while preserving area-level random effects and sampling error variance.
#'
#' @param formula Object of class \code{formula} describing the relationship
#'   between the direct survey estimate and auxiliary covariates.
#' @param data A data frame containing survey variables and area identifiers.
#' @param vardir Character string indicating the column name for direct sampling
#'   variances (\eqn{\psi_i}), or a numeric vector of variances.
#' @param domain Optional character string specifying the domain/area identifier
#'   column. If \code{NULL}, sequential indices are used.
#' @param subarea Optional character string specifying a nested sub-area
#'   identifier column for two-level nested MERF (Torabi & Rao, 2014).
#' @param spatial Optional row-standardized spatial adjacency matrix (\eqn{W})
#'   for Spatial MERF.
#' @param engine Character string specifying the Random Forest engine:
#'   \code{"ranger"} (fast C++ multithreaded, default) or \code{"randomForest"}.
#' @param num_trees Number of trees in the Random Forest ensemble (default: 500).
#' @param mtry Number of variables randomly sampled as candidates at each split.
#'   Default is \code{max(1, floor(sqrt(ncol(X))))}.
#' @param min_node_size Minimum size of terminal nodes (default: 5).
#' @param max_iter Maximum number of EM iterations (default: 25).
#' @param tol Convergence tolerance for the random effects vector
#'   (default: \code{1e-3}).
#' @param mse_type Uncertainty estimation method: \code{"bootstrap"} computes
#'   parametric bootstrap MSE (Krennmair & Schmid, 2022); \code{"none"} returns
#'   point estimates with plug-in analytical approximations.
#' @param B Number of bootstrap replications for MSE estimation (default: 50).
#' @param smooth_vardir Logical; whether to apply Generalized Variance Function
#'   (GVF) smoothing to \code{vardir} prior to estimation (default: \code{FALSE}).
#' @param gvf_method Character specifying the GVF smoothing method if
#'   \code{smooth_vardir = TRUE} (\code{"log_linear"}, \code{"power"},
#'   \code{"ratio"}, or \code{"loess"}).
#' @param num_threads Number of threads for the \code{ranger} engine. Default
#'   is \code{NULL} (all available CPU threads).
#' @param seed Optional integer seed for reproducibility.
#' @param verbose Logical; if \code{TRUE}, prints progress during iterations.
#' @return An object of class \code{fastsaegpu_merf} containing:
#' \describe{
#'   \item{estimates}{Data frame of domain-level SAE estimates, synthetic RF
#'     predictions, random effects, shrinkage factors, MSE, RSE, and 95\% CIs.}
#'   \item{df_hb}{Alias to \code{estimates} containing column \code{hb} for
#'     full interoperability with \code{benchmark()}.}
#'   \item{hyperparams}{List containing \code{sigma2_u}, \code{loglik},
#'     \code{iterations}, \code{converged}, and nested variance if applicable.}
#'   \item{importance}{Named vector of variable importance scores.}
#'   \item{forest}{The final fitted Random Forest object.}
#'   \item{is_nested}{Logical indicating whether two-level nested structure was used.}
#'   \item{is_spatial}{Logical indicating whether spatial structure was used.}
#'   \item{gvf}{The GVF smoothing model if \code{smooth_vardir = TRUE}, else \code{NULL}.}
#'   \item{call}{The matched call.}
#' }
#'
#' @references
#'   Krennmair, P., & Schmid, T. (2022). Flexible small area estimation using
#'   random forests: A semi-parametric approach. *Journal of the Royal
#'   Statistical Society: Series C (Applied Statistics)*, 71(5), 1865-1894.
#'
#'   Hajjem, A., Bellavance, F., & Larocque, D. (2014). Mixed-effects random
#'   forest for clustered data. *Journal of Statistical Computation and
#'   Simulation*, 84(6), 1313-1329.
#'
#'   Bukhari, A. S., Notodiputro, K. A., Indahwati, & Fitrianto, A. (2025).
#'   A modified Fay-Herriot model with machine learning for small area estimation
#'   of per capita expenditure among the poor and agricultural households.
#'   *IPB University Research*.
#'
#'   Torabi, M., & Rao, J. N. K. (2014). On small area estimation under a
#'   two-level model. *Survey Methodology*, 40(1), 43-56.
#'
#' @examples
#' \donttest{
#' set.seed(42)
#' D <- 30
#' x1 <- rnorm(D)
#' x2 <- runif(D)
#' # Non-linear true relationship
#' y_true <- 2 * sin(pi * x1) + 1.5 * x2^2 + rnorm(D, 0, 0.3)
#' vardir <- rep(0.04, D)
#' y_dir <- y_true + rnorm(D, 0, sqrt(vardir))
#'
#' df <- data.frame(
#'   domain = paste0("area_", 1:D),
#'   y = y_dir,
#'   vardir = vardir,
#'   x1 = x1,
#'   x2 = x2
#' )
#'
#' # Fit MERF Area Model
#' fit_rf <- merf_area(
#'   formula = y ~ x1 + x2,
#'   data = df,
#'   vardir = "vardir",
#'   domain = "domain",
#'   num_trees = 100,
#'   mse_type = "none",
#'   seed = 123
#' )
#' print(fit_rf)
#' }
#' @export
merf_area <- function(formula,
                      data,
                      vardir,
                      domain = NULL,
                      subarea = NULL,
                      spatial = NULL,
                      engine = c("ranger", "randomForest"),
                      num_trees = 500,
                      mtry = NULL,
                      min_node_size = 5,
                      max_iter = 25,
                      tol = 1e-3,
                      mse_type = c("bootstrap", "none"),
                      B = 50,
                      smooth_vardir = FALSE,
                      gvf_method = c("log_linear", "power", "ratio", "loess"),
                      num_threads = NULL,
                      seed = NULL,
                      verbose = FALSE) {

  cl <- match.call()
  engine <- match.arg(engine)
  mse_type <- match.arg(mse_type)
  gvf_method <- match.arg(gvf_method)

  if (!is.null(seed)) {
    set.seed(seed)
  }

  # --- 1. Data Validation & Preprocessing ---
  if (!is.data.frame(data)) {
    cli::cli_abort("{.arg data} must be a data frame.")
  }

  # Extract Response & Design Matrix
  mf <- stats::model.frame(formula, data = data)
  y <- stats::model.response(mf)
  terms_obj <- stats::terms(mf)
  X <- stats::model.matrix(terms_obj, data = data)

  # Remove Intercept column if present because decision trees split on covariates directly
  int_idx <- which(colnames(X) == "(Intercept)")
  if (length(int_idx) > 0) {
    X <- X[, -int_idx, drop = FALSE]
  }

  D <- length(y)
  if (nrow(X) != D) {
    cli::cli_abort("Mismatch between response length ({D}) and covariate rows ({nrow(X)}).")
  }
  if (ncol(X) == 0) {
    cli::cli_abort("Formula must include at least one covariate.")
  }

  if (is.null(mtry)) {
    mtry <- max(1, floor(sqrt(ncol(X))))
  }

  # Extract vardir
  if (is.character(vardir) && length(vardir) == 1) {
    if (!vardir %in% names(data)) {
      cli::cli_abort("Column {.val {vardir}} not found in {.arg data}.")
    }
    psi <- data[[vardir]]
  } else if (is.numeric(vardir) && length(vardir) == D) {
    psi <- as.numeric(vardir)
  } else {
    cli::cli_abort("{.arg vardir} must be a column name in {.arg data} or a numeric vector of length {D}.")
  }

  if (any(is.na(psi)) || any(psi <= 0)) {
    cli::cli_abort("All sampling variances in {.arg vardir} must be strictly positive and non-missing.")
  }

  # Optional GVF Variance Smoothing
  gvf_obj <- NULL
  if (isTRUE(smooth_vardir)) {
    if (verbose) cli::cli_inform("Applying GVF smoothing to stabilize {.arg vardir} ({gvf_method})...")
    gvf_obj <- gvf_smooth(y = y, vardir = psi, method = gvf_method)
    psi <- gvf_obj$vardir_smooth
  }

  # Domain extraction
  if (!is.null(domain) && is.character(domain) && length(domain) == 1) {
    if (!domain %in% names(data)) {
      cli::cli_abort("Column {.val {domain}} not found in {.arg data}.")
    }
    domain_vec <- as.character(data[[domain]])
  } else {
    domain_vec <- paste0("area_", seq_len(D))
  }

  # Check Two-Level Nested Subarea
  is_nested <- FALSE
  major_area_vec <- NULL
  subarea_vec <- NULL
  if (!is.null(subarea) && is.character(subarea) && length(subarea) == 1) {
    if (!subarea %in% names(data)) {
      cli::cli_abort("Column {.val {subarea}} not found in {.arg data}.")
    }
    sub_raw <- as.character(data[[subarea]])
    dom_raw <- domain_vec

    # Auto-detection hierarchy logic:
    # Coarser level (fewer categories) -> Major Area (cluster)
    # Finer level (more categories) -> Sub-Area
    u_dom <- length(unique(dom_raw))
    u_sub <- length(unique(sub_raw))

    if (u_sub < u_dom) {
      if (verbose) cli::cli_inform("Hierarchy auto-swap: mapping {.val {subarea}} as major area and {.val {domain}} as sub-area.")
      major_area_vec <- sub_raw
      subarea_vec <- dom_raw
    } else {
      major_area_vec <- dom_raw
      subarea_vec <- sub_raw
    }
    is_nested <- TRUE
  }

  # Spatial Adjacency
  is_spatial <- FALSE
  if (!is.null(spatial)) {
    if (is.matrix(spatial) || inherits(spatial, "Matrix")) {
      W <- as.matrix(spatial)
      if (nrow(W) != D || ncol(W) != D) {
        cli::cli_abort("Spatial matrix {.arg spatial} must have dimensions {D} x {D}.")
      }
      is_spatial <- TRUE
    } else {
      cli::cli_abort("{.arg spatial} must be a {D} x {D} adjacency matrix.")
    }
  }

  # Check Engine
  if (engine == "ranger") {
    if (!requireNamespace("ranger", quietly = TRUE)) {
      cli::cli_warn("Package {.pkg ranger} is not installed. Falling back to {.pkg randomForest}.")
      engine <- "randomForest"
    }
  }
  if (engine == "randomForest") {
    if (!requireNamespace("randomForest", quietly = TRUE)) {
      cli::cli_abort("Neither {.pkg ranger} nor {.pkg randomForest} is available. Please install {.pkg ranger}.")
    }
  }

  # --- 2. Random Forest Helper Function ---
  fit_rf <- function(X_mat, target_vec, trees = num_trees, seed_iter = NULL) {
    df_rf <- as.data.frame(X_mat)
    df_rf$.target <- target_vec
    var_names <- setdiff(names(df_rf), ".target")

    if (engine == "ranger") {
      rf_fit <- ranger::ranger(
        formula = .target ~ .,
        data = df_rf,
        num.trees = trees,
        mtry = mtry,
        min.node.size = min_node_size,
        importance = "permutation",
        num.threads = num_threads,
        seed = seed_iter
      )
      preds <- stats::predict(rf_fit, data = df_rf)$predictions
      vimp <- rf_fit$variable.importance
      return(list(fit = rf_fit, pred = preds, importance = vimp))
    } else {
      rf_fit <- randomForest::randomForest(
        x = X_mat,
        y = target_vec,
        ntree = trees,
        mtry = mtry,
        nodesize = min_node_size,
        importance = TRUE
      )
      preds <- stats::predict(rf_fit, newdata = X_mat)
      vimp <- rf_fit$importance[, 1]
      return(list(fit = rf_fit, pred = preds, importance = vimp))
    }
  }

  # --- 3. Expectation-Maximization (EM) Loop ---
  if (verbose) cli::cli_inform("Initializing MERF EM algorithm (max_iter = {max_iter}, tol = {tol})...")

  if (!is_nested && !is_spatial) {
    # Standard Area-Level MERF (Krennmair & Schmid, 2022)
    u <- rep(0, D)
    sigma2_u <- max(0.01, stats::var(y) - mean(psi))
    loglik_val <- -Inf
    converged <- FALSE
    final_rf <- NULL

    for (iter in seq_len(max_iter)) {
      # Step 1: Adjusted response
      y_star <- y - u

      # Step 2: Fit Random Forest
      rf_res <- fit_rf(X, y_star, trees = num_trees, seed_iter = seed)
      f_hat <- rf_res$pred
      final_rf <- rf_res

      # Step 3: Residuals
      r <- y - f_hat

      # Step 4: Profile Log-Likelihood optimization for sigma2_u
      nll_sigma2 <- function(s2) {
        v <- s2 + psi
        if (any(v <= 0)) return(1e10)
        0.5 * sum(log(v) + (r^2) / v)
      }

      opt_res <- stats::optimize(
        nll_sigma2,
        interval = c(0, max(10 * stats::var(y), 10))
      )
      sigma2_u_new <- opt_res$minimum

      # Step 5: Update random effects and shrinkage
      gamma <- sigma2_u_new / (sigma2_u_new + psi)
      u_new <- gamma * r

      # Check convergence: relative root-mean-squared change in random effects
      delta_u <- sqrt(mean((u_new - u)^2)) / (stats::sd(y) + 1e-6)
      u <- u_new
      sigma2_u <- sigma2_u_new
      loglik_val <- -opt_res$objective

      if (verbose) {
        cli::cli_text("  Iteration {iter}: sigma2_u = {round(sigma2_u, 5)}, delta_u = {format.default(delta_u, scientific = TRUE, digits = 3)}")
      }

      if (delta_u < tol) {
        converged <- TRUE
        break
      }
    }

    # Point Estimates
    theta_merf <- f_hat + u
    gamma_vec <- sigma2_u / (sigma2_u + psi)

  } else if (is_nested) {
    # Two-Level Nested MERF (Torabi & Rao, 2014)
    # y_{jk} = f(x_{jk}) + u_j + v_{jk} + e_{jk}
    major_areas <- unique(major_area_vec)
    J <- length(major_areas)
    area_idx_list <- split(seq_len(D), major_area_vec)

    u_major <- stats::setNames(rep(0, J), major_areas)
    v_sub <- rep(0, D)
    sigma2_area <- max(0.01, 0.5 * (stats::var(y) - mean(psi)))
    sigma2_sub <- max(0.01, 0.5 * (stats::var(y) - mean(psi)))
    converged <- FALSE
    final_rf <- NULL

    for (iter in seq_len(max_iter)) {
      u_expanded <- u_major[major_area_vec]
      y_star <- y - u_expanded - v_sub

      rf_res <- fit_rf(X, y_star, trees = num_trees, seed_iter = seed)
      f_hat <- rf_res$pred
      final_rf <- rf_res

      r <- y - f_hat

      # Fast analytical profile log-likelihood for (sigma2_area, sigma2_sub)
      nll_nested <- function(par) {
        s2_a <- par[1]
        s2_s <- par[2]
        if (s2_a < 0 || s2_s < 0) return(1e10)

        nll <- 0
        for (j in seq_len(J)) {
          idx_j <- area_idx_list[[j]]
          psi_j <- psi[idx_j]
          r_j <- r[idx_j]
          d_j <- s2_s + psi_j
          w_j <- 1 / d_j
          W_sum <- sum(w_j)
          denom <- 1 + s2_a * W_sum

          log_det_Vj <- sum(log(d_j)) + log(max(1e-12, denom))
          quad_form <- sum(w_j * (r_j^2)) - (s2_a / denom) * (sum(w_j * r_j)^2)
          nll <- nll + 0.5 * (log_det_Vj + quad_form)
        }
        nll
      }

      opt_res <- stats::optim(
        par = c(sigma2_area, sigma2_sub),
        fn = nll_nested,
        method = "L-BFGS-B",
        lower = c(1e-6, 1e-6)
      )

      sigma2_area_new <- opt_res$par[1]
      sigma2_sub_new <- opt_res$par[2]

      # Update BLUPs analytically
      u_major_new <- numeric(J)
      for (j in seq_len(J)) {
        idx_j <- area_idx_list[[j]]
        psi_j <- psi[idx_j]
        r_j <- r[idx_j]
        w_j <- 1 / (sigma2_sub_new + psi_j)
        u_major_new[j] <- (sigma2_area_new / (1 + sigma2_area_new * sum(w_j))) * sum(w_j * r_j)
      }
      names(u_major_new) <- major_areas

      u_expanded_new <- u_major_new[major_area_vec]
      gamma_sub <- sigma2_sub_new / (sigma2_sub_new + psi)
      v_sub_new <- gamma_sub * (r - u_expanded_new)

      u_tot_new <- u_expanded_new + v_sub_new
      u_tot_old <- u_expanded + v_sub
      delta_u <- sqrt(mean((u_tot_new - u_tot_old)^2)) / (stats::sd(y) + 1e-6)

      u_major <- u_major_new
      v_sub <- v_sub_new
      sigma2_area <- sigma2_area_new
      sigma2_sub <- sigma2_sub_new
      loglik_val <- -opt_res$value

      if (verbose) {
        cli::cli_text("  Iteration {iter}: sigma2_area = {round(sigma2_area, 5)}, sigma2_sub = {round(sigma2_sub, 5)}, delta_u = {format.default(delta_u, scientific = TRUE, digits = 3)}")
      }

      if (delta_u < tol) {
        converged <- TRUE
        break
      }
    }

    u <- u_major[major_area_vec] + v_sub
    sigma2_u <- sigma2_area
    theta_merf <- f_hat + u
    gamma_vec <- (sigma2_area + sigma2_sub) / (sigma2_area + sigma2_sub + psi)

  } else if (is_spatial) {
    # Spatial MERF: u ~ N(0, sigma2_u * (I - rho W)^(-1) ((I - rho W)^(-1))')
    u <- rep(0, D)
    sigma2_u <- max(0.01, stats::var(y) - mean(psi))
    rho_val <- 0.2
    converged <- FALSE
    final_rf <- NULL

    I_mat <- diag(D)
    for (iter in seq_len(max_iter)) {
      y_star <- y - u
      rf_res <- fit_rf(X, y_star, trees = num_trees, seed_iter = seed)
      f_hat <- rf_res$pred
      final_rf <- rf_res

      r <- y - f_hat

      nll_spatial <- function(par) {
        s2 <- par[1]
        rho <- par[2]
        if (s2 <= 0 || rho <= -0.98 || rho >= 0.98) return(1e10)

        A <- I_mat - rho * W
        A_inv <- tryCatch(solve(A), error = function(e) NULL)
        if (is.null(A_inv)) return(1e10)
        Sigma_u <- s2 * (A_inv %*% t(A_inv))
        V <- Sigma_u + diag(psi)

        L <- tryCatch(chol(V), error = function(e) NULL)
        if (is.null(L)) return(1e10)
        log_det <- 2 * sum(log(diag(L)))
        v_inv_r <- backsolve(L, forwardsolve(t(L), r))
        0.5 * (log_det + sum(r * v_inv_r))
      }

      opt_res <- stats::optim(
        par = c(sigma2_u, rho_val),
        fn = nll_spatial,
        method = "L-BFGS-B",
        lower = c(1e-5, -0.95),
        upper = c(max(10 * stats::var(y), 10), 0.95)
      )

      sigma2_u_new <- opt_res$par[1]
      rho_new <- opt_res$par[2]

      # Compute Spatial BLUP
      A_inv <- solve(I_mat - rho_new * W)
      Sigma_u <- sigma2_u_new * (A_inv %*% t(A_inv))
      V <- Sigma_u + diag(psi)
      u_new <- as.vector(Sigma_u %*% solve(V, r))

      delta_u <- sqrt(mean((u_new - u)^2)) / (stats::sd(y) + 1e-6)
      u <- u_new
      sigma2_u <- sigma2_u_new
      rho_val <- rho_new
      loglik_val <- -opt_res$value

      if (verbose) {
        cli::cli_text("  Iteration {iter}: sigma2_u = {round(sigma2_u, 5)}, rho = {round(rho_val, 4)}, delta_u = {format.default(delta_u, scientific = TRUE, digits = 3)}")
      }

      if (delta_u < tol) {
        converged <- TRUE
        break
      }
    }

    theta_merf <- f_hat + u
    gamma_vec <- sigma2_u / (sigma2_u + psi)
  }

  # --- 4. Parametric Bootstrap for MSE Estimation ---
  if (mse_type == "bootstrap" && B > 0) {
    if (verbose) cli::cli_inform("Computing Parametric Bootstrap MSE with {B} replications...")

    boot_sq_err <- matrix(0, nrow = D, ncol = B)
    # Use reduced trees and max_iter for faster bootstrap iterations
    boot_trees <- min(150, num_trees)
    boot_max_iter <- min(8, max_iter)

    for (b in seq_len(B)) {
      seed_b <- if (!is.null(seed)) (seed + 1000 + b) else NULL
      if (!is.null(seed_b)) set.seed(seed_b)

      # Generate synthetic true population parameter
      if (!is_nested && !is_spatial) {
        u_b <- stats::rnorm(D, 0, sqrt(sigma2_u))
        theta_b <- f_hat + u_b
      } else if (is_nested) {
        u_maj_b <- stats::rnorm(J, 0, sqrt(sigma2_area))
        names(u_maj_b) <- major_areas
        v_sub_b <- stats::rnorm(D, 0, sqrt(sigma2_sub))
        theta_b <- f_hat + u_maj_b[major_area_vec] + v_sub_b
      } else if (is_spatial) {
        A_inv <- solve(diag(D) - rho_val * W)
        u_b <- as.vector(A_inv %*% stats::rnorm(D, 0, sqrt(sigma2_u)))
        theta_b <- f_hat + u_b
      }

      # Generate pseudo-sample survey estimate
      e_b <- stats::rnorm(D, 0, sqrt(psi))
      y_b <- theta_b + e_b

      # Fast MERF re-fit on pseudo-sample
      u_boot <- rep(0, D)
      s2_boot <- sigma2_u
      for (b_iter in seq_len(boot_max_iter)) {
        y_star_b <- y_b - u_boot
        rf_b <- fit_rf(X, y_star_b, trees = boot_trees, seed_iter = seed_b)
        f_b <- rf_b$pred
        r_b <- y_b - f_b

        opt_b <- stats::optimize(
          function(s2) {
            v <- s2 + psi
            if (any(v <= 0)) return(1e10)
            0.5 * sum(log(v) + (r_b^2) / v)
          },
          interval = c(0, max(10 * stats::var(y_b), 10))
        )
        s2_boot <- opt_b$minimum
        g_b <- s2_boot / (s2_boot + psi)
        u_boot <- g_b * r_b
      }
      theta_hat_b <- f_b + u_boot
      boot_sq_err[, b] <- (theta_hat_b - theta_b)^2
    }

    mse_est <- rowMeans(boot_sq_err)
    sd_est <- sqrt(mse_est)
    rse_est <- abs(sd_est / theta_merf) * 100
    ci_lower <- theta_merf - 1.96 * sd_est
    ci_upper <- theta_merf + 1.96 * sd_est

  } else {
    # Analytical Plug-in Approximation
    mse_est <- (1 - gamma_vec) * sigma2_u
    sd_est <- sqrt(pmax(1e-8, mse_est))
    rse_est <- abs(sd_est / theta_merf) * 100
    ci_lower <- theta_merf - 1.96 * sd_est
    ci_upper <- theta_merf + 1.96 * sd_est
  }

  # --- 5. Assemble Return Object ---
  df_estimates <- data.frame(
    domain = domain_vec,
    y = y,
    merf = theta_merf,
    hb = theta_merf,          # Compatible alias for benchmark()
    rf_pred = f_hat,
    random_effect = u,
    gamma = gamma_vec,
    sd = sd_est,
    mse = mse_est,
    rse = rse_est,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    vardir = psi,
    stringsAsFactors = FALSE
  )

  if (is_nested) {
    df_estimates$subarea <- subarea_vec
  }

  hyperparams <- list(
    sigma2_u = sigma2_u,
    loglik = loglik_val,
    iterations = iter,
    converged = converged
  )

  if (is_nested) {
    hyperparams$sigma2_area <- sigma2_area
    hyperparams$sigma2_subarea <- sigma2_sub
    hyperparams$icc_nested <- sigma2_area / (sigma2_area + sigma2_sub)
  }
  if (is_spatial) {
    hyperparams$rho_spatial <- rho_val
  }

  # Sorted Variable Importance
  vimp_sorted <- sort(final_rf$importance, decreasing = TRUE)

  res <- structure(
    list(
      estimates = df_estimates,
      df_hb = df_estimates,       # Compatible alias for benchmark()
      data = data,                # Original data for benchmarking weights and groups
      hyperparams = hyperparams,
      importance = vimp_sorted,
      forest = final_rf$fit,
      is_nested = is_nested,
      is_spatial = is_spatial,
      engine = engine,
      mse_type = mse_type,
      B = if (mse_type == "bootstrap") B else 0,
      family = "gaussian",
      gvf = gvf_obj,
      call = cl
    ),
    class = c("fastsaegpu_merf", "fastsaegpu_model")
  )

  res
}
