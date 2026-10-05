#' Mixed Effects Random Forest Small Area Estimation (MERF / FH-RF)
#'
#' Fits an enhanced Mixed Effects Random Forest (MERF) model for area-level
#' Small Area Estimation based on Krennmair & Schmid (2022), Hajjem et al. (2014),
#' and Bukhari et al. (2025). The linear fixed-effects component of the classical
#' Fay-Herriot model is replaced by a flexible non-parametric Random Forest,
#' while preserving area-level random effects and sampling error variance.
#'
#' Enhanced methodological features:
#' \itemize{
#'   \item \strong{Precision-Weighted Tree Splitting}: Prioritizes splits on areas
#'     with higher survey precision using inverse-variance weights (\eqn{w_i \propto 1 / (\sigma_u^2 + \psi_i)}).
#'   \item \strong{Out-of-Bag (OOB) Residuals in EM Loop}: Uses honest OOB predictions
#'     during EM iterations to eliminate in-sample overfitting and prevent artificial
#'     shrinkage of area random effect variance \eqn{\sigma_u^2}.
#'   \item \strong{Automated Feature Screening}: Prunes noise covariates based on
#'     permutation importance thresholding.
#'   \item \strong{Hyperparameter Auto-Tuning}: Grid-searches optimal \code{mtry}
#'     and \code{min_node_size} via minimum OOB prediction error.
#'   \item \strong{Parametric Bootstrap MSE}: Computes empirical bootstrap MSE, RSE,
#'     and 95\% confidence intervals (Krennmair & Schmid, 2022).
#'   \item \strong{Two-Level Nested & Spatial Extensions}: Supports nested sub-areas
#'     (Torabi & Rao, 2014) and spatial autoregressive SAR correlation.
#' }
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
#' @param weighted Logical; if \code{TRUE} (default), uses inverse-variance
#'   precision case weights (\eqn{w_i \propto 1 / (\sigma_u^2 + \psi_i)}) during
#'   Random Forest tree splitting to prioritize areas with higher survey precision.
#' @param use_oob Logical; if \code{TRUE} (default), computes Out-Of-Bag (OOB)
#'   residuals (\eqn{r_i^{\text{OOB}} = y_i - \hat{f}_{\text{OOB}}(\mathbf{x}_i)})
#'   during EM iterations to prevent artificial shrinkage of area random effect
#'   variance \eqn{\sigma_u^2} (Krennmair & Schmid, 2022).
#' @param feature_screening Logical; if \code{TRUE}, performs automated noise
#'   covariate pruning based on permutation importance after initial EM
#'   iterations (default: \code{FALSE}).
#' @param importance_threshold Numeric threshold for feature screening
#'   (default: \code{0.0}). Covariates with permutation importance \eqn{\le}
#'   threshold are pruned.
#' @param tune_params Logical; if \code{TRUE}, performs fast grid search for
#'   optimal \code{mtry} and \code{min_node_size} via minimum OOB prediction
#'   error on the initial response (default: \code{FALSE}).
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
#'   \item{selected_vars}{Character vector of active covariates retained after screening.}
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
#' # Fit MERF Area Model with precision weighting and OOB residuals
#' fit_rf <- merf_area(
#'   formula = y ~ x1 + x2,
#'   data = df,
#'   vardir = "vardir",
#'   domain = "domain",
#'   weighted = TRUE,
#'   use_oob = TRUE,
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
                      weighted = TRUE,
                      use_oob = TRUE,
                      feature_screening = FALSE,
                      importance_threshold = 0.0,
                      tune_params = FALSE,
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

  # --- 2. Hyperparameter Auto-Tuning (Optional) ---
  P <- ncol(X)
  if (isTRUE(tune_params) && engine == "ranger") {
    if (verbose) cli::cli_inform("Tuning Random Forest hyperparameters via minimum OOB prediction error...")
    mtry_cands <- unique(pmax(1, pmin(P, c(floor(sqrt(P) / 2), floor(sqrt(P)), floor(P / 2), P))))
    node_cands <- unique(pmax(1, c(3, 5, 10)))

    df_tune <- as.data.frame(X)
    df_tune$.target <- y
    best_oob <- Inf
    best_mtry <- max(1, floor(sqrt(P)))
    best_node <- min_node_size

    for (m_c in mtry_cands) {
      for (n_c in node_cands) {
        rf_try <- tryCatch(
          ranger::ranger(
            formula = .target ~ .,
            data = df_tune,
            num.trees = min(150, num_trees),
            mtry = m_c,
            min.node.size = n_c,
            importance = "none",
            num.threads = num_threads,
            seed = seed
          ),
          error = function(e) NULL
        )
        if (!is.null(rf_try) && !is.na(rf_try$prediction.error)) {
          if (rf_try$prediction.error < best_oob) {
            best_oob <- rf_try$prediction.error
            best_mtry <- m_c
            best_node <- n_c
          }
        }
      }
    }
    mtry <- best_mtry
    min_node_size <- best_node
    if (verbose) cli::cli_inform("Auto-tuned hyperparameters: mtry = {mtry}, min_node_size = {min_node_size} (OOB MSE: {round(best_oob, 5)})")
  } else if (is.null(mtry)) {
    mtry <- max(1, floor(sqrt(P)))
  }

  # --- 3. Random Forest Helper Function ---
  fit_rf <- function(X_mat, target_vec, trees = num_trees, cur_weights = NULL, seed_iter = NULL) {
    df_rf <- as.data.frame(X_mat)
    df_rf$.target <- target_vec
    p_cur <- ncol(X_mat)
    cur_mtry <- min(mtry, p_cur)

    # Normalize weights if provided
    cw <- NULL
    if (!is.null(cur_weights) && isTRUE(weighted)) {
      cw <- cur_weights / mean(cur_weights)
    }

    if (engine == "ranger") {
      rf_fit <- ranger::ranger(
        formula = .target ~ .,
        data = df_rf,
        num.trees = trees,
        mtry = cur_mtry,
        min.node.size = min_node_size,
        importance = "permutation",
        case.weights = cw,
        num.threads = num_threads,
        seed = seed_iter
      )
      in_preds <- stats::predict(rf_fit, data = df_rf)$predictions
      oob_preds <- rf_fit$predictions
      if (any(is.na(oob_preds))) {
        oob_preds <- ifelse(is.na(oob_preds), in_preds, oob_preds)
      }
      vimp <- rf_fit$variable.importance
      return(list(fit = rf_fit, pred = in_preds, pred_oob = oob_preds, importance = vimp))
    } else {
      rf_fit <- randomForest::randomForest(
        x = X_mat,
        y = target_vec,
        ntree = trees,
        mtry = cur_mtry,
        nodesize = min_node_size,
        importance = TRUE
      )
      in_preds <- stats::predict(rf_fit, newdata = X_mat)
      oob_preds <- rf_fit$predicted
      if (any(is.na(oob_preds))) {
        oob_preds <- ifelse(is.na(oob_preds), in_preds, oob_preds)
      }
      vimp <- rf_fit$importance[, 1]
      return(list(fit = rf_fit, pred = in_preds, pred_oob = oob_preds, importance = vimp))
    }
  }

  # --- 4. Expectation-Maximization (EM) Loop ---
  if (verbose) cli::cli_inform("Initializing MERF EM algorithm (max_iter = {max_iter}, tol = {tol})...")

  X_active <- X
  screened_done <- FALSE

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

      # Step 1b: Precision weights
      weights_iter <- if (isTRUE(weighted)) 1 / (sigma2_u + psi) else NULL

      # Step 2: Fit Random Forest
      rf_res <- fit_rf(X_active, y_star, trees = num_trees, cur_weights = weights_iter, seed_iter = seed)
      final_rf <- rf_res

      # Step 2b: Feature Screening (Noise pruning after iteration 1)
      if (isTRUE(feature_screening) && !screened_done && iter == 1 && ncol(X_active) > 1) {
        vimp <- rf_res$importance
        keep_vars <- names(vimp)[vimp > importance_threshold]
        if (length(keep_vars) == 0) {
          # Keep at least the top variable
          keep_vars <- names(sort(vimp, decreasing = TRUE))[1]
        }
        if (length(keep_vars) < ncol(X_active)) {
          if (verbose) {
            pruned_count <- ncol(X_active) - length(keep_vars)
            cli::cli_inform("Feature screening: pruned {pruned_count} noise covariates. Retained: {paste(keep_vars, collapse = ', ')}")
          }
          X_active <- X_active[, keep_vars, drop = FALSE]
          # Refit with active covariates
          rf_res <- fit_rf(X_active, y_star, trees = num_trees, cur_weights = weights_iter, seed_iter = seed)
          final_rf <- rf_res
        }
        screened_done <- TRUE
      }

      f_in <- rf_res$pred
      f_hat_eval <- if (isTRUE(use_oob)) rf_res$pred_oob else f_in

      # Step 3: Residuals
      r <- y - f_hat_eval

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

      # Check convergence
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

    # Point Estimates: use in-sample predictions for synthetic baseline plus random effects
    f_hat <- final_rf$pred
    theta_merf <- f_hat + u
    gamma_vec <- sigma2_u / (sigma2_u + psi)

  } else if (is_nested) {
    # Two-Level Nested MERF (Torabi & Rao, 2014)
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

      weights_iter <- if (isTRUE(weighted)) 1 / (sigma2_area + sigma2_sub + psi) else NULL
      rf_res <- fit_rf(X_active, y_star, trees = num_trees, cur_weights = weights_iter, seed_iter = seed)
      final_rf <- rf_res

      # Feature Screening
      if (isTRUE(feature_screening) && !screened_done && iter == 1 && ncol(X_active) > 1) {
        vimp <- rf_res$importance
        keep_vars <- names(vimp)[vimp > importance_threshold]
        if (length(keep_vars) == 0) keep_vars <- names(sort(vimp, decreasing = TRUE))[1]
        if (length(keep_vars) < ncol(X_active)) {
          X_active <- X_active[, keep_vars, drop = FALSE]
          rf_res <- fit_rf(X_active, y_star, trees = num_trees, cur_weights = weights_iter, seed_iter = seed)
          final_rf <- rf_res
        }
        screened_done <- TRUE
      }

      f_in <- rf_res$pred
      f_hat_eval <- if (isTRUE(use_oob)) rf_res$pred_oob else f_in
      r <- y - f_hat_eval

      # Analytical profile log-likelihood for (sigma2_area, sigma2_sub)
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

    f_hat <- final_rf$pred
    u <- u_major[major_area_vec] + v_sub
    sigma2_u <- sigma2_area
    theta_merf <- f_hat + u
    gamma_vec <- (sigma2_area + sigma2_sub) / (sigma2_area + sigma2_sub + psi)

  } else if (is_spatial) {
    # Spatial MERF — eigenvalue trick for fast log-det (Ord 1975)
    eig_W <- eigen(W, symmetric = FALSE, only.values = TRUE)$values
    eig_W <- Re(eig_W)
    u <- rep(0, D)
    sigma2_u <- max(0.01, stats::var(y) - mean(psi))
    rho_val <- 0.2
    converged <- FALSE
    final_rf <- NULL

    I_mat <- diag(D)
    for (iter in seq_len(max_iter)) {
      y_star <- y - u
      weights_iter <- if (isTRUE(weighted)) 1 / (sigma2_u + psi) else NULL
      rf_res <- fit_rf(X_active, y_star, trees = num_trees, cur_weights = weights_iter, seed_iter = seed)
      final_rf <- rf_res

      # Feature Screening
      if (isTRUE(feature_screening) && !screened_done && iter == 1 && ncol(X_active) > 1) {
        vimp <- rf_res$importance
        keep_vars <- names(vimp)[vimp > importance_threshold]
        if (length(keep_vars) == 0) keep_vars <- names(sort(vimp, decreasing = TRUE))[1]
        if (length(keep_vars) < ncol(X_active)) {
          X_active <- X_active[, keep_vars, drop = FALSE]
          rf_res <- fit_rf(X_active, y_star, trees = num_trees, cur_weights = weights_iter, seed_iter = seed)
          final_rf <- rf_res
        }
        screened_done <- TRUE
      }

      f_in <- rf_res$pred
      f_hat_eval <- if (isTRUE(use_oob)) rf_res$pred_oob else f_in
      r <- y - f_hat_eval

      A_inv_r_cached <- NULL
      nll_spatial <- function(par) {
        s2 <- par[1]
        rho <- par[2]
        if (s2 <= 0 || rho <= -0.98 || rho >= 0.98) return(1e10)

        # Fast log-det via eigenvalues: log|I - rho*W| = sum(log(1 - rho*lambda_i))
        ld_A <- sum(log(pmax(1 - rho * eig_W, 1e-8)))
        # Use Woodbury / direct solve without forming Sigma_u explicitly
        A <- I_mat - rho * W
        # Solve A^{-1} * r efficiently
        A_inv_r <- tryCatch(solve(A, r), error = function(e) NULL)
        if (is.null(A_inv_r)) return(1e10)
        # V = s2*(A'A)^{-1} + diag(psi) — compute log-det and quadratic form via Cholesky of V
        # For moderate D, form V directly but reuse ld_A
        A_inv <- tryCatch(solve(A), error = function(e) NULL)
        if (is.null(A_inv)) return(1e10)
        Sigma_u <- s2 * (A_inv %*% t(A_inv))
        V <- Sigma_u + diag(psi)

        L <- tryCatch(chol(V), error = function(e) NULL)
        if (is.null(L)) return(1e10)
        log_det_V <- 2 * sum(log(diag(L)))
        v_inv_r <- backsolve(L, forwardsolve(t(L), r))
        0.5 * (log_det_V + sum(r * v_inv_r))
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

    f_hat <- final_rf$pred
    theta_merf <- f_hat + u
    gamma_vec <- sigma2_u / (sigma2_u + psi)
  }

  # --- 5. Parametric Bootstrap for MSE Estimation ---
  if (mse_type == "bootstrap" && B > 0) {
    if (verbose) cli::cli_inform("Computing Parametric Bootstrap MSE with {B} replications...")

    boot_trees <- min(150, num_trees)
    boot_max_iter <- min(8, max_iter)

    # Single bootstrap replication helper (structure-aware)
    .one_boot <- function(b) {
      seed_b <- if (!is.null(seed)) (seed + 1000 + b) else NULL
      if (!is.null(seed_b)) set.seed(seed_b)

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
      e_b <- stats::rnorm(D, 0, sqrt(psi))
      y_b <- theta_b + e_b

      if (!is_nested && !is_spatial) {
        # Standard 1-level bootstrap re-fit
        u_boot <- rep(0, D)
        s2_boot <- sigma2_u
        for (b_iter in seq_len(boot_max_iter)) {
          y_star_b <- y_b - u_boot
          w_b <- if (isTRUE(weighted)) 1 / (s2_boot + psi) else NULL
          rf_b <- fit_rf(X_active, y_star_b, trees = boot_trees, cur_weights = w_b, seed_iter = seed_b)
          f_b_eval <- if (isTRUE(use_oob)) rf_b$pred_oob else rf_b$pred
          r_b <- y_b - f_b_eval
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
        theta_hat_b <- rf_b$pred + u_boot

      } else if (is_nested) {
        # Nested bootstrap re-fit: re-estimate sigma2_area & sigma2_subarea
        u_maj_boot <- stats::setNames(rep(0, J), major_areas)
        v_sub_boot <- rep(0, D)
        s2a_boot <- sigma2_area
        s2s_boot <- sigma2_sub
        for (b_iter in seq_len(boot_max_iter)) {
          u_exp_boot <- u_maj_boot[major_area_vec]
          y_star_b <- y_b - u_exp_boot - v_sub_boot
          w_b <- if (isTRUE(weighted)) 1 / (s2a_boot + s2s_boot + psi) else NULL
          rf_b <- fit_rf(X_active, y_star_b, trees = boot_trees, cur_weights = w_b, seed_iter = seed_b)
          f_b_eval <- if (isTRUE(use_oob)) rf_b$pred_oob else rf_b$pred
          r_b <- y_b - f_b_eval
          nll_nested_b <- function(par) {
            s2_a <- par[1]; s2_s <- par[2]
            if (s2_a < 0 || s2_s < 0) return(1e10)
            nll <- 0
            for (jj in seq_len(J)) {
              idx_jj <- area_idx_list[[jj]]
              psi_jj <- psi[idx_jj]; r_jj <- r_b[idx_jj]
              d_jj <- s2_s + psi_jj; w_jj <- 1 / d_jj
              W_sum_jj <- sum(w_jj); denom_jj <- 1 + s2_a * W_sum_jj
              nll <- nll + 0.5 * (sum(log(d_jj)) + log(max(1e-12, denom_jj)) + sum(w_jj * r_jj^2) - (s2_a / denom_jj) * sum(w_jj * r_jj)^2)
            }
            nll
          }
          opt_b <- stats::optim(par = c(s2a_boot, s2s_boot), fn = nll_nested_b, method = "L-BFGS-B", lower = c(1e-6, 1e-6))
          s2a_boot <- opt_b$par[1]; s2s_boot <- opt_b$par[2]
          for (jj in seq_len(J)) {
            idx_jj <- area_idx_list[[jj]]
            w_jj <- 1 / (s2s_boot + psi[idx_jj])
            u_maj_boot[jj] <- (s2a_boot / (1 + s2a_boot * sum(w_jj))) * sum(w_jj * r_b[idx_jj])
          }
          u_exp_new <- u_maj_boot[major_area_vec]
          v_sub_boot <- (s2s_boot / (s2s_boot + psi)) * (r_b - u_exp_new)
        }
        theta_hat_b <- rf_b$pred + u_maj_boot[major_area_vec] + v_sub_boot

      } else if (is_spatial) {
        # Spatial bootstrap re-fit: re-estimate sigma2_u & rho
        u_boot_sp <- rep(0, D)
        s2_boot_sp <- sigma2_u; rho_boot <- rho_val
        I_D <- diag(D)
        for (b_iter in seq_len(boot_max_iter)) {
          y_star_b <- y_b - u_boot_sp
          w_b <- if (isTRUE(weighted)) 1 / (s2_boot_sp + psi) else NULL
          rf_b <- fit_rf(X_active, y_star_b, trees = boot_trees, cur_weights = w_b, seed_iter = seed_b)
          f_b_eval <- if (isTRUE(use_oob)) rf_b$pred_oob else rf_b$pred
          r_b <- y_b - f_b_eval
          nll_sp_b <- function(par) {
            s2 <- par[1]; rho_p <- par[2]
            if (s2 <= 0 || rho_p <= -0.98 || rho_p >= 0.98) return(1e10)
            A_p <- I_D - rho_p * W
            A_inv_p <- tryCatch(solve(A_p), error = function(e) NULL)
            if (is.null(A_inv_p)) return(1e10)
            Sigma_p <- s2 * (A_inv_p %*% t(A_inv_p))
            V_p <- Sigma_p + diag(psi)
            L_p <- tryCatch(chol(V_p), error = function(e) NULL)
            if (is.null(L_p)) return(1e10)
            0.5 * (2 * sum(log(diag(L_p))) + sum(r_b * backsolve(L_p, forwardsolve(t(L_p), r_b))))
          }
          opt_b <- stats::optim(par = c(s2_boot_sp, rho_boot), fn = nll_sp_b, method = "L-BFGS-B", lower = c(1e-5, -0.95), upper = c(max(10 * stats::var(y_b), 10), 0.95))
          s2_boot_sp <- opt_b$par[1]; rho_boot <- opt_b$par[2]
          A_inv_n <- solve(I_D - rho_boot * W)
          Sigma_n <- s2_boot_sp * (A_inv_n %*% t(A_inv_n))
          V_n <- Sigma_n + diag(psi)
          u_boot_sp <- as.vector(Sigma_n %*% solve(V_n, r_b))
        }
        theta_hat_b <- rf_b$pred + u_boot_sp
      }
      (theta_hat_b - theta_b)^2
    }

    # Parallel when possible (mclapply on Unix, else serial)
    # Respect CRAN check limit: _R_CHECK_LIMIT_CORES_ / R_CHECK_LIMIT_CORES
    use_parallel <- .Platform$OS.type == "unix" && B >= 4 && requireNamespace("parallel", quietly = TRUE) &&
      identical(Sys.getenv("_R_CHECK_LIMIT_CORES_", unset = ""), "") &&
      identical(Sys.getenv("R_CHECK_LIMIT_CORES", unset = ""), "")
    if (use_parallel) {
      nc <- min(parallel::detectCores(logical = FALSE), B, 4L)
      boot_list <- parallel::mclapply(seq_len(B), .one_boot, mc.cores = nc)
      boot_sq_err <- do.call(cbind, boot_list)
    } else {
      boot_sq_err <- vapply(seq_len(B), .one_boot, numeric(D))
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

  # --- 6. Assemble Return Object ---
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
    converged = converged,
    weighted = weighted,
    use_oob = use_oob,
    feature_screening = feature_screening,
    tune_params = tune_params
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
      selected_vars = colnames(X_active),
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
