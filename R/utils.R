utils::globalVariables(c(".data", "original", "benchmarked", "Importance", "Variable", "merf", "y"))

#' Extract variable from data frame or use as-is
#' @noRd
.get_variable <- function(data, variable) {
  if (is.character(variable) && length(variable) == 1) {
    if (variable %in% colnames(data)) {
      return(data[[variable]])
    } else {
      cli::cli_abort('variable "{variable}" is not found in data')
    }
  } else if (inherits(variable, "formula")) {
    v_names <- all.vars(variable)
    if (length(v_names) == 1 && v_names %in% colnames(data)) {
      return(data[[v_names]])
    } else {
      cli::cli_abort("formula does not reference a valid single column in data")
    }
  } else if (length(variable) == nrow(data)) {
    return(variable)
  } else {
    cli::cli_abort("variable is not valid or length does not match data ({length(variable)} vs {nrow(data)})")
  }
}

#' Convert and validate spatial adjacency matrix
#' @noRd
.convert_spatial_weights <- function(W, n_domains, domain_names = NULL) {
  if (is.null(W)) {
    cli::cli_abort("Spatial matrix {.arg W} must be provided when spatial modeling is active.")
  }
  
  if (inherits(W, "listw")) {
    if (requireNamespace("spdep", quietly = TRUE)) {
      W_mat <- spdep::listw2mat(W)
    } else {
      cli::cli_abort("Package {.pkg spdep} is required to convert a listw object.")
    }
  } else if (inherits(W, "nb")) {
    if (requireNamespace("spdep", quietly = TRUE)) {
      W_mat <- spdep::nb2mat(W, style = "B", zero.policy = TRUE)
    } else {
      cli::cli_abort("Package {.pkg spdep} is required to convert an nb object.")
    }
  } else if (is.matrix(W) || inherits(W, "Matrix")) {
    W_mat <- as.matrix(W)
  } else {
    cli::cli_abort("Unsupported spatial object type for {.arg W}. Expected matrix, Matrix, nb, or listw.")
  }
  
  if (!is.null(domain_names) && !is.null(rownames(W_mat))) {
    dom_chr <- as.character(domain_names)
    if (all(dom_chr %in% rownames(W_mat))) {
      W_mat <- W_mat[dom_chr, dom_chr, drop = FALSE]
    }
  }
  
  if (nrow(W_mat) != n_domains || ncol(W_mat) != n_domains) {
    cli::cli_abort("W must be a {n_domains} x {n_domains} matrix.")
  }
  
  adj_mat <- (W_mat > 0 | t(W_mat) > 0) * 1.0
  diag(adj_mat) <- 0.0
  
  # Calculate ICAR scaling factor (Riebler et al., 2016)
  deg <- rowSums(adj_mat)
  L <- diag(deg) - adj_mat
  eig <- eigen(L, symmetric = TRUE)
  pos_idx <- which(eig$values > 1e-6)
  if (length(pos_idx) > 0) {
    V <- eig$vectors[, pos_idx, drop = FALSE]
    lambda_inv <- 1 / eig$values[pos_idx]
    Q_inv_diag <- rowSums((V^2) * rep(lambda_inv, each = nrow(V)))
    scale_factor <- exp(mean(log(pmax(Q_inv_diag, 1e-8))))
  } else {
    scale_factor <- 1.0
  }
  
  list(adj_mat = adj_mat, scale_factor = scale_factor)
}
