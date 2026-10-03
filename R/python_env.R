#' Check if NumPyro and JAX environment is available
#'
#' @param device Character: "auto", "metal", "cuda", or "cpu".
#' @return Logical indicating whether the environment is properly configured.
#' @export
check_numpyro_available <- function(device = "auto") {
  if (device == "cpu" || (device == "auto" && Sys.info()["sysname"] == "Darwin")) {
    if (Sys.getenv("JAX_PLATFORMS") == "") {
      Sys.setenv(JAX_PLATFORMS = "cpu")
    }
  }
  
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    cli::cli_alert_warning("Package {.pkg reticulate} is not installed.")
    return(FALSE)
  }

  if (!reticulate::py_available()) {
    try(reticulate::use_virtualenv("r-numpyro-gpu", required = FALSE), silent = TRUE)
  }
  
  has_np <- reticulate::py_module_available("numpyro")
  has_jax <- reticulate::py_module_available("jax")
  
  if (!has_np || !has_jax) {
    return(FALSE)
  }
  
  tryCatch({
    jax <- reticulate::import("jax")
    devs <- jax$devices()
    cli::cli_alert_success("JAX devices found: {paste(vapply(devs, as.character, character(1)), collapse = ', ')}")
    TRUE
  }, error = function(e) {
    FALSE
  })
}

#' Setup or initialize NumPyro Python environment with GPU support
#'
#' @param envname Name of the conda or virtualenv environment. Default "r-numpyro-gpu".
#' @param method "auto", "virtualenv", or "conda".
#' @param device "auto", "metal" (Apple Silicon), "cuda" (NVIDIA), or "cpu".
#' @export
setup_numpyro_env <- function(envname = "r-numpyro-gpu", method = "auto", device = "auto") {
  if (!requireNamespace("reticulate", quietly = TRUE)) {
    cli::cli_abort("Package {.pkg reticulate} is required. Install via install.packages('reticulate').")
  }
  
  sys_info <- Sys.info()
  is_mac <- sys_info["sysname"] == "Darwin"
  is_arm <- sys_info["machine"] == "arm64"
  
  if (device == "auto") {
    if (is_mac && is_arm) {
      device <- "metal"
    } else {
      has_nvidia <- nzchar(Sys.which("nvidia-smi"))
      device <- if (has_nvidia) "cuda" else "cpu"
    }
  }
  
  cli::cli_h2("Configuring NumPyro environment for target device: {.val {device}}")
  
  pkgs <- c("numpy", "scipy", "numpyro")
  if (device == "metal") {
    pkgs <- c(pkgs, "jax==0.5.0", "jaxlib==0.5.0", "jax-metal==0.1.1")
  } else if (device == "cuda") {
    pkgs <- c(pkgs, "jax[cuda12]")
  } else {
    pkgs <- c(pkgs, "jax", "jaxlib")
  }
  
  cli::cli_alert_info("Creating/updating virtual environment {.val {envname}} with: {paste(pkgs, collapse = ', ')}")
  reticulate::virtualenv_create(envname)
  reticulate::virtualenv_install(envname, packages = pkgs)
  reticulate::use_virtualenv(envname, required = TRUE)
  
  cli::cli_alert_success("NumPyro GPU environment successfully configured!")
  invisible(TRUE)
}

#' Load and return the internal Python NumPyro backend
#' @noRd
.get_numpyro_backend <- function(device = "auto") {
  if (!check_numpyro_available(device = device)) {
    cli::cli_abort(c(
      "NumPyro and JAX are not available in the current Python environment.",
      "i" = "Run {.code setup_numpyro_env(device = '{device}')} or configure a Python environment with jax and numpyro."
    ))
  }
  
  # Robust multi-path resolution
  candidates <- c(
    system.file("python", "hb_area_numpyro.py", package = "fastsaegpu"),
    file.path(getwd(), "inst", "python", "hb_area_numpyro.py"),
    file.path(dirname(getwd()), "mcmc", "inst", "python", "hb_area_numpyro.py"),
    Sys.getenv("FASTSAEGPU_PYTHON_PATH", "")
  )
  candidates <- candidates[nzchar(candidates) & file.exists(candidates)]
  
  if (length(candidates) == 0) {
    cli::cli_abort("Backend script {.file hb_area_numpyro.py} not found. Ensure the package is installed.")
  }
  
  backend <- reticulate::import_from_path("hb_area_numpyro", path = dirname(candidates[1]))
  backend
}
