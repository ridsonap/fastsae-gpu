#' fastsaegpu: GPU-Accelerated Hierarchical Bayesian Small Area Estimation
#'
#' Hierarchical Bayesian area-level Small Area Estimation (SAE) with
#' spatio-temporal dynamics powered by 'NumPyro' and 'JAX' on NVIDIA (CUDA)
#' GPUs or the CPU. The Apple Silicon (Metal) backend is detected, but NUTS
#' currently samples on the CPU there.
#' Provides spatial, temporal, and spatio-temporal interactions for
#' continuous, count, and bounded survey outcomes using Markov Chain
#' Monte Carlo (MCMC) sampling.
#'
#' @docType package
#' @name fastsaegpu-package
#' @aliases fastsaegpu
#' @keywords internal
"_PACKAGE"
