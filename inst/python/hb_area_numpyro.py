"""
GPU-Accelerated Hierarchical Bayesian Small Area Estimation (SAE) via NumPyro (JAX).
Supports Apple Silicon (Metal) and NVIDIA (CUDA) with graceful CPU fallback.
Implements non-centered parameterization, stable AR(1)/RW(1) operators,
and fast Kronecker tensor contraction for spatio-temporal interactions.
"""

import os
import sys
import numpy as np
from scipy.special import logsumexp

def init_jax_environment(device="auto"):
    """Initializes JAX on the requested accelerator with graceful CPU fallback."""
    import jax
    active_dev_str = "cpu"
    preferred = device
    if preferred == "auto":
        preferred = "metal" if sys.platform == "darwin" else "cuda"
    
    try:
        if preferred in ("metal", "cuda", "gpu"):
            backend_target = "metal" if preferred == "metal" else "cuda"
            try:
                devs = jax.devices(backend_target)
                if len(devs) > 0:
                    jax.config.update("jax_platform_name", backend_target)
                    active_dev_str = f"{backend_target}:{str(devs[0])}"
                else:
                    jax.config.update("jax_platform_name", "cpu")
                    active_dev_str = "cpu (fallback)"
            except Exception:
                jax.config.update("jax_platform_name", "cpu")
                active_dev_str = "cpu (fallback)"
        else:
            jax.config.update("jax_platform_name", "cpu")
            active_dev_str = "cpu"
    except Exception:
        jax.config.update("jax_platform_name", "cpu")
        active_dev_str = "cpu (fallback)"
        
    import jax.numpy as jnp
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_median
    
    return jax, jnp, numpyro, dist, MCMC, NUTS, init_to_median, active_dev_str

def build_temporal_operator(T, temporal, rho, jnp):
    """
    Constructs a numerically stable non-centered 1D temporal operator (T x T).
    Avoids 0**0 gradient NaN singularity in JAX reverse-mode AD.
    """
    if temporal in ("none", "iid") or T <= 1:
        return jnp.eye(T)
    elif temporal == "rw1":
        # Cumulative sum with sum-to-zero projection across time periods
        # Ensures strict identifiability with fixed intercept
        L_cumsum = jnp.tril(jnp.ones((T, T)))
        P_zero = jnp.eye(T) - (1.0 / T) * jnp.ones((T, T))
        return jnp.dot(P_zero, L_cumsum)
    elif temporal == "ar1":
        # Lower triangular Toeplitz AR(1) operator
        idx = jnp.arange(T)
        diff = idx[:, None] - idx[None, :]
        mask = diff > 0
        # Clip powers to avoid rho**0 singularity at rho = 0
        powers_clipped = jnp.where(mask, diff, 1)
        # Explicit 1.0 on diagonal
        L = jnp.where(mask, rho ** powers_clipped, jnp.eye(T))
        # Scale column 0 for stationary distribution
        scale_col0 = 1.0 / jnp.sqrt(jnp.maximum(1.0 - rho**2, 1e-4))
        L = L.at[:, 0].multiply(scale_col0)
        return L
    return jnp.eye(T)

def fit_numpyro_hb(
    y,
    X,
    domain_idx,
    time_idx=None,
    vardir=None,
    trials=None,
    exposure=None,
    D=1,
    T=1,
    family="gaussian",
    spatial="none",
    temporal="none",
    st_interaction="none",
    W_adj=None,
    scale_factor=1.0,
    num_warmup=500,
    num_samples=1000,
    num_chains=2,
    device="auto",
    seed=42
):
    jax, jnp, numpyro, dist, MCMC, NUTS, init_to_median, dev_str = init_jax_environment(device)

    N, P = X.shape
    y_np = np.asarray(y, dtype=np.float32)
    X_jnp = jnp.asarray(X, dtype=jnp.float32)
    domain_jnp = jnp.asarray(domain_idx, dtype=jnp.int32)
    time_jnp = jnp.asarray(time_idx, dtype=jnp.int32) if time_idx is not None else None
    
    # Missing / unsampled domains mask
    observed_mask = ~np.isnan(y_np)
    obs_idx = np.where(observed_mask)[0]
    y_obs = jnp.asarray(y_np[obs_idx], dtype=jnp.float32)

    # Convert optional vectors safely
    vardir_np = np.asarray(vardir, dtype=np.float32) if vardir is not None else None
    trials_np = np.asarray(trials, dtype=np.float32) if trials is not None else None
    exposure_np = np.asarray(exposure, dtype=np.float32) if exposure is not None else None

    vardir_obs = jnp.asarray(vardir_np[obs_idx], dtype=jnp.float32) if vardir_np is not None else None
    trials_obs = jnp.asarray(trials_np[obs_idx], dtype=jnp.float32) if trials_np is not None else None
    exposure_obs = jnp.asarray(exposure_np[obs_idx], dtype=jnp.float32) if exposure_np is not None else None

    # Spatial Spectral Decomposition (ICAR basis)
    icar_basis = None
    icar_rank = 0
    if spatial in ("besag", "bym2") and W_adj is not None:
        deg = np.sum(W_adj, axis=1)
        L_spatial = np.diag(deg) - W_adj
        evals, evecs = np.linalg.eigh(L_spatial)
        pos = evals > 1e-6
        evals_pos = evals[pos]
        evecs_pos = evecs[:, pos]
        # Unit-variance scaled ICAR basis (Riebler et al., 2016)
        icar_basis = jnp.asarray(evecs_pos / np.sqrt(evals_pos * scale_factor), dtype=jnp.float32)
        icar_rank = icar_basis.shape[1]

    def model():
        # Regression coefficients
        beta = numpyro.sample("beta", dist.Normal(0.0, 10.0).expand([P]))
        linpred_fixed = jnp.dot(X_jnp, beta)

        # 1. Spatial Random Effect
        u_spatial = jnp.zeros(D)
        # Avoid double-counting spatial field if separable spatio-temporal structure is active
        if st_interaction != "separable":
            if spatial == "none":
                sigma_s = numpyro.sample("sigma_s", dist.HalfNormal(1.0))
                z_s = numpyro.sample("z_s", dist.Normal(0.0, 1.0).expand([D]))
                u_spatial = sigma_s * z_s
                numpyro.deterministic("sigma2_u", sigma_s ** 2)
            elif spatial == "besag":
                sigma_s = numpyro.sample("sigma_s", dist.HalfNormal(1.0))
                z_icar = numpyro.sample("z_icar", dist.Normal(0.0, 1.0).expand([icar_rank]))
                u_spatial = sigma_s * jnp.dot(icar_basis, z_icar)
                numpyro.deterministic("sigma2_u", sigma_s ** 2)
            elif spatial == "bym2":
                sigma_s = numpyro.sample("sigma_s", dist.HalfNormal(1.0))
                phi = numpyro.sample("phi", dist.Beta(1.0, 1.0))
                z_iid = numpyro.sample("z_iid", dist.Normal(0.0, 1.0).expand([D]))
                z_icar = numpyro.sample("z_icar", dist.Normal(0.0, 1.0).expand([icar_rank]))
                u_icar = jnp.dot(icar_basis, z_icar)
                u_spatial = sigma_s * (jnp.sqrt(1.0 - phi) * z_iid + jnp.sqrt(phi) * u_icar)
                numpyro.deterministic("sigma2_u", sigma_s ** 2)
                numpyro.deterministic("phi_est", phi)

        # 2. Temporal Random Effect
        u_temporal = jnp.zeros(T)
        rho_t = 0.0
        if temporal != "none" and T > 1:
            sigma_t = numpyro.sample("sigma_t", dist.HalfNormal(1.0))
            numpyro.deterministic("sigma2_t", sigma_t ** 2)
            if temporal == "ar1":
                rho_t = numpyro.sample("rho_t", dist.Uniform(-0.99, 0.99))
            C_T = build_temporal_operator(T, temporal, rho_t, jnp)
            z_t = numpyro.sample("z_t", dist.Normal(0.0, 1.0).expand([T]))
            u_temporal = sigma_t * jnp.dot(C_T, z_t)

        # 3. Spatio-Temporal Interaction
        u_st = jnp.zeros(N)
        if st_interaction != "none" and temporal != "none" and T > 1:
            sigma_st = numpyro.sample("sigma_st", dist.HalfNormal(1.0))
            C_T = build_temporal_operator(T, temporal, rho_t, jnp)

            if st_interaction == "type1":
                # Unstructured space x Unstructured time (I_D (x) I_T)
                z_st = numpyro.sample("z_st", dist.Normal(0.0, 1.0).expand([D, T]))
                u_st_mat = sigma_st * z_st
                u_st = u_st_mat[domain_jnp, time_jnp]
            elif st_interaction in ("type2", "domain-specific"):
                # Unstructured space x Structured time (I_D (x) Q_t)
                z_st = numpyro.sample("z_st", dist.Normal(0.0, 1.0).expand([D, T]))
                u_st_mat = sigma_st * jnp.matmul(z_st, C_T.T)
                u_st = u_st_mat[domain_jnp, time_jnp]
            elif st_interaction == "type3":
                # Structured space x Unstructured time (Q_s (x) I_T)
                if icar_basis is not None:
                    z_st = numpyro.sample("z_st", dist.Normal(0.0, 1.0).expand([icar_rank, T]))
                    u_st_mat = sigma_st * jnp.matmul(icar_basis, z_st)
                    u_st = u_st_mat[domain_jnp, time_jnp]
            elif st_interaction in ("type4", "separable"):
                # Structured space x Structured time (Q_s (x) Q_t via fast Kronecker contraction)
                if icar_basis is not None:
                    z_st = numpyro.sample("z_st", dist.Normal(0.0, 1.0).expand([icar_rank, T]))
                    temp_filt = jnp.matmul(z_st, C_T.T)
                    u_st_mat = sigma_st * jnp.matmul(icar_basis, temp_filt)
                    u_st = u_st_mat[domain_jnp, time_jnp]

        # Total latent linear predictor
        rand_total = u_spatial[domain_jnp]
        if time_jnp is not None:
            rand_total = rand_total + u_temporal[time_jnp] + u_st
        else:
            rand_total = rand_total + u_st

        eta = linpred_fixed + rand_total
        numpyro.deterministic("linear_pred", eta)
        numpyro.deterministic("rand_eff", rand_total)

        # Likelihood and observation sampling
        if family == "gaussian":
            mu = eta
            numpyro.deterministic("hb_est", mu)
            if vardir_obs is not None:
                scale_obs = jnp.sqrt(vardir_obs)
            else:
                sigma_e = numpyro.sample("sigma_e", dist.HalfNormal(1.0))
                scale_obs = sigma_e
            numpyro.sample("y_obs", dist.Normal(mu[obs_idx], scale_obs), obs=y_obs)

        elif family == "binomial":
            p = jax.nn.sigmoid(eta)
            numpyro.deterministic("hb_est", p)
            numpyro.sample("y_obs", dist.Binomial(total_count=trials_obs, probs=p[obs_idx]), obs=y_obs)

        elif family == "poisson":
            rate = jnp.exp(eta)
            numpyro.deterministic("hb_est", rate)
            lam = rate[obs_idx] * (exposure_obs if exposure_obs is not None else 1.0)
            numpyro.sample("y_obs", dist.Poisson(rate=lam), obs=y_obs)

        elif family == "beta":
            p = jax.nn.sigmoid(eta)
            numpyro.deterministic("hb_est", p)
            if vardir_obs is not None:
                phi_beta = jnp.maximum((y_obs * (1.0 - y_obs) / vardir_obs) - 1.0, 1.0)
            elif trials_obs is not None:
                phi_beta = jnp.maximum(trials_obs - 1.0, 1.0)
            else:
                phi_beta = numpyro.sample("phi_beta", dist.HalfNormal(10.0))
            a = jnp.maximum(p[obs_idx] * phi_beta, 1e-4)
            b = jnp.maximum((1.0 - p[obs_idx]) * phi_beta, 1e-4)
            numpyro.sample("y_obs", dist.Beta(a, b), obs=y_obs)

    # Run MCMC on GPU/Accelerator
    rng_key = jax.random.PRNGKey(seed)
    kernel = NUTS(model, init_strategy=init_to_median, target_accept_prob=0.85, max_tree_depth=10)
    mcmc = MCMC(
        kernel,
        num_warmup=num_warmup,
        num_samples=num_samples,
        num_chains=num_chains,
        chain_method="vectorized" if num_chains > 1 else "sequential"
    )
    mcmc.run(rng_key)
    samples = mcmc.get_samples()

    # Extract posterior statistics
    beta_samples = np.asarray(samples["beta"])
    beta_mean = np.mean(beta_samples, axis=0).tolist()
    beta_sd = np.std(beta_samples, axis=0).tolist()
    beta_ci_lower = np.percentile(beta_samples, 2.5, axis=0).tolist()
    beta_ci_upper = np.percentile(beta_samples, 97.5, axis=0).tolist()

    hb_samples = np.asarray(samples["hb_est"])
    hb_mean = np.mean(hb_samples, axis=0).tolist()
    hb_sd = np.std(hb_samples, axis=0).tolist()
    hb_ci_lower = np.percentile(hb_samples, 2.5, axis=0).tolist()
    hb_ci_upper = np.percentile(hb_samples, 97.5, axis=0).tolist()

    linpred_samples = np.asarray(samples["linear_pred"])
    linpred_mean = np.mean(linpred_samples, axis=0).tolist()

    rand_eff_samples = np.asarray(samples["rand_eff"])
    rand_eff_mean = np.mean(rand_eff_samples, axis=0).tolist()

    # Hyperparameters
    hyperparams = {}
    if "sigma2_u" in samples:
        hyperparams["sigma2_u"] = float(np.mean(samples["sigma2_u"]))
    if "sigma2_t" in samples:
        hyperparams["sigma2_t"] = float(np.mean(samples["sigma2_t"]))
    if "phi_est" in samples:
        hyperparams["phi"] = float(np.mean(samples["phi_est"]))
    if "rho_t" in samples:
        hyperparams["rho_t"] = float(np.mean(samples["rho_t"]))

    # Accurate Pointwise Log-Likelihood, WAIC, and DIC
    S_total = hb_samples.shape[0]
    y_obs_val = y_np[obs_idx]
    
    if family == "gaussian":
        sigma_val = np.sqrt(vardir_np[obs_idx]) if vardir_np is not None else float(np.mean(samples.get("sigma_e", 1.0)))
        mu_post = hb_samples[:, obs_idx]
        ll = -0.5 * np.log(2.0 * np.pi * sigma_val**2) - 0.5 * ((y_obs_val - mu_post) / sigma_val)**2
        mu_mean = np.mean(mu_post, axis=0)
        ll_mean = -0.5 * np.log(2.0 * np.pi * sigma_val**2) - 0.5 * ((y_obs_val - mu_mean) / sigma_val)**2
    elif family == "binomial":
        p_post = np.clip(hb_samples[:, obs_idx], 1e-6, 1.0 - 1e-6)
        n_val = trials_np[obs_idx]
        from scipy.special import gammaln
        log_comb = gammaln(n_val + 1) - gammaln(y_obs_val + 1) - gammaln(n_val - y_obs_val + 1)
        ll = log_comb + y_obs_val * np.log(p_post) + (n_val - y_obs_val) * np.log(1.0 - p_post)
        p_mean = np.mean(p_post, axis=0)
        ll_mean = log_comb + y_obs_val * np.log(p_mean) + (n_val - y_obs_val) * np.log(1.0 - p_mean)
    elif family == "poisson":
        rate_post = np.clip(hb_samples[:, obs_idx], 1e-6, 1e8)
        e_val = exposure_np[obs_idx] if exposure_np is not None else 1.0
        lam_post = rate_post * e_val
        from scipy.special import gammaln
        ll = y_obs_val * np.log(lam_post) - lam_post - gammaln(y_obs_val + 1)
        lam_mean = np.mean(lam_post, axis=0)
        ll_mean = y_obs_val * np.log(lam_mean) - lam_mean - gammaln(y_obs_val + 1)
    else:
        ll = np.zeros((S_total, len(obs_idx)))
        ll_mean = np.zeros(len(obs_idx))

    lppd = float(np.sum(logsumexp(ll, axis=0) - np.log(S_total)))
    p_waic = float(np.sum(np.var(ll, axis=0, ddof=1)))
    waic = float(-2.0 * (lppd - p_waic))

    D_bar = float(-2.0 * np.mean(np.sum(ll, axis=1)))
    D_hat = float(-2.0 * np.sum(ll_mean))
    p_d = float(D_bar - D_hat)
    dic = float(D_bar + p_d)

    return {
        "beta_mean": beta_mean,
        "beta_sd": beta_sd,
        "beta_ci_lower": beta_ci_lower,
        "beta_ci_upper": beta_ci_upper,
        "hb_mean": hb_mean,
        "hb_sd": hb_sd,
        "hb_ci_lower": hb_ci_lower,
        "hb_ci_upper": hb_ci_upper,
        "linpred_mean": linpred_mean,
        "rand_eff_mean": rand_eff_mean,
        "hyperparameters": hyperparams,
        "waic": waic,
        "p_waic": p_waic,
        "dic": dic,
        "p_dic": p_d,
        "device_used": dev_str
    }
