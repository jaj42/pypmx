"""
Calculate Empirical Bayes / Posthoc individualised model parameters
"""

import numpy as np
from scipy.optimize import minimize


def pk_model(t, dose, theta, eta):
    CL = theta[0] * np.exp(eta[0])
    Vd = theta[1] * np.exp(eta[1])
    return (dose / Vd) * np.exp(-(CL / Vd) * t)


def neg_log_posterior(
    eta, times, obs, dose, theta, omega_inv, sigma, error_model="proportional"
):
    """
    Exact MAP objective:
      -log f(y|η) - log f(η)

    f(y|η) is a product of Gaussians
    f(η)   is N(0, Ω).
    """
    eta = np.array(eta)
    f = pk_model(times, dose, theta, eta)  # individual predictions

    # --- Residual variance (depends on error model) ---
    if error_model == "additive":
        var = np.full_like(f, sigma**2)  # constant → log term drops out
    elif error_model == "proportional":
        var = (sigma * f) ** 2  # η-dependent → log term matters
    elif error_model == "combined":
        sigma_add, sigma_prop = sigma
        var = sigma_add**2 + (sigma_prop * f) ** 2
    else:
        raise ValueError(f"Unknown error model: {error_model}")

    # Gaussian log-likelihood
    # -log f(y|η) = 0.5 * Σ [ (y - f)²/var + log(var) ]
    nll = 0.5 * np.sum((obs - f) ** 2 / var + np.log(var))

    # Prior / shrinkage penalty N(0,OMEGA)
    # -log f(η) = 0.5 * η' Ω⁻¹ η
    prior = 0.5 * eta @ omega_inv @ eta

    return nll + prior


def compute_ebe(
    times, obs, dose, theta, omega, sigma, error_model="proportional", eta_init=None
):
    omega_inv = np.linalg.inv(omega)
    eta_init = np.zeros(omega.shape[0]) if eta_init is None else eta_init

    result = minimize(
        neg_log_posterior,
        x0=eta_init,
        args=(times, obs, dose, theta, omega_inv, sigma, error_model),
        method="L-BFGS-B",
        # options={"gtol": 1e-8},
    )
    return result.x, result.fun, result


# ── Example ───────────────────────────────────────────────────────────────────
THETA = np.array([5.0, 50.0])
OMEGA = np.diag([0.09, 0.04])
SIGMA_PROP = 0.10
DOSE = 100.0
TIMES = np.array([0.5, 1, 2, 4, 8, 12, 24])
OBS = np.array([1.71, 1.42, 1.05, 0.56, 0.18, 0.06, 0.004])

eta_hat, obj, res = compute_ebe(
    TIMES, OBS, DOSE, THETA, OMEGA, SIGMA_PROP, error_model="proportional"
)

print(f"ETA1 (CL): {eta_hat[0]:+.4f}")
print(f"ETA2 (Vd): {eta_hat[1]:+.4f}")
print(f"Ind. CL = {THETA[0] * np.exp(eta_hat[0]):.3f} L/h")
print(f"Ind. Vd = {THETA[1] * np.exp(eta_hat[1]):.3f} L")
