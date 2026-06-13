"""
Calculate Empirical Bayes / Posthoc individualised model parameters
"""

import numpy as np
from scipy.optimize import minimize


def neg_log_posterior(
    eta, model, times, obs, dosing, theta, omega_inv, sigma_prop, sigma_add
):
    """
    Exact MAP objective:
      -log f(y|η) - log f(η)

    f(y|η) is a product of Gaussians
    f(η)   is N(0, Ω).

    ``model`` is the user-supplied prediction function (see ``compute_ebe``).
    """
    eta = np.array(eta)
    f = model(times, dosing, theta, eta)  # individual predictions

    # --- Residual variance  ---
    var = sigma_add**2 + (sigma_prop * f) ** 2

    # Gaussian log-likelihood
    # -log f(y|η) = 0.5 * Σ [ (y - f)²/var + log(var) ]
    nll = 0.5 * np.sum((obs - f) ** 2 / var + np.log(var))

    # Prior / shrinkage penalty N(0,OMEGA)
    # -log f(η) = 0.5 * η' Ω⁻¹ η
    prior = 0.5 * eta @ omega_inv @ eta

    return nll + prior


def compute_ebe(
    model, times, obs, dosing, theta, omega, sigma_prop=0, sigma_add=0, eta_init=None
):
    """
    Estimate individual random effects (Empirical Bayes / posthoc estimates) by
    MAP optimisation of ``neg_log_posterior``.

    Parameters
    ----------
    model : callable
        User-supplied prediction function ``model(times, dosing, theta, eta) -> ndarray``
        returning individual predictions evaluated at each measurement time in ``times``.
        It is fully generic: it owns the parameterisation (how ``eta`` maps to individual
        parameters) and how ``dosing`` is interpreted.
    times, obs : array_like
        Measurement times and the observations recorded at them.
    dosing : opaque
        Generic, user-defined dosing schedule passed through unchanged to ``model``. May be
        a scalar fixed dose or a time series of infusion rates (e.g. ``(dose_times,
        dose_rates)`` held zero-order between dose times). Its time axis is independent of
        ``times`` — dosing times and measurement times need not coincide.
        A ``model`` that integrates a time series can reuse
        ``pypmx.utils.continuize_order_zero(dose_times, dose_rates)`` to build the ``Inf(t)``
        zero-order-hold callable expected by ``pypmx.standard_models.pk_3comp``, integrate it
        with ``scipy.integrate.solve_ivp`` (cf. ``pypmx.ode.simulate_pkpd``) and evaluate the
        solution at ``times``.
    theta : array_like
        Population (fixed-effect) parameters.
    omega : ndarray
        Random-effect covariance matrix Ω.
    sigma_prop, sigma_add : float
        Proportional and additive residual error standard deviations.
    eta_init : array_like, optional
        Starting point for the optimiser (defaults to zeros).

    Returns
    -------
    (eta_hat, objective, result)
        The MAP estimate of ``eta``, the objective value, and the raw ``scipy`` result.
    """
    omega_inv = np.linalg.inv(omega)
    eta_init = np.zeros(omega.shape[0]) if eta_init is None else eta_init

    result = minimize(
        neg_log_posterior,
        x0=eta_init,
        args=(model, times, obs, dosing, theta, omega_inv, sigma_prop, sigma_add),
        method="L-BFGS-B",
        # options={"gtol": 1e-8},
    )
    return result.x, result.fun, result


if __name__ == "__main__":

    def model(t, dosing, theta, eta):
        CL = theta[0] * np.exp(eta[0])
        Vd = theta[1] * np.exp(eta[1])
        return (dosing / Vd) * np.exp(-(CL / Vd) * t)

    # ── Example ───────────────────────────────────────────────────────────────────
    THETA = np.array([5.0, 50.0])
    OMEGA = np.diag([0.09, 0.04])
    SIGMA_PROP = 0.10
    DOSE = 100.0
    TIMES = np.array([0.5, 1, 2, 4, 8, 12, 24])
    OBS = np.array([1.71, 1.42, 1.05, 0.56, 0.18, 0.06, 0.004])

    eta_hat, obj, res = compute_ebe(
        model, TIMES, OBS, DOSE, THETA, OMEGA, sigma_prop=SIGMA_PROP
    )

    print(f"ETA1 (CL): {eta_hat[0]:+.4f}")
    print(f"ETA2 (Vd): {eta_hat[1]:+.4f}")
    print(f"Ind. CL = {THETA[0] * np.exp(eta_hat[0]):.3f} L/h")
    print(f"Ind. Vd = {THETA[1] * np.exp(eta_hat[1]):.3f} L")
