"""Eigendecomposition solver for linear ODE systems.

The system is ``x' = S·x + B·r(t) + f``, where ``r(t)`` is a piecewise-constant
input rate (an infusion) and ``f`` is a constant forcing term (e.g. endogenous
production). Diagonalising ``S = V·diag(-λ)·V⁻¹`` decouples the system into
independent modes; each one is propagated exactly over a constant-rate step:

    a_i(t + dt) = a_i(t)·e^{-λ_i·dt} + v_i·(α_B,i·r + α_f,i)·φ(λ_i, dt)

with ``α_B = V⁻¹B``, ``α_f = V⁻¹f`` and ``φ(λ, dt) = (1 - e^{-λ·dt})/λ``, which
tends to ``dt`` as λ → 0, so zero eigenvalues (pure integrators) are safe.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pypmx.utils import rate_at

FloatArray = NDArray[np.float64]


class Modes(NamedTuple):
    """Modal form of ``x' = S·x + B·r + f``.

    The state in compartment space is ``x = Σ_i a_i``, where column ``i`` of the
    modal state matrix ``a`` (n_states × n_modes) is the contribution of mode i.
    """

    lambdas: NDArray  # decay constants, -eigenvalues of S (n_modes,)
    eigvecs: NDArray  # right eigenvectors, one per column (n_states, n_modes)
    alpha_B: NDArray  # V⁻¹·B: input weights per mode (n_modes,)
    alpha_f: NDArray  # V⁻¹·f: forcing weights per mode (n_modes,)
    y0_mod: NDArray  # initial state per mode (n_states, n_modes)


def eigendecomposition(
    S: ArrayLike,
    B: ArrayLike,
    f: ArrayLike | None = None,
    y0: ArrayLike | None = None,
    real_eigenvalues: bool = True,
) -> Modes:
    """Decompose ``x' = S·x + B·r + f`` into its modes.

    Modes are sorted by decay constant, slowest first. The modal weights are
    *not* divided by λ (see :func:`phi`), so a zero eigenvalue stays finite.
    With ``real_eigenvalues`` (the default, true for mammillary PK systems) the
    imaginary parts are dropped; otherwise everything stays complex.
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float).ravel()
    n = S.shape[0]

    eigvals, eigvecs = np.linalg.eig(S)
    if real_eigenvalues:
        eigvals = eigvals.real
        eigvecs = eigvecs.real
    lambdas = -eigvals
    order = np.argsort(lambdas)
    lambdas = lambdas[order]
    eigvecs = eigvecs[:, order]

    Vinv = np.linalg.inv(eigvecs)
    alpha_B = Vinv @ B
    if f is None:
        alpha_f = np.zeros_like(alpha_B)
    else:
        alpha_f = Vinv @ np.asarray(f, dtype=float).ravel()
    if y0 is None:
        y0_mod = np.zeros((n, n), dtype=eigvecs.dtype)
    else:
        y0_mod = eigvecs * (Vinv @ np.asarray(y0, dtype=float).ravel())

    return Modes(lambdas, eigvecs, alpha_B, alpha_f, y0_mod)


def phi(lambdas: NDArray, dt: float) -> NDArray:
    """``(1 - e^{-λ·dt})/λ``, continued to ``dt`` at λ = 0."""
    zero = lambdas == 0
    safe = np.where(zero, 1.0, lambdas)
    return np.where(zero, dt, -np.expm1(-lambdas * dt) / safe)


def eig_advan(
    S: ArrayLike,
    B: ArrayLike,
    meas_time: ArrayLike,
    infu_time: ArrayLike,
    infu_rate: ArrayLike,
    y0: ArrayLike | None = None,
    f: ArrayLike | None = None,
    lag: float = 0.0,
    real_eigenvalues: bool = True,
) -> FloatArray:
    """Solve ``x' = S·x + B·r(t - lag) + f`` and return x at ``meas_time``.

    ``r`` is piecewise constant: ``infu_rate[i]`` from ``infu_time[i]`` on, the
    last rate persists, and the rate is 0 before ``infu_time[0]``. The state
    ``y0`` (default 0) holds at the earliest infusion or measurement time. The
    breakpoints are the measurement times and the *lagged* infusion times, so
    the solution is exact for any lag.

    Returns an array of shape (n_meas, n_states).
    """
    meas_time = np.atleast_1d(np.asarray(meas_time, dtype=float))
    infu_time = np.atleast_1d(np.asarray(infu_time, dtype=float))
    infu_rate = np.atleast_1d(np.asarray(infu_rate, dtype=float))
    modes = eigendecomposition(S, B, f, y0, real_eigenvalues)

    t0 = min(infu_time.min(), meas_time.min())
    lagged_time = infu_time + lag
    all_times = np.unique(np.concatenate([[t0], lagged_time, meas_time]))
    # Compare against the lagged times themselves: ``(t + lag) - lag`` can
    # round below ``t`` and pick the previous rate for a whole segment.
    rates = rate_at(all_times[:-1], lagged_time, infu_rate)

    A = modes.y0_mod
    states = np.empty((len(all_times),) + A.shape, dtype=A.dtype)
    states[0] = A
    for k, (dt, rate) in enumerate(zip(np.diff(all_times), rates), start=1):
        source = modes.eigvecs * (modes.alpha_B * rate + modes.alpha_f)
        A = A * np.exp(-modes.lambdas * dt) + source * phi(modes.lambdas, dt)
        states[k] = A

    res = states[np.searchsorted(all_times, meas_time)].sum(axis=-1)
    return np.real(res)


def calculate_impulse_response(
    modes: Modes, t: ArrayLike, state: int = 0
) -> FloatArray:
    """Response of ``state`` to a unit bolus through B at t = 0."""
    t = np.asarray(t, dtype=float)
    weights = modes.eigvecs[state, :] * modes.alpha_B
    decay = np.exp(-modes.lambdas[:, np.newaxis] * t[np.newaxis, :])
    return np.real(weights @ decay)


def convolve(uir: NDArray, inf: NDArray, dt: float) -> NDArray:
    npoints = len(uir)
    e_res = np.convolve(uir, inf, mode="full")
    return e_res[:npoints] * dt
