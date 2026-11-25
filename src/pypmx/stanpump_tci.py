"""STANPUMP-style target-controlled infusion (TCI).

The controller works on a modal state: the plasma (``Ap``) and effect-site
(``Ae``) concentrations are each a sum of exponential modes, advanced every
``delta_seconds`` with :func:`update`. The modes come from the unified PK model
(:class:`pypmx.pk.LinearPKModel` or :class:`pypmx.pk.PKParameters`) through
:func:`pypmx.eigen.eigendecomposition`, ordered STANPUMP-style (PK modes fastest
first, ke0 last).

PK models are in minutes; the controller runs in **seconds**. The conversion
happens here only (:func:`decompose`): lambdas are in s⁻¹ and rates in
amount/s.

This is the reference port of STANPUMP; :mod:`pypmx.tci` is the modern
controller (continuous UDFs, fixed-point peak time, Brent's method).
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

from pypmx.eigen import Modes, eigendecomposition
from pypmx.pk import LinearPKModel, PKParameters, system_matrix

FloatArray = NDArray[np.float64]

SECONDS_PER_MINUTE = 60.0


class UDFs(NamedTuple):
    """Unit disposition functions, one value per second."""

    p_udf: FloatArray  # Cp after a 1/s infusion from t = 0
    e_udf: FloatArray  # Ce after a 1/s infusion for delta_seconds


class Coefs(NamedTuple):
    """Steady-state concentration per mode for a 1/s infusion."""

    p_coef: FloatArray
    e_coef: FloatArray


class TCISetup(NamedTuple):
    """Everything :func:`model` needs, from :func:`prepare`."""

    lambdas: FloatArray  # s⁻¹, PK modes fastest first, ke0 last
    coefs: Coefs
    udfs: UDFs
    peak_time: int | None  # s, time to peak Ce after a delta_seconds bolus
    delta_seconds: int


def stanpump_order(modes: Modes, ke0: float) -> Modes:
    """Reorder modes STANPUMP-style: PK modes (fastest first), then ke0 last.

    The ke0 mode is the one whose decay constant is closest to ``ke0``.
    """
    lambdas = modes.lambdas
    ke0_idx = int(np.argmin(np.abs(lambdas - ke0)))
    pk_idx = [i for i in np.argsort(-lambdas.real) if i != ke0_idx]
    order = np.array([*pk_idx, ke0_idx])
    return Modes(
        lambdas[order],
        modes.eigvecs[:, order],
        modes.alpha_B[order],
        modes.alpha_f[order],
        modes.y0_mod[:, order],
    )


def _pk_parameters(model: LinearPKModel | PKParameters) -> PKParameters:
    if isinstance(model, PKParameters):
        return model
    if model.units.time != "min":
        raise ValueError(f"TCI expects a model in minutes, got {model.units.time!r}")
    return model.pk


def decompose(model: LinearPKModel | PKParameters) -> tuple[FloatArray, Coefs]:
    """Lambdas (s⁻¹) and per-mode coefficients for the TCI controller.

    ``model`` is a :class:`LinearPKModel` or :class:`PKParameters` in minutes.
    There is one mode per PK compartment plus one for ke0 when the model has an
    effect site. Without an effect site ``e_coef`` is all zeros.
    """
    p = _pk_parameters(model)
    S = system_matrix(p) / SECONDS_PER_MINUTE
    B = np.zeros(S.shape[0])
    B[0] = 1.0
    modes = eigendecomposition(S, B)
    if p.has_effect_site:
        modes = stanpump_order(modes, p.ke0 / SECONDS_PER_MINUTE)
    else:
        modes = Modes(*(m[..., ::-1] for m in modes))  # fastest first
    # Step response of state j: Σ_i v_ji·α_i/λ_i·(1 - e^{-λ_i·t}).
    weights = modes.eigvecs * (modes.alpha_B / modes.lambdas)
    p_coef = weights[0] / p.V1
    if p.has_effect_site:
        p_coef[-1] = 0.0  # the ke0 mode does not feed back into the PK block
        e_coef = weights[-1]
    else:
        e_coef = np.zeros_like(p_coef)
    return modes.lambdas, Coefs(p_coef, e_coef)


def calculate_udfs(
    lambdas: FloatArray, coefs: Coefs, delta_seconds: int
) -> tuple[UDFs, int | None]:
    """Unit disposition functions (s⁻¹ lambdas) and the effect-site peak time.

    ``peak_time`` is ``None`` when the model has no effect site.
    """
    p_udf = np.zeros(200)
    e_udf = np.zeros(45 * 60)  # 45 minutes

    decay = np.exp(-lambdas)

    # Plasma UDF (step response)
    A = np.zeros(len(lambdas))
    for i in range(len(p_udf)):
        p_udf[i] = np.sum(A)
        A = A * decay + coefs.p_coef * (1 - decay)

    if not np.any(coefs.e_coef):
        return UDFs(p_udf, e_udf), None

    # Effect-site UDF until its peak: 1/s for delta_seconds, then washout.
    A = np.zeros(len(lambdas))
    for i in range(delta_seconds):
        e_udf[i] = np.sum(A)
        A = A * decay + coefs.e_coef * (1 - decay)
    e_udf[delta_seconds] = np.sum(A)

    prior = e_udf[delta_seconds]
    for i in range(delta_seconds + 1, len(e_udf)):
        A = A * decay
        e_udf[i] = np.sum(A)
        if prior >= e_udf[i]:
            break
        prior = e_udf[i]
    else:
        warnings.warn("e_udf: no effect-site peak within 45 min", stacklevel=2)
    return UDFs(p_udf, e_udf), i - 1


def prepare(model: LinearPKModel | PKParameters, delta_seconds: int = 10) -> TCISetup:
    """Lambdas, coefficients, UDFs and peak time of a PK model, for :func:`model`."""
    lambdas, coefs = decompose(model)
    udfs, peak_time = calculate_udfs(lambdas, coefs, delta_seconds)
    return TCISetup(lambdas, coefs, udfs, peak_time, delta_seconds)


def model(
    Ap: FloatArray,
    Ae: FloatArray,
    lambdas: FloatArray,
    udfs: UDFs,
    coefs: Coefs,
    desired: float,
    peak_time: int | None,
    delta_seconds: int,
    target_effect: bool = True,
) -> float:
    """
    Calculate required infusion rate to achieve desired concentration
    Returns rate in units/second
    """
    p_udf = udfs.p_udf
    e_udf = udfs.e_udf
    first_dose = Ap[0] == 0

    if target_effect and peak_time is None:
        raise ValueError("Effect-site targeting needs a model with an effect site")

    # Check if we should use plasma targeting or are very close to target
    target_delta_threshold = 0.05  # 5% close to target -> switch to plasma targetting
    close_to_target = abs(desired - np.sum(Ae)) < desired * target_delta_threshold
    if not target_effect or close_to_target:
        # Plasma targeting mode
        if first_dose:
            return desired / p_udf[delta_seconds]
        result = virtual_model(Ap, delta_seconds, peak_time, lambdas, False)
        if desired > result:
            return (desired - result) / p_udf[delta_seconds]
        return 0.0

    # Effect site targeting mode
    assert peak_time is not None
    if first_dose:
        return desired / e_udf[peak_time]

    # Check if pump should be off (already at/above target)
    if virtual_model(Ae, delta_seconds, peak_time, lambdas, True) > desired:
        return 0.0

    # Set convergence criteria
    min_dif = desired * 0.0000001  # High precision for simulation

    # Initial settings
    temp_peak = peak_time
    if temp_peak <= delta_seconds:
        temp_peak = delta_seconds + 1

    # Initial rate calculation
    rate = (desired - virtual_model(Ae, temp_peak, peak_time, lambdas, True)) / e_udf[
        temp_peak
    ]

    temp_peak = find_peak(temp_peak, rate, Ae, peak_time, lambdas, e_udf, delta_seconds)
    current = (
        virtual_model(Ae, temp_peak, peak_time, lambdas, True) + e_udf[temp_peak] * rate
    )

    # Iterate until convergence
    iteration_count = 0
    max_iterations = 5000  # Safety limit

    while abs(current - desired) > min_dif and iteration_count < max_iterations:
        rate = (
            desired - virtual_model(Ae, temp_peak, peak_time, lambdas, True)
        ) / e_udf[temp_peak]

        temp_peak = find_peak(
            temp_peak, rate, Ae, peak_time, lambdas, e_udf, delta_seconds
        )
        current = (
            virtual_model(Ae, temp_peak, peak_time, lambdas, True)
            + e_udf[temp_peak] * rate
        )
        iteration_count += 1

    if iteration_count >= max_iterations:
        warnings.warn(
            f"TCI model did not converge after {max_iterations} iterations",
            stacklevel=2,
        )

    # Handle very small rates near peak time
    # XXX Maybe this is not needed here but should be handled by the pump rate resolution
    if rate < 0.00001 and temp_peak < delta_seconds * 2:
        rate = 0

    return max(0.0, rate)


def virtual_model(
    A: FloatArray,
    t: int,
    peak_time: int | None,
    lambdas: FloatArray,
    target_effect: bool,
) -> float:
    """
    Predict drug concentration at time t given current compartment amounts
    """
    # Probably this function should be replaced by update() with rate = 0 +/- lru_cache
    if target_effect and peak_time is not None and t > peak_time + 1:
        raise ValueError("Error: Time error in virtual_model")

    decay = np.exp(-lambdas * t)
    return float(np.sum(A * decay))


def find_peak(
    current_time: int,
    rate: float,
    Ae: FloatArray,
    peak_time: int,
    lambdas: FloatArray,
    e_udf: FloatArray,
    delta_seconds: int,
) -> int:
    """Find the time when effect site concentration will peak"""

    # Calculate concentrations at three time points
    current = (
        virtual_model(Ae, current_time, peak_time, lambdas, True)
        + e_udf[current_time] * rate
    )
    earlier = (
        virtual_model(Ae, current_time - 1, peak_time, lambdas, True)
        + e_udf[current_time - 1] * rate
    )
    later = (
        virtual_model(Ae, current_time + 1, peak_time, lambdas, True)
        + e_udf[current_time + 1] * rate
    )

    # Search for peak
    while current < earlier or current < later:
        if current < earlier:
            if current_time == delta_seconds:
                return current_time
            current_time -= 1
            later = current
            current = earlier
            earlier = (
                virtual_model(Ae, current_time, peak_time, lambdas, True)
                + e_udf[current_time] * rate
            )
        else:
            current_time += 1
            earlier = current
            current = later
            later = (
                virtual_model(Ae, current_time + 1, peak_time, lambdas, True)
                + e_udf[current_time + 1] * rate
            )

    return current_time


def update(
    A: FloatArray,
    lambdas: FloatArray,
    coefs: FloatArray,
    rate: float,  # rate in units/second
    dt: int,  # time in seconds
) -> FloatArray:
    ls = np.exp(-lambdas * dt)
    return A * ls + coefs * rate * (1 - ls)
