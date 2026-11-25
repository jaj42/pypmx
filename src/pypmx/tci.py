"""Target-controlled infusion (TCI) on the modal solution of the PK system.

Every ``delta_seconds`` the controller chooses the constant rate for the next
step. It relies on the superposition principle: the concentration ``s`` seconds
from now is the *free course* of the current state (no infusion) plus the rate
times the *unit disposition function* (UDF), the response to 1/s for one step:

    C(s) = C_free(s) + rate·udf(s)

Both come from the eigendecomposition of the system matrix
(:mod:`pypmx.eigen`), so any linear model works and both are continuous in
``s``; no per-structure closed forms and no per-second tables are needed.

* **Plasma target**: aim at the end of the step,
  ``rate = (target - Cp_free(δ)) / udf_cp(δ)``.
* **Effect-site target**: aim at the *peak* of Ce, which depends on the rate.
  The peak time ``t`` is the fixed point of
  ``t ↦ argmax_s Ce_free(s) + rate(t)·udf_ce(s)`` with
  ``rate(t) = (target - Ce_free(t)) / udf_ce(t)``; the argmax is found with
  Brent's method (``minimize_scalar``), the fixed point with
  ``scipy.optimize.fixed_point``. As in STANPUMP, the controller switches to
  plasma targeting once Ce is within ``plasma_switch`` of the target, which
  removes the rate wobble caused by discrete steps.

Endogenous production (``model.forcing``) is part of the free course, and the
body starts at ``model.steady_state()``. With an infusion lag (``model.lag``)
a rate chosen now only enters the body ``lag`` later, so the controller plans
from the state at ``now + lag``, which earlier decisions fully determine.

PK models are in minutes; the controller runs in **seconds** (rates in
amount/s), the conversion happens in :func:`_controller_system` only.
:mod:`pypmx.stanpump_tci` is the reference port of the original algorithm.
"""

from __future__ import annotations

from typing import Literal, NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import fixed_point, minimize_scalar

from pypmx.eigen import eigendecomposition, phi
from pypmx.pk import LinearPKModel, PKParameters, system_matrix

FloatArray = NDArray[np.float64]

SECONDS_PER_MINUTE = 60.0
PEAK_SEARCH_LIMIT = 45 * 60.0  # s, the unit effect-site peak must come earlier
PEAK_TIME_TOLERANCE = 1e-3  # s, for the argmax (Ce is flat at its peak)
FIXED_POINT_TOLERANCE = 1e-6  # relative, on the peak time


class TCIResult(NamedTuple):
    """Closed-loop run, one row per step.

    ``rate[k]`` is chosen at ``t[k]`` and runs for one step; it enters the
    body ``lag`` seconds later, so ``cp[k]``/``ce[k]`` are the concentrations
    at ``t[k] + lag + delta_seconds``.
    """

    t: FloatArray  # s, decision times
    rate: FloatArray  # amount/s
    cp: FloatArray
    ce: FloatArray | None


class _System(NamedTuple):
    pk: PKParameters
    S: FloatArray  # s⁻¹
    f: FloatArray  # amount/s
    lag: float  # s
    x0: FloatArray  # state before any infusion


def _controller_system(model: LinearPKModel | PKParameters) -> _System:
    """The model's ``x' = S·x + B·r + f``, converted to seconds."""
    if isinstance(model, PKParameters):
        n = model.n_cmt + int(model.has_effect_site)
        S = system_matrix(model) / SECONDS_PER_MINUTE
        return _System(model, S, np.zeros(n), 0.0, np.zeros(n))
    if model.units.time != "min":
        raise ValueError(f"TCI expects a model in minutes, got {model.units.time!r}")
    return _System(
        model.pk,
        model.system_matrix / SECONDS_PER_MINUTE,
        model.forcing / SECONDS_PER_MINUTE,
        model.lag * SECONDS_PER_MINUTE,
        model.steady_state(),
    )


class TCI:
    """Target-controlled infusion of a linear PK model (see module docstring).

    ``target`` selects plasma or effect-site targeting. :meth:`run` does a
    whole closed loop; :meth:`rate` and :meth:`step` do one step, e.g. to clip
    the rate to what a pump can deliver before stepping.
    """

    def __init__(
        self,
        model: LinearPKModel | PKParameters,
        delta_seconds: float = 10,
        target: Literal["effect", "plasma"] = "effect",
        plasma_switch: float = 0.05,
    ):
        system = _controller_system(model)
        if target == "effect" and not system.pk.has_effect_site:
            raise ValueError("Effect-site targeting needs a model with an effect site")
        self.V1 = system.pk.V1
        self.has_effect_site = system.pk.has_effect_site
        self.lag = system.lag
        self.delta_seconds = float(delta_seconds)
        self.target = target
        self.plasma_switch = plasma_switch

        B = np.zeros(len(system.x0))
        B[0] = 1.0
        modes = eigendecomposition(system.S, B, system.f)
        self._lambdas = modes.lambdas
        self._V = modes.eigvecs
        self._Vinv = np.linalg.inv(modes.eigvecs)
        self._alpha_B = modes.alpha_B
        self._alpha_f = modes.alpha_f

        self.time = 0.0  # s, time of the next decision
        self.state = system.x0  # body state at time + lag
        self.peak_time = self._unit_peak_time() if self.has_effect_site else None

    # ---- the model ------------------------------------------------------

    def advance(self, x: FloatArray, rate: float, dt: float) -> FloatArray:
        """State ``dt`` seconds after ``x`` with a constant ``rate`` (exact)."""
        lam = self._lambdas
        modal = self._Vinv @ x
        modal = modal * np.exp(-lam * dt) + (
            self._alpha_B * rate + self._alpha_f
        ) * phi(lam, dt)
        return self._V @ modal

    def unit_response(self, t: float) -> FloatArray:
        """State at ``t`` after 1/s during one step, from an empty body.

        Infusion only, no endogenous production: the UDF of every compartment.
        """
        lam, delta = self._lambdas, self.delta_seconds
        if t <= delta:
            modal = self._alpha_B * phi(lam, t)
        else:
            modal = self._alpha_B * phi(lam, delta) * np.exp(-lam * (t - delta))
        return self._V @ modal

    def cp(self, x: FloatArray) -> float:
        return float(x[0] / self.V1)

    def ce(self, x: FloatArray) -> float:
        return float(x[-1])

    def _unit_peak_time(self) -> float:
        """Time of the maximum of the effect-site UDF."""
        res = minimize_scalar(
            lambda t: -self.ce(self.unit_response(t)),
            bounds=(self.delta_seconds, PEAK_SEARCH_LIMIT),
            method="bounded",
            options={"xatol": PEAK_TIME_TOLERANCE},
        )
        return float(res.x)

    # ---- the controller -------------------------------------------------

    def rate(self, target: float) -> float:
        """Rate (amount/s) for the next step to reach ``target``."""
        if self.target == "plasma":
            return self._plasma_rate(target)
        if abs(target - self.ce(self.state)) < self.plasma_switch * target:
            return self._plasma_rate(target)
        return self._effect_rate(target)

    def _plasma_rate(self, target: float) -> float:
        """Rate that brings Cp to ``target`` at the end of the step."""
        delta = self.delta_seconds
        cp_free = self.cp(self.advance(self.state, 0.0, delta))
        rate = (target - cp_free) / self.cp(self.unit_response(delta))
        return max(0.0, rate)

    def _effect_rate(self, target: float) -> float:
        """Rate whose Ce peak is at ``target``."""
        x, delta = self.state, self.delta_seconds
        assert self.peak_time is not None

        def ce_free(s: float) -> float:
            return self.ce(self.advance(x, 0.0, s))

        def udf(s: float) -> float:
            return self.ce(self.unit_response(s))

        if ce_free(delta) > target:  # already above target: let it decay
            return 0.0

        def rate_for(peak: float) -> float:
            """Rate that puts Ce at ``target`` at time ``peak``."""
            return (target - ce_free(peak)) / udf(peak)

        def next_peak(peak: float) -> float:
            """When Ce actually peaks with ``rate_for(peak)``."""
            rate = rate_for(float(peak))
            if rate <= 0:
                return peak
            res = minimize_scalar(
                lambda s: -(ce_free(s) + rate * udf(s)),
                bounds=(delta, self.peak_time),
                method="bounded",
                options={"xatol": PEAK_TIME_TOLERANCE},
            )
            return res.x

        peak = fixed_point(
            next_peak,
            self.peak_time,
            xtol=FIXED_POINT_TOLERANCE,
            method="iteration",
        )
        return max(0.0, rate_for(float(peak)))

    def step(self, rate: float) -> None:
        """Infuse ``rate`` (amount/s) for one step."""
        self.state = self.advance(self.state, rate, self.delta_seconds)
        self.time += self.delta_seconds

    def run(self, targets: NDArray | list[float]) -> TCIResult:
        """Closed loop, one target per step, from the current state."""
        t, rates, cp, ce = [], [], [], []
        for target in targets:
            t.append(self.time)
            rate = self.rate(target)
            self.step(rate)
            rates.append(rate)
            cp.append(self.cp(self.state))
            ce.append(self.ce(self.state))
        return TCIResult(
            np.array(t),
            np.array(rates),
            np.array(cp),
            np.array(ce) if self.has_effect_site else None,
        )
