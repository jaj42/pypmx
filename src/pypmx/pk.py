"""Common base for linear (mammillary) PK models.

Every model is a 1-, 2- or 3-compartment mammillary system with an optional
effect site. The state vector is in **amounts** for the PK compartments and in
**concentration** for the effect site (mixed units are fine for a linear
system, and Ce needs no effect-site volume):

    A1' = r(t - lag) + f1 - (k10 + k12 + k13)·A1 + k21·A2 + k31·A3
    A2' = k12·A1 - k21·A2
    A3' = k13·A1 - k31·A3
    Ce' = ke0·(A1/V1 - Ce)

so ``x' = S·x + B·r(t - lag) + f`` with ``B = e₀``. Models without an effect
site (``ke0 == 0``) omit the Ce state. Observations: ``Cp = A1/V1``,
``Ce = x[-1]``.

A drug model subclasses :class:`LinearPKModel`, maps its covariates to macro
parameters and stores them as ``self.pk`` (:class:`PKParameters`). The base
class provides the system matrix, the ODE right-hand side and an eigen-based
simulator (:func:`pypmx.eigen.eig_advan`).
"""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Mapping
from math import isfinite, nan
from typing import Any, ClassVar, NamedTuple, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray

from pypmx.eigen import eig_advan
from pypmx.iiv import RandomEffects
from pypmx.utils import filter_valid

FloatArray = NDArray[np.float64]
C = TypeVar("C", bound=NamedTuple)


class Units(NamedTuple):
    time: str
    cp: str
    effect: str | None = None


class Covariates(NamedTuple):
    """Default demographic covariates. Drugs define their own when needed."""

    age: float = 35
    weight: float = 70
    height: float = 170
    sex: str = "m"


class PKParameters(NamedTuple):
    """Macro parameters of a mammillary model.

    ``Q2 == 0`` means 1 compartment, ``Q3 == 0`` means 2 compartments and
    ``ke0 == 0`` means no effect site.
    """

    V1: float
    CL: float
    V2: float = nan
    V3: float = nan
    Q2: float = 0.0
    Q3: float = 0.0
    ke0: float = 0.0

    @classmethod
    def from_micro(
        cls,
        V1: float,
        k10: float,
        k12: float = 0.0,
        k21: float = 0.0,
        k13: float = 0.0,
        k31: float = 0.0,
        ke0: float = 0.0,
    ) -> PKParameters:
        """Macro parameters of a model defined by V1 and micro rate constants.

        ``CL = k10·V1``, ``Q = k1i·V1`` and ``Vi = Q/ki1``. Published peripheral
        volumes are often rounded and slightly inconsistent with the rate
        constants; deriving them keeps the dynamics exact.
        """
        Q2 = k12 * V1
        Q3 = k13 * V1
        return cls(
            V1=V1,
            CL=k10 * V1,
            V2=Q2 / k21 if Q2 else nan,
            V3=Q3 / k31 if Q3 else nan,
            Q2=Q2,
            Q3=Q3,
            ke0=ke0,
        )

    @property
    def n_cmt(self) -> int:
        if self.Q2 == 0:
            if self.Q3 != 0:
                raise ValueError("Q3 is set but Q2 is 0; use Q2 for 2-comp models")
            return 1
        return 2 if self.Q3 == 0 else 3

    @property
    def has_effect_site(self) -> bool:
        return self.ke0 != 0

    @property
    def k10(self) -> float:
        return self.CL / self.V1

    @property
    def k12(self) -> float:
        return self.Q2 / self.V1

    @property
    def k21(self) -> float:
        return self.Q2 / self.V2 if self.Q2 else 0.0

    @property
    def k13(self) -> float:
        return self.Q3 / self.V1

    @property
    def k31(self) -> float:
        return self.Q3 / self.V3 if self.Q3 else 0.0


def system_matrix(p: PKParameters) -> FloatArray:
    """``S`` of ``x' = S·x + B·r``: amounts, plus Ce when ``ke0 != 0``."""
    n_cmt = p.n_cmt
    n = n_cmt + int(p.has_effect_site)
    S = np.zeros((n, n))
    S[0, 0] = -(p.k10 + p.k12 + p.k13)
    for i, (k1i, ki1) in enumerate(((p.k12, p.k21), (p.k13, p.k31))[: n_cmt - 1]):
        S[0, i + 1] = ki1
        S[i + 1, 0] = k1i
        S[i + 1, i + 1] = -ki1
    if p.has_effect_site:
        S[-1, 0] = p.ke0 / p.V1
        S[-1, -1] = -p.ke0
    return S


class Prediction(NamedTuple):
    t: FloatArray
    cp: FloatArray
    ce: FloatArray | None
    states: FloatArray  # (n_times, n_states), amounts + optional Ce
    effect: FloatArray | None = None  # pd(ce), for models that define a PD


class LinearPKModel(RandomEffects):
    """Base class for linear PK models (see module docstring).

    Subclasses set ``units``, ``covariate_ranges`` (studied ``(low, high)``
    per covariate, a warning is issued outside), optionally ``options``
    (names of constructor options shown by ``repr``), and in ``__init__``:
    ``covars`` (via :meth:`_set_covariates`) and ``pk``. Models with thetas,
    PD values or derived covariates also set ``parameters`` (a drug-specific
    ``Parameters`` NamedTuple, declared in the subclass as
    ``parameters: Parameters``). Models with a PD override :meth:`pd`, and
    :meth:`simulate` then fills ``Prediction.effect``.

    Models with published IIV set ``omega`` (:class:`pypmx.iiv.Omega`) over
    ``pk`` and/or ``parameters`` field names; ``sample(rng)`` and
    ``with_eta(eta)`` then return individual models (see :mod:`pypmx.iiv`).
    """

    units: ClassVar[Units]
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {}
    options: ClassVar[tuple[str, ...]] = ()  # constructor options shown in repr

    covars: NamedTuple
    pk: PKParameters

    # ---- structure -------------------------------------------------------

    @property
    def n_states(self) -> int:
        return self.pk.n_cmt + int(self.pk.has_effect_site)

    @property
    def system_matrix(self) -> FloatArray:
        return system_matrix(self.pk)

    @property
    def input_matrix(self) -> FloatArray:
        B = np.zeros(self.n_states)
        B[0] = 1.0
        return B

    @property
    def forcing(self) -> FloatArray:
        """Constant input (amount/time), e.g. endogenous production."""
        return np.zeros(self.n_states)

    @property
    def lag(self) -> float:
        return 0.0

    def pd(self, ce: FloatArray) -> FloatArray | None:
        """Effect as a function of Ce. ``None`` when the model has no PD."""
        return None

    def steady_state(self) -> FloatArray:
        """State with no infusion: ``-S⁻¹f`` (zeros when there is no forcing)."""
        f = self.forcing
        if not np.any(f):
            return np.zeros(self.n_states)
        return np.linalg.solve(self.system_matrix, -f).astype(np.float64)

    # ---- simulation ------------------------------------------------------

    def ode(self, t: float, y: ArrayLike, Inf: Callable[[float], float]) -> FloatArray:
        """Right-hand side for ``solve_ivp(m.ode, ..., args=(Inf,))``."""
        rate = Inf(t - self.lag)
        return (
            self.system_matrix @ np.asarray(y, dtype=float)
            + self.input_matrix * rate
            + self.forcing
        )

    def simulate(
        self,
        infu_time: ArrayLike,
        infu_rate: ArrayLike,
        meas_time: ArrayLike,
        y0: ArrayLike | None = None,
    ) -> Prediction:
        """Exact solution for a piecewise-constant infusion (see ``eig_advan``).

        ``y0`` (default :meth:`steady_state`) holds at the earliest infusion or
        measurement time.
        """
        t = np.atleast_1d(np.asarray(meas_time, dtype=float))
        states = eig_advan(
            self.system_matrix,
            self.input_matrix,
            t,
            infu_time,
            infu_rate,
            y0=self.steady_state() if y0 is None else y0,
            f=self.forcing,
            lag=self.lag,
        )
        cp = states[:, 0] / self.pk.V1
        ce = states[:, -1] if self.pk.has_effect_site else None
        effect = None if ce is None else self.pd(ce)
        return Prediction(t, cp, ce, states, effect)

    def with_parameters(self, **kw: float):
        """Copy of the model with some ``pk`` fields replaced."""
        new = copy.copy(self)
        new.pk = self.pk._replace(**kw)
        return new

    # ---- IIV hooks (pypmx.iiv.RandomEffects) ------------------------------

    def _get_values(self, names: tuple[str, ...]) -> dict[str, float]:
        params = getattr(self, "parameters", None)
        return {
            n: getattr(self.pk if n in PKParameters._fields else params, n)
            for n in names
        }

    def _set_values(self, values: Mapping[str, float]) -> None:
        pk_kw = {k: v for k, v in values.items() if k in PKParameters._fields}
        self.pk = self.pk._replace(**pk_kw)
        if other := {k: v for k, v in values.items() if k not in pk_kw}:
            params: Any = getattr(self, "parameters")
            self.parameters = params._replace(**other)

    # ---- covariates ------------------------------------------------------

    def _set_covariates(self, covars: Mapping[str, Any] | None, cls: type[C]) -> C:
        """Build ``self.covars`` from a mapping (unknown keys are ignored)."""
        c = cls(**filter_valid(covars, cls)) if covars is not None else cls()
        self.covars = c
        self._check_ranges(c._asdict(), stacklevel=4)
        return c

    def _check_ranges(self, values: Mapping[str, Any], stacklevel: int = 3) -> None:
        """Warn for each value outside its ``covariate_ranges`` entry."""
        for name, (lo, hi) in self.covariate_ranges.items():
            value = values.get(name)
            if value is None or not isfinite(value):
                continue
            if not lo <= value <= hi:
                warnings.warn(
                    f"{type(self).__name__}: {name} {value:.4g} outside studied "
                    f"range ({lo:g}-{hi:g})",
                    stacklevel=stacklevel,
                )

    def __repr__(self) -> str:
        covars = getattr(self, "covars", None)
        args = [] if covars is None else [repr(covars._asdict())]
        args += [f"{name}={getattr(self, name)!r}" for name in self.options]
        return f"{type(self).__name__}({', '.join(args)})"
