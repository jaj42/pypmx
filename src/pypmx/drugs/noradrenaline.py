"""Noradrenaline (norepinephrine) PK/PD model.

2-compartment PK with endogenous production, an infusion lag and an effect
site driving mean arterial pressure through a sigmoid Emax model.

Units: weight (kg), time (min), amounts (nmol), volumes (L), clearances
(L/min), concentrations (nmol/L), MAP (mmHg). For reference, 5 µg/h of
noradrenaline is 0.261 nmol/min (5.911 nmol/µg).
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

import numpy as np
from numpy.typing import ArrayLike

from pypmx.pk import FloatArray, LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    weight: float = 70


class Noradrenaline(LinearPKModel):
    """2 compartment noradrenaline PK/PD model with endogenous production.

    ``A1' = r(t - lag) + EPROD - (k10 + k12)·A1 + k21·A2`` with
    ``lag = PKLAG + PDLAG``; without infusion the state sits at the
    endogenous steady state ``Cp = EPROD/CL``. ``MAP = pd(Ce)``.

    All parameters are scaled allometrically to ``weight`` (reference 70 kg):
    volumes ^1, CL/Q/EPROD ^0.75, ke0 ^0.25.
    """

    units: ClassVar[Units] = Units(time="min", cp="nmol/l", effect="MAP (mmHg)")

    class Thetas(NamedTuple):
        """Typical values for a 70 kg patient."""

        V1: float = 1.08
        V2: float = 1.39
        CL: float = 1.49
        Q: float = 0.319
        EPROD: float = 1.36
        KE0: float = 1.29
        E0: float = 55.8
        EC50: float = 54.2
        EMAX: float = 200
        GAMMA: float = 1.34
        PKLAG: float = 0.546
        PDLAG: float = -0.188
        weight_ref: float = 70

    class Parameters(NamedTuple):
        """Individual non-PK parameters (EPROD is weight-scaled)."""

        EPROD: float
        E0: float
        EC50: float
        EMAX: float
        GAMMA: float
        PKLAG: float
        PDLAG: float

    thetas: ClassVar[Thetas] = Thetas()
    parameters: Parameters

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        th = self.thetas

        def scale(exponent: float) -> float:
            return (c.weight / th.weight_ref) ** exponent

        self.pk = PKParameters(
            V1=th.V1 * scale(1),
            V2=th.V2 * scale(1),
            CL=th.CL * scale(0.75),
            Q2=th.Q * scale(0.75),
            ke0=th.KE0 * scale(0.25),
        )
        self.parameters = self.Parameters(
            EPROD=th.EPROD * scale(0.75),
            E0=th.E0,
            EC50=th.EC50,
            EMAX=th.EMAX,
            GAMMA=th.GAMMA,
            PKLAG=th.PKLAG,
            PDLAG=th.PDLAG,
        )

    @property
    def forcing(self) -> FloatArray:
        f = np.zeros(self.n_states)
        f[0] = self.parameters.EPROD
        return f

    @property
    def lag(self) -> float:
        return self.parameters.PKLAG + self.parameters.PDLAG

    def pd(self, ce: ArrayLike, E0: float | None = None) -> FloatArray:
        """MAP (mmHg) from Ce: sigmoid Emax on top of ``E0``."""
        p = self.parameters
        e0 = p.E0 if E0 is None else E0
        ce_g = np.asarray(ce, dtype=float) ** p.GAMMA
        return e0 + p.EMAX * ce_g / (p.EC50**p.GAMMA + ce_g)

    def with_parameters(self, **kw: float):
        """Copy with some ``pk`` and/or ``parameters`` fields replaced."""
        pk_kw = {k: v for k, v in kw.items() if k in PKParameters._fields}
        other = {k: v for k, v in kw.items() if k not in pk_kw}
        new = copy.copy(self)
        new.pk = self.pk._replace(**pk_kw)
        new.parameters = self.parameters._replace(**other)
        return new
