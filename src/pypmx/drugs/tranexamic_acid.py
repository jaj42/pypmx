import warnings
from functools import partial
from math import nan
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from pypmx.standard_models import pk_2comp


class Units(NamedTuple):
    time: str
    cp: str
    effect: str | None = None


class Covariates(NamedTuple):
    age: int = 25
    weight: float = 70
    height: float = 170
    sex: str = "f"


class GrassinDelyle:
    """2 compartment tranexamic acid PK model (no effect site)

    Units:
    weight (kg), time (min), volumes (L), clearances (L/min),
    concentrations (mg/L, serum)

    Reference:
    Grassin-Delyle S, Semeraro M, Lamy E, Urien S, Runge I, Foissac F, Bouazza N,
    Treluyer JM, Arribas M, Roberts I, Shakur-Still H.
    Pharmacokinetics of tranexamic acid after intravenous, intramuscular, and
    oral routes: a prospective, randomised, crossover trial in healthy volunteers.
    Br J Anaesth. 2022;128(3):465-472. doi:10.1016/j.bja.2021.10.054

    Parameters (Table 2, per 70 kg): CL 10.1 L/h, Q 15.6 L/h, Vc 7.7 L,
    Vp 10.8 L. Clearances scale with (BW/70)^0.75 and volumes with (BW/70)^1.
    Age, sex and ethnicity had no effect. Hour-based values are divided by 60.

    Population: 15 healthy volunteers (11 F / 4 M), median age 25 y [22-29],
    median weight 64.2 kg [56.0-71.8]. Inclusion criteria 18-45 y, 50-100 kg,
    BMI 18-30. A warning is issued outside the inclusion ranges.

    Only the i.v. part of the model is implemented. The i.m. (ka 1.25 /h, F 1.0)
    and oral (ka 0.28 /h, lag 0.43 h, F 0.47) absorption models are out of scope.
    No effect site, so ke0 = 0.
    """

    units: Units = Units(time="min", cp="mg/L")

    class Parameters(NamedTuple):
        V1: float = nan
        V2: float = nan
        Q2: float = nan
        CL: float = nan
        k12: float = nan
        k21: float = nan
        k10: float = nan
        ke0: float = 0
        # Reference values (per 70 kg), converted to L/min
        clref: float = 10.1 / 60
        q2ref: float = 15.6 / 60
        v1ref: float = 7.7
        v2ref: float = 10.8
        weight_ref: float = 70

    def __repr__(self):
        return f"Tranexamic acid Grassin-Delyle ({self.covars})"

    def __init__(self, covars: Mapping[str, Any] | None = None):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        c = self.covars
        if not 50 <= c.weight <= 100:
            warnings.warn(
                f"Weight {c.weight} kg outside studied range (50-100 kg)", stacklevel=2
            )
        if not 18 <= c.age <= 45:
            warnings.warn(
                f"Age {c.age} y outside studied range (18-45 y)", stacklevel=2
            )

        p = self.Parameters()
        wt = c.weight / p.weight_ref
        v1 = p.v1ref * wt
        v2 = p.v2ref * wt
        CL = p.clref * wt**0.75
        Q2 = p.q2ref * wt**0.75

        self.parameters = p._replace(
            V1=v1, V2=v2,
            Q2=Q2, CL=CL,
            k12=Q2 / v1, k21=Q2 / v2,
            k10=CL / v1,
        )  # fmt: skip
        self.ode = partial(pk_2comp, p=self.parameters._asdict())

    @property
    def system_matrix(self):
        p = self.parameters
        return np.array(
            [
                [ -(p.k10 + p.k12),  p.k12,      0],
                [            p.k21, -p.k21,      0],
                [            p.ke0,      0, -p.ke0],
            ]
        )  # fmt: skip
