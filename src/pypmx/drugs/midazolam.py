import warnings
from functools import partial
from math import nan
from collections.abc import Mapping
from typing import Any, NamedTuple

import numpy as np

from pypmx.leanbodymass import dubois
from pypmx.standard_models import pk_3comp


class Units(NamedTuple):
    time: str
    cp: str
    effect: str | None = None


class Covariates(NamedTuple):
    age: int = 64
    weight: float = 86
    height: float = 175
    sex: str = "m"


class Zomorodi:
    """3 compartment intravenous midazolam PK model (no effect site)

    Units:
    weight (kg), height (cm), time (min), volumes (L), clearances (L/min),
    doses (ug), concentrations (ng/ml = ug/L, arterial plasma)

    Reference:
    Zomorodi K, Donner A, Somma J, Barr J, Sladen R, Ramsay J, Geller E,
    Shafer SL. Population pharmacokinetics of midazolam administered by
    target-controlled infusion for sedation following coronary artery bypass
    grafting. Anesthesiology. 1998;89:1418-29.
    doi:10.1097/00000542-199812000-00020.

    Models (constructor option `bsa_adjusted`):
    - False (default): simple MEM model (Table 3, abstract). V1 32.2,
      V2 53, V3 245 L; Cl1 0.43, Cl2 0.56, Cl3 0.39 L/min. No covariates.
      The authors preferred it because cross-validation did not confirm the
      better performance of the BSA-adjusted model.
    - True: BSA-adjusted MEM model (Table 4). V1 33 L,
      V2 = 3.32 + 32.1 * BSA L, V3 365 L; Cl1 = 0.0889 + 0.151 * BSA L/min,
      Cl2 0.622, Cl3 0.264 L/min.

    Assumptions:
    - The paper does not state its BSA formula. Du Bois is used
      (`leanbodymass.dubois`); it reproduces the Table 2 mean BSA of each
      centre from the mean height and weight to within 0.01 m^2.
    - The default covariates (64 y, 86 kg, 175 cm, male) are rounded
      population means from Table 2. They only matter for the BSA model.

    Population: 63 analysed post-CABG ICU patients (7 F / 56 M by centre
    totals), age 60-67 +/- 7-8 y, weight 84-87 +/- 14-15 kg (centre means
    +/- SD, Table 2). Ranges are not reported, so a warning is issued
    outside approximately mean +/- 2 SD: age 45-80 y, weight 55-115 kg.

    No effect site (the PD paper is not used), so ke0 = 0.
    """

    units: Units = Units(time="min", cp="ng/ml")

    class Parameters(NamedTuple):
        V1: float = nan
        V2: float = nan
        V3: float = nan
        CL: float = nan
        Q2: float = nan
        Q3: float = nan
        k12: float = nan
        k21: float = nan
        k13: float = nan
        k31: float = nan
        k10: float = nan
        ke0: float = 0
        BSA: float = nan

    def __repr__(self):
        return f"Midazolam Zomorodi ({self.covars}, bsa_adjusted={self.bsa_adjusted})"

    def __init__(
        self,
        covars: Mapping[str, Any] | None = None,
        bsa_adjusted: bool = False,
    ):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        self.bsa_adjusted = bsa_adjusted
        c = self.covars
        if not 45 <= c.age <= 80:
            warnings.warn(
                f"Age {c.age} y outside approximate studied range (45-80 y)",
                stacklevel=2,
            )
        if not 55 <= c.weight <= 115:
            warnings.warn(
                f"Weight {c.weight} kg outside approximate studied range (55-115 kg)",
                stacklevel=2,
            )

        bsa = dubois(c.height, c.weight)
        if bsa_adjusted:
            v1, v2, v3 = 33.0, 3.32 + 32.1 * bsa, 365.0
            CL, Q2, Q3 = 0.0889 + 0.151 * bsa, 0.622, 0.264
        else:
            v1, v2, v3 = 32.2, 53.0, 245.0
            CL, Q2, Q3 = 0.43, 0.56, 0.39

        self.parameters = self.Parameters(
            V1=v1, V2=v2, V3=v3,
            CL=CL, Q2=Q2, Q3=Q3,
            k12=Q2 / v1, k21=Q2 / v2,
            k13=Q3 / v1, k31=Q3 / v3,
            k10=CL / v1,
            BSA=bsa,
        )  # fmt: skip
        self.ode = partial(pk_3comp, p=self.parameters._asdict())

    @property
    def system_matrix(self):
        p = self.parameters
        k123 = p.k10 + p.k12 + p.k13
        return np.array(
            [
                [ -k123,  p.k12,  p.k13,      0],
                [ p.k21, -p.k21,      0,      0],
                [ p.k31,      0, -p.k31,      0],
                [ p.ke0,      0,      0, -p.ke0],
            ]
        )  # fmt: skip
