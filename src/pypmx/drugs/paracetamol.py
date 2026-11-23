import warnings
from functools import partial
from math import nan
from typing import NamedTuple

import numpy as np

from pypmx.leanbodymass import janmahasation
from pypmx.standard_models import pk_3comp


class Units(NamedTuple):
    time: str
    cp: str
    effect: str | None = None


class Covariates(NamedTuple):
    age: int = 25
    weight: float = 70
    height: float = 170
    sex: str = "m"


# Covariate factors (Table 3): (on CL, on V2)
SETTINGS = {
    "none": (1.0, 1.0),
    "pregnancy": (1.14, 1.0),
    "abdominal": (0.715, 1.0),
    "orthopaedic": (1.0, 0.649),
    "icu": (1.0, 1.51),
}


class Allegaert:
    """3 compartment intravenous paracetamol PK model (no effect site)

    Units:
    weight (kg), height (cm), time (min), volumes (L), clearances (L/min),
    concentrations (mg/L, plasma/serum)

    Reference:
    Allegaert K, Olkkola KT, Owens KH, Van de Velde M, de Maat MM, Anderson BJ.
    Covariates of intravenous paracetamol pharmacokinetics in adults.
    BMC Anesthesiol. 2014;14:77. doi:10.1186/1471-2253-14-77. PMID 25342929.

    Parameters (Table 3, per 70 kg): CL 16.7 L/h, V1 24.6 L, Q2 67.3 L/h,
    V2 23.1 L, Q3 2.04 L/h, V3 30.6 L. Hour-based values are divided by 60.

    Size: normal fat mass NFM = FFM + Ffat * (TBW - FFM), with FFM from the
    Janmahasatian WHS formula (`leanbodymass.janmahasation`, algebraically
    identical to the paper's WHSmax/WHS50). Ffat_CL = 1 (fixed, i.e. TBW) and
    Ffat_VOL = 0.778. Clearances scale with (NFM_CL/70)^0.75 and volumes with
    (NFM_VOL/70)^1.

    Covariates (constructor option `setting`):
    - "none" (default): healthy adult
    - "pregnancy": CL x 1.14 (third trimester)
    - "abdominal": CL x 0.715 (abdominal surgery)
    - "orthopaedic": V2 x 0.649 (orthopaedic surgery)
    - "icu": V2 x 1.51 (intensive care)
    Independently of setting, V2 x 0.838 if age > 60 y.
    Postpartum and high-dependency care had no effect.

    Assumptions:
    - Table 3 gives no units for Q3 and V3. They are taken as per 70 kg and
      scaled like the other clearances (Q3) and volumes (V3).
    - Intercompartmental clearances Q2 and Q3 use NFM_CL (Ffat = 1), and all
      volumes use NFM_VOL. The paper only states Ffat_CL and Ffat_VOL.
    - The standard size is 70 kg of NFM.
    - The age factor on CL (F_AGECL) was dropped from the final model (Table 2)
      and is not used.
    - The Figure 2 caption ("2-compartment model") is a typo: the final model
      has three compartments (Methods, Table 2).

    Population: 189 adults (healthy volunteers, pregnant/postpartum women,
    orthopaedic and abdominal surgery, high-dependency and intensive care),
    weight 49.2-120 kg, age 19-88.5 y. A warning is issued outside these
    ranges and when "pregnancy" is used with sex "m".

    No effect site, so ke0 = 0.
    """

    units: Units = Units(time="min", cp="mg/L")

    class Parameters(NamedTuple):
        V1: float = nan
        V2: float = nan
        V3: float = nan
        Q2: float = nan
        Q3: float = nan
        CL: float = nan
        k12: float = nan
        k21: float = nan
        k13: float = nan
        k31: float = nan
        k10: float = nan
        ke0: float = 0
        # Reference values (per 70 kg), converted to L/min
        clref: float = 16.7 / 60
        q2ref: float = 67.3 / 60
        q3ref: float = 2.04 / 60
        v1ref: float = 24.6
        v2ref: float = 23.1
        v3ref: float = 30.6
        ffat_cl: float = 1.0
        ffat_vol: float = 0.778
        fage_v2: float = 0.838
        weight_ref: float = 70

    def __repr__(self):
        return f"Paracetamol Allegaert ({self.covars}, setting={self.setting})"

    def __init__(
        self,
        covars: dict[str, int | float | str] | None = None,
        setting: str = "none",
    ):
        if setting not in SETTINGS:
            raise ValueError(
                f"Unknown setting {setting!r}, expected one of {list(SETTINGS)}"
            )
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        self.setting = setting
        c = self.covars
        if not 49.2 <= c.weight <= 120:
            warnings.warn(
                f"Weight {c.weight} kg outside studied range (49.2-120 kg)",
                stacklevel=2,
            )
        if not 19 <= c.age <= 88.5:
            warnings.warn(
                f"Age {c.age} y outside studied range (19-88.5 y)", stacklevel=2
            )
        if setting == "pregnancy" and c.sex == "m":
            warnings.warn("Setting 'pregnancy' used with sex 'm'", stacklevel=2)

        p = self.Parameters()
        ffm = janmahasation(c.height, c.weight, c.sex)
        nfm_cl = ffm + p.ffat_cl * (c.weight - ffm)
        nfm_vol = ffm + p.ffat_vol * (c.weight - ffm)
        f_cl = (nfm_cl / p.weight_ref) ** 0.75
        f_vol = nfm_vol / p.weight_ref

        fset_cl, fset_v2 = SETTINGS[setting]
        fage_v2 = p.fage_v2 if c.age > 60 else 1.0

        v1 = p.v1ref * f_vol
        v2 = p.v2ref * f_vol * fage_v2 * fset_v2
        v3 = p.v3ref * f_vol
        CL = p.clref * f_cl * fset_cl
        Q2 = p.q2ref * f_cl
        Q3 = p.q3ref * f_cl

        self.parameters = p._replace(
            V1=v1, V2=v2, V3=v3,
            Q2=Q2, Q3=Q3, CL=CL,
            k12=Q2 / v1, k21=Q2 / v2,
            k13=Q3 / v1, k31=Q3 / v3,
            k10=CL / v1,
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
