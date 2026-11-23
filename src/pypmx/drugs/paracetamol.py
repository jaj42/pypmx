import warnings
from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

from pypmx.leanbodymass import janmahasation
from pypmx.pk import LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    age: float = 25
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


class Allegaert(LinearPKModel):
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

    units: ClassVar[Units] = Units(time="min", cp="mg/L")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "weight": (49.2, 120),
        "age": (19, 88.5),
    }
    options: ClassVar[tuple[str, ...]] = ("setting",)

    class Parameters(NamedTuple):
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

    parameters: Parameters

    def __init__(
        self,
        covars: Mapping[str, Any] | None = None,
        setting: str = "none",
    ):
        if setting not in SETTINGS:
            raise ValueError(
                f"Unknown setting {setting!r}, expected one of {list(SETTINGS)}"
            )
        self.setting = setting
        c = self._set_covariates(covars, Covariates)
        if setting == "pregnancy" and c.sex == "m":
            warnings.warn("Setting 'pregnancy' used with sex 'm'", stacklevel=2)

        p = self.parameters = self.Parameters()
        ffm = janmahasation(c.height, c.weight, c.sex)
        nfm_cl = ffm + p.ffat_cl * (c.weight - ffm)
        nfm_vol = ffm + p.ffat_vol * (c.weight - ffm)
        f_cl = (nfm_cl / p.weight_ref) ** 0.75
        f_vol = nfm_vol / p.weight_ref

        fset_cl, fset_v2 = SETTINGS[setting]
        fage_v2 = p.fage_v2 if c.age > 60 else 1.0

        self.pk = PKParameters(
            V1=p.v1ref * f_vol,
            V2=p.v2ref * f_vol * fage_v2 * fset_v2,
            V3=p.v3ref * f_vol,
            CL=p.clref * f_cl * fset_cl,
            Q2=p.q2ref * f_cl,
            Q3=p.q3ref * f_cl,
        )
