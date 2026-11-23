from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

from pypmx.pk import LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    age: float = 25
    weight: float = 70
    height: float = 170
    sex: str = "f"


class GrassinDelyle(LinearPKModel):
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

    units: ClassVar[Units] = Units(time="min", cp="mg/L")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "weight": (50, 100),
        "age": (18, 45),
    }

    class Parameters(NamedTuple):
        # Reference values (per 70 kg), converted to L/min
        clref: float = 10.1 / 60
        q2ref: float = 15.6 / 60
        v1ref: float = 7.7
        v2ref: float = 10.8
        weight_ref: float = 70

    parameters: Parameters

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        p = self.parameters = self.Parameters()
        wt = c.weight / p.weight_ref
        self.pk = PKParameters(
            V1=p.v1ref * wt,
            V2=p.v2ref * wt,
            CL=p.clref * wt**0.75,
            Q2=p.q2ref * wt**0.75,
        )
