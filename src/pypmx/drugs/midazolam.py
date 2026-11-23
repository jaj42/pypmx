from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

from pypmx.leanbodymass import dubois
from pypmx.pk import LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    age: float = 64
    weight: float = 86
    height: float = 175
    sex: str = "m"


class Zomorodi(LinearPKModel):
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

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "age": (45, 80),  # approximate: mean +/- 2 SD
        "weight": (55, 115),
    }
    options: ClassVar[tuple[str, ...]] = ("bsa_adjusted",)

    class Parameters(NamedTuple):
        BSA: float  # derived covariate (m^2, Du Bois)

    parameters: Parameters

    def __init__(
        self,
        covars: Mapping[str, Any] | None = None,
        bsa_adjusted: bool = False,
    ):
        self.bsa_adjusted = bsa_adjusted
        c = self._set_covariates(covars, Covariates)
        bsa = dubois(c.height, c.weight)
        self.parameters = self.Parameters(BSA=bsa)
        if bsa_adjusted:
            self.pk = PKParameters(
                V1=33.0, V2=3.32 + 32.1 * bsa, V3=365.0,
                CL=0.0889 + 0.151 * bsa, Q2=0.622, Q3=0.264,
            )  # fmt: skip
        else:
            self.pk = PKParameters(
                V1=32.2, V2=53.0, V3=245.0,
                CL=0.43, Q2=0.56, Q3=0.39,
            )  # fmt: skip
