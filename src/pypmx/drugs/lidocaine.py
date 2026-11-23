from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

from pypmx.leanbodymass import bmi
from pypmx.pk import LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    age: float = 45
    weight: float = 75
    height: float = 174
    sex: str = "m"
    cardiac_output: float = 5.8


class Kuipers(LinearPKModel):
    """3 compartment intravenous lidocaine PK model (no effect site)

    Units:
    weight (kg), height (cm), cardiac output (L/min), time (min),
    volumes (L), clearances (L/min), doses (mg),
    concentrations (mg/L = ug/ml, arterial whole blood)

    Reference:
    Kuipers JA, Boer F, de Roode A, Olofsen E, Bovill JG, Burm AGL.
    Modeling population pharmacokinetics of lidocaine: should cardiac output
    be included as a patient factor? Anesthesiology. 2001;94:566-73.
    doi:10.1097/00000542-200104000-00007.

    Models (constructor option `model`, Tables 2 and 3):
    - "A": no covariates. V1 1.29, V2 7.39, V3 57.3 L;
      CL 0.646, Q2 0.898, Q3 0.750 L/min.
    - "B": cardiac output. V2 = 2.33 + 0.928 * CO, CL = 0.402 + 0.0422 * CO;
      V1 1.30, V3 57.4 L, Q2 0.918, Q3 0.745 L/min.
    - "C": demographics. V3 = 9.37 + 1.91 * QI, CL = 0.258 + 0.00517 * wt,
      Q2 = 1.721 - 0.0158 * age, Q3 = 0.687 - 0.0107 * age + 0.0220 * QI;
      V1 1.31, V2 8.05 L.
    - "D" (default, best fit): V2 = 2.21 + 1.01 * CO, V3 = 9.02 + 1.92 * QI,
      CL = 0.138 + 0.00410 * wt + 0.0349 * CO, Q2 = 1.697 - 0.0156 * age,
      Q3 = 0.691 - 0.0107 * age + 0.0218 * QI; V1 1.29 L.
    Q2 and Q3 are the paper's "CL (rapid distribution)" and
    "CL (slow distribution)". QI is the Quetelet index (BMI, kg/m^2),
    computed from height and weight. Cardiac output only affects models B and
    D; it was measured by thoracic electrical bioimpedance, averaged over the
    sampling period.

    Assumptions:
    - The Table 3 linear equations are used as published. They are the
      paper's median-centred covariate model expanded at the Table 1 medians
      (45 y, 75 kg, QI 24.8, CO 5.8 L/min) and reproduce the Table 2 typical
      values there (e.g. model D CL 0.648 vs 0.647 L/min).
    - Default height 174 cm gives QI 24.8 at 75 kg (Table 1 median). The
      defaults are the Table 1 medians; sex is not a covariate.
    - The models were fitted to a single 1 mg/kg bolus with up to 6 h of
      sampling; long-infusion predictions (as in the paper's Fig. 3) are
      extrapolations.

    Population: 31 ASA I-II adults (17 M / 14 F) under propofol-remifentanil
    anaesthesia for ophthalmic surgery, without cardiovascular disease.
    Median (range): age 45 (19-76) y, weight 75 (51-102) kg,
    QI 24.8 (20.4-33.3) kg/m^2, CO 5.8 (3.6-10.0) L/min (Table 1).
    A warning is issued outside these ranges.

    No effect site, so ke0 = 0.
    """

    units: ClassVar[Units] = Units(time="min", cp="mg/L")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "age": (19, 76),
        "weight": (51, 102),
        "QI": (20.4, 33.3),  # Quetelet index, derived from height and weight
        "cardiac_output": (3.6, 10.0),
    }
    options: ClassVar[tuple[str, ...]] = ("model",)
    models = ("A", "B", "C", "D")

    class Parameters(NamedTuple):
        QI: float  # derived covariate: Quetelet index (BMI, kg/m^2)

    parameters: Parameters

    def __init__(
        self,
        covars: Mapping[str, Any] | None = None,
        model: str = "D",
    ):
        if model not in self.models:
            raise ValueError(f"model must be one of {self.models}, got {model!r}")
        self.model = model
        c = self._set_covariates(covars, Covariates)
        qi = bmi(c.height, c.weight)
        self._check_ranges({"QI": qi})
        self.parameters = self.Parameters(QI=qi)
        co = c.cardiac_output

        if model == "A":
            v1, v2, v3 = 1.29, 7.39, 57.3
            CL, Q2, Q3 = 0.646, 0.898, 0.750
        elif model == "B":
            v1, v2, v3 = 1.30, 2.33 + 0.928 * co, 57.4
            CL, Q2, Q3 = 0.402 + 0.0422 * co, 0.918, 0.745
        elif model == "C":
            v1, v2, v3 = 1.31, 8.05, 9.37 + 1.91 * qi
            CL = 0.258 + 0.00517 * c.weight
            Q2 = 1.721 - 0.0158 * c.age
            Q3 = 0.687 - 0.0107 * c.age + 0.0220 * qi
        else:
            v1, v2, v3 = 1.29, 2.21 + 1.01 * co, 9.02 + 1.92 * qi
            CL = 0.138 + 0.00410 * c.weight + 0.0349 * co
            Q2 = 1.697 - 0.0156 * c.age
            Q3 = 0.691 - 0.0107 * c.age + 0.0218 * qi

        self.pk = PKParameters(V1=v1, V2=v2, V3=v3, CL=CL, Q2=Q2, Q3=Q3)
