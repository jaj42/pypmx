from collections.abc import Mapping
from typing import Any, ClassVar

from pypmx.pk import Covariates, LinearPKModel, PKParameters, Units


class Maitre(LinearPKModel):
    """
    Maitre alfentanil model

    Units:
    age (years), weight (kg), time (min), volumes (L), clearances (L/min),
    dose (ug), concentrations (ng/ml)

    Reference:
    Maitre PO, Vozeh S, Heykants J, Thomson DA, Stanski DR. Population
    pharmacokinetics of alfentanil: the average dose-plasma concentration
    relationship and interindividual variability in patients. Anesthesiology.
    1987;66(1):3-12. ke0 0.77 /min as in STANPUMP.

    V1 = 0.111·weight (men) or 0.128·weight (women); CL = 0.356 L/min and
    k31 = 0.0126 /min, both decreasing after 40 years. ``V2 = V1·k12/k21``,
    ``V3 = V1·k13/k31``.
    """

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        if c.sex not in ("m", "f"):
            raise ValueError(f"Unknown value for sex: {c.sex!r}")
        v1 = (0.111 if c.sex == "m" else 0.128) * c.weight
        over40 = max(c.age - 40, 0)
        cl = 0.356 - 0.00269 * over40
        self.pk = PKParameters.from_micro(
            V1=v1,
            k10=cl / v1,
            k12=0.104,
            k21=0.0673,
            k13=0.017,
            k31=0.0126 - 0.000113 * over40,
            ke0=0.77,
        )
