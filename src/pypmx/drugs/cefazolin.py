from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

from pypmx.pk import LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    CrCL: float | None = None  # creatinine clearance (ml/min), None if unknown


class Lanoiselee(LinearPKModel):
    """
    Lanoiselée cefazolin model (2-compartment, total hip arthroplasty)

    Units:
    CrCL (ml/min), time (min), volumes (L), clearances (L/min), dose (mg),
    concentrations (mg/L)

    Reference:
    Lanoiselée J, Chaux R, Hodin S, Bourayou S, Gibert A, Philippot R,
    Molliex S, Zufferey PJ, Delavenne X, Ollier E. Population pharmacokinetic
    model of cefazolin in total hip arthroplasty. Sci Rep. 2021;11(1):19763.
    doi:10.1038/s41598-021-99162-7.

    With ``CrCL`` the covariate model is used (CL = 2.86·(CrCL/80)^0.79 L/h),
    without it the base model (CL = 2.87 L/h).
    """

    units: ClassVar[Units] = Units(time="min", cp="mg/L")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        if c.CrCL is None:
            cl = 2.87  # L/h, model 1 (no covariate)
        elif c.CrCL > 0:
            cl = 2.86 * (c.CrCL / 80) ** 0.79  # L/h, model 2
        else:
            raise ValueError(f"CrCL must be positive (got {c.CrCL})")
        self.pk = PKParameters(V1=5.2, V2=4.56, CL=cl / 60, Q2=10.9 / 60)
