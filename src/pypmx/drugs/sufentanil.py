from collections.abc import Mapping
from typing import Any, ClassVar

from pypmx.pk import LinearPKModel, PKParameters, Units


class Gepts(LinearPKModel):
    """
    Gepts sufentanil model (no covariates)

    Units:
    time (min), volumes (L), clearances (L/min), dose (ug),
    concentrations (ng/ml)

    Reference:
    Gepts E, Shafer SL, Camu F, Stanski DR, Woestenborghs R, Van Peer A,
    Heykants JJ. Linearity of pharmacokinetics and model estimation of
    sufentanil. Anesthesiology. 1995;83(6):1194-1204.

    ke0 0.112 /min as implemented by Fresenius, see
    https://blog.tivatrainer.com/sufentanil-another-mistake-in-the-implementation-in-commercial-target-controlled-infusion-systems/

    Defined by V1 = 14.3 L and the micro rate constants; ``V2 = Q2/k21``
    (63.4 L, published 63.1) and ``V3 = Q3/k31`` (251.9 L, published 261.6).
    """

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        # ``covars`` is accepted for a uniform constructor; the model has none.
        self.pk = PKParameters.from_micro(
            V1=14.3,
            k10=0.0645,
            k12=0.1086,
            k21=0.0245,
            k13=0.0229,
            k31=0.0013,
            ke0=0.112,
        )
