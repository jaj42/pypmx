from collections.abc import Mapping
from typing import Any, ClassVar, NamedTuple

from pypmx.pk import Covariates, LinearPKModel, PKParameters, Units

# Colin PJ, Hannivoort LN, Eleveld DJ, Reyntjens KMEM, Absalom AR, Vereecke HEM, Struys MMRF.
# Dexmedetomidine pharmacokinetic-pharmacodynamic modelling in healthy volunteers:
# 1. Influence of arousal on bispectral index and sedation.
# Br J Anaesth. 2017 Aug 1;119(2):200-210. doi: 10.1093/bja/aex085. PMID: 28854538.
# Built on top of the Hannivoort PK model.
COLIN_EFFECTS = {
    # ke0 (min⁻¹), C50 (ng/ml)
    "moaas": {"ke0": 0.0428, "ce50": 0.428},
    "bis": {"ke0": 0.120, "ce50": 2.63},
}


class Hannivoort(LinearPKModel):
    """3 compartment dexmedetomidine PK model

    Units:
    weight (kg), time (min), volumes (L), clearances (L/min), concentrations (ng/ml)

    Reference:
    Hannivoort LN, Eleveld DJ, Proost JH, Reyntjens KMEM, Absalom AR, Vereecke HEM, Struys MMRF.
    Development of an Optimized Pharmacokinetic Model of Dexmedetomidine Using
    Target-controlled Infusion in Healthy Volunteers.
    Anesthesiology. 2015 Aug;123(2):357-67. doi:10.1097/ALN.0000000000000740

    Effect site (ke0, C50) from Colin et al. 2017 (BJA 119:200), either for
    MOAA/S (effect="moaas", default) or BIS (effect="bis").
    """

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")
    options: ClassVar[tuple[str, ...]] = ("effect",)

    class Parameters(NamedTuple):
        v1ref: float = 1.78
        v2ref: float = 30.3
        v3ref: float = 52.0
        clref: float = 0.686
        q2ref: float = 2.98
        q3ref: float = 0.602
        weight_ref: float = 70
        # PD (from the selected Colin effect)
        ce50: float = 0.0

    parameters: Parameters

    def __init__(
        self,
        covars: Mapping[str, Any] | None = None,
        effect: str = "moaas",
    ):
        if effect not in COLIN_EFFECTS:
            raise ValueError(
                f"Unknown effect {effect!r}, expected one of {list(COLIN_EFFECTS)}"
            )
        self.effect = effect
        c = self._set_covariates(covars, Covariates)
        p = self.Parameters(ce50=COLIN_EFFECTS[effect]["ce50"])
        self.parameters = p

        wt = c.weight / p.weight_ref
        V2 = p.v2ref * wt
        V3 = p.v3ref * wt
        self.pk = PKParameters(
            V1=p.v1ref * wt,
            V2=V2,
            V3=V3,
            CL=p.clref * wt**0.75,
            Q2=p.q2ref * (V2 / p.v2ref) ** 0.75,
            Q3=p.q3ref * (V3 / p.v3ref) ** 0.75,
            ke0=COLIN_EFFECTS[effect]["ke0"],
        )


class Dyck(LinearPKModel):
    """3 compartment dexmedetomidine PK model (no effect site)

    Units:
    height (cm), time (min), volumes (L), clearances (L/min), concentrations (ng/ml)

    Reference:
    Dyck JB, Maze M, Haack C, Azarnoff DL, Vuorilehto L, Shafer SL.
    Computer-controlled infusion of intravenous dexmedetomidine hydrochloride
    in adult human volunteers.
    Anesthesiology. 1993 May;78(5):821-8. doi:10.1097/00000542-199305000-00003
    """

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        self.pk = PKParameters(
            V1=7.99,
            V2=13.8,
            V3=187,
            CL=0.00791 * c.height - 0.928,
            Q2=2.26,
            Q3=1.99,
        )
