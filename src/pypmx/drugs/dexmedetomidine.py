from functools import partial
from math import nan
from typing import NamedTuple

import numpy as np

from pypmx.standard_models import pk_3comp


class Units(NamedTuple):
    time: str
    cp: str
    effect: str | None = None


class Covariates(NamedTuple):
    age: int = 35
    weight: float = 70
    height: float = 170
    sex: str = "m"


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


class Hannivoort:
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

    units: Units = Units(time="min", cp="ng/ml")

    class Parameters(NamedTuple):
        # PK
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
        ke0: float = nan
        v1ref: float = 1.78
        v2ref: float = 30.3
        v3ref: float = 52.0
        clref: float = 0.686
        q2ref: float = 2.98
        q3ref: float = 0.602
        # PD
        ce50: float = nan
        # Reference values
        weight_ref: float = 70

    def __repr__(self):
        return f"Dexmedetomidine Hannivoort ({self.covars}, effect={self.effect})"

    def __init__(
        self,
        covars: dict[str, int | float | str] | None = None,
        effect: str = "moaas",
    ):
        if effect not in COLIN_EFFECTS:
            raise ValueError(
                f"Unknown effect {effect!r}, expected one of {list(COLIN_EFFECTS)}"
            )
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        self.effect = effect
        self.units = Units(time="min", cp="ng/ml", effect=effect)
        p = self.Parameters()
        c = self.covars

        wt = c.weight / p.weight_ref
        v1 = p.v1ref * wt
        v2 = p.v2ref * wt
        v3 = p.v3ref * wt

        CL = p.clref * wt**0.75
        Q2 = p.q2ref * (v2 / p.v2ref) ** 0.75
        Q3 = p.q3ref * (v3 / p.v3ref) ** 0.75

        self.parameters = p._replace(
            V1=v1, V2=v2, V3=v3,
            Q2=Q2, Q3=Q3, CL=CL,
            k12=Q2 / v1, k21=Q2 / v2,
            k13=Q3 / v1, k31=Q3 / v3,
            k10=CL / v1,
            ke0=COLIN_EFFECTS[effect]["ke0"],
            ce50=COLIN_EFFECTS[effect]["ce50"],
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


class Dyck:
    """3 compartment dexmedetomidine PK model (no effect site)

    Units:
    height (cm), time (min), volumes (L), clearances (L/min), concentrations (ng/ml)

    Reference:
    Dyck JB, Maze M, Haack C, Azarnoff DL, Vuorilehto L, Shafer SL.
    Computer-controlled infusion of intravenous dexmedetomidine hydrochloride
    in adult human volunteers.
    Anesthesiology. 1993 May;78(5):821-8. doi:10.1097/00000542-199305000-00003
    """

    units: Units = Units(time="min", cp="ng/ml")

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

    def __repr__(self):
        return f"Dexmedetomidine Dyck ({self.covars})"

    def __init__(self, covars: dict[str, int | float | str] | None = None):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        c = self.covars

        v1 = 7.99
        v2 = 13.8
        v3 = 187

        CL = 0.00791 * c.height - 0.928
        Q2 = 2.26
        Q3 = 1.99

        self.parameters = self.Parameters(
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
