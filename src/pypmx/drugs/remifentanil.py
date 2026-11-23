from collections.abc import Mapping
from math import exp
from typing import Any, ClassVar

from pypmx import leanbodymass
from pypmx.iiv import Omega
from pypmx.pk import Covariates, LinearPKModel, PKParameters, Units


class Minto(LinearPKModel):
    """
    Minto remifentanil model

    Units:
    age (years), weight (kg), height (cm), time (min), volumes (L),
    clearances (L/min), concentrations (ng/ml)

    Reference:
    Minto CF, Schnider TW, Egan TD, Youngs E, Lemmens HJ, Gambus PL, Billard V,
    Hoke JF, Moore KH, Hermann DJ, Muir KT, Mandema JW, Shafer SL.
    Influence of age and gender on the pharmacokinetics and pharmacodynamics of
    remifentanil. I. Model development. Anesthesiology. 1997;86(1):10-23.

    Lean body mass from the James formula.
    """

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        lbm = leanbodymass.james(c.height, c.weight, c.sex)
        age = c.age - 40
        self.pk = PKParameters(
            V1=5.1 - 0.0201 * age + 0.072 * (lbm - 55),
            V2=9.82 - 0.0811 * age + 0.108 * (lbm - 55),
            V3=5.42,
            CL=2.6 - 0.0162 * age + 0.0191 * (lbm - 55),
            Q2=2.05 - 0.0301 * age,
            Q3=0.076 - 0.00113 * age,
            ke0=0.595 - 0.007 * age,
        )


def _ageing(i: float, age: float) -> float:
    return exp(i * (age - 35))


def _sigmoid(x: float, e50: float, y: float) -> float:
    return (x**y) / ((x**y) + (e50**y))


class Eleveld(LinearPKModel):
    """
    Eleveld universal remifentanil model

    Units:
    Age (years)
    Weight (Kg)

    Reference:
    An Allometric Model of Remifentanil Pharmacokinetics and Pharmacodynamics.
    Eleveld DJ, Proost JH, Vereecke H, Absalom AR, Olofsen E, Vuyk J, Struys MMRF.
    Anesthesiology. 2017 Jun;126(6):1005-1018.
    doi: 10.1097/ALN.0000000000001634.

    IIV: exponential on V1, V2, V3, CL, Q2, Q3 (diagonal $OMEGA of the
    published model; CV 33.0, 35.0, 111.7, 14.1, 23.7, 57.5 %).
    """

    units: ClassVar[Units] = Units(time="min", cp="ng/ml")
    omega: ClassVar[Omega | None] = Omega.from_variances(
        {
            "V1": 1.035760e-01,
            "V2": 1.153660e-01,
            "V3": 8.097060e-01,
            "CL": 1.973920e-02,
            "Q2": 5.466160e-02,
            "Q3": 2.854850e-01,
        }
    )

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)

        v1ref = 5.81
        v2ref = 8.82
        v3ref = 5.03

        clref = 2.58
        q2ref = 1.72
        q3ref = 0.124

        theta1 = 2.88
        theta2 = -0.00554
        theta3 = -0.00327
        theta4 = -0.0315
        theta5 = 0.470
        theta6 = -0.0260

        ffm = leanbodymass.alsallami(c.age, c.height, c.weight, c.sex)
        ffmref = leanbodymass.alsallami(35, 170, 70, "m")
        Fsize = ffm / ffmref

        mat = _sigmoid(c.weight, theta1, 2)
        matref = _sigmoid(70, theta1, 2)
        Fmat = mat / matref

        if c.sex == "m":
            Fsex = 1
        elif c.sex == "f":
            Fsex = 1 + theta5 * _sigmoid(c.age, 12, 6) * (1 - _sigmoid(c.age, 45, 6))
        else:
            raise ValueError(f"Unknown sex {c.sex}")

        v1 = v1ref * Fsize * _ageing(theta2, c.age)
        v2 = v2ref * Fsize * _ageing(theta3, c.age) * Fsex
        v3 = v3ref * Fsize * _ageing(theta4, c.age) * exp(theta6 * (c.weight - 70))

        self.pk = PKParameters(
            V1=v1,
            V2=v2,
            V3=v3,
            CL=clref * Fsize**0.75 * Fmat * Fsex * _ageing(theta3, c.age),
            Q2=q2ref * (v2 / v2ref) ** 0.75 * _ageing(theta2, c.age) * Fsex,
            Q3=q3ref * (v3 / v3ref) ** 0.75 * _ageing(theta2, c.age),
            ke0=1.09 * _ageing(-0.0289, c.age),
        )
