from functools import partial
from math import exp, nan
from typing import NamedTuple

import numpy as np

from pypmx import leanbodymass
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


class Minto:
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
        ke0: float = nan

    def __init__(self, covars: dict[str, int | float | str] | None = None):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        c = self.covars
        lean_body_mass = leanbodymass.james(c.height, c.weight, c.sex)

        v1 = 5.1 - 0.0201 * (c.age - 40) + 0.072 * (lean_body_mass - 55)
        v2 = 9.82 - 0.0811 * (c.age - 40) + 0.108 * (lean_body_mass - 55)
        v3 = 5.42

        k10 = (2.6 - 0.0162 * (c.age - 40) + 0.0191 * (lean_body_mass - 55)) / v1
        k12 = (2.05 - 0.0301 * (c.age - 40)) / v1
        k13 = (0.076 - 0.00113 * (c.age - 40)) / v1
        k21 = k12 * (v1 / v2)
        k31 = k13 * (v1 / v3)

        ke0 = 0.595 - 0.007 * (c.age - 40)

        self.parameters = self.Parameters(
            V1=v1, V2=v2, V3=v3,
            Q2 = k12 * v1,
            Q3 = k13 * v1,
            CL = k10 * v1,
            k12=k12, k21=k21,
            k13=k13, k31=k31,
            k10=k10, ke0=ke0,
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


class Eleveld:
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
        ke0: float = nan

    def __init__(self, covars: dict[str, int | float | str] | None = None):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        c = self.covars

        def ageing(i, age):
            return exp(i * (age - 35))

        def sigmoid(x, e50, y):
            sig = (x**y) / ((x**y) + (e50**y))
            return sig

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

        mat = sigmoid(c.weight, theta1, 2)
        matref = sigmoid(70, theta1, 2)
        Fmat = mat / matref

        if c.sex == "m":
            Fsex = 1
        elif c.sex == "f":
            Fsex = 1 + theta5 * sigmoid(c.age, 12, 6) * (1 - sigmoid(c.age, 45, 6))
        else:
            raise ValueError(f"Unknown sex {c.sex}")

        v1 = v1ref * Fsize * ageing(theta2, c.age)
        v2 = v2ref * Fsize * ageing(theta3, c.age) * Fsex
        v3 = v3ref * Fsize * ageing(theta4, c.age) * exp(theta6 * (c.weight - 70))

        CL = clref * Fsize**0.75 * Fmat * Fsex * ageing(theta3, c.age)
        Q2 = q2ref * (v2 / v2ref) ** 0.75 * ageing(theta2, c.age) * Fsex
        Q3 = q3ref * (v3 / v3ref) ** 0.75 * ageing(theta2, c.age)

        ke0 = 1.09 * ageing(-0.0289, c.age)

        self.parameters = self.Parameters(
            V1=v1, V2=v2, V3=v3,
            Q2=Q2, Q3=Q3, CL=CL,
            k12=Q2 / v1, k21=Q2 / v2,
            k13=Q3 / v1, k31=Q3 / v3,
            k10=CL / v1, ke0=ke0,
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

    def bsv(self):
        # $OMEGA
        # 1.035760e-01 ; CV=33.0% V1
        # 1.153660e-01 ; CV=35.0% V2
        # 8.097060e-01 ; CV=111.7% V3
        # 1.973920e-02 ; CV=14.1% CL
        # 5.466160e-02 ; CV=23.7% Q2
        # 2.854850e-01 ; CV=57.5% Q3
        pass
