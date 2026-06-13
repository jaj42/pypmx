import warnings
from functools import partial
from math import exp, nan
from typing import NamedTuple

import numpy as np
from scipy.special import expit

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
    opiates: bool = True


class Eleveld:
    units: Units = Units(time="min", cp="ug/ml")

    class Parameters(NamedTuple):
        """
        PK parameters
        | Parameter | Variable Name | Description                                                  |
        | --------- | ------------- | ------------------------------------------------------------ |
        | THETA(1)  | V1ref         | Reference volume of central compartment (adult) = 6.283 L    |
        | THETA(2)  | V2ref         | Reference volume of peripheral compartment 1 = 25.501 L      |
        | THETA(3)  | V3ref         | Reference volume of peripheral compartment 2 = 272.817 L     |
        | THETA(4)  | CLref_male    | Reference clearance (male) = 1.790 L/min                     |
        | THETA(5)  | Q2ref         | Reference inter-compartmental clearance 2 = 1.750 L/min      |
        | THETA(6)  | Q3ref         | Reference inter-compartmental clearance 3 = 1.109 L/min      |
        | THETA(7)  | σ_log         | Log-error                                                    |
        | THETA(8)  | E50_CL        | Maturation CL E50 = 42.3 weeks                               |
        | THETA(9)  | γ_CL          | Maturation CL slope = 9.05                                   |
        | THETA(10) | β_V2_age      | V2 decline with age                                          |
        | THETA(11) | β_CL_age_tech | CL decline with age with opiates                             |
        | THETA(12) | E50_V1        | V1 sigmoid E50 = 33.6 kg                                     |
        | THETA(13) | β_V3_age_tech | V3 decline with age with opiates                             |
        | THETA(14) | E50_Q3        | Maturation Q3 E50 = 68.3 weeks                               |
        | THETA(15) | CLref_female  | Reference clearance (female) = 2.100 L/min                   |
        | THETA(16) | κ_Q2_child    | Q2 child factor = 1.304                                      |
        | THETA(17) | θ_V1_venous   | V1 extra (max, venous) = 1.419 L                             |
        | THETA(18) | θ_Q2_venous   | Q2 less for venous = 0.681 L/min                             |

        PD parameters
        | Parameter | Description                       | Typical Value |
        | --------- | --------------------------------- | ------------- |
        | THETA(1)  | log(EC₅₀) base value              | 1.125         |
        | THETA(2)  | log(k_{e0}) for arterial sampling | -1.922        |
        | THETA(3)  | E_{max} (maximum effect)          | 92.98         |
        | THETA(4)  | log(γ) for arterial sampling      | 0.388         |
        | THETA(5)  | log(residual error SD)            | 2.083         |
        | THETA(6)  | Age effect on absorption lag      | 0.0517        |
        | THETA(7)  | Age effect on EC₅₀                | -0.00635      |
        | THETA(8)  | log(k_{e0}) for venous sampling   | 0.216         |
        | THETA(9)  | log(γ₁) for venous sampling       | 0.639         |
        """

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
        v1ref: float = 6.283
        v2ref: float = 25.501
        v3ref: float = 272.817
        clref_male: float = 1.790
        q2ref: float = 1.750
        q3ref: float = 1.109
        sigma_log: float = 0.191
        e50_cl: float = 42.3
        gamma_cl: float = 9.05
        beta_v2_age: float = -0.0156
        beta_cl_age_tech: float = -0.00286
        e50_v1: float = 33.6
        beta_v3_age_tech: float = -0.0138
        e50_q3: float = 68.3
        clref_female: float = 2.100
        kappa_q2_child: float = 1.304
        v1_venous: float = 1.419
        q2_venous: float = 0.681
        # PD
        ce50: float = nan
        bis_delay: float = nan
        ce50_ref: float = 3.08
        ke0_ref: float = 0.14631
        bis_ref: float = 92.98
        gamma_high_ce: float = 1.4740
        ruv_pd: float = 8.0285
        age_pdlag: float = 0.0517
        age_ce50: float = -0.00635
        ke0_venous: float = 1.24
        gamma_low_ce: float = 1.8946
        # Reference values
        age_ref: int = 35
        weight_ref: float = 70
        height_ref: int = 170
        sex_ref: str = "m"

    def __repr__(self):
        return f"Propofol Eleveld ({self.covars})"

    def __init__(self, covars: dict[str, int | float | str] | None = None):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        self.parameters = self.Parameters()
        p = self.parameters
        c = self.covars

        # post menstrual age
        pma = self.covars.age * 52 + 40
        pmaref = self.parameters.age_ref * 52 + 40

        # clearance maturation
        clmat = self.sigmoid(pma, p.e50_cl, p.gamma_cl)
        clmatref = self.sigmoid(pmaref, p.e50_cl, p.gamma_cl)

        # q3 maturation
        q3mat = self.sigmoid(pma, p.e50_q3, 1)
        q3matref = self.sigmoid(pmaref, p.e50_q3, 1)

        # fat free mass
        ffm = leanbodymass.alsallami(c.age, c.height, c.weight, c.sex)
        ffmref = leanbodymass.alsallami(
            p.age_ref, p.height_ref, p.weight_ref, p.sex_ref
        )

        v1 = p.v1ref * (self.central(c.weight) / self.central(p.weight_ref))
        v2 = p.v2ref * (c.weight / p.weight_ref) * self.ageing(p.beta_v2_age)
        v3 = p.v3ref * (ffm / ffmref)

        if c.sex == "m":
            CL = p.clref_male * (c.weight / p.weight_ref) ** 0.75 * (clmat / clmatref)
        else:
            CL = p.clref_female * (c.weight / p.weight_ref) ** 0.75 * (clmat / clmatref)

        if c.opiates:
            v3 *= exp(p.beta_v3_age_tech * c.age)
            CL *= exp(p.beta_cl_age_tech * c.age)

        Q2 = p.q2ref * (v2 / p.v2ref) ** 0.75 * (1 + p.kappa_q2_child * (1 - q3mat))
        Q3 = p.q3ref * (v3 / p.v3ref) ** 0.75 * (q3mat / q3matref)

        ke0 = p.ke0_ref * ((c.weight / p.weight_ref) ** -0.25)
        ce50 = p.ce50_ref * self.ageing(p.age_ce50)
        bis_delay = 15 + exp(p.age_pdlag * c.age)

        self.parameters = self.parameters._replace(
            V1=v1,
            V2=v2,
            V3=v3,
            Q2=Q2,
            Q3=Q3,
            CL=CL,
            k12=Q2 / v1,
            k21=Q2 / v2,
            k13=Q3 / v1,
            k31=Q3 / v3,
            k10=CL / v1,
            ke0=ke0,
            ce50=ce50,
            bis_delay=bis_delay,
        )
        self.ode = partial(pk_3comp, p=self.parameters._asdict())

    def ageing(self, x):
        return exp(x * (self.covars.age - self.parameters.age_ref))

    def sigmoid(self, x, e50, y):
        return (x**y) / ((x**y) + (e50**y))

    def central(self, x):
        return self.sigmoid(x, self.parameters.e50_v1, 1)

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


class Schnider:
    units: Units = Units(time="min", cp="ug/ml")

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

    def __init__(self, covars: Covariates):
        self.covars = Covariates(**covars) if covars is not None else Covariates()
        c = self.covars
        lean_body_mass = leanbodymass.james(c.height, c.weight, c.sex)

        v1 = 4.27
        v2 = 18.9 - 0.391 * (c.age - 53)
        v3 = 238

        k10 = (
            0.443
            + 0.0107 * (c.weight - 77)
            - 0.0159 * (lean_body_mass - 59)
            + 0.0062 * (c.height - 177)
        )
        k12 = 0.302 - 0.0056 * (c.age - 53)
        k13 = 0.196
        k21 = 1.29 - 0.024 * (c.age - 53) / v2
        k31 = 0.0035

        ke0 = 0.456

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


class Marsh:
    units: Units = Units(time="min", cp="ug/ml")

    def __init__(self, covars: Covariates):
        weight = covars.weight

        self.v1 = 0.228 * weight
        self.v2 = 0.463 * weight
        self.v3 = 2.893 * weight

        self.k10 = 0.119
        self.k12 = 0.112
        self.k13 = 0.042
        self.k21 = 0.055
        self.k31 = 0.0031

        self.keo = 0.26


class Kataria:
    units: Units = Units(time="min", cp="ug/ml")

    def __init__(self, covars: Covariates):
        age = covars.age
        weight = covars.weight

        if not 2.99 < age < 12:
            warnings.warn("Age out of range of model validation (3-11)")

        self.v1 = 0.38 * weight
        self.v2 = (0.59 * weight) + (3.1 * age) - 13
        self.v3 = 6.12 * weight

        self.Q1 = 0.037 * weight
        self.Q2 = 0.063 * weight
        self.Q3 = 0.025 * weight

        self.k10 = self.Q1 / self.v1
        self.k12 = self.Q2 / self.v1
        self.k13 = self.Q3 / self.v1
        self.k21 = (self.k12 * self.v1) / self.v2
        self.k31 = (self.k13 * self.v1) / self.v3

        self.keo = 0


class Paedfusor:
    units: Units = Units(time="min", cp="ug/ml")

    def __init__(self, covars: Covariates):
        age = covars.age
        weight = covars.weight

        if age < 1:
            warnings.warn("age below that for which model is intended")
        elif age > 12:
            warnings.warn("Warning: Patient older than intended for model")

        self.v1 = 0.46 * weight
        self.v2 = 0.95 * weight
        self.v3 = 5.85 * weight

        self.k10 = 0.1527 * (weight ** (-0.3))
        self.k12 = 0.114
        self.k13 = 0.042
        self.k21 = 0.055
        self.k31 = 0.0033

        self.keo = 0
