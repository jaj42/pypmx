from collections.abc import Mapping
from math import exp, nan
from typing import Any, ClassVar, NamedTuple

from pypmx import leanbodymass
from pypmx.pk import LinearPKModel, PKParameters, Units


class Covariates(NamedTuple):
    age: float = 35
    weight: float = 70
    height: float = 170
    sex: str = "m"
    opiates: bool = True


def _sigmoid(x: float, e50: float, y: float) -> float:
    return (x**y) / ((x**y) + (e50**y))


class Eleveld(LinearPKModel):
    """
    Eleveld propofol model (PK and BIS PD)

    Units:
    age (years), weight (kg), height (cm), time (min), volumes (L),
    clearances (L/min), concentrations (ug/ml)

    Reference:
    Eleveld DJ, Colin P, Absalom AR, Struys MMRF. Pharmacokinetic-
    pharmacodynamic model for propofol for broad application in anaesthesia
    and sedation. Br J Anaesth. 2018;120(5):942-959.
    doi:10.1016/j.bja.2018.01.018.

    Arterial PK; ``opiates`` (default True) selects the age-related decline in
    CL and V3 seen with concomitant opiates. ``parameters`` holds the thetas
    and the derived PD values ``ce50`` and ``bis_delay``.
    """

    units: ClassVar[Units] = Units(time="min", cp="ug/ml")

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
        ce50: float = nan  # derived
        bis_delay: float = nan  # derived
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

    parameters: Parameters

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        p = self.Parameters()

        def ageing(x: float) -> float:
            return exp(x * (c.age - p.age_ref))

        def central(x: float) -> float:
            return _sigmoid(x, p.e50_v1, 1)

        # post menstrual age
        pma = c.age * 52 + 40
        pmaref = p.age_ref * 52 + 40

        # clearance maturation
        clmat = _sigmoid(pma, p.e50_cl, p.gamma_cl)
        clmatref = _sigmoid(pmaref, p.e50_cl, p.gamma_cl)

        # q3 maturation
        q3mat = _sigmoid(pma, p.e50_q3, 1)
        q3matref = _sigmoid(pmaref, p.e50_q3, 1)

        # fat free mass
        ffm = leanbodymass.alsallami(c.age, c.height, c.weight, c.sex)
        ffmref = leanbodymass.alsallami(
            p.age_ref, p.height_ref, p.weight_ref, p.sex_ref
        )

        v1 = p.v1ref * (central(c.weight) / central(p.weight_ref))
        v2 = p.v2ref * (c.weight / p.weight_ref) * ageing(p.beta_v2_age)
        v3 = p.v3ref * (ffm / ffmref)

        clref = p.clref_male if c.sex == "m" else p.clref_female
        CL = clref * (c.weight / p.weight_ref) ** 0.75 * (clmat / clmatref)

        if c.opiates:
            v3 *= exp(p.beta_v3_age_tech * c.age)
            CL *= exp(p.beta_cl_age_tech * c.age)

        Q2 = p.q2ref * (v2 / p.v2ref) ** 0.75 * (1 + p.kappa_q2_child * (1 - q3mat))
        Q3 = p.q3ref * (v3 / p.v3ref) ** 0.75 * (q3mat / q3matref)

        self.pk = PKParameters(
            V1=v1,
            V2=v2,
            V3=v3,
            CL=CL,
            Q2=Q2,
            Q3=Q3,
            ke0=p.ke0_ref * ((c.weight / p.weight_ref) ** -0.25),
        )
        self.parameters = p._replace(
            ce50=p.ce50_ref * ageing(p.age_ce50),
            bis_delay=15 + exp(p.age_pdlag * c.age),
        )


class Schnider(LinearPKModel):
    """
    Schnider propofol model

    Units:
    age (years), weight (kg), height (cm), time (min), volumes (L),
    clearances (L/min), concentrations (ug/ml)

    Reference:
    Schnider TW, Minto CF, Gambus PL, Andresen C, Goodale DB, Shafer SL,
    Youngs EJ. The influence of method of administration and covariates on the
    pharmacokinetics of propofol in adult volunteers. Anesthesiology.
    1998;88(5):1170-82. ke0 0.456 /min from Schnider et al. 1999 (PD).

    Defined, as in STANPUMP and TCI pumps, by V1 and the micro rate constants
    (k10, k12, k13, k21, k31; lean body mass from the James formula). The
    published V2/V3 are rounded and slightly inconsistent with k21/k31, so
    here ``V2 = Q2/k21`` and ``V3 = Q3/k31`` (Q = k·V1). This only affects the
    reported volumes, not the concentrations.
    """

    units: ClassVar[Units] = Units(time="min", cp="ug/ml")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        lbm = leanbodymass.james(c.height, c.weight, c.sex)

        v1 = 4.27
        k10 = (
            0.443
            + 0.0107 * (c.weight - 77)
            - 0.0159 * (lbm - 59)
            + 0.0062 * (c.height - 177)
        )
        k12 = 0.302 - 0.0056 * (c.age - 53)
        k13 = 0.196
        k21 = (1.29 - 0.024 * (c.age - 53)) / (18.9 - 0.391 * (c.age - 53))
        k31 = 0.0035

        self.pk = PKParameters.from_micro(
            V1=v1, k10=k10, k12=k12, k21=k21, k13=k13, k31=k31, ke0=0.456
        )


class Marsh(LinearPKModel):
    """
    Marsh propofol model (as in the Diprifusor)

    Units:
    weight (kg), time (min), volumes (L), clearances (L/min),
    concentrations (ug/ml)

    Reference:
    Marsh B, White M, Morton N, Kenny GN. Pharmacokinetic model driven infusion
    of propofol in children. Br J Anaesth. 1991;67(1):41-48.
    ke0 0.26 /min as in the Diprifusor.

    Defined by V1 = 0.228·weight and the micro rate constants;
    ``V2 = Q2/k21`` (≈ 0.464·weight) and ``V3 = Q3/k31`` (≈ 3.09·weight).
    """

    units: ClassVar[Units] = Units(time="min", cp="ug/ml")

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        self.pk = PKParameters.from_micro(
            V1=0.228 * c.weight,
            k10=0.119,
            k12=0.112,
            k21=0.055,
            k13=0.042,
            k31=0.0031,
            ke0=0.26,
        )


class Kataria(LinearPKModel):
    """
    Kataria paediatric propofol model (no effect site)

    Units:
    age (years), weight (kg), time (min), volumes (L), clearances (L/min),
    concentrations (ug/ml)

    Reference:
    Kataria BK, Ved SA, Nicodemus HF, Hoy GR, Lea D, Dubois MY, Mandema JW,
    Shafer SL. The pharmacokinetics of propofol in children using three
    different data analysis approaches. Anesthesiology. 1994;80(1):104-122.
    """

    units: ClassVar[Units] = Units(time="min", cp="ug/ml")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {"age": (3, 12)}

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        self.pk = PKParameters(
            V1=0.38 * c.weight,
            V2=0.59 * c.weight + 3.1 * c.age - 13,
            V3=6.12 * c.weight,
            CL=0.037 * c.weight,
            Q2=0.063 * c.weight,
            Q3=0.025 * c.weight,
        )


class Paedfusor(LinearPKModel):
    """
    Paedfusor propofol model (no effect site)

    Units:
    age (years), weight (kg), time (min), volumes (L), clearances (L/min),
    concentrations (ug/ml)

    Reference:
    Absalom A, Amutike D, Lal A, White M, Kenny GN. Accuracy of the
    'Paedfusor' in children undergoing cardiac surgery or catheterization.
    Br J Anaesth. 2003;91(4):507-513.

    Defined by V1 = 0.46·weight and the micro rate constants
    (k10 = 0.1527·weight^-0.3); ``V2 = Q2/k21``, ``V3 = Q3/k31``. Only the
    1-12 year form is implemented.
    """

    units: ClassVar[Units] = Units(time="min", cp="ug/ml")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {"age": (1, 12)}

    def __init__(self, covars: Mapping[str, Any] | None = None):
        c = self._set_covariates(covars, Covariates)
        self.pk = PKParameters.from_micro(
            V1=0.46 * c.weight,
            k10=0.1527 * c.weight**-0.3,
            k12=0.114,
            k21=0.055,
            k13=0.042,
            k31=0.0033,
        )
