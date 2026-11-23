from collections.abc import Mapping
from math import exp, log, nan
from typing import ClassVar, NamedTuple

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp

from pypmx.iiv import Omega, RandomEffects
from pypmx.utils import ConcentrationFunction, filter_valid


class State(NamedTuple):
    HRi: float
    SVi: float
    TPR: float


class State_array(NamedTuple):
    HRi: NDArray
    SVi: NDArray
    TPR: NDArray


class Units(NamedTuple):
    time: str
    cp_propfol: str
    cp_remifentanil: str
    effect: str


class Covariates(NamedTuple):
    age: float = 35
    base_hr: float = 56
    base_sv: float = 82.2
    base_tpr: float = 0.016


class Parameters(NamedTuple):
    # Emax
    ec50_int_hr: float
    ec50_prop_sv: float
    ec50_prop_tpr: float
    ec50_remi_tpr: float
    emax_prop_sv_typ: float
    emax_prop_tpr: float
    emax_remi_tpr: float
    gamma_prop: float
    gamma_remi: float
    kout: float
    # Feedback
    fb: float
    hr_sv: float
    # HDE HR
    k: float
    ltde_hr: float
    ltde_sv: float
    # Interaction
    int_hr: float
    int_sv: float
    int_tpr: float
    sl_remi_hr: float
    sl_remi_sv: float
    # Covars
    age_emax_sv: float
    # Baseline
    base_hr: float = nan
    base_sv: float = nan
    base_tpr: float = nan
    # Calculated
    kin_tpr: float = nan
    kin_sv: float = nan
    kin_hr: float = nan
    emax_prop_sv: float = nan
    base_map: float = nan


class PropofolRemifentanil(RandomEffects):
    units: Units = Units(
        time="min", cp_propfol="ug/ml", cp_remifentanil="ng/ml", effect="mmHg"
    )
    parameters: Parameters = Parameters(
        # Emax
        ec50_int_hr=0.2,
        ec50_prop_sv=0.44,
        ec50_prop_tpr=3.21,
        ec50_remi_tpr=4.59,
        emax_prop_sv_typ=-0.15,
        emax_prop_tpr=-0.78,
        emax_remi_tpr=-1,
        gamma_prop=1.83,
        gamma_remi=1,
        kout=0.072,
        # Feedback
        fb=0.66,
        hr_sv=0.312,
        # HDE HR
        k=0.067,
        ltde_hr=0.12,
        ltde_sv=0.09,
        # Interaction
        int_hr=-0.12,
        int_sv=-0.21,
        int_tpr=1,
        sl_remi_hr=0.033,
        sl_remi_sv=0.058,
        # Covars
        age_emax_sv=0.033,
    )

    omega: ClassVar[Omega | None] = Omega.from_cv(
        {
            "base_sv": 18.3,
            "base_tpr": 23.3,
            "base_hr": 15.67,
            "ec50_prop_tpr": 74.34,
            "emax_remi_tpr": 75.30,
            "sl_remi_hr": 6.18,
            "sl_remi_sv": 9.34,
        },
        {
            ("base_sv", "base_tpr"): -0.58,
            ("base_tpr", "base_hr"): -0.65,
            ("sl_remi_hr", "sl_remi_sv"): 0.57,
        },
    )
    # residual proportional error CV (%)
    residual_error: ClassVar[dict[str, float]] = {
        "map": 11.04,
        "hr": 10.42,
        "sv": 17.12,
    }

    def __init__(
        self,
        Cp_prop: ConcentrationFunction,
        Cp_remi: ConcentrationFunction,
        covariates: dict | None = None,
        y0: State | None = None,
        random_effects: bool = False,
        rng: np.random.Generator | None = None,
    ):
        """``random_effects=True`` samples the IIV (``omega``) and the residual
        error once, from ``rng``. Use :meth:`sample` for a copy instead."""
        self._Cp_prop = Cp_prop
        self._Cp_remi = Cp_remi
        if covariates is None:
            covariates = {}
        self._covars = covars = Covariates(**filter_valid(covariates, Covariates))
        self._y0_arg = y0
        self.parameters = self.parameters._replace(**filter_valid(covars, Parameters))
        self._derive()
        if random_effects:
            self._apply_eta(self._require_omega().sample(rng))
            self.ruv = self.sample_ruv(rng)
        else:
            self.ruv = {"map": 0, "hr": 0, "sv": 0}

    def _get_values(self, names: tuple[str, ...]) -> dict[str, float]:
        return {n: getattr(self.parameters, n) for n in names}

    def _set_values(self, values: Mapping[str, float]) -> None:
        self.parameters = self.parameters._replace(**values)
        self._derive()

    def sample_ruv(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        rng = np.random.default_rng() if rng is None else rng
        vars = ["map", "hr", "sv"]
        cov = np.diag((np.array([self.residual_error[k] for k in vars]) / 100) ** 2)
        values = rng.multivariate_normal([0, 0, 0], cov)
        return dict(zip(vars, map(float, values)))

    def _derive(self) -> None:
        """Derived parameters and default ``y0`` from the base parameters."""
        p = self.parameters
        self._y0 = (
            self._y0_arg
            if self._y0_arg is not None
            else State(HRi=p.base_hr, SVi=p.base_sv, TPR=p.base_tpr)
        )
        self.parameters = self.parameters._replace(
            kin_tpr=p.kout * p.base_tpr,
            kin_sv=p.kout * p.base_sv,
            kin_hr=p.kout * p.base_hr,
            emax_prop_sv=p.emax_prop_sv_typ
            * exp(p.age_emax_sv * (self._covars.age - 35)),
            base_map=p.base_hr
            * (1 + p.ltde_hr)
            * p.base_sv
            * (1 + p.ltde_sv)
            * p.base_tpr,
        )

    def calculate_map(self, t: float, y: State):
        p = self.parameters
        TDE_hr = (p.ltde_hr * p.base_hr) * exp(-p.k * t)
        TDE_sv = (p.ltde_sv * p.base_sv) * exp(-p.k * t)
        HR = y.HRi + TDE_hr
        SV = (y.SVi + TDE_sv) * (1 - p.hr_sv * log(HR / (p.base_hr * (1 + p.ltde_hr))))
        return HR * SV * y.TPR

    def ode(self, t: float, y0: list[float]) -> tuple[float, float, float]:
        HRi, SVi, TPR = y0
        # Ensure positive state variables
        HRi = max(HRi, 0.1)
        SVi = max(SVi, 0.1)
        TPR = max(TPR, 0.0001)

        p = self.parameters
        Cp_prop = self._Cp_prop(t)
        Cp_remi = self._Cp_remi(t)
        MAP = self.calculate_map(t, State(HRi=HRi, SVi=SVi, TPR=TPR))
        RMAP = MAP / p.base_map

        eff_prop_sv = (p.emax_prop_sv * Cp_prop) / (p.ec50_prop_sv + Cp_prop)
        eff_remi_sv = (
            p.sl_remi_sv + (p.int_sv * Cp_prop) / (p.ec50_prop_sv + Cp_prop)
        ) * Cp_remi
        eff_prop_tpr = (
            (p.emax_prop_tpr + (p.int_tpr * Cp_remi) / (p.ec50_remi_tpr + Cp_remi))
            * Cp_prop**p.gamma_prop
        ) / (p.ec50_prop_tpr**p.gamma_prop + Cp_prop**p.gamma_prop)
        eff_remi_tpr = (p.emax_remi_tpr * Cp_remi**p.gamma_remi) / (
            p.ec50_remi_tpr**p.gamma_remi + Cp_remi**p.gamma_remi
        )
        eff_remi_hr = (
            p.sl_remi_hr + (p.int_hr * Cp_prop) / (p.ec50_int_hr + Cp_prop)
        ) * Cp_remi

        with np.errstate(invalid="ignore"):
            ddt_TPR = p.kin_tpr * RMAP ** (-p.fb) * (
                1 + eff_prop_tpr
            ) - p.kout * TPR * (1 - eff_remi_tpr)
            ddt_SVi = p.kin_sv * RMAP ** (-p.fb) * (1 + eff_prop_sv) - p.kout * SVi * (
                1 - eff_remi_sv
            )
            ddt_HRi = p.kin_hr * RMAP ** (-p.fb) - p.kout * HRi * (1 - eff_remi_hr)

        return ddt_HRi, ddt_SVi, ddt_TPR

    def solve_ode(self, t) -> tuple[State_array, NDArray]:
        t_span = [min(t), max(t)]
        y0 = list(self._y0)
        res = solve_ivp(
            self.ode,
            t_span=t_span,
            t_eval=t,
            y0=y0,
        )
        return State_array(HRi=res.y[0, :], SVi=res.y[1, :], TPR=res.y[2, :]), res.t

    def __call__(self, t: ArrayLike) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        p = self.parameters

        # Use new time points because ODE solver may return less points when it fails evaluation
        y, t_new = self.solve_ode(t)

        TDE_hr = (p.ltde_hr * p.base_hr) * np.exp(-p.k * t_new)
        TDE_sv = (p.ltde_sv * p.base_sv) * np.exp(-p.k * t_new)
        HR = y.HRi + TDE_hr
        SV = (y.SVi + TDE_sv) * (
            1 - p.hr_sv * np.log(HR / (p.base_hr * (1 + p.ltde_hr)))
        )

        MAP = HR * SV * y.TPR

        return (
            t_new,
            HR * (1 + self.ruv["hr"]),
            SV * (1 + self.ruv["sv"]),
            MAP * (1 + self.ruv["map"]),
        )
