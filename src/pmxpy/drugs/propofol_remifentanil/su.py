from collections.abc import Callable
from functools import cached_property
from math import exp, log, nan
from typing import NamedTuple, Union

import numpy as np
from numpy.typing import ArrayLike
from scipy.integrate import solve_ivp


class State(NamedTuple):
    HRi: float
    SVi: float
    TPR: float


class State_array(NamedTuple):
    HRi: ArrayLike
    SVi: ArrayLike
    TPR: ArrayLike


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


def filter_valid(
    data: Union[dict[str, float], NamedTuple], typeclass: NamedTuple
) -> dict[str, float]:
    valid_fields = set(typeclass._fields)
    try:
        data_fields = data._fields
        return {k: getattr(data, k) for k in set(data_fields) & valid_fields}
    except AttributeError:
        data_fields = data.keys()
        return {k: data[k] for k in set(data_fields) & valid_fields}


class PropofolRemifentanil:
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

    random_effects: dict[str, float] = {
        # coefficients of variation (CV)
        "base_sv": 18.3,
        "base_tpr": 23.3,
        "base_hr": 15.67,
        "ec50_prop_tpr": 74.34,
        "emax_remi_tpr": 75.30,
        "sl_remi_hr": 6.18,
        "sl_remi_sv": 9.34,
        # correlation between random effects
        "corr_base_sv-base_tpr": -0.58,
        "corr_base_hr-base_tpr": -0.65,
        "corr_sl_remi_hr-sl_remi_sv": 0.57,
        # residual proportional error CV
        "map": 11.04,
        "hr": 10.42,
        "sv": 17.12,
    }

    def __init__(
        self,
        Cp_prop: Callable[[float], float],
        Cp_remi: Callable[[float], float],
        covariates: dict | None = None,
        y0: State | None = None,
        random_effects: bool = False,
    ):
        self._Cp_prop = Cp_prop
        self._Cp_remi = Cp_remi
        if covariates is None:
            covariates = {}
        covars = Covariates(**filter_valid(covariates, Covariates))
        self.parameters = self.parameters._replace(**filter_valid(covars, Parameters))
        if random_effects:
            self.parameters = self.parameters._replace(**self.sample_etas())
            self.ruv = self.sample_ruv()
        else:
            self.ruv = {"map": 0, "hr": 0, "sv": 0}
        p = self.parameters
        self._y0 = (
            y0
            if y0 is not None
            else State(HRi=p.base_hr, SVi=p.base_sv, TPR=p.base_tpr)
        )
        self.parameters = self.parameters._replace(
            kin_tpr=p.kout * p.base_tpr,
            kin_sv=p.kout * p.base_sv,
            kin_hr=p.kout * p.base_hr,
            emax_prop_sv=p.emax_prop_sv_typ * exp(p.age_emax_sv * (covars.age - 35)),
            base_map=p.base_hr
            * (1 + p.ltde_hr)
            * p.base_sv
            * (1 + p.ltde_sv)
            * p.base_tpr,
        )

    @cached_property
    def covariance(self):
        params = [
            "base_sv",
            "base_tpr",
            "base_hr",
            "ec50_prop_tpr",
            "emax_remi_tpr",
            "sl_remi_hr",
            "sl_remi_sv",
        ]
        coeff_matrix = np.array([self.random_effects[k] for k in params])
        # Calculate SD (omega) from coefficients of variation
        sd = np.sqrt(np.log((coeff_matrix / 100) ** 2 + 1))
        # Calculate scale factor for correlation matrix
        sd_outer = np.outer(sd, sd)

        # Add correlation between random effects
        corr = np.diag(np.ones_like(sd))
        corr[0, 1] = self.random_effects["corr_base_sv-base_tpr"]
        corr[1, 2] = self.random_effects["corr_base_hr-base_tpr"]
        corr[5, 6] = self.random_effects["corr_sl_remi_hr-sl_remi_sv"]

        # Calculate covariance matrix
        corr_sym = np.triu(corr) + np.triu(corr, 1).T
        covariance_matrix = corr_sym * sd_outer
        return params, covariance_matrix

    def sample_etas(self):
        vars, cov = self.covariance
        mu = np.array([getattr(self.parameters, k) for k in vars])
        eta = np.random.multivariate_normal(np.zeros_like(mu), cov)
        values = mu * np.exp(eta)
        return dict(zip(vars, values))

    def sample_ruv(self):
        vars = ["map", "hr", "sv"]
        cv = np.array([self.random_effects[k] for k in vars])
        cov = np.diag((cv / 100) ** 2)
        values = np.random.multivariate_normal([0, 0, 0], cov)
        return dict(zip(vars, values))

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

    def solve_ode(self, t) -> tuple[State_array, ArrayLike]:
        t_span = [min(t), max(t)]
        y0 = list(self._y0)
        res = solve_ivp(
            self.ode,
            t_span=t_span,
            t_eval=t,
            y0=y0,
        )
        return State_array(HRi=res.y[0, :], SVi=res.y[1, :], TPR=res.y[2, :]), res.t

    def calculate_at(
        self, t: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
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
