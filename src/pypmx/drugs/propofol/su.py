from collections.abc import Callable
from functools import cached_property
from math import exp, log
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp


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
    cp: str
    effect_map: str
    effect_hr: str
    effect_sv: str
    effect_tpr: str


class Covariates(NamedTuple):
    age: float = 35


class Propofol:
    units: Units = Units(
        time="min",
        cp="ug/ml",
        effect_map="mmHg",
        effect_hr="min^-1",
        effect_sv="ml",
        effect_tpr="mmHg*ml/min",
    )

    parameters: dict[str, float] = {
        # Baseline
        "base_hr": 53.6,
        "base_sv": 80.4,
        "base_tpr": 0.0176,
        # Emax
        "EC50_sv": 0.338,
        "EC50_tpr": 2.96,
        "Emax_sv_typ": -0.235,
        "Emax_tpr": -0.851,
        "gamma": 1.76,
        "kout": 0.0754,
        # Feedback
        "FB": 0.965,
        "HR_SV": 0.312,
        # TDE HR
        "k": 0.0358,
        "LTDE": 0.188,
        # Covars
        "age_emax_sv": 0.0258,
    }

    random_effects: dict[str, float] = {
        # coefficients of variation
        "base_sv": 18.0,
        "base_hr": 17.0,
        "base_tpr": 27.3,
        "EC50_tpr": 63.7,
        # correlation between random effects
        "corr_base_sv-base_tpr": -0.662,
        "corr_base_hr-base_tpr": -0.586,
        # residual additive error variances
        "map": 55.2,
        "hr": 33.8,
        "sv": 72.3,
    }

    def __init__(
        self,
        Cp: Callable[[float], float],
        covariates: dict | None = None,
        y0: State | None = None,
        random_effects: bool = False,
    ):
        p = self.parameters
        self._Cp = Cp

        if covariates is None or "age" not in covariates:
            self._covars: Covariates = Covariates()
        else:
            self._covars: Covariates = Covariates(age=covariates["age"])

        if random_effects:
            self.parameters.update(self.sample_etas())
            self.ruv = self.sample_ruv()
        else:
            self.ruv = {"map": 0, "hr": 0, "sv": 0}
        self.parameters.update(
            {
                "kin_tpr": p["kout"] * p["base_tpr"],
                "kin_sv": p["kout"] * p["base_sv"],
                "kin_hr": p["kout"] * p["base_hr"],
                "Emax_sv": p["Emax_sv_typ"]
                * exp(p["age_emax_sv"] * (self._covars.age - 35)),
                "base_map": p["base_hr"]
                * (1 + p["LTDE"])
                * p["base_sv"]
                * p["base_tpr"],
            }
        )
        self._y0 = (
            y0
            if y0 is not None
            else State(HRi=p["base_hr"], SVi=p["base_sv"], TPR=p["base_tpr"])
        )

    @cached_property
    def covariance(self):
        params = ["base_sv", "base_tpr", "base_hr", "EC50_tpr"]
        cv = np.array([self.random_effects[k] for k in params])
        # Calculate SD from coefficients of variation
        sd = np.sqrt(np.log((cv / 100) ** 2 + 1))
        # Calculate scale factor for correlation matrix
        sd_outer = np.outer(sd, sd)

        # Add correlation between random effects
        corr = np.diag(np.ones_like(sd))
        corr[0, 1] = self.random_effects["corr_base_sv-base_tpr"]
        corr[1, 2] = self.random_effects["corr_base_hr-base_tpr"]

        # Calculate covariance matrix
        corr_sym = np.triu(corr) + np.triu(corr, 1).T
        covariance_matrix = corr_sym * sd_outer
        return params, covariance_matrix

    def sample_etas(self):
        vars, cov = self.covariance
        mu = np.array([self.parameters[k] for k in vars])
        eta = np.random.multivariate_normal(np.zeros_like(mu), cov)
        values = mu * np.exp(eta)
        return dict(zip(vars, values))

    def sample_ruv(self):
        vars = ["map", "hr", "sv"]
        cov = np.diag([self.random_effects[k] for k in vars])
        values = np.random.multivariate_normal([0, 0, 0], cov)
        return dict(zip(vars, values))

    def calculate_map(self, t: float, y: State):
        p = self.parameters
        TDE = (p["LTDE"] * p["base_hr"]) * exp(-p["k"] * t)
        HR = y.HRi + TDE
        HR = max(HR, 1)  # make sure HR stays positive
        SV = y.SVi * (1 - p["HR_SV"] * log(HR / (p["base_hr"] * (1 + p["LTDE"]))))
        SV = max(SV, 1)  # make sure SV stays positive
        return HR * SV * y.TPR

    def ode(self, t: float, y0: list[float]) -> tuple[float, float, float]:
        HRi, SVi, TPR = y0
        p = self.parameters
        Cp = self._Cp
        MAP = self.calculate_map(t, State(HRi=HRi, SVi=SVi, TPR=TPR))
        RMAP = MAP / p["base_map"]
        ddt_TPR = (
            p["kin_tpr"]
            * RMAP ** (-p["FB"])
            * (
                1
                + (p["Emax_tpr"] * Cp(t) ** p["gamma"])
                / (p["EC50_tpr"] ** p["gamma"] + Cp(t) ** p["gamma"])
            )
            - p["kout"] * TPR
        )
        ddt_SVi = (
            p["kin_sv"]
            * RMAP ** (-p["FB"])
            * (1 + (p["Emax_sv"] * Cp(t)) / (p["EC50_sv"] + Cp(t)))
            - p["kout"] * SVi
        )
        ddt_HRi = p["kin_hr"] * RMAP ** (-p["FB"]) - p["kout"] * HRi
        return ddt_HRi, ddt_SVi, ddt_TPR

    def solve_ode(self, t) -> tuple[NDArray, State_array]:
        t_span = [min(t), max(t)]
        y0 = [self._y0.HRi, self._y0.SVi, self._y0.TPR]
        res = solve_ivp(
            self.ode,
            t_span=t_span,
            t_eval=t,
            y0=y0,
        )
        return res.t, State_array(HRi=res.y[0, :], SVi=res.y[1, :], TPR=res.y[2, :])

    def calculate_at(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        p = self.parameters

        ode_t, y = self.solve_ode(t)

        TDE = p["LTDE"] * p["base_hr"] * np.exp(-p["k"] * ode_t)
        HR = y.HRi + TDE
        SV = y.SVi * (1 - p["HR_SV"] * np.log(HR / (p["base_hr"] * (1 + p["LTDE"]))))

        MAP = HR * SV * y.TPR

        return (
            ode_t,
            HR + self.ruv["hr"],
            SV + self.ruv["sv"],
            MAP + self.ruv["map"],
        )
