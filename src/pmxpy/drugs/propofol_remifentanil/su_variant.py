from collections.abc import Mapping
from functools import cached_property
from math import exp, log, nan
from typing import Any, NamedTuple, Protocol, TypeVar

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_ivp


class ConcentrationFunction(Protocol):
    def __call__(self, t: float, /) -> float: ...


FloatArray = NDArray[np.floating[Any]]


class State(NamedTuple):
    HRi: float
    SVi: float
    TPR: float


class StateArray(NamedTuple):
    HRi: FloatArray
    SVi: FloatArray
    TPR: FloatArray


class Units(NamedTuple):
    time: str
    cp_propfol: str
    cp_remifentanil: str
    effect: str


class Covariates(NamedTuple):
    age: float = 35
    base_hr: float = exp(4.06)
    base_sv: float = exp(4.49)
    base_tpr: float = exp(-4.08)


class Parameters(NamedTuple):
    # Drug effect parameters
    ec50_int_hr_prop: float
    ec50_prop_sv: float
    ec50_prop_tpr: float
    ec50_remi_tpr: float
    emax_prop_sv_lgt: float
    emax_prop_tpr: float
    emax_remi_tpr: float
    gam_tpr_prop: float
    kout: float
    # Feedback
    fb: float
    hr_sv: float
    # Time-dependent effects
    k: float
    tde_hr: float
    tde_sv: float
    # Interaction parameters
    int_emax_hr_prop: float
    int_emax_sv_prop: float
    int_emax_tpr_remi: float
    sl_remi_hr: float
    sl_remi_sv: float
    # Covariates
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


T = TypeVar("T", bound=NamedTuple)


def filter_valid(
    data: Mapping[str, float] | tuple, typeclass: type[T]
) -> dict[str, float]:
    valid_fields = set(typeclass._fields)
    if hasattr(data, "_fields"):
        return {k: getattr(data, k) for k in set(data._fields) & valid_fields}
    return {k: data[k] for k in set(data.keys()) & valid_fields}


class PropofolRemifentanil:
    units: Units = Units(
        time="min", cp_propfol="ug/ml", cp_remifentanil="ng/ml", effect="mmHg"
    )
    parameters: Parameters = Parameters(
        # Drug effect parameters
        ec50_int_hr_prop=exp(-1.5),
        ec50_prop_sv=exp(-0.764),
        ec50_prop_tpr=exp(1.17),
        ec50_remi_tpr=exp(-0.708),
        emax_prop_sv_lgt=exp(-1.66),
        emax_prop_tpr=exp(-0.238),
        emax_remi_tpr=0.372,
        gam_tpr_prop=exp(0.582),
        kout=exp(-2.58),
        # Feedback
        fb=exp(-0.331),
        hr_sv=0.312,
        # Time-dependent effects
        k=exp(-1.85),
        tde_hr=exp(-2.69),
        tde_sv=exp(-3.93),
        # Interaction parameters
        int_emax_hr_prop=exp(-2.07),
        int_emax_sv_prop=exp(-1.58),
        int_emax_tpr_remi=exp(-0.943),
        sl_remi_hr=0.0306,
        sl_remi_sv=0.0459,
        # Covariates
        age_emax_sv=0.478 / 10,
    )

    random_effects: dict[str, float] = {
        # variance / covariance
        "base_sv": 0.0339,
        "base_tpr": 0.0536,
        "base_hr": 0.0241,
        "ec50_prop_tpr": 0.444,
        "emax_remi_tpr": 0.0792,
        "sl_remi_hr": 0.0041,
        "sl_remi_sv": 0.0093,
        # random effects covariance
        "cov_base_sv-base_tpr": -0.0249,
        "cov_base_hr-base_tpr": -0.0235,
        "cov_sl_remi_hr-sl_remi_sv": 0.0038,
        # residual proportional error variance
        "map": 0.0125,
        "hr": 0.0106,
        "sv": 0.0291,
    }

    def __init__(
        self,
        Cp_prop: ConcentrationFunction,
        Cp_remi: ConcentrationFunction,
        covariates: dict[str, float] | None = None,
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
            else State(
                HRi=p.base_hr * exp(p.tde_hr),
                SVi=p.base_sv * exp(p.tde_sv),
                TPR=p.base_tpr,
            )
        )

        emax_prop_sv_lgt_age = p.emax_prop_sv_lgt * exp(
            p.age_emax_sv * (covars.age - 35)
        )

        self.parameters = self.parameters._replace(
            kin_tpr=p.kout * p.base_tpr,
            kin_sv=p.kout * p.base_sv,
            kin_hr=p.kout * p.base_hr,
            emax_prop_sv=emax_prop_sv_lgt_age / (1 + emax_prop_sv_lgt_age),
            base_map=p.base_hr * p.base_sv * p.base_tpr,
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
        cov = np.diag([self.random_effects[k] for k in params])
        cov[0, 1] = self.random_effects["cov_base_sv-base_tpr"]
        cov[1, 2] = self.random_effects["cov_base_hr-base_tpr"]
        cov[5, 6] = self.random_effects["cov_sl_remi_hr-sl_remi_sv"]
        cov_sym = np.triu(cov) + np.triu(cov, 1).T
        return params, cov_sym

    def sample_etas(self):
        vars, cov = self.covariance
        mu = np.array([getattr(self.parameters, k) for k in vars])
        eta = np.random.multivariate_normal(np.zeros_like(mu), cov)
        additive_params = {"emax_remi_tpr", "sl_remi_hr", "sl_remi_sv"}
        values = {}
        for i, var in enumerate(vars):
            if var in additive_params:
                values[var] = mu[i] + eta[i]
            else:
                values[var] = mu[i] * np.exp(eta[i])
        return values

    def sample_ruv(self):
        vars = ["map", "hr", "sv"]
        cov = np.diag([self.random_effects[k] for k in vars])
        values = np.random.multivariate_normal([0, 0, 0], cov)
        return dict(zip(vars, values))

    def calculate_map(self, t: float, y: State):
        p = self.parameters
        HR = y.HRi
        SV = y.SVi * (1 - p.hr_sv * log(HR / p.base_hr))
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

        # We can use a closed form expression for TDE
        TDE = exp(-p.k * t)

        eff_prop_sv = (p.emax_prop_sv * Cp_prop) / (p.ec50_prop_sv + Cp_prop)

        sv_int_prop = (p.int_emax_sv_prop * Cp_prop) / (p.ec50_prop_sv + Cp_prop)
        eff_remi_sv = (p.sl_remi_sv - sv_int_prop) * Cp_remi
        eff_remi_sv = min(eff_remi_sv, 0.999)

        tpr_int_remi = (p.int_emax_tpr_remi * Cp_remi) / (p.ec50_remi_tpr + Cp_remi)
        tpr_prop_unt = p.emax_prop_tpr - tpr_int_remi
        eff_prop_tpr = (tpr_prop_unt * Cp_prop**p.gam_tpr_prop) / (
            p.ec50_prop_tpr**p.gam_tpr_prop + Cp_prop**p.gam_tpr_prop
        )

        eff_remi_tpr = (p.emax_remi_tpr * Cp_remi) / (p.ec50_remi_tpr + Cp_remi)
        eff_remi_tpr = max(eff_remi_tpr, -0.999)

        hr_int_prop = (p.int_emax_hr_prop * Cp_prop) / (p.ec50_int_hr_prop + Cp_prop)
        eff_remi_hr = (p.sl_remi_hr - hr_int_prop) * Cp_remi
        eff_remi_hr = min(eff_remi_hr, 0.999)

        ddt_TPR = p.kin_tpr * RMAP ** (-p.fb) * (1 - eff_prop_tpr) - p.kout * TPR * (
            1 + eff_remi_tpr
        )

        ddt_SVi = p.kin_sv * RMAP ** (-p.fb) * (1 - eff_prop_sv) * exp(
            TDE * p.tde_sv
        ) - p.kout * SVi * (1 - eff_remi_sv)

        ddt_HRi = p.kin_hr * RMAP ** (-p.fb) * exp(TDE * p.tde_hr) - p.kout * HRi * (
            1 - eff_remi_hr
        )

        return ddt_HRi, ddt_SVi, ddt_TPR

    def solve_ode(self, t: ArrayLike) -> tuple[StateArray, FloatArray]:
        t_span = [float(np.min(t)), float(np.max(t))]
        y0 = list(self._y0)
        res = solve_ivp(
            self.ode,
            t_span=t_span,
            t_eval=t,
            y0=y0,
        )
        return StateArray(HRi=res.y[0, :], SVi=res.y[1, :], TPR=res.y[2, :]), res.t

    def calculate_at(
        self, t: ArrayLike
    ) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        p = self.parameters

        y, t_new = self.solve_ode(t)

        HR = y.HRi
        SV = y.SVi * (1 - p.hr_sv * np.log(HR / p.base_hr))

        MAP = HR * SV * y.TPR

        return (
            t_new,
            HR * (1 + self.ruv["hr"]),
            SV * (1 + self.ruv["sv"]),
            MAP * (1 + self.ruv["map"]),
        )
