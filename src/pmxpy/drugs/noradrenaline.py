from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp

from pmxpy.utils import ConcentrationFunction

# # 5 ug/h = <Quantity(0.261015371, 'nanomole / minute')>
# df["Babynoradrenaline-rate"] *= 0.261015371

# <Quantity(5.91086417, 'nanomole / microgram')>


class Noradrenaline:
    _thetas = {
        "V1": 1.08,
        "V2": 1.39,
        "CL": 1.49,
        "Q": 0.319,
        "EPROD": 1.36,
        "E0": 55.8,
        "EC50": 54.2,
        "KE0": 1.29,
        "GAMMA": 1.34,
        "EMAX": 200,
        "PKLAG": 0.546,
        "PDLAG": -0.188,
    }
    ode_info = {"ndim": 3}

    def __init__(
        self,
        covars: dict[str, float] | None = None,
        t0: float = 0,
        y0: list[float] | None = None,
        rate: float | ConcentrationFunction = 0.0,
        params: dict[str, float] | None = None,
    ) -> None:
        self._t0 = t0
        self._covars = covars
        if isinstance(rate, Callable):
            self._rate = rate
        else:
            self._rate = lambda _: rate
        self.calculate_parameters()
        self.update_parameters(params)
        if y0 is None:
            p = self._params
            endo = p["EPROD"] / p["CL"]
            self._y0 = np.full(self.ode_info["ndim"], endo)
            self._y0[0] *= p["V1"]
            self._y0[1] *= p["V2"]
        else:
            self._y0 = np.array(y0)

    def calculate_parameters(self):
        weightref = 70
        if self._covars is not None:
            weight = self._covars["weight"]
        else:
            weight = 70

        def scale(i):
            return (weight / weightref) ** i

        p = self._thetas.copy()
        p["V1"] *= scale(1)
        p["V2"] *= scale(1)
        p["CL"] *= scale(0.75)
        p["Q"] *= scale(0.75)
        p["EPROD"] *= scale(0.75)
        p["KE0"] *= scale(0.25)
        self._params = p

    def update_parameters(self, params: dict[str, float] | None = None):
        if params is not None:
            self._params.update(params)
        p = self._params
        self._params["K10"] = p["CL"] / p["V1"]
        self._params["K12"] = p["Q"] / p["V1"]
        self._params["K21"] = p["Q"] / p["V2"]

    def get_state(self):
        return self._y0

    def get_param(self, param):
        return self._params[param]

    def ode(self, t, y, rate_func):
        A1, A2, Ce = y
        p = self._params
        Tlag = p["PKLAG"] + p["PDLAG"]
        ddt_A1 = (
            A2 * p["K21"]
            - A1 * (p["K10"] + p["K12"])
            + p["EPROD"]
            + rate_func(t - Tlag)
        )
        ddt_A2 = A1 * p["K12"] - A2 * p["K21"]
        ddt_Ce = p["KE0"] * (A1 / p["V1"] - Ce)
        return ddt_A1, ddt_A2, ddt_Ce

    def solve_ode(
        self, t: NDArray, y0: list[float] | None = None
    ) -> tuple[NDArray, NDArray, NDArray]:
        if y0 is None:
            y0 = list(self.get_state())
        t = np.array(t)
        t_span = [min(t), max(t)]
        res = solve_ivp(
            self.ode,
            t_span=t_span,
            t_eval=t,
            y0=y0,
            args=(self._rate,),
            # max_step=0.1,
            dense_output=True,
        )
        return res.t, res.y[0, :], res.y[2, :]

    def pd(self, Ce: NDArray, E0: float | None = None) -> NDArray:
        if E0 is None:
            E0 = self._params["E0"]
        gamma = self._params["GAMMA"]
        Emax = self._params["EMAX"]
        EC50 = self._params["EC50"]
        return E0 + Emax * Ce**gamma / (EC50**gamma + Ce**gamma)

    def __call__(self, t: NDArray) -> tuple[NDArray, NDArray, NDArray, NDArray]:
        ode_t, A1, Ce = self.solve_ode(t)
        Cp = A1 / self._params["V1"]
        MAP = self.pd(Ce)
        return ode_t, Cp, Ce, MAP

    def copy_with(
        self,
        t0: float | None = None,
        y0: list[float] | None = None,
        rate: float | ConcentrationFunction = 0.0,
        params: dict[str, float] | None = None,
    ):
        if t0 is None:
            t0 = self._t0
        if rate is None:
            rate = self._rate
        if params is None:
            params = self._params.copy()
        return Noradrenaline(self._covars, t0, y0, rate, params)
