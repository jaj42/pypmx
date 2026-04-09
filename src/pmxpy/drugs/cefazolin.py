from datetime import datetime, timedelta
from typing import List

import numpy as np
from scipy.integrate import solve_ivp


class Cefazolin:
    dose_unit = "milligram"
    rate_unit = "milligram / second"
    plasma_unit = "milligram / liter"

    def __init__(
        self, CrCL: int, t0: datetime | None = None, y0: List[float] = [0, 0]
    ) -> None:
        """
        Lanoiselée J, Chaux R, Hodin S, Bourayou S, Gibert A, Philippot R, Molliex S, Zufferey PJ, Delavenne X, Ollier E.
        Population pharmacokinetic model of cefazolin in total hip arthroplasty.
        Sci Rep. 2021 Oct 5;11(1):19763. doi: 10.1038/s41598-021-99162-7. PMID: 34611213; PMCID: PMC8492877.
        """
        self._dt = t0 if t0 is not None else datetime.now()
        self._y = y0
        self._rate = 0
        theta_1 = 2.86
        theta_2 = 0.79
        Q = 10.9  # L/h
        if CrCL:
            # Model 2 (with covar)
            CL = theta_1 * (CrCL / 80) ** theta_2  # L/h
        else:
            # Model 1 (without covar)
            CL = 2.87  # L/h
        params = {
            "CL": CL / 3600,  # L/h -> L/second
            "Vc": 5.2,  # L
            "Q": Q / 3600,  # L/h -> L/second
            "Vp": 4.56,  # L
        }
        k10 = params["CL"] / params["Vc"]
        k12 = params["Q"] / params["Vc"]
        k21 = (k12 * params["Vc"]) / params["Vp"]
        self._params = params | {"k10": k10, "k12": k12, "k21": k21}

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        self.run_pk()
        self._rate = float(value)

    def give_bolus(self, dose):
        self._y[0] += dose / self._params["Vc"]

    def pk(self, t, y, Inf, p):
        C1, C2 = y
        ddt_C1 = (
            Inf + C2 * p["Vp"] * p["k21"] - C1 * p["Vc"] * (p["k12"] + p["k10"])
        ) / p["Vc"]
        ddt_C2 = (C1 * p["Vc"] * p["k12"] - C2 * p["Vp"] * p["k21"]) / p["Vp"]
        return ddt_C1, ddt_C2

    @property
    def cc(self):
        return self._y[0]

    def run_pk(self, dt=None):
        if dt is None:
            dt = datetime.now()
        t = (dt - self._dt).total_seconds()
        res = solve_ivp(
            self.pk,
            t_span=[0, t],
            t_eval=[t],
            y0=self._y,
            args=(self._rate, self._params),
        )
        self._dt = dt
        ytmp = np.squeeze(res.y)
        if len(ytmp) == len(self._y):
            self._y = ytmp
        else:
            # TODO: log error
            pass
        return dt, self.cc

    def simulate_pk(self, dt, bolus=None, inf_rate=None, npoints=100):
        if bolus is None and inf_rate is None:
            # Reset
            return ([], [])
        t0 = self._dt
        y0 = self._y.copy()
        t = (dt - t0).total_seconds()
        t_span = [0, t]
        if bolus:
            y0[0] += bolus / self._params["Vc"]
        rate = inf_rate if inf_rate is not None else self._rate
        sol_pk = solve_ivp(
            self.pk, t_span=t_span, y0=y0, args=(rate, self._params), dense_output=True
        )

        t = np.linspace(*t_span, npoints)
        z = sol_pk.sol(t)
        Cc = z[0]

        t_out = [t0 + timedelta(seconds=_t) for _t in t]

        return (t_out, Cc)
