from datetime import datetime, timedelta
from typing import List

import numpy as np
from scipy.integrate import solve_ivp


class Sufentanil:
    # Reference from the literature : Gepts et al.: Anesthesiology 1995, 83, 1194-1204
    # Rate constants are converted from min^-1 to s^-1
    # This uses the ke0 implemented by Fresenius
    # See https://blog.tivatrainer.com/sufentanil-another-mistake-in-the-implementation-in-commercial-target-controlled-infusion-systems/
    _params = {
        "V1": 14.3,
        "V2": 63.1,
        "V3": 261.6,
        "k10": 0.0645 / 60,
        "k12": 0.1086 / 60,
        "k13": 0.0229 / 60,
        "k21": 0.0245 / 60,
        "k31": 0.0013 / 60,
        "ke0": 0.112 / 60,
    }
    dose_unit = "microgram"
    plasma_unit = "ng / ml"

    def __init__(
        self, t0: datetime | None = None, y0: List[float] = [0, 0, 0, 0]
    ) -> None:
        self._dt = t0 if t0 is not None else datetime.now()
        self._y = y0
        self._rate = 0

    @property
    def rate(self):
        return self._rate

    @rate.setter
    def rate(self, value):
        self.run_pk()
        self._rate = float(value)

    def give_bolus(self, dose):
        self._y[0] += dose / self._params["V1"]

    def pk(self, t, y, Inf, p):
        C1, C2, C3, Ce = y
        ddt_C1 = (
            Inf
            + C2 * p["V2"] * p["k21"]
            + C3 * p["V3"] * p["k31"]
            - C1 * p["V1"] * (p["k12"] + p["k13"] + p["k10"])
        ) / p["V1"]
        ddt_C2 = (C1 * p["V1"] * p["k12"] - C2 * p["V2"] * p["k21"]) / p["V2"]
        ddt_C3 = (C1 * p["V1"] * p["k13"] - C3 * p["V3"] * p["k31"]) / p["V3"]
        ddt_Ce = p["ke0"] * (C1 - Ce)
        return ddt_C1, ddt_C2, ddt_C3, ddt_Ce

    @property
    def cc(self):
        return self._y[0]

    @property
    def ce(self):
        return self._y[3]

    def run_pkpd(self, dt=None):
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
        return dt, self.cc, self.ce

    def simulate_pkpd(self, dt, bolus=None, inf_rate=None, npoints=100):
        if bolus is None and inf_rate is None:
            # Reset
            return ([], [], [])
        t0 = self._dt
        y0 = self._y.copy()
        t = (dt - t0).total_seconds()
        t_span = [0, t]
        if bolus:
            y0[0] += bolus / self._params["V1"]
        rate = inf_rate if inf_rate is not None else self._rate
        sol_pk = solve_ivp(
            self.pk, t_span=t_span, y0=y0, args=(rate, self._params), dense_output=True
        )

        t = np.linspace(*t_span, npoints)
        z = sol_pk.sol(t)
        Cc = z[0]
        Ce = z[3]

        t_out = [t0 + timedelta(seconds=_t) for _t in t]

        return (t_out, Cc, Ce)
