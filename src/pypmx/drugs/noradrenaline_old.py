from datetime import datetime, timedelta
from typing import List

import numpy as np
from pkinsight.models.base import Model
from scipy.integrate import solve_ivp


class Noradrenaline(Model):
    _params = {
        "S0": 0.811,
        "Tlag": 25.3,
        "ka": 0.0243,
        "V1": 0.486,
        "k13": 0.0595,
        "k31": 0.0386,
        "k10": 0.0501,
        "beta": 1.21,
        "E0": 57.1,
        "Emax": 56.4,
        "EC50": 15.7,
        "n": 1.46,
        "omega": 1.01,
        "zeta": 19.4,
        "imass": 2.12,
        "p": 0.628,
    }

    def __init__(self, t0: datetime | None = None, y0: List[float] = [0, 0, 0]) -> None:
        super().__init__(t0, y0)

    def pk(self, t, y, Inf, p):
        Ad, Ac, Ap = y
        # TODO: add Tlag
        # ddt_Ad = -p["ka"] * (Ad ** p["beta"]) + I(t - p["Tlag"])
        ddt_Ad = -p["ka"] * (Ad ** p["beta"]) + Inf
        ddt_Ac = (
            p["ka"] * (Ad ** p["beta"]) - p["k13"] * Ac + p["k31"] * Ap - p["k10"] * Ac
        )
        ddt_Ap = p["k13"] * Ac - p["k31"] * Ap
        return ddt_Ad, ddt_Ac, ddt_Ap

    def pd_wk(self, t, y, E0, Cc, p):
        E, dE = y
        ddt_E = dE
        ddt_dE = (
            p["imass"] * (Cc(t) - p["S0"])
            - 2 * p["zeta"] * p["omega"] * dE
            - (p["omega"] ** 2) * (E - E0)
        )
        return ddt_E, ddt_dE

    def pd_emax(self, Cc):
        E0 = self._params["E0"]
        n = self._params["n"]
        Emax = self._params["Emax"]
        EC50 = self._params["EC50"]
        num = (Cc**n) * Emax
        den = Cc**n + EC50**n
        return E0 + num / den

    def calc_cc(self, Ac=None):
        if Ac is None:
            Ac = self._y[1]
        return self._params["S0"] + Ac / self._params["V1"]

    def calc_effect(self, Cc=None):
        if Cc is None:
            Cc = self.calc_cc()
        return self.pd_emax(Cc)

    def simulate_pkpd(self, dt, bolus=None, inf_rate=None, npoints=100):
        if bolus is None and inf_rate is None:
            # Reset
            return ([], [], [])
        t0 = self._dt
        y0 = self._y.copy()
        t = (dt - t0).total_seconds()
        t_span = [0, t]
        if bolus:
            y0[0] += bolus
        rate = inf_rate if inf_rate is not None else self._rate
        sol_pk = solve_ivp(
            self.pk, t_span=t_span, y0=y0, args=(rate, self._params), dense_output=True
        )

        t = np.linspace(*t_span, npoints)
        z = sol_pk.sol(t)
        Ac = z[1]

        calc_cc_v = np.vectorize(self.calc_cc)
        Cc = calc_cc_v(Ac)

        calc_effect_v = np.vectorize(self.calc_effect)
        E = calc_effect_v(Cc)

        t_out = [t0 + timedelta(seconds=_t) for _t in t]

        return (t_out, Cc, E)
