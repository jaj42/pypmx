from scipy.integrate import solve_ivp
from typing import Dict, List, Tuple
import numpy as np


def pk_3comp(t, y, Inf, p):
    A1, A2, A3, Ce = y
    ddt_A1 = Inf + A2 * p.k21 + A3 * p.k31 - A1 * (p.k12 + p.k13 + p.k10)
    ddt_A2 = A1 * p.k12 - A2 * p.k21
    ddt_A3 = A1 * p.k13 - A3 * p.k31
    ddt_Ce = p.ke0 * (A1 / p.vc - Ce)
    return ddt_A1, ddt_A2, ddt_A3, ddt_Ce


def simulate_pkpd(
    y0,
    dt: int,  # seconds
    inf_rate: int,  # ug/s
    params: Dict[str, float],
    npoints: int = 100,
    odefunc=pk_3comp,
):
    t_span = [0, int(dt)]
    rate = inf_rate
    sol_pk = solve_ivp(
        odefunc, t_span=t_span, y0=y0, args=(rate, params), dense_output=True
    )
    t = np.linspace(*t_span, npoints)
    z = sol_pk.sol(t)
    return (t, z)


def run(params, schema: List[Tuple[int, float]]):
    t0 = 0
    y0 = [0, 0, 0, 0]
    ts = []
    ys = []
    for dt, rate in schema:
        # Convert to seconds
        dt *= 60
        rate /= 60

        t, y = simulate_pkpd(y0, dt, rate, params)
        t += t0
        ts.append(t)
        ys.append(y.T)

        y0 = y[:, -1]
        t0 = t[-1]
    return np.concat(ts), np.concat(ys)
