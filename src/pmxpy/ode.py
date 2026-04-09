import bisect

import numpy as np
from scipy.integrate import solve_ivp


def solve_ode(model, t_span, rate_func, y0: list[float] | None = None):
    if y0 is None:
        y0 = model.get_state()
    max_step = 0.1
    # We should probably use odeint. Even if it is deprecated, it is better.
    res = solve_ivp(
        model.ode, t_span, y0, dense_output=True, args=(rate_func,), max_step=max_step
    )
    return res.sol


def build_piecewise_rate_old(rates: dict[float, float], r0: float | None = None):
    sorted_times = sorted(rates.keys())
    sorted_rates = [rates[t] for t in sorted_times]

    def piecewise_rate(t):
        idx = bisect.bisect_right(sorted_times, t) - 1
        if idx < 0:
            return r0
        return sorted_rates[idx]

    return np.vectorize(piecewise_rate)


def build_piecewise_rate(infu_time, infu_rate, r0: float = 0):
    def run(t):
        if t < infu_time[0]:
            return r0
        idx = np.searchsorted(infu_time, t, side="right") - 1
        idx = np.clip(idx, 0, len(infu_rate) - 1)
        return infu_rate[idx]

    return run
