from scipy.integrate import solve_ivp


def solve_ode(model, t_span, rate_func, y0: list[float] | None = None):
    if y0 is None:
        y0 = model.get_state()
    max_step = 0.1
    res = solve_ivp(
        model.ode, t_span, y0, dense_output=True, args=(rate_func,), max_step=max_step
    )
    return res.sol
