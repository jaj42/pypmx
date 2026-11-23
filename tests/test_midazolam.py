import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx import eigen
from pypmx.drugs import ZomorodiMidazolam as Model
from pypmx.leanbodymass import dubois

PK_NAMES = ("V1", "V2", "V3", "CL", "Q2", "Q3", "k12", "k21", "k13", "k31", "k10")


@pytest.mark.parametrize(
    "covars",
    [
        {},
        {"weight": 55, "age": 45, "height": 150, "sex": "f"},
        {"weight": 115, "age": 80, "height": 195, "sex": "m"},
    ],
)
@pytest.mark.parametrize("bsa_adjusted", [False, True])
def test_parameters_finite_positive(covars, bsa_adjusted):
    p = Model(covars, bsa_adjusted=bsa_adjusted).pk
    for name in PK_NAMES:
        value = getattr(p, name)
        assert np.isfinite(value) and value > 0, name
    assert p.ke0 == 0 and p.n_cmt == 3


def test_simple_model_table3():
    p = Model().pk
    assert (p.V1, p.V2, p.V3) == (32.2, 53, 245)
    assert (p.CL, p.Q2, p.Q3) == (0.43, 0.56, 0.39)
    # Table 3 derived rate constants (min^-1)
    assert p.k10 == pytest.approx(0.013, abs=5e-4)
    assert p.k12 == pytest.approx(0.017, abs=5e-4)
    assert p.k13 == pytest.approx(0.012, abs=5e-4)
    assert p.k21 == pytest.approx(0.011, abs=5e-4)
    assert p.k31 == pytest.approx(0.0016, abs=5e-5)
    # Simple model ignores covariates
    assert Model({"weight": 60, "height": 160}).pk.V2 == p.V2


def test_bsa_model_table4():
    c = {"weight": 80, "height": 180}
    bsa = dubois(180, 80)
    m = Model(c, bsa_adjusted=True)
    p = m.pk
    assert m.parameters.BSA == pytest.approx(bsa)
    assert p.V1 == 33 and p.V3 == 365
    assert p.V2 == pytest.approx(3.32 + 32.1 * bsa)
    assert p.CL == pytest.approx(0.0889 + 0.151 * bsa)
    assert (p.Q2, p.Q3) == (0.622, 0.264)
    # At a typical BSA (~2 m^2) the clearance is close to the simple model's
    assert p.CL == pytest.approx(0.43, rel=0.12)


@pytest.mark.parametrize(
    "height, weight, bsa", [(170, 86, 1.97), (176, 84, 1.99), (178, 87, 2.05)]
)
def test_dubois_matches_table2(height, weight, bsa):
    assert dubois(height, weight) == pytest.approx(bsa, abs=0.02)


def test_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model()
        Model(bsa_adjusted=True)
    with pytest.warns(UserWarning, match="age"):
        Model({"age": 30})
    with pytest.warns(UserWarning, match="weight"):
        Model({"weight": 130})


@pytest.mark.parametrize("bsa_adjusted", [False, True])
def test_simulate_bolus_matches_solve_ivp(bsa_adjusted):
    m = Model({"weight": 95, "height": 182}, bsa_adjusted=bsa_adjusted)
    t = np.linspace(0, 2880, 2881)
    cp = m.simulate([0.0], [0.0], t, y0=[1.0, 0, 0]).cp
    sol = solve_ivp(
        m.ode,
        (0, 2880),
        [1.0, 0, 0],
        t_eval=t,
        args=(lambda t: 0.0,),
        method="LSODA",
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(sol.y[0] / m.pk.V1, cp, rtol=1e-6, atol=1e-12)


def test_terminal_half_life_15h():
    """Abstract: calculated elimination half-life 15 h (Table 3: 905 min)."""
    m = Model()
    rates = eigen.eigendecomposition(m.system_matrix, m.input_matrix).lambdas
    assert len(rates) == 3  # no effect site, no zero mode
    t_half = np.log(2) / rates[0]
    assert t_half / 60 == pytest.approx(15, rel=0.03)
    assert t_half == pytest.approx(905.088, rel=0.01)
    # Table 3 exponents alpha 0.048, beta 0.006, gamma 0.0007 min^-1
    np.testing.assert_allclose(rates, [0.0007, 0.006, 0.048], rtol=0.1)


def test_10h_infusion_recovery():
    """3 mg/h for 10 h (30 mg), then stop (Table 2: 8-12 h, 29-48 mg).

    The end-of-infusion level stays below Css = R / CL, and the concentration
    falls monotonically but slowly for the next day.
    """
    m = Model()
    rate = 3000 / 60  # ug/min
    t_inf = 600

    def inf(t):
        return rate if t < t_inf else 0.0

    t = np.arange(0, 2881, 1.0)
    sol = solve_ivp(m.ode, (0, 2880), [0, 0, 0], t_eval=t, args=(inf,), max_step=1)
    cp = sol.y[0] / m.pk.V1
    css = rate / m.pk.CL  # ng/ml
    assert 0.5 * css < cp[t_inf] < css
    assert 50 < cp[t_inf] < 120  # sedative range in the paper (Fig. 1)
    post = cp[t >= t_inf]
    assert np.all(np.diff(post) <= 1e-9)
    # Slow decline: still > 10 % of end-of-infusion level 12 h after stopping
    assert cp[t_inf + 720] > 0.1 * cp[t_inf]
