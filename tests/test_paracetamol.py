import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx import eigen
from pypmx.drugs import AllegaertParacetamol as Model
from pypmx.leanbodymass import janmahasation

PK_NAMES = ("V1", "V2", "V3", "Q2", "Q3", "CL", "k12", "k21", "k13", "k31", "k10")


def simulate_q6h(model, t_end, dose=1000, tau=360, t_inf=15, dt=0.1):
    def inf(t):
        return dose / t_inf if (t % tau) < t_inf else 0.0

    t = np.arange(0, t_end + dt, dt)
    sol = solve_ivp(
        model.ode, (0, t_end), [0, 0, 0, 0], t_eval=t, args=(inf,), max_step=1
    )
    return t, sol.y[0] / model.parameters.V1


@pytest.mark.parametrize(
    "covars",
    [
        {},
        {"weight": 49.2, "age": 19, "height": 150, "sex": "f"},
        {"weight": 120, "age": 88.5, "height": 190, "sex": "m"},
    ],
)
@pytest.mark.parametrize(
    "setting", ["none", "pregnancy", "abdominal", "orthopaedic", "icu"]
)
def test_parameters_finite_positive(covars, setting):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Model(covars, setting=setting).parameters
    for name in PK_NAMES:
        value = getattr(p, name)
        assert np.isfinite(value) and value > 0, name
    assert p.ke0 == 0


def test_size_scaling_nfm():
    c = {"weight": 90, "height": 175, "sex": "m", "age": 40}
    p = Model(c).parameters
    ffm = janmahasation(175, 90, "m")
    nfm_vol = ffm + 0.778 * (90 - ffm)
    # Ffat_CL = 1 -> TBW
    assert p.CL == pytest.approx(16.7 / 60 * (90 / 70) ** 0.75)
    assert p.Q2 == pytest.approx(67.3 / 60 * (90 / 70) ** 0.75)
    assert p.Q3 == pytest.approx(2.04 / 60 * (90 / 70) ** 0.75)
    assert p.V1 == pytest.approx(24.6 * nfm_vol / 70)
    assert p.V2 == pytest.approx(23.1 * nfm_vol / 70)
    assert p.V3 == pytest.approx(30.6 * nfm_vol / 70)


def test_janmahasation_matches_paper_whs():
    # FFM = WHSmax * H^2 * TBW / (WHS50 * H^2 + TBW), H in m
    for sex, whs_max, whs_50 in (("m", 42.92, 30.93), ("f", 37.99, 35.98)):
        h, tbw = 1.72, 80
        ffm = whs_max * h**2 * tbw / (whs_50 * h**2 + tbw)
        assert janmahasation(172, tbw, sex) == pytest.approx(ffm, rel=1e-3)


@pytest.mark.parametrize(
    "setting, f_cl, f_v2",
    [
        ("pregnancy", 1.14, 1),
        ("abdominal", 0.715, 1),
        ("orthopaedic", 1, 0.649),
        ("icu", 1, 1.51),
    ],
)
def test_setting_factors(setting, f_cl, f_v2):
    c = {"sex": "f", "age": 30}
    ref = Model(c).parameters
    p = Model(c, setting=setting).parameters
    assert p.CL == pytest.approx(ref.CL * f_cl)
    assert p.V2 == pytest.approx(ref.V2 * f_v2)
    for name in ("V1", "V3", "Q2", "Q3"):
        assert getattr(p, name) == pytest.approx(getattr(ref, name)), name


def test_age_factor_on_v2():
    young = Model({"age": 60}).parameters
    old = Model({"age": 61}).parameters
    assert old.V2 == pytest.approx(young.V2 * 0.838)
    assert old.CL == pytest.approx(young.CL)


def test_unknown_setting_raises():
    with pytest.raises(ValueError):
        Model(setting="postpartum")


def test_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model()
        Model({"sex": "f"}, setting="pregnancy")
    with pytest.warns(UserWarning, match="Weight"):
        Model({"weight": 140})
    with pytest.warns(UserWarning, match="Age"):
        Model({"age": 18})
    with pytest.warns(UserWarning, match="pregnancy"):
        Model({"sex": "m"}, setting="pregnancy")


def test_ode_bolus_matches_eigen_impulse_response():
    m = Model({"age": 68, "weight": 80, "height": 180}, setting="icu")
    p = m.parameters
    lambdas, coefs = eigen.eigendecomposition(m.system_matrix, p.V1, p.ke0)
    # ke0 = 0 gives a zero eigenvalue (last); its coefficients are nan, drop it.
    t = np.linspace(0, 1440, 1441)
    cp_eigen = eigen.calculate_impulse_response(lambdas[:-1], coefs[0, :-1], t)
    sol = solve_ivp(
        m.ode,
        (0, 1440),
        [1.0, 0, 0, 0],
        t_eval=t,
        args=(lambda t: 0.0,),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(sol.y[0] / p.V1, cp_eigen, rtol=1e-6, atol=1e-9)


def test_1g_q6h_healthy_volunteer():
    """25 y, 70 kg, 1 g q6h infused over 15 min for 36 h (paper simulation).

    CL-based average steady-state concentration is 1000 / (16.7 * 6) = 10 mg/L.
    The paper reports a mean of 11.6 (SD 1.9) mg/L across pathologies and a
    target of 10 mg/L.
    """
    m = Model({"age": 25, "weight": 70})
    t, cp = simulate_q6h(m, 36 * 60)
    last = cp[t >= 30 * 60]
    assert last.mean() == pytest.approx(1000 / (16.7 * 6), rel=0.05)
    assert 8 < cp.mean() < 11.6
    assert 20 < cp.max() < 50  # peaks after a 1 g infusion
    assert last.min() > 1  # troughs stay measurable
