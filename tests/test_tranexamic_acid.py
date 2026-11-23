import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx import eigen
from pypmx.drugs import GrassinDelyleTranexamicAcid as Model


def _unit_bolus(m):
    """Input vector of a unit bolus into the central compartment (concentration
    space: the bolus raises C1 by 1/V1)."""
    B = np.zeros(m.system_matrix.shape[0])
    B[0] = 1 / m.parameters.V1
    return B


def simulate(model, inf, t_end, dt=0.1):
    t = np.arange(0, t_end + dt, dt)
    sol = solve_ivp(
        model.ode, (0, t_end), [0, 0, 0], t_eval=t, args=(inf,), max_step=0.5
    )
    return t, sol.y[0] / model.parameters.V1


@pytest.mark.parametrize("weight", [50, 64.2, 70, 100])
def test_parameters_finite_positive(weight):
    p = Model({"weight": weight}).parameters
    for name in ("V1", "V2", "Q2", "CL", "k12", "k21", "k10"):
        value = getattr(p, name)
        assert np.isfinite(value) and value > 0, name
    assert p.ke0 == 0


def test_reference_values_at_70kg():
    p = Model({"weight": 70}).parameters
    assert p.CL == pytest.approx(10.1 / 60)
    assert p.Q2 == pytest.approx(15.6 / 60)
    assert p.V1 == pytest.approx(7.7)
    assert p.V2 == pytest.approx(10.8)


def test_allometric_scaling():
    ref = Model({"weight": 70}).parameters
    p = Model({"weight": 91}).parameters
    f = 91 / 70
    assert p.CL == pytest.approx(ref.CL * f**0.75)
    assert p.Q2 == pytest.approx(ref.Q2 * f**0.75)
    assert p.V1 == pytest.approx(ref.V1 * f)
    assert p.V2 == pytest.approx(ref.V2 * f)


def test_warnings_outside_studied_range():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model({"weight": 64, "age": 30})
    with pytest.warns(UserWarning, match="Weight"):
        Model({"weight": 120})
    with pytest.warns(UserWarning, match="Age"):
        Model({"age": 70})


def test_ode_bolus_matches_eigen_impulse_response():
    m = Model({"weight": 64.2})
    p = m.parameters
    modes = eigen.eigendecomposition(m.system_matrix, _unit_bolus(m))
    t = np.linspace(0, 600, 601)
    cp_eigen = eigen.calculate_impulse_response(modes, t)

    sol = solve_ivp(
        m.ode,
        (0, 600),
        [1.0, 0, 0],
        t_eval=t,
        args=(lambda t: 0.0,),
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(sol.y[0] / p.V1, cp_eigen, rtol=1e-6, atol=1e-9)


def test_1g_iv_over_10min_matches_paper():
    """1 g over ~10 min in a 64.2 kg volunteer (paper median weight).

    Paper: median observed peak 57.5 mg/L, but the first i.v. sample was taken at
    T0+5 min, i.e. mid-infusion; Fig. 1 shows ~100 mg/L at the start. Time above
    10 mg/L was 2.9 h (model-derived, Supplementary Table 2).
    """
    m = Model({"weight": 64.2})
    t, cp = simulate(m, lambda t: 100.0 if t < 10 else 0.0, t_end=24 * 60)

    c5 = cp[np.searchsorted(t, 5)]
    assert 40 < c5 < 80  # paper median observed peak 57.5 mg/L at 5 min

    assert t[np.argmax(cp)] == pytest.approx(10, abs=0.2)  # peak at end of infusion
    assert 80 < cp.max() < 130

    hours_above_10 = (cp > 10).sum() * (t[1] - t[0]) / 60
    assert hours_above_10 == pytest.approx(2.9, abs=0.3)

    assert cp[-1] < 1  # paper Fig. 1: ~1 mg/L at 24 h
