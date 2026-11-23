"""Noradrenaline on LinearPKModel (plan step 4)."""

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx.drugs import Noradrenaline
from pypmx.pk import LinearPKModel
from pypmx.utils import continuize_order_zero

INFU_TIME = np.array([0.0, 2.0, 60.0])
INFU_RATE = np.array([40.0, 20.0, 0.0])  # nmol/min
T = np.linspace(0, 240, 481)


def test_structure_and_scaling():
    m = Noradrenaline()
    assert isinstance(m, LinearPKModel)
    assert m.pk.n_cmt == 2 and m.pk.has_effect_site and m.n_states == 3
    assert m.pk == m.pk._replace(V1=1.08, V2=1.39, CL=1.49, Q2=0.319, ke0=1.29)
    assert m.lag == pytest.approx(0.546 - 0.188)

    heavy = Noradrenaline({"weight": 140})
    assert heavy.pk.V1 == pytest.approx(2 * 1.08)
    assert heavy.pk.CL == pytest.approx(2**0.75 * 1.49)
    assert heavy.pk.ke0 == pytest.approx(2**0.25 * 1.29)
    assert heavy.parameters.EPROD == pytest.approx(2**0.75 * 1.36)
    # Endogenous concentration EPROD/CL is weight-independent
    assert heavy.steady_state()[0] / heavy.pk.V1 == pytest.approx(1.36 / 1.49)


def test_baseline_stays_at_endogenous_steady_state():
    m = Noradrenaline()
    y0 = m.steady_state()
    cp_endo = 1.36 / 1.49
    np.testing.assert_allclose(y0, [cp_endo * 1.08, cp_endo * 1.39, cp_endo])
    pred = m.simulate([0.0], [0.0], T)
    assert pred.ce is not None and pred.effect is not None
    np.testing.assert_allclose(pred.cp, cp_endo, rtol=1e-12)
    np.testing.assert_allclose(pred.ce, cp_endo, rtol=1e-12)
    p = m.parameters
    map0 = p.E0 + p.EMAX * cp_endo**p.GAMMA / (p.EC50**p.GAMMA + cp_endo**p.GAMMA)
    np.testing.assert_allclose(pred.effect, map0, rtol=1e-12)
    assert 55.8 < map0 < 60  # endogenous NA barely raises MAP above E0


def test_simulate_matches_solve_ivp_with_lag():
    m = Noradrenaline({"weight": 30})
    pred = m.simulate(INFU_TIME, INFU_RATE, T)
    sol = solve_ivp(
        m.ode,
        (0, T[-1]),
        m.steady_state(),
        t_eval=T,
        args=(continuize_order_zero(INFU_TIME, INFU_RATE),),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
    )
    assert sol.success
    np.testing.assert_allclose(pred.states, sol.y.T, rtol=1e-6, atol=1e-9)
    assert pred.effect is not None
    np.testing.assert_allclose(pred.effect, m.pd(sol.y[2]), rtol=1e-6)


def test_lag_shifts_response():
    m = Noradrenaline()
    no_lag = m.with_parameters(PKLAG=0.0, PDLAG=0.0)
    assert no_lag.lag == 0 and m.lag > 0
    assert no_lag.pk == m.pk
    lagged = m.simulate(INFU_TIME, INFU_RATE, T + m.lag)
    ref = no_lag.simulate(INFU_TIME, INFU_RATE, T)
    np.testing.assert_allclose(lagged.cp, ref.cp, rtol=1e-10)
    assert lagged.effect is not None and ref.effect is not None
    np.testing.assert_allclose(lagged.effect, ref.effect, rtol=1e-10)
    # Nothing happens before the lag has elapsed
    early = m.simulate(INFU_TIME, INFU_RATE, [0.0, 0.9 * m.lag])
    np.testing.assert_allclose(early.cp, 1.36 / 1.49, rtol=1e-12)


def test_infusion_response_is_reasonable():
    m = Noradrenaline()
    pred = m.simulate(INFU_TIME, INFU_RATE, T)
    assert pred.effect is not None
    cp_endo = 1.36 / 1.49
    # Near steady state at 60 min: Cp ≈ (EPROD + rate)/CL
    i60 = np.searchsorted(T, 60.0)
    assert pred.cp[i60] == pytest.approx(cp_endo + 20.0 / 1.49, rel=0.02)
    # MAP rises with the infusion, peaks below E0 + Emax, and washes out
    assert pred.effect.max() > pred.effect[0] + 10
    assert pred.effect.max() < 55.8 + 200
    assert pred.effect[-1] == pytest.approx(pred.effect[0], rel=1e-3)


def test_with_parameters():
    m = Noradrenaline()
    m2 = m.with_parameters(CL=2.0, EPROD=2.0, E0=60.0)
    assert m2.pk.CL == 2.0 and m2.parameters.EPROD == 2.0
    assert m2.steady_state()[2] == pytest.approx(1.0)
    assert m2.pd(0.0) == pytest.approx(60.0)
    assert m.pk.CL == pytest.approx(1.49) and m.parameters.E0 == 55.8
    with pytest.raises(ValueError):
        m.with_parameters(nonsense=1.0)


def test_repr():
    assert repr(Noradrenaline({"weight": 80})) == "Noradrenaline({'weight': 80})"
