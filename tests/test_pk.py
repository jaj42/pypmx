"""Sanity checks for the LinearPKModel base class (plan.md step 2)."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx.pk import Covariates, LinearPKModel, PKParameters, Units
from pypmx.standard_models import pk_2comp, pk_3comp
from pypmx.utils import continuize_order_zero

THREE = dict(V1=4.0, V2=20.0, V3=150.0, CL=1.5, Q2=2.0, Q3=0.8)
TWO = dict(V1=4.0, V2=20.0, CL=1.5, Q2=2.0)
ONE = dict(V1=10.0, CL=0.5)


class Toy(LinearPKModel):
    units: ClassVar[Units] = Units(time="min", cp="mg/L")
    covariate_ranges: ClassVar[dict[str, tuple[float, float]]] = {
        "weight": (40, 120),
        "age": (18, 80),
    }

    def __init__(
        self, pk: Mapping[str, float], covars: Mapping[str, Any] | None = None
    ):
        self._set_covariates(covars, Covariates)
        self.pk = PKParameters(**pk)


class Forced(Toy):
    """Toy model with an endogenous production into A1."""

    production = 0.3

    @property
    def forcing(self):
        f = np.zeros(self.n_states)
        f[0] = self.production
        return f


def test_micro_constants():
    p = PKParameters(**THREE, ke0=0.4)
    assert p.k10 == pytest.approx(1.5 / 4)
    assert p.k12 == pytest.approx(2 / 4)
    assert p.k21 == pytest.approx(2 / 20)
    assert p.k13 == pytest.approx(0.8 / 4)
    assert p.k31 == pytest.approx(0.8 / 150)
    assert p.n_cmt == 3 and p.has_effect_site


def test_n_cmt():
    assert PKParameters(**ONE).n_cmt == 1
    assert PKParameters(**TWO).n_cmt == 2
    assert PKParameters(**THREE).n_cmt == 3
    p = PKParameters(**ONE)
    assert p.k21 == 0 and p.k31 == 0  # no nan from V2 = nan
    with pytest.raises(ValueError):
        _ = PKParameters(V1=1, CL=1, V3=10, Q3=1).n_cmt


@pytest.mark.parametrize(
    "pk, n_cmt", [(ONE, 1), (TWO, 2), (THREE, 3)], ids=["1cmt", "2cmt", "3cmt"]
)
@pytest.mark.parametrize("ke0", [0.0, 0.4], ids=["no_ce", "ce"])
def test_matrix_shape_and_eigenvalues(pk, n_cmt, ke0):
    m = Toy({**pk, "ke0": ke0})
    S = m.system_matrix
    n = n_cmt + (ke0 > 0)
    assert S.shape == (n, n)
    assert m.n_states == n
    np.testing.assert_array_equal(m.input_matrix, np.eye(n)[0])
    eig = np.sort(np.linalg.eigvals(S).real)
    assert np.all(eig < 0)  # no zero eigenvalue, even without effect site
    if ke0:
        # block lower-triangular: PK eigenvalues ∪ {-ke0}
        pk_eig = np.linalg.eigvals(S[:-1, :-1]).real
        np.testing.assert_allclose(eig, np.sort(np.append(pk_eig, -ke0)))
    # mass balance: elimination only from A1 via k10
    np.testing.assert_allclose(S[:n_cmt, :n_cmt].sum(axis=0)[0], -m.pk.k10)
    np.testing.assert_allclose(S[:n_cmt, :n_cmt].sum(axis=0)[1:], 0, atol=1e-15)


@pytest.mark.parametrize("pk", [ONE, TWO, THREE], ids=["1cmt", "2cmt", "3cmt"])
@pytest.mark.parametrize("ke0", [0.0, 0.4], ids=["no_ce", "ce"])
def test_ode_matches_system_matrix(pk, ke0):
    m = Toy({**pk, "ke0": ke0})
    rng = np.random.default_rng(0)
    y = rng.uniform(0, 10, m.n_states)
    dy = m.ode(1.0, y, lambda t: 3.0)
    np.testing.assert_allclose(dy, m.system_matrix @ y + 3.0 * m.input_matrix)


@pytest.mark.parametrize(
    "pk, ref_ode", [(TWO, pk_2comp), (THREE, pk_3comp)], ids=["2cmt", "3cmt"]
)
def test_ode_matches_standard_models(pk, ref_ode):
    """Same amounts + Ce system as the legacy pk_2comp/pk_3comp."""
    m = Toy({**pk, "ke0": 0.4})
    p = m.pk
    params = dict(
        V1=p.V1, k10=p.k10, k12=p.k12, k21=p.k21, k13=p.k13, k31=p.k31, ke0=p.ke0
    )
    y = np.random.default_rng(1).uniform(0, 10, m.n_states)
    inf = lambda t: 2.5
    np.testing.assert_allclose(m.ode(0.0, y, inf), ref_ode(0.0, y, inf, params))


@pytest.mark.parametrize("pk", [ONE, TWO, THREE], ids=["1cmt", "2cmt", "3cmt"])
@pytest.mark.parametrize("ke0", [0.0, 0.4], ids=["no_ce", "ce"])
def test_simulate_matches_solve_ivp(pk, ke0):
    m = Toy({**pk, "ke0": ke0})
    infu_time = np.array([0.0, 2.0, 60.0])
    infu_rate = np.array([50.0, 5.0, 0.0])
    t = np.linspace(0, 240, 481)
    pred = m.simulate(infu_time, infu_rate, t)

    inf = continuize_order_zero(infu_time, infu_rate)
    sol = solve_ivp(
        m.ode,
        (0, 240),
        np.zeros(m.n_states),
        t_eval=t,
        args=(inf,),
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5,
    )
    np.testing.assert_allclose(pred.states, sol.y.T, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(pred.cp, sol.y[0] / m.pk.V1, rtol=1e-6, atol=1e-8)
    if ke0:
        assert pred.ce is not None
        np.testing.assert_allclose(pred.ce, sol.y[-1], rtol=1e-6, atol=1e-8)
    else:
        assert pred.ce is None


def test_one_comp_analytic():
    """Constant infusion into a 1-comp model: Cp = R/CL·(1 - e^{-k t})."""
    m = Toy(ONE)
    t = np.linspace(0, 200, 101)
    pred = m.simulate([0.0], [2.0], t)
    k = m.pk.k10
    np.testing.assert_allclose(pred.cp, 2.0 / m.pk.CL * -np.expm1(-k * t), rtol=1e-12)


def test_effect_site_reaches_plasma_steady_state():
    m = Toy({**THREE, "ke0": 0.4})
    pred = m.simulate([0.0], [3.0], [1e5])
    assert pred.ce is not None
    np.testing.assert_allclose(pred.cp, 3.0 / m.pk.CL, rtol=1e-9)
    np.testing.assert_allclose(pred.ce, pred.cp, rtol=1e-9)


def test_steady_state_default_zero_and_forced():
    assert not np.any(Toy(THREE).steady_state())
    m = Forced({**THREE, "ke0": 0.4})
    y0 = m.steady_state()
    # endogenous Css = production / CL, all compartments at equilibrium with Cp
    cp = m.production / m.pk.CL
    np.testing.assert_allclose(y0, [cp * m.pk.V1, cp * m.pk.V2, cp * m.pk.V3, cp])
    # starting from it with no infusion, nothing moves
    pred = m.simulate([0.0], [0.0], np.linspace(0, 500, 11))
    np.testing.assert_allclose(pred.states, np.tile(y0, (11, 1)), rtol=1e-10)


def test_lag_shifts_response():
    class Lagged(Toy):
        @property
        def lag(self):
            return 5.0

    t = np.linspace(0, 100, 201)
    base = Toy(THREE).simulate([0.0, 1.0], [10.0, 0.0], t)
    lagged = Lagged(THREE).simulate([0.0, 1.0], [10.0, 0.0], t + 5.0)
    np.testing.assert_allclose(lagged.cp, base.cp, rtol=1e-10, atol=1e-14)


def test_with_parameters_is_a_copy():
    m = Toy(THREE)
    m2 = m.with_parameters(CL=3.0)
    assert m.pk.CL == 1.5 and m2.pk.CL == 3.0
    assert m2.pk.V1 == m.pk.V1 and m2.covars == m.covars
    assert m2.system_matrix[0, 0] < m.system_matrix[0, 0]
    css = m2.simulate([0.0], [3.0], [1e5]).cp
    np.testing.assert_allclose(css, 1.0, rtol=1e-9)


def test_covariates_and_range_warnings():
    m = Toy(ONE, {"weight": 80, "unknown": 1})
    assert m.covars == Covariates(weight=80)
    assert Toy(ONE).covars == Covariates()
    with pytest.warns(UserWarning, match="weight 150 outside"):
        Toy(ONE, {"weight": 150})
    with pytest.warns(UserWarning, match="age 10 outside"):
        Toy(ONE, {"age": 10})


def test_repr():
    assert repr(Toy(ONE)).startswith("Toy({'age': 35")
