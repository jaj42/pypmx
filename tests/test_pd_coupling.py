"""PD-only models (Su haemodynamics) driven by a PK ``Prediction``."""

import math

import numpy as np
import pytest

from pypmx.drugs import (
    EleveldPropofol,
    KatariaPropofol,
    MintoRemifentanil,
    SuPropofolHemodynamics,
    SuPropofolRemifentanilHemodynamics,
    SuPropofolRemifentanilHemodynamicsVariant,
)
from pypmx.pk import Prediction
from pypmx.utils import continuize_order_one

T = np.linspace(0, 120, 241)
# 2-min loading, maintenance, stop at 60 min
INFU_TIME = [0.0, 2.0, 60.0]
PROP_RATE = [100.0, 8.0, 0.0]  # mg/min
REMI_RATE = [20.0, 15.0, 0.0]  # ug/min


@pytest.fixture(scope="module")
def prop() -> Prediction:
    return EleveldPropofol({}).simulate(INFU_TIME, PROP_RATE, T)


@pytest.fixture(scope="module")
def remi() -> Prediction:
    return MintoRemifentanil({}).simulate(INFU_TIME, REMI_RATE, T)


def test_cp_function_interpolates(prop):
    f = prop.cp_function()
    np.testing.assert_array_equal(f(T), prop.cp)
    mid = 0.5 * (T[10] + T[11])
    assert f(mid) == pytest.approx(0.5 * (prop.cp[10] + prop.cp[11]))
    assert math.isnan(f(-1.0))
    assert f(T[-1] + 10) == prop.cp[-1]
    assert prop.cp_function(left=0.0, right=0.0)(-1.0) == 0.0


def test_ce_function(prop):
    assert prop.ce is not None
    np.testing.assert_array_equal(prop.ce_function()(T), prop.ce)


def test_ce_function_without_effect_site():
    pred = KatariaPropofol({"age": 8, "weight": 25}).simulate([0], [10.0], T)
    assert pred.ce is None
    with pytest.raises(ValueError, match="effect site"):
        pred.ce_function()


def test_cp_function_is_accurate_on_fine_grid(prop):
    """Linear interpolation of a 0.5-min grid vs the exact solution.

    The error is ~14% during the 2-min loading phase (Cp is strongly curved
    there) and <1% from 5 min on: the grid must resolve fast changes.
    """
    t_fine = np.linspace(0, 120, 2401)
    exact = EleveldPropofol({}).simulate(INFU_TIME, PROP_RATE, t_fine).cp
    approx = prop.cp_function()(t_fine)
    after = t_fine >= 5
    np.testing.assert_allclose(approx[after], exact[after], rtol=1e-2)
    np.testing.assert_allclose(approx, exact, rtol=0.2)


def test_su_propofol_from_prediction(prop):
    m = SuPropofolHemodynamics(prop.cp_function(), {"age": 50})
    t, hr, sv, map_ = m.calculate_at(T)
    np.testing.assert_array_equal(t, T)
    # Same result as a hand-built interpolant of the same samples.
    ref = SuPropofolHemodynamics(continuize_order_one(prop.t, prop.cp), {"age": 50})
    np.testing.assert_array_equal(map_, ref.calculate_at(T)[3])
    assert np.all(np.isfinite(hr)) and np.all(np.isfinite(sv))
    # Propofol lowers MAP during the infusion.
    assert map_[T == 30][0] < map_[0]


@pytest.mark.parametrize(
    "cls",
    [SuPropofolRemifentanilHemodynamics, SuPropofolRemifentanilHemodynamicsVariant],
)
def test_su_propofol_remifentanil_from_predictions(cls, prop, remi):
    m = cls(prop.cp_function(), remi.cp_function(), {"age": 50})
    t, hr, sv, map_ = m(T)
    np.testing.assert_array_equal(t, T)
    for x in (hr, sv, map_):
        assert np.all(np.isfinite(x)) and np.all(x > 0)
    assert map_[T == 30][0] < map_[0]
    # With no drug the model stays near its baseline; with drugs MAP is lower.
    zero = cls(lambda t: 0.0, lambda t: 0.0, {"age": 50})
    assert map_[T == 30][0] < zero(T)[3][T == 30][0]
