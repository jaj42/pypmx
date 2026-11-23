"""Shared sanity checks for the drug models migrated to LinearPKModel (step 3)."""

import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx.drugs import (
    AllegaertParacetamol,
    DyckDexmedetomidine,
    EleveldPropofol,
    GrassinDelyleTranexamicAcid,
    HannivoortDexmedetomidine,
    KuipersLidocaine,
    ZomorodiMidazolam,
)
from pypmx.drugs.propofol import Schnider
from pypmx.drugs.remifentanil import Eleveld as EleveldRemifentanil
from pypmx.drugs.remifentanil import Minto
from pypmx.pk import LinearPKModel
from pypmx.utils import continuize_order_zero

# (factory, n_cmt, has effect site, (loading, maintenance) rate)
MODELS = {
    "EleveldPropofol": (lambda: EleveldPropofol(), 3, True, (50.0, 8.0)),
    "Schnider": (lambda: Schnider(), 3, True, (50.0, 8.0)),
    "EleveldRemifentanil": (lambda: EleveldRemifentanil(), 3, True, (25.0, 15.0)),
    "Minto": (lambda: Minto(), 3, True, (25.0, 15.0)),
    "Hannivoort": (lambda: HannivoortDexmedetomidine(), 3, True, (10.0, 1.0)),
    "HannivoortBIS": (
        lambda: HannivoortDexmedetomidine(effect="bis"),
        3,
        True,
        (10.0, 1.0),
    ),
    "Dyck": (lambda: DyckDexmedetomidine(), 3, False, (10.0, 1.0)),
    "GrassinDelyle": (lambda: GrassinDelyleTranexamicAcid(), 2, False, (500.0, 1.0)),
    "Allegaert": (lambda: AllegaertParacetamol(), 3, False, (500.0, 5.0)),
    "Zomorodi": (lambda: ZomorodiMidazolam(), 3, False, (1000.0, 50.0)),
    "Kuipers": (lambda: KuipersLidocaine(), 3, False, (50.0, 2.0)),
}


@pytest.fixture(params=list(MODELS), ids=list(MODELS))
def case(request):
    factory, n_cmt, effect, rates = MODELS[request.param]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # defaults are inside the studied ranges
        m = factory()
    return m, n_cmt, effect, rates


def test_structure(case):
    m, n_cmt, effect, _ = case
    assert isinstance(m, LinearPKModel)
    assert m.pk.n_cmt == n_cmt
    assert m.pk.has_effect_site == effect
    assert m.n_states == n_cmt + effect
    names = ["V1", "V2", "CL", "Q2"] + (["V3", "Q3"] if n_cmt == 3 else [])
    for name in names:
        value = getattr(m.pk, name)
        assert np.isfinite(value) and value > 0, name
    assert np.all(np.linalg.eigvals(m.system_matrix).real < 0)
    assert m.units.time == "min"


def test_simulate_matches_solve_ivp(case):
    m, _, effect, (load, maint) = case
    infu_time = np.array([0.0, 2.0, 60.0])
    infu_rate = np.array([load, maint, 0.0])
    t = np.linspace(0, 240, 241)
    pred = m.simulate(infu_time, infu_rate, t)
    sol = solve_ivp(
        m.ode,
        (0, 240),
        np.zeros(m.n_states),
        t_eval=t,
        args=(continuize_order_zero(infu_time, infu_rate),),
        method="LSODA",
        rtol=1e-10,
        atol=1e-12,
        max_step=0.5,
    )
    np.testing.assert_allclose(pred.cp, sol.y[0] / m.pk.V1, rtol=1e-5, atol=1e-9)
    if effect:
        assert pred.ce is not None
        np.testing.assert_allclose(pred.ce, sol.y[-1], rtol=1e-5, atol=1e-9)
    else:
        assert pred.ce is None


def test_steady_state_concentration(case):
    """Long constant infusion: Cp (and Ce) → rate / CL."""
    m, _, effect, (_, maint) = case
    t_end = 200 * np.log(2) / min(-np.linalg.eigvals(m.system_matrix).real)
    pred = m.simulate([0.0], [maint], [t_end])
    np.testing.assert_allclose(pred.cp[-1], maint / m.pk.CL, rtol=1e-8)
    if effect:
        assert pred.ce is not None
        np.testing.assert_allclose(pred.ce[-1], maint / m.pk.CL, rtol=1e-8)


def test_repr_shows_covariates_and_options():
    assert repr(KuipersLidocaine()).endswith(", model='D')")
    assert "'cardiac_output': 5.8" in repr(KuipersLidocaine())
    assert repr(HannivoortDexmedetomidine(effect="bis")).endswith("effect='bis')")
    assert repr(AllegaertParacetamol()).endswith("setting='none')")
    assert repr(ZomorodiMidazolam()).endswith("bsa_adjusted=False)")
    assert repr(Minto()).startswith("Minto({'age': 35")


def test_derived_parameters():
    assert EleveldPropofol().parameters.ce50 == pytest.approx(3.08)  # age 35
    assert HannivoortDexmedetomidine(effect="bis").parameters.ce50 == 2.63
    assert HannivoortDexmedetomidine().pk.ke0 == 0.0428


def test_schnider_volumes_consistent_with_rate_constants():
    p = Schnider({"age": 53, "weight": 77, "height": 177}).pk
    assert p.V1 == 4.27
    assert p.k21 == pytest.approx(1.29 / 18.9)
    assert p.k31 == pytest.approx(0.0035)
    # published V2 18.9 / V3 238 L, slightly inconsistent with k12/k13
    assert p.V2 == pytest.approx(18.9, rel=1e-3)
    assert p.V3 == pytest.approx(238, rel=0.01)
