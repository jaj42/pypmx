"""Modern TCI (pypmx.tci) against the STANPUMP port (pypmx.stanpump_tci)."""

from pathlib import Path

import numpy as np
import pytest

from pypmx import stanpump_tci
from pypmx.drugs import (
    EleveldPropofol,
    GeptsSufentanil,
    KatariaPropofol,
    MarshPropofol,
    MintoRemifentanil,
    Noradrenaline,
)
from pypmx.pk import LinearPKModel, PKParameters
from pypmx.tci import TCI

DELTA = 10
GOLDEN = Path(__file__).parent / "golden" / "tci" / "runs.npz"

# Blog example (202602_udf): Ce 0.2 until 500 s, 0.3 until 1000 s, then 0.1.
_t = np.arange(400) * DELTA
BLOG_TARGETS = np.where(_t <= 500, 0.2, np.where(_t <= 1000, 0.3, 0.1))

# Eleveld propofol as printed in the blog (202602_moderntci), 35 y, 75 kg, 175 cm.
ELEVELD_BLOG = PKParameters.from_micro(
    V1=6.283,
    k10=0.2577585580908108,
    k12=0.2913939174276039,
    k13=0.12286871331825001,
    k21=0.07179436034656034,
    k31=0.004586708876899454,
    ke0=0.14631,
)

CASES = {
    "GeptsSufentanil": (GeptsSufentanil(), BLOG_TARGETS),
    "EleveldBlog": (ELEVELD_BLOG, BLOG_TARGETS * 15),
    "EleveldPropofol": (EleveldPropofol(), BLOG_TARGETS * 15),
    "MarshPropofol": (MarshPropofol(), BLOG_TARGETS * 15),
    "MintoRemifentanil": (MintoRemifentanil(), BLOG_TARGETS * 20),
    # STANPUMP has no endogenous production nor lag: compare without them.
    "NoradrenalineNoForcingNoLag": (
        Noradrenaline().with_parameters(EPROD=0, PKLAG=0, PDLAG=0),
        BLOG_TARGETS * 20,
    ),
}


def stanpump_run(model, targets, target_effect: bool):
    """Closed loop with the STANPUMP port (as in test_stanpump_tci.py)."""
    setup = stanpump_tci.prepare(model, DELTA)
    n = len(setup.lambdas)
    Ap, Ae = np.zeros(n), np.zeros(n)
    rates, cp, ce = [], [], []
    for target in targets:
        rate = stanpump_tci.model(
            Ap,
            Ae,
            setup.lambdas,
            setup.udfs,
            setup.coefs,
            target,
            setup.peak_time,
            DELTA,
            target_effect=target_effect,
        )
        Ap = stanpump_tci.update(Ap, setup.lambdas, setup.coefs.p_coef, rate, DELTA)
        Ae = stanpump_tci.update(Ae, setup.lambdas, setup.coefs.e_coef, rate, DELTA)
        rates.append(rate)
        cp.append(Ap.sum())
        ce.append(Ae.sum())
    return setup.peak_time, np.array(rates), np.array(cp), np.array(ce)


# ---- same output as STANPUMP ----------------------------------------------


def test_blog_peak_times():
    """Continuous peak times quoted in the blog (STANPUMP: 428 s and 195 s)."""
    assert TCI(GeptsSufentanil()).peak_time == pytest.approx(427.99173, abs=1e-3)
    assert TCI(ELEVELD_BLOG).peak_time == pytest.approx(194.50040, abs=1e-3)


@pytest.mark.parametrize("key", CASES)
def test_peak_time_rounds_to_stanpump(key):
    model, _ = CASES[key]
    stanpump_peak = stanpump_tci.prepare(model, DELTA).peak_time
    peak_time = TCI(model, DELTA).peak_time
    assert peak_time is not None
    assert round(peak_time) == stanpump_peak


@pytest.mark.parametrize("key", CASES)
def test_plasma_targeting_identical_to_stanpump(key):
    """Same formula, both exact: identical up to rounding."""
    model, targets = CASES[key]
    _, rate, cp, ce = stanpump_run(model, targets, target_effect=False)
    res = TCI(model, DELTA, target="plasma").run(targets)
    assert res.ce is not None
    np.testing.assert_allclose(res.rate, rate, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(res.cp, cp, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(res.ce, ce, rtol=1e-10, atol=1e-14)


def _assert_effect_close(res, rate, ce, targets):
    """Continuous peak time vs STANPUMP's whole seconds: nearly the same."""
    assert res.ce is not None
    np.testing.assert_allclose(res.ce, ce, atol=1e-4 * targets.max())
    np.testing.assert_allclose(res.rate, rate, atol=1e-4 * rate.max())
    assert res.rate.sum() == pytest.approx(rate.sum(), rel=1e-5)


@pytest.mark.parametrize("key", CASES)
def test_effect_targeting_matches_stanpump(key):
    model, targets = CASES[key]
    _, rate, _, ce = stanpump_run(model, targets, target_effect=True)
    res = TCI(model, DELTA).run(targets)
    _assert_effect_close(res, rate, ce, targets)


@pytest.mark.parametrize(
    "key", ["EleveldPropofol", "MarshPropofol", "MintoRemifentanil"]
)
def test_golden_targets_match_stanpump(key):
    model, _ = CASES[key]
    targets = np.load(GOLDEN)["targets"]
    _, rate, _, ce = stanpump_run(model, targets, target_effect=True)
    res = TCI(model, DELTA).run(targets)
    _assert_effect_close(res, rate, ce, targets)


# ---- sanity of the controller ---------------------------------------------


@pytest.mark.parametrize("key", [k for k in CASES if k != "EleveldBlog"])
def test_concentrations_match_simulate(key):
    """The controller's Cp/Ce equal the model's own simulation of its rates."""
    model, targets = CASES[key]
    assert isinstance(model, LinearPKModel)
    tci = TCI(model, DELTA)
    res = tci.run(targets)
    assert res.ce is not None
    t_meas = (res.t + tci.lag + DELTA) / 60
    pred = model.simulate(res.t / 60, res.rate * 60, t_meas)
    assert pred.ce is not None
    np.testing.assert_allclose(res.cp, pred.cp, rtol=1e-9)
    np.testing.assert_allclose(res.ce, pred.ce, rtol=1e-9)


def test_first_bolus_peaks_at_target():
    """One step at the first rate, then nothing: Ce peaks at target at peak_time."""
    model = GeptsSufentanil()
    tci = TCI(model, DELTA)
    rate = tci.rate(0.2)
    t = np.linspace(0, 900, 9001)  # s
    pred = model.simulate([0, DELTA / 60], [rate * 60, 0], t / 60)
    assert pred.ce is not None
    assert pred.ce.max() == pytest.approx(0.2, rel=1e-9)
    assert t[pred.ce.argmax()] == pytest.approx(tci.peak_time, abs=0.1)


def test_plasma_target_without_effect_site():
    """Kataria has no ke0: plasma targeting holds Cp; effect targeting raises."""
    model = KatariaPropofol({"age": 8, "weight": 25})
    res = TCI(model, DELTA, target="plasma").run(np.full(6 * 60, 3.0))
    np.testing.assert_allclose(res.cp, 3.0, rtol=1e-9)
    assert res.ce is None
    assert np.all(np.diff(res.rate[1:]) < 0)  # peripheral compartments fill up
    with pytest.raises(ValueError, match="effect site"):
        TCI(model, DELTA, target="effect")


# ---- noradrenaline: endogenous production and lag --------------------------

NORA_TARGETS = np.repeat([0.5, 3.0, 6.0, 2.0], 60)  # nmol/L, 10 min each


def test_noradrenaline_starts_at_endogenous_level():
    model = Noradrenaline()
    tci = TCI(model, DELTA)
    baseline = model.parameters.EPROD / model.pk.CL
    assert tci.ce(tci.state) == pytest.approx(baseline)
    assert tci.lag == pytest.approx(model.lag * 60)
    res = tci.run(np.full(30, 0.5))  # below baseline: nothing to infuse
    assert res.ce is not None
    np.testing.assert_array_equal(res.rate, 0.0)
    np.testing.assert_allclose(res.ce, baseline, rtol=1e-12)


@pytest.mark.parametrize("weight", [70, 30])
def test_noradrenaline_reaches_and_holds_targets(weight):
    tci = TCI(Noradrenaline({"weight": weight}), DELTA)
    baseline = tci.ce(tci.state)
    res = tci.run(NORA_TARGETS)
    assert res.ce is not None
    ceiling = np.maximum(np.maximum.accumulate(NORA_TARGETS), baseline)
    # Switching to plasma targeting within 5 % of target lets Ce run up to
    # ~0.6 % above it (STANPUMP does the same).
    assert np.all(res.ce <= ceiling * 1.01)
    for start in (60, 120):  # target increases: reached and held within 2.5 min
        held = slice(start + 15, start + 60)
        np.testing.assert_allclose(res.ce[held], NORA_TARGETS[held], rtol=2e-3)
    assert np.all(res.rate[180:185] == 0)  # target decrease: stop and wait
    np.testing.assert_allclose(res.ce[-20:], 2.0, rtol=2e-3)


def test_noradrenaline_lag_matches_simulate():
    """Lag and forcing: the planning state equals the body ``lag`` s later."""
    model = Noradrenaline({"weight": 30})
    tci = TCI(model, DELTA)
    res = tci.run(NORA_TARGETS)
    assert res.ce is not None
    t_meas = (res.t + tci.lag + DELTA) / 60
    pred = model.simulate(res.t / 60, res.rate * 60, t_meas)
    assert pred.ce is not None
    np.testing.assert_allclose(res.cp, pred.cp, rtol=1e-9)
    np.testing.assert_allclose(res.ce, pred.ce, rtol=1e-9)
