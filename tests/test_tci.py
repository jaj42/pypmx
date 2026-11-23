from pathlib import Path

import numpy as np
import pytest

from pypmx import tci
from pypmx.drugs import (
    EleveldPropofol,
    KatariaPropofol,
    MarshPropofol,
    MintoRemifentanil,
)

GOLDEN = Path(__file__).parent / "golden" / "tci" / "runs.npz"
DELTA = 10
MODELS = {
    "EleveldPropofol": EleveldPropofol,
    "MarshPropofol": MarshPropofol,
    "MintoRemifentanil": MintoRemifentanil,
}


def run(setup: tci.TCISetup, targets, target_effect: bool):
    """Closed-loop TCI, one controller call per delta_seconds step."""
    n = len(setup.lambdas)
    Ap, Ae = np.zeros(n), np.zeros(n)
    rates, cp, ce = [], [], []
    for target in targets:
        rate = tci.model(
            Ap,
            Ae,
            setup.lambdas,
            setup.udfs,
            setup.coefs,
            target,
            setup.peak_time,
            setup.delta_seconds,
            target_effect=target_effect,
        )
        Ap = tci.update(Ap, setup.lambdas, setup.coefs.p_coef, rate, DELTA)
        Ae = tci.update(Ae, setup.lambdas, setup.coefs.e_coef, rate, DELTA)
        rates.append(rate)
        cp.append(Ap.sum())
        ce.append(Ae.sum())
    return {"rate": np.array(rates), "cp": np.array(cp), "ce": np.array(ce)}


@pytest.fixture(scope="module")
def golden():
    return np.load(GOLDEN)


@pytest.mark.parametrize("key", MODELS)
def test_setup_matches_pre_refactor(golden, key):
    setup = tci.prepare(MODELS[key](), DELTA)
    np.testing.assert_allclose(setup.lambdas, golden[f"{key}/lambdas"], rtol=1e-10)
    for name in ("p_coef", "e_coef"):
        np.testing.assert_allclose(
            getattr(setup.coefs, name), golden[f"{key}/{name}"], rtol=1e-9, atol=1e-12
        )
    for name in ("p_udf", "e_udf"):
        np.testing.assert_allclose(
            getattr(setup.udfs, name), golden[f"{key}/{name}"], rtol=1e-9, atol=1e-12
        )
    assert setup.peak_time == int(golden[f"{key}/peak_time"])


@pytest.mark.parametrize("mode", ["plasma", "effect"])
@pytest.mark.parametrize("key", MODELS)
def test_rates_match_pre_refactor(golden, key, mode):
    setup = tci.prepare(MODELS[key](), DELTA)
    res = run(setup, golden["targets"], target_effect=mode == "effect")
    for name, values in res.items():
        np.testing.assert_allclose(
            values, golden[f"{key}/{mode}/{name}"], rtol=1e-9, atol=1e-12
        )


def test_pk_parameters_input_same_as_model():
    m = EleveldPropofol()
    a, b = tci.prepare(m), tci.prepare(m.pk)
    np.testing.assert_array_equal(a.lambdas, b.lambdas)
    np.testing.assert_array_equal(a.coefs.p_coef, b.coefs.p_coef)
    assert a.peak_time == b.peak_time


@pytest.mark.parametrize("cls", [EleveldPropofol, MarshPropofol, MintoRemifentanil])
def test_seconds_boundary(cls):
    """Lambdas in s⁻¹, coefficients sum to Css of a 1/s infusion (60/CL_min)."""
    m = cls()
    lambdas, coefs = tci.decompose(m)
    pk_eigs = -np.linalg.eigvals(m.system_matrix)
    np.testing.assert_allclose(np.sort(lambdas * 60), np.sort(pk_eigs), rtol=1e-10)
    assert coefs.p_coef.sum() == pytest.approx(60 / m.pk.CL, rel=1e-10)
    if m.pk.has_effect_site:
        assert lambdas[-1] * 60 == pytest.approx(m.pk.ke0)
        assert coefs.e_coef.sum() == pytest.approx(60 / m.pk.CL, rel=1e-10)
    else:
        assert not np.any(coefs.e_coef)
    assert np.all(np.diff(lambdas[: m.pk.n_cmt]) < 0)  # PK modes fastest first


@pytest.mark.parametrize("key", MODELS)
def test_tci_cp_matches_simulate(key):
    """The controller's Cp/Ce equal the model's own simulation of its rates."""
    m = MODELS[key]()
    setup = tci.prepare(m, DELTA)
    targets = np.repeat([3.0, 4.0, 2.0], 60)
    res = run(setup, targets, target_effect=True)
    t_step = np.arange(len(targets)) * DELTA / 60  # min
    pred = m.simulate(t_step, res["rate"] * 60, t_step + DELTA / 60)
    np.testing.assert_allclose(res["cp"], pred.cp, rtol=1e-9)
    assert pred.ce is not None
    np.testing.assert_allclose(res["ce"], pred.ce, rtol=1e-9)


def test_plasma_target_without_effect_site():
    """Kataria has no ke0: plasma targeting reaches and holds Cp; Ce raises."""
    m = KatariaPropofol({"age": 8, "weight": 25})
    setup = tci.prepare(m, DELTA)
    assert setup.peak_time is None
    res = run(setup, np.full(6 * 60 * 4, 3.0), target_effect=False)  # 4 h
    np.testing.assert_allclose(res["cp"], 3.0, rtol=1e-9)
    # Maintenance decreases towards Css·CL (amount/min → per second) while the
    # peripheral compartments fill.
    maintenance = res["rate"][1:]
    assert np.all(np.diff(maintenance) < 0)
    assert maintenance[-1] > 3.0 * m.pk.CL / 60
    with pytest.raises(ValueError, match="effect site"):
        run(setup, [3.0], target_effect=True)


def test_effect_target_reached_at_peak():
    """First effect-site bolus brings Ce to target at peak_time."""
    setup = tci.prepare(EleveldPropofol(), DELTA)
    assert setup.peak_time is not None
    rate = tci.model(
        np.zeros(4),
        np.zeros(4),
        setup.lambdas,
        setup.udfs,
        setup.coefs,
        3.0,
        setup.peak_time,
        DELTA,
    )
    assert rate * setup.udfs.e_udf[setup.peak_time] == pytest.approx(3.0)
    assert setup.udfs.e_udf.argmax() == setup.peak_time
