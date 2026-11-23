"""Shared IIV API (plan step 7): Omega, with_eta and sample on all models."""

from math import log

import numpy as np
import pytest

from pypmx.drugs import (
    EleveldPropofol,
    EleveldRemifentanil,
    Noradrenaline,
    SuPropofolHemodynamics,
    SuPropofolRemifentanilHemodynamics,
    SuPropofolRemifentanilHemodynamicsVariant,
)
from pypmx.iiv import Omega

# ---- Omega -----------------------------------------------------------------


def legacy_cv_covariance(cv, corr_pairs):
    """The covariance construction the Su models used before step 7."""
    sd = np.sqrt(np.log((np.asarray(cv) / 100) ** 2 + 1))
    corr = np.diag(np.ones_like(sd))
    for (i, j), r in corr_pairs.items():
        corr[i, j] = r
    corr_sym = np.triu(corr) + np.triu(corr, 1).T
    return corr_sym * np.outer(sd, sd)


def test_from_cv_matches_legacy_su_propofol():
    expected = legacy_cv_covariance(
        [18.0, 27.3, 17.0, 63.7], {(0, 1): -0.662, (1, 2): -0.586}
    )
    omega = SuPropofolHemodynamics.omega
    assert omega is not None
    assert omega.names == ("base_sv", "base_tpr", "base_hr", "EC50_tpr")
    np.testing.assert_allclose(omega.cov, expected, rtol=1e-14)


def test_from_cv_matches_legacy_su_propofol_remifentanil():
    expected = legacy_cv_covariance(
        [18.3, 23.3, 15.67, 74.34, 75.30, 6.18, 9.34],
        {(0, 1): -0.58, (1, 2): -0.65, (5, 6): 0.57},
    )
    omega = SuPropofolRemifentanilHemodynamics.omega
    assert omega is not None
    np.testing.assert_allclose(omega.cov, expected, rtol=1e-14)


def test_from_variances_su_variant():
    omega = SuPropofolRemifentanilHemodynamicsVariant.omega
    assert omega is not None
    assert omega.cov[0, 1] == omega.cov[1, 0] == -0.0249
    assert omega.cov[1, 2] == omega.cov[2, 1] == -0.0235
    assert omega.cov[5, 6] == omega.cov[6, 5] == 0.0038
    assert omega.variances["ec50_prop_tpr"] == 0.444
    assert omega.additive == {"emax_remi_tpr", "sl_remi_hr", "sl_remi_sv"}


def test_cv_round_trip():
    omega = Omega.from_cv({"CL": 30.0})
    assert np.sqrt(np.expm1(omega.cov[0, 0])) == pytest.approx(0.30)


def test_validation():
    with pytest.raises(ValueError, match="symmetric"):
        Omega(["a", "b"], [[1, 0.1], [0, 1]])
    with pytest.raises(ValueError, match="2x2"):
        Omega(["a", "b"], [[1]])
    with pytest.raises(np.linalg.LinAlgError):
        Omega(["a", "b"], [[1, 2], [2, 1]])  # not positive definite
    with pytest.raises(ValueError, match="additive"):
        Omega(["a"], [[1]], additive=["b"])
    with pytest.raises(ValueError, match="diagonal"):
        Omega.from_variances({"a": 1}, {("a", "a"): 0.5})
    omega = Omega(["a"], [[1]])
    with pytest.raises(ValueError, match="not in omega"):
        omega.apply({"a": 1.0}, {"b": 0.1})


def test_apply_exponential_and_additive():
    omega = Omega.from_variances({"a": 0.1, "b": 0.2}, additive=["b"])
    out = omega.apply({"a": 2.0, "b": 3.0}, {"a": log(1.5), "b": -0.5})
    assert out == pytest.approx({"a": 3.0, "b": 2.5})
    assert omega.apply({"a": 2.0, "b": 3.0}, {}) == {"a": 2.0, "b": 3.0}


def test_draw_moments():
    omega = Omega.from_variances({"a": 0.1, "b": 0.3}, {("a", "b"): 0.05})
    eta = omega.draw(np.random.default_rng(1), n=200_000)
    assert eta.shape == (200_000, 2)
    np.testing.assert_allclose(eta.mean(axis=0), 0, atol=5e-3)
    np.testing.assert_allclose(np.cov(eta.T), omega.cov, rtol=2e-2, atol=2e-3)


# ---- LinearPKModel ---------------------------------------------------------


def test_eleveld_remifentanil_omega_cvs():
    """CV% of the published $OMEGA (from the NONMEM output comments)."""
    omega = EleveldRemifentanil.omega
    assert omega is not None
    cv = {k: 100 * np.sqrt(np.expm1(v)) for k, v in omega.variances.items()}
    expected = {"V1": 33.0, "V2": 35.0, "V3": 111.7, "CL": 14.1, "Q2": 23.7}
    for k, v in expected.items():
        assert cv[k] == pytest.approx(v, abs=0.1)
    assert cv["Q3"] == pytest.approx(57.5, abs=0.5)


def test_with_eta_linear_model():
    m = EleveldRemifentanil()
    assert m.with_eta({}).pk == m.pk
    ind = m.with_eta({"CL": log(2), "V1": -log(2)})
    assert ind.pk.CL == pytest.approx(2 * m.pk.CL)
    assert ind.pk.V1 == pytest.approx(m.pk.V1 / 2)
    assert ind.pk.ke0 == m.pk.ke0 and ind.pk.Q2 == m.pk.Q2
    assert m.pk.CL == EleveldRemifentanil().pk.CL  # original untouched
    assert ind.covars == m.covars


def test_sample_reproducible_and_distributed():
    m = EleveldRemifentanil()
    a = m.sample(np.random.default_rng(42))
    b = m.sample(np.random.default_rng(42))
    assert a.pk == b.pk
    assert a.pk != m.pk

    rng = np.random.default_rng(0)
    log_cl = np.array([log(m.sample(rng).pk.CL / m.pk.CL) for _ in range(4000)])
    omega = m.omega
    assert omega is not None
    assert log_cl.mean() == pytest.approx(0, abs=0.01)
    assert log_cl.var() == pytest.approx(omega.variances["CL"], rel=0.1)


def test_sampled_individual_simulates():
    """A higher clearance gives a lower steady state (Css = rate / CL)."""
    m = EleveldRemifentanil()
    ind = m.with_eta({"CL": log(1.25)})
    t = np.array([0.0, 2000.0])
    css_typ = m.simulate([0], [10.0], t).cp[-1]
    css_ind = ind.simulate([0], [10.0], t).cp[-1]
    assert css_typ == pytest.approx(10.0 / m.pk.CL, rel=1e-6)
    assert css_ind == pytest.approx(css_typ / 1.25, rel=1e-6)
    assert np.all(
        np.isfinite(m.sample(np.random.default_rng(3)).simulate([0], [10.0], t).cp)
    )


def test_model_without_omega_raises():
    m = EleveldPropofol()
    with pytest.raises(NotImplementedError, match="no IIV"):
        m.sample(np.random.default_rng(0))
    with pytest.raises(NotImplementedError):
        m.with_eta({})


class _NoradrenalineIIV(Noradrenaline):
    """Omega over a pk field (CL) and a non-PK parameter (EPROD)."""

    omega = Omega.from_variances({"CL": 0.1, "EPROD": 0.2})


def test_omega_on_pk_and_parameters():
    m = _NoradrenalineIIV({"weight": 70})
    ind = m.with_eta({"CL": log(2), "EPROD": log(3)})
    assert ind.pk.CL == pytest.approx(2 * m.pk.CL)
    assert ind.parameters.EPROD == pytest.approx(3 * m.parameters.EPROD)
    assert m.parameters.EPROD == Noradrenaline({"weight": 70}).parameters.EPROD
    # Endogenous steady state Cp = EPROD / CL follows both etas.
    cp0 = ind.simulate([0], [0.0], [0.0]).cp[0]
    assert cp0 == pytest.approx(1.5 * m.parameters.EPROD / m.pk.CL, rel=1e-9)


# ---- Su PD models ----------------------------------------------------------


def cp_prop(t):
    return 3.0 if t >= 0 else 0.0


def cp_remi(t):
    return 2.0 if t >= 0 else 0.0


T = np.linspace(0, 60, 61)


def su_models():
    return {
        "propofol": lambda **kw: SuPropofolHemodynamics(cp_prop, {"age": 50}, **kw),
        "propofol_remifentanil": lambda **kw: SuPropofolRemifentanilHemodynamics(
            cp_prop, cp_remi, {"age": 50}, **kw
        ),
        "variant": lambda **kw: SuPropofolRemifentanilHemodynamicsVariant(
            cp_prop, cp_remi, {"age": 50}, **kw
        ),
    }


def run(m):
    if isinstance(m, SuPropofolHemodynamics):
        return m.calculate_at(T)
    return m(T)


def param(m, name):
    p = m.parameters
    return p[name] if isinstance(p, dict) else getattr(p, name)


@pytest.mark.parametrize("name", list(su_models()))
def test_su_with_zero_eta_is_typical(name):
    make = su_models()[name]
    m = make()
    for a, b in zip(run(m), run(m.with_eta({}))):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("name", list(su_models()))
def test_su_sample_rederives(name):
    m = su_models()[name]()
    ind = m.with_eta({"base_hr": log(1.2)})
    assert param(ind, "base_hr") == pytest.approx(1.2 * param(m, "base_hr"))
    kin = "kin_hr"
    assert param(ind, kin) == pytest.approx(param(ind, "kout") * param(ind, "base_hr"))
    assert param(ind, "base_map") == pytest.approx(1.2 * param(m, "base_map"))
    assert ind._y0.HRi == pytest.approx(1.2 * m._y0.HRi)
    assert param(m, "base_hr") == param(su_models()[name](), "base_hr")
    # Baseline HR at t=0 rises with base_hr (no residual error).
    assert run(ind)[1][0] == pytest.approx(1.2 * run(m)[1][0], rel=1e-6)


def test_su_variant_additive_eta():
    m = su_models()["variant"]()
    ind = m.with_eta({"sl_remi_hr": 0.01})
    assert ind.parameters.sl_remi_hr == pytest.approx(m.parameters.sl_remi_hr + 0.01)


@pytest.mark.parametrize("name", list(su_models()))
def test_su_random_effects_flag(name):
    make = su_models()[name]
    a = make(random_effects=True, rng=np.random.default_rng(7))
    b = make(random_effects=True, rng=np.random.default_rng(7))
    assert a.parameters == b.parameters
    assert a.ruv == b.ruv
    assert param(a, "base_hr") != param(make(), "base_hr")
    hr = run(a)[1]
    assert np.all(np.isfinite(hr)) and np.all(hr > 0)


def test_su_propofol_typical_not_mutated():
    """The old code updated the class-level dict when sampling."""
    before = dict(SuPropofolHemodynamics.typical)
    SuPropofolHemodynamics(cp_prop, random_effects=True, rng=np.random.default_rng(0))
    SuPropofolHemodynamics(cp_prop).sample(np.random.default_rng(1))
    assert SuPropofolHemodynamics.typical == before
