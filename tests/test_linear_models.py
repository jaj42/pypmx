"""Shared sanity checks for the drug models migrated to LinearPKModel (steps 3, 5)."""

import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx.drugs import (
    AllegaertParacetamol,
    Cefazolin,
    DyckDexmedetomidine,
    EleveldPropofol,
    EleveldRemifentanil,
    GeptsSufentanil,
    GrassinDelyleTranexamicAcid,
    HannivoortDexmedetomidine,
    KatariaPropofol,
    KuipersLidocaine,
    MaitreAlfentanil,
    MarshPropofol,
    PaedfusorPropofol,
    ZomorodiMidazolam,
)
from pypmx.drugs import MintoRemifentanil as Minto
from pypmx.drugs import SchniderPropofol as Schnider
from pypmx.pk import LinearPKModel, PKParameters
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
    "Marsh": (lambda: MarshPropofol(), 3, True, (50.0, 8.0)),
    "Kataria": (
        lambda: KatariaPropofol({"age": 6, "weight": 20}),
        3,
        False,
        (15.0, 2.5),
    ),
    "Paedfusor": (
        lambda: PaedfusorPropofol({"age": 6, "weight": 20}),
        3,
        False,
        (15.0, 2.5),
    ),
    "Gepts": (lambda: GeptsSufentanil(), 3, True, (10.0, 0.5)),
    "Cefazolin": (lambda: Cefazolin(), 2, False, (1000.0, 10.0)),
    "CefazolinCrCL": (lambda: Cefazolin({"CrCL": 80}), 2, False, (1000.0, 10.0)),
    "Maitre": (lambda: MaitreAlfentanil(), 3, True, (500.0, 50.0)),
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


def terminal_half_life(m: LinearPKModel) -> float:
    """Slowest PK disposition half-life (min)."""
    return np.log(2) / min(-np.linalg.eigvals(m.system_matrix).real)


def test_from_micro_roundtrip():
    p = PKParameters.from_micro(
        V1=4.0, k10=0.1, k12=0.2, k21=0.05, k13=0.03, k31=0.003, ke0=0.5
    )
    for name, value in dict(k10=0.1, k12=0.2, k21=0.05, k13=0.03, k31=0.003).items():
        assert getattr(p, name) == pytest.approx(value), name
    assert (p.CL, p.Q2, p.V2) == pytest.approx((0.4, 0.8, 16.0))
    one = PKParameters.from_micro(V1=10.0, k10=0.1)
    assert one.n_cmt == 1 and one.CL == pytest.approx(1.0)


def test_marsh_volumes():
    """Marsh: V1 0.228, derived V2 ≈ 0.464 and V3 ≈ 3.09 L/kg, ke0 0.26."""
    p = MarshPropofol({"weight": 80}).pk
    assert p.V1 == pytest.approx(0.228 * 80)
    assert p.V2 / 80 == pytest.approx(0.464, rel=1e-3)  # published 0.463
    assert p.V3 / 80 == pytest.approx(3.09, rel=1e-3)
    assert p.k10 == pytest.approx(0.119)
    assert p.ke0 == 0.26


def test_paediatric_propofol_ranges():
    with pytest.warns(UserWarning, match="Kataria: age 2"):
        KatariaPropofol({"age": 2, "weight": 12})
    with pytest.warns(UserWarning, match="Paedfusor: age 14"):
        PaedfusorPropofol({"age": 14, "weight": 50})
    p = KatariaPropofol({"age": 5, "weight": 20}).pk
    assert (p.V1, p.V2, p.CL) == pytest.approx((7.6, 0.59 * 20 + 15.5 - 13, 0.74))
    # Paedfusor k10 scales with weight^-0.3
    assert PaedfusorPropofol({"age": 5, "weight": 20}).pk.k10 == pytest.approx(
        0.1527 * 20**-0.3
    )


def test_gepts_sufentanil():
    m = GeptsSufentanil()
    assert m.pk.V1 == 14.3
    assert m.pk.k10 == pytest.approx(0.0645)
    assert m.pk.ke0 == 0.112
    assert m.pk.V2 == pytest.approx(63.1, rel=0.01)  # published 63.1 L
    assert m.pk.V3 == pytest.approx(261.6, rel=0.05)  # published 261.6 L
    assert 600 < terminal_half_life(m) < 900  # Gepts: 769 min
    assert repr(m) == "Gepts()"


def test_cefazolin_crcl():
    base, ref = Cefazolin(), Cefazolin({"CrCL": 80})
    assert base.pk.CL == pytest.approx(2.87 / 60)
    assert ref.pk.CL == pytest.approx(2.86 / 60)  # reference CrCL 80 ml/min
    low, high = Cefazolin({"CrCL": 40}).pk.CL, Cefazolin({"CrCL": 130}).pk.CL
    assert low < ref.pk.CL < high
    assert low == pytest.approx(2.86 / 60 * 0.5**0.79)
    assert (ref.pk.V1, ref.pk.V2, ref.pk.Q2) == pytest.approx((5.2, 4.56, 10.9 / 60))
    assert 100 < terminal_half_life(ref) < 200  # cefazolin t½β ≈ 2-2.5 h
    with pytest.raises(ValueError, match="CrCL"):
        Cefazolin({"CrCL": 0})


def test_maitre_alfentanil():
    m = MaitreAlfentanil()  # 35 y, 70 kg, male
    assert m.pk.V1 == pytest.approx(0.111 * 70)
    assert m.pk.CL == pytest.approx(0.356)
    assert m.pk.V2 == pytest.approx(0.111 * 70 * 0.104 / 0.0673)
    assert MaitreAlfentanil({"sex": "f"}).pk.V1 == pytest.approx(0.128 * 70)
    old = MaitreAlfentanil({"age": 70})
    assert old.pk.CL == pytest.approx(0.356 - 0.00269 * 30)
    assert old.pk.k31 == pytest.approx(0.0126 - 0.000113 * 30)
    # Maitre: t½β ≈ 1.5 h in adults, prolonged with age
    assert 70 < terminal_half_life(m) < 120
    assert terminal_half_life(old) > terminal_half_life(m)
    with pytest.raises(ValueError, match="sex"):
        MaitreAlfentanil({"sex": "x"})
