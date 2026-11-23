import warnings

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx import eigen
from pypmx.drugs import KuipersLidocaine as Model

PK_NAMES = ("V1", "V2", "V3", "CL", "Q2", "Q3", "k12", "k21", "k13", "k31", "k10")
MODELS = ("A", "B", "C", "D")
# Table 1 medians: 45 y, 75 kg, QI 24.8 (height chosen to give it), CO 5.8
MEDIAN = {"age": 45, "weight": 75, "height": 100 * (75 / 24.8) ** 0.5}

# Table 2 typical values (V1, V2, V3, CL, Q2, Q3) per model
TABLE2 = {
    "A": (1.29, 7.39, 57.3, 0.646, 0.898, 0.750),
    "B": (1.30, 7.70, 57.4, 0.647, 0.918, 0.745),
    "C": (1.31, 8.05, 56.8, 0.646, 1.010, 0.752),
    "D": (1.29, 8.04, 56.7, 0.647, 0.997, 0.751),
}


@pytest.mark.parametrize(
    "covars",
    [
        {},
        {"age": 19, "weight": 51, "height": 158, "cardiac_output": 3.6},
        {"age": 76, "weight": 102, "height": 175, "cardiac_output": 10.0},
        {"age": 76, "weight": 51, "height": 158, "cardiac_output": 3.6},
    ],
)
@pytest.mark.parametrize("model", MODELS)
def test_parameters_finite_positive(covars, model):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = Model(covars, model=model).pk
    for name in PK_NAMES:
        value = getattr(p, name)
        assert np.isfinite(value) and value > 0, name
    assert p.ke0 == 0 and p.n_cmt == 3


@pytest.mark.parametrize("model", MODELS)
def test_table3_reproduces_table2_at_medians(model):
    m = Model(MEDIAN, model=model)
    assert m.parameters.QI == pytest.approx(24.8)
    p = m.pk
    got = (p.V1, p.V2, p.V3, p.CL, p.Q2, p.Q3)
    np.testing.assert_allclose(got, TABLE2[model], rtol=0.01)


def test_model_d_cl_at_medians():
    assert Model(MEDIAN).pk.CL == pytest.approx(0.648, abs=5e-4)


def test_covariate_dependence():
    base = Model(MEDIAN)
    # model A ignores covariates
    a = Model({"cardiac_output": 9, "weight": 95}, model="A").pk
    assert a == Model(model="A").pk
    # CO affects B and D only
    for m in MODELS:
        p0 = Model(MEDIAN, model=m).pk
        p1 = Model({**MEDIAN, "cardiac_output": 8}, model=m).pk
        if m in "BD":
            assert p1.CL > p0.CL and p1.V2 > p0.V2
        else:
            assert p1 == p0
    # age lowers distribution clearances in D
    old = Model({**MEDIAN, "age": 70}).pk
    assert old.Q2 < base.pk.Q2 and old.Q3 < base.pk.Q3
    assert old.CL == base.pk.CL


def test_invalid_model():
    with pytest.raises(ValueError):
        Model(model="E")


def test_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        Model()
        Model(MEDIAN)
    with pytest.warns(UserWarning, match="age"):
        Model({"age": 80})
    with pytest.warns(UserWarning, match="weight"):
        Model({"weight": 110, "height": 185})
    with pytest.warns(UserWarning, match="QI"):
        Model({"weight": 75, "height": 150})
    with pytest.warns(UserWarning, match="cardiac_output"):
        Model({"cardiac_output": 2.5})


@pytest.mark.parametrize("model", MODELS)
def test_simulate_bolus_matches_solve_ivp(model):
    m = Model(
        {"age": 60, "weight": 85, "height": 180, "cardiac_output": 4.5}, model=model
    )
    t = np.linspace(0, 720, 1441)
    cp = m.simulate([0.0], [0.0], t, y0=[1.0, 0, 0]).cp
    sol = solve_ivp(
        m.ode,
        (0, 720),
        [1.0, 0, 0],
        t_eval=t,
        args=(lambda t: 0.0,),
        method="LSODA",
        rtol=1e-10,
        atol=1e-14,
    )
    np.testing.assert_allclose(sol.y[0] / m.pk.V1, cp, rtol=1e-6, atol=1e-12)


def _fig3_infusion(m, t_end=800):
    """Fig. 3 regimen: 50 mg/min for 2 min, then 2.5 mg/min."""

    def inf(t):
        return 50.0 if t < 2 else 2.5

    t = np.arange(0, t_end + 1, 1.0)
    sol = solve_ivp(m.ode, (0, t_end), [0, 0, 0], t_eval=t, args=(inf,), max_step=0.5)
    return t, sol.y[0] / m.pk.V1


def test_fig3_models_agree_in_average_patient():
    """Fig. 3b: average patient (45 y, 75 kg, QI 25), CO 6 L/min.

    The four models give very similar profiles; Css = 2.5 / CL ~ 3.8-3.9 mg/L.
    """
    covars = {
        "age": 45,
        "weight": 75,
        "height": 100 * (75 / 25) ** 0.5,
        "cardiac_output": 6,
    }
    profiles = {}
    for name in MODELS:
        m = Model(covars, model=name)
        t, cp = _fig3_infusion(m)
        profiles[name] = cp
        css = 2.5 / m.pk.CL
        assert 3.7 < css < 4.0
        assert cp[-1] < css
        assert cp[-1] > 0.75 * css  # mostly at steady state after 800 min
        # Fig. 3 y-axis is 0-6 mg/L; only the loading peak exceeds it
        assert np.all(cp[t >= 10] < 6.0)
    ref = profiles["D"]
    for name in "ABC":
        np.testing.assert_allclose(profiles[name][5:], ref[5:], rtol=0.15)


def test_fig3_cardiac_output_effect():
    """Low CO gives higher concentrations with model D than with A; high CO lower."""
    covars = {"age": 45, "weight": 75, "height": 173}
    _, a = _fig3_infusion(Model({**covars, "cardiac_output": 6}, model="A"))
    _, d_low = _fig3_infusion(Model({**covars, "cardiac_output": 3.6}))
    _, d_high = _fig3_infusion(Model({**covars, "cardiac_output": 10}))
    assert d_low[-1] > 1.05 * a[-1]
    assert d_high[-1] < 0.95 * a[-1]


def test_bolus_1mg_per_kg():
    """1 mg/kg (75 mg) bolus: blood levels fall to sub-therapeutic within an hour
    and to tens of ng/ml by 6 h (assay range 11.7-1404 ng/ml after the peak)."""
    m = Model()
    modes = eigen.eigendecomposition(m.system_matrix, m.input_matrix)
    t = np.array([2.0, 10, 60, 360])
    cp = 75 * eigen.calculate_impulse_response(modes, t) / m.pk.V1
    assert np.all(np.diff(cp) < 0)
    assert 1.0 < cp[1] < 5.0
    assert 0.2 < cp[2] < 1.0
    assert 0.01 < cp[3] < 0.2
