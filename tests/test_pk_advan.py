"""Cross-check the ADVAN closed forms (``advan/pk_advan.py``) against ``eig_advan``.

The two solvers share no code: ``pk_advan`` uses the Abuhelwa et al. (2015)
closed-form equations, ``eig_advan`` an eigendecomposition of ``S``.
"""

import numpy as np
import pytest

from pypmx import pk
from pypmx.advan import pk_advan as pa
from pypmx.drugs import (
    DyckDexmedetomidine,
    EleveldPropofol,
    GrassinDelyleTranexamicAcid,
)
from pypmx.eigen import eig_advan

RTOL = 1e-10
T = np.array([0.0, 0.1, 0.7, 3.0, 10.0, 60.0, 240.0])

CLASSES = {
    1: (pa.OneCompartmentIVBolus, pa.OneCompartmentIVInfusion, pa.OneCompartmentOral),
    2: (pa.TwoCompartmentIVBolus, pa.TwoCompartmentIVInfusion, pa.TwoCompartmentOral),
    3: (
        pa.ThreeCompartmentIVBolus,
        pa.ThreeCompartmentIVInfusion,
        pa.ThreeCompartmentOral,
    ),
}


def random_parameters(rng: np.random.Generator, n_cmt: int) -> pa.Parameters:
    """Random rate constants (1/min), including peripheral elimination k20/k30."""
    k = dict(zip(["k10", "k12", "k13", "k21", "k31"], rng.uniform(0.01, 0.5, 5)))
    k["ka"] = rng.uniform(0.6, 2.0)
    k["k20"], k["k30"] = rng.uniform(0.0, 0.02, 2)
    if n_cmt < 3:
        k.update(k13=0.0, k31=0.0, k30=0.0)
    if n_cmt < 2:
        k.update(k12=0.0, k21=0.0, k20=0.0)
    return pa.Parameters(**k)


def disposition_matrix(p: pa.Parameters, n_cmt: int) -> np.ndarray:
    K = np.array(
        [
            [-(p.k10 + p.k12 + p.k13), p.k21, p.k31],
            [p.k12, -(p.k20 + p.k21), 0.0],
            [p.k13, 0.0, -(p.k30 + p.k31)],
        ]
    )
    return K[:n_cmt, :n_cmt]


def oral_matrix(p: pa.Parameters, n_cmt: int) -> np.ndarray:
    """Depot (state 0) feeding the central compartment (state 1)."""
    S = np.zeros((n_cmt + 1, n_cmt + 1))
    S[1:, 1:] = disposition_matrix(p, n_cmt)
    S[0, 0] = -p.ka
    S[1, 0] = p.ka
    return S


def advance(model, A0, t, **kw) -> np.ndarray:
    return np.array([np.atleast_1d(model.advance(*A0, tt, **kw)) for tt in t])


def assert_close(got, ref):
    scale = np.max(np.abs(ref))
    np.testing.assert_allclose(got, ref, rtol=RTOL, atol=RTOL * scale)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_cmt", [1, 2, 3])
def test_bolus(n_cmt, seed):
    p = random_parameters(np.random.default_rng(seed), n_cmt)
    A0 = np.arange(1.0, n_cmt + 1)  # all compartments start non-empty
    S = disposition_matrix(p, n_cmt)
    ref = eig_advan(S, np.eye(n_cmt)[0], T, [0.0], [0.0], y0=A0)
    assert_close(advance(CLASSES[n_cmt][0](p), A0, T), ref)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_cmt", [1, 2, 3])
def test_infusion(n_cmt, seed):
    p = random_parameters(np.random.default_rng(seed), n_cmt)
    A0 = np.arange(1.0, n_cmt + 1)
    S = disposition_matrix(p, n_cmt)
    ref = eig_advan(S, np.eye(n_cmt)[0], T, [0.0], [2.5], y0=A0)
    assert_close(advance(CLASSES[n_cmt][1](p), A0, T, R=2.5), ref)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("n_cmt", [1, 2, 3])
def test_oral(n_cmt, seed):
    p = random_parameters(np.random.default_rng(seed), n_cmt)
    A0 = np.arange(1.0, n_cmt + 2)  # depot + disposition compartments
    S = oral_matrix(p, n_cmt)
    ref = eig_advan(S, np.eye(n_cmt + 1)[0], T, [0.0], [0.0], y0=A0)
    assert_close(advance(CLASSES[n_cmt][2](p), A0, T), ref)


def test_three_comp_oral_first_peripheral():
    """Regression: Eq. 38 lost its λ2 term, so X3 was ~15% off."""
    p = random_parameters(np.random.default_rng(0), 3)
    A0 = np.array([100.0, 0.0, 0.0, 0.0])
    ref = eig_advan(oral_matrix(p, 3), np.eye(4)[0], T, [0.0], [0.0], y0=A0)
    got = advance(pa.ThreeCompartmentOral(p), A0, T)
    assert_close(got[:, 2], ref[:, 2])


def to_advan(p: pk.PKParameters) -> pa.Parameters:
    return pa.Parameters(k10=p.k10, k12=p.k12, k13=p.k13, k21=p.k21, k31=p.k31)


@pytest.mark.parametrize("n_cmt", [1, 2, 3])
def test_calculate_lambda(n_cmt):
    p = random_parameters(np.random.default_rng(3), n_cmt)
    lambdas = pa.calculate_lambda(p)
    expected = -np.linalg.eigvals(disposition_matrix(p, n_cmt)).real
    np.testing.assert_allclose(lambdas[:n_cmt], np.sort(expected)[::-1], rtol=1e-12)
    np.testing.assert_allclose(lambdas[n_cmt:], 0.0, atol=1e-15)


def chain_infusions(model, A0, infu_time, infu_rate, meas_time) -> np.ndarray:
    """Step the ADVAN model segment by segment, as a NONMEM-style event loop."""
    bps = np.unique(np.concatenate([infu_time, meas_time]))
    A = np.asarray(A0, dtype=float)
    out = {bps[0]: A}
    for a, b in zip(bps[:-1], bps[1:]):
        rate = infu_rate[np.searchsorted(infu_time, a, side="right") - 1]
        A = np.atleast_1d(model.advance(*A, b - a, R=rate))
        out[b] = A
    return np.array([out[t] for t in meas_time])


REGIMEN = (np.array([0.0, 2.0, 60.0]), np.array([100.0, 10.0, 0.0]))


@pytest.mark.parametrize(
    "model, n_cmt",
    [
        (EleveldPropofol({}), 3),
        (DyckDexmedetomidine({}), 3),
        (GrassinDelyleTranexamicAcid({}), 2),
        (pk.PKParameters(V1=10.0, CL=0.5), 1),
    ],
    ids=[
        "EleveldPropofol",
        "DyckDexmedetomidine",
        "GrassinDelyleTranexamicAcid",
        "1comp",
    ],
)
def test_drug_regimen(model, n_cmt):
    """Loading + maintenance + washout: same amounts as ``LinearPKModel``'s S."""
    p = model.pk if isinstance(model, pk.LinearPKModel) else model
    assert p.n_cmt == n_cmt
    infu_time, infu_rate = REGIMEN
    t = np.linspace(0, 240, 97)
    got = chain_infusions(
        CLASSES[n_cmt][1](to_advan(p)), np.zeros(n_cmt), infu_time, infu_rate, t
    )
    S = pk.system_matrix(p)
    ref = eig_advan(S, np.eye(len(S))[0], t, infu_time, infu_rate)
    assert_close(got, ref[:, :n_cmt])
    if isinstance(model, pk.LinearPKModel):
        cp = model.simulate(infu_time, infu_rate, t).cp
        np.testing.assert_allclose(got[:, 0] / p.V1, cp, rtol=RTOL, atol=1e-12)
