import numpy as np
import pytest
from scipy.integrate import solve_ivp

from pypmx import eigen
from pypmx.tci import stanpump_order
from pypmx.utils import rate_at


def three_comp_ce(rng: np.random.Generator):
    """Random 3-comp mammillary system in amounts + effect site (Ce)."""
    V1, V2, V3 = rng.uniform(2, 50, 3)
    CL, Q2, Q3 = rng.uniform(0.2, 3, 3)
    ke0 = rng.uniform(0.1, 1)
    k10, k12, k13, k21, k31 = CL / V1, Q2 / V1, Q3 / V1, Q2 / V2, Q3 / V3
    S = np.array(
        [
            [-(k10 + k12 + k13), k21, k31, 0],
            [k12, -k21, 0, 0],
            [k13, 0, -k31, 0],
            [ke0 / V1, 0, 0, -ke0],
        ]
    )
    B = np.array([1.0, 0, 0, 0])
    return S, B


def solve_reference(S, B, meas_time, infu_time, infu_rate, y0=None, f=None, lag=0.0):
    """solve_ivp, restarted at each breakpoint so the rate is smooth per segment."""
    n = S.shape[0]
    f = np.zeros(n) if f is None else np.asarray(f)
    y = np.zeros(n) if y0 is None else np.asarray(y0, dtype=float)
    t0 = min(np.min(infu_time), np.min(meas_time))
    bps = np.unique(np.concatenate([[t0], np.asarray(infu_time) + lag, meas_time]))
    out = {bps[0]: y}
    for a, b in zip(bps[:-1], bps[1:]):
        r = float(rate_at(a - lag, infu_time, infu_rate))
        sol = solve_ivp(
            lambda t, x: S @ x + B * r + f, (a, b), y, rtol=1e-11, atol=1e-13
        )
        y = sol.y[:, -1]
        out[b] = y
    return np.array([out[t] for t in meas_time])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_reconstructs_system_matrix(seed):
    S, B = three_comp_ce(np.random.default_rng(seed))
    m = eigen.eigendecomposition(S, B)
    S_rec = m.eigvecs @ np.diag(-m.lambdas) @ np.linalg.inv(m.eigvecs)
    np.testing.assert_allclose(S_rec, S, atol=1e-12)
    np.testing.assert_allclose(m.eigvecs @ m.alpha_B, B, atol=1e-12)
    assert np.all(np.diff(m.lambdas) >= 0)  # slowest mode first
    assert np.all(m.lambdas > 0)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(
    "infu_time, infu_rate",
    [
        ([0.0, 0.01], [1000.0, 0.0]),  # bolus-like short infusion
        ([0.0, 2.0, 60.0, 90.0], [50.0, 5.0, 10.0, 0.0]),  # multi-step regimen
    ],
    ids=["bolus", "regimen"],
)
def test_matches_solve_ivp(seed, infu_time, infu_rate):
    S, B = three_comp_ce(np.random.default_rng(seed))
    t = np.linspace(0, 240, 241)
    res = eigen.eig_advan(S, B, t, infu_time, infu_rate)
    ref = solve_reference(S, B, t, infu_time, infu_rate)
    assert res.shape == (len(t), 4)
    np.testing.assert_allclose(res, ref, rtol=1e-7, atol=1e-10)


def test_nonzero_initial_state_and_unaligned_times():
    S, B = three_comp_ce(np.random.default_rng(3))
    y0 = [10.0, 5.0, 2.0, 0.1]
    t = np.array([0.0, 0.7, 3.3, 47.9, 100.0, 100.0, 333.3])
    infu_time, infu_rate = [1.5, 12.25], [7.0, 0.0]
    res = eigen.eig_advan(S, B, t, infu_time, infu_rate, y0=y0)
    ref = solve_reference(S, B, t, infu_time, infu_rate, y0=y0)
    np.testing.assert_allclose(res[0], y0)
    np.testing.assert_allclose(res, ref, rtol=1e-7, atol=1e-10)


def test_forcing_steady_state_is_constant():
    S, B = three_comp_ce(np.random.default_rng(4))
    f = np.array([0.5, 0, 0, 0])
    y_ss = np.linalg.solve(S, -f)
    t = np.linspace(0, 500, 51)
    res = eigen.eig_advan(S, B, t, [0.0], [0.0], y0=y_ss, f=f)
    np.testing.assert_allclose(res, np.tile(y_ss, (len(t), 1)), rtol=1e-10)


def test_forcing_with_infusion_and_lag_matches_solve_ivp():
    S, B = three_comp_ce(np.random.default_rng(5))
    f = np.array([0.5, 0, 0, 0])
    y_ss = np.linalg.solve(S, -f)
    t = np.linspace(0, 200, 201)
    infu_time, infu_rate, lag = [10.0, 40.0], [3.0, 0.0], 1.7
    res = eigen.eig_advan(S, B, t, infu_time, infu_rate, y0=y_ss, f=f, lag=lag)
    ref = solve_reference(S, B, t, infu_time, infu_rate, y0=y_ss, f=f, lag=lag)
    np.testing.assert_allclose(res, ref, rtol=1e-7, atol=1e-10)


def test_lag_shifts_response():
    S, B = three_comp_ce(np.random.default_rng(6))
    t = np.linspace(0, 120, 241)
    infu_time, infu_rate = [0.0, 5.0], [10.0, 0.0]
    lagged = eigen.eig_advan(S, B, t, infu_time, infu_rate, lag=2.5)
    shifted = eigen.eig_advan(S, B, t - 2.5, infu_time, infu_rate)
    shifted[t < 2.5] = 0.0
    np.testing.assert_allclose(lagged, shifted, rtol=1e-10, atol=1e-12)


def test_lag_rounding_keeps_the_right_rate():
    """``(t + lag) - lag`` rounds below t here; each segment keeps its own rate."""
    S, B = three_comp_ce(np.random.default_rng(7))
    lag = 21.48 / 60
    infu_time = np.arange(20) / 6  # every 10 s, in min
    assert np.any((infu_time + lag) - lag < infu_time)
    infu_rate = np.arange(20.0)[::-1]
    meas = infu_time + lag + 1 / 6
    lagged = eigen.eig_advan(S, B, meas, infu_time, infu_rate, lag=lag)
    shifted = eigen.eig_advan(S, B, meas, infu_time + lag, infu_rate)
    np.testing.assert_allclose(lagged, shifted, rtol=1e-12)


def test_zero_eigenvalue_is_finite():
    # Pure integrator: x' = r  =>  x = r·t during the infusion.
    t = np.linspace(0, 10, 11)
    res = eigen.eig_advan([[0.0]], [1.0], t, [0.0, 4.0], [2.0, 0.0])
    np.testing.assert_allclose(res[:, 0], 2.0 * np.minimum(t, 4.0))

    # 1-comp with an accumulating sink (k10 = 0 in the sink row): x2 = dose - x1.
    k = 0.3
    S = np.array([[-k, 0.0], [k, 0.0]])
    res = eigen.eig_advan(S, [1.0, 0.0], t, [0.0, 0.5], [20.0, 0.0])
    assert np.all(np.isfinite(res))
    np.testing.assert_allclose(res.sum(axis=1), 20.0 * np.minimum(t, 0.5))
    x1 = 20 * (1 - np.exp(-k * 0.5)) / k * np.exp(-k * (t - 0.5))
    np.testing.assert_allclose(res[1:, 0], x1[1:], rtol=1e-12)


def test_phi_limit():
    lambdas = np.array([0.0, 1e-300, 1e-8, 1.0])
    np.testing.assert_allclose(
        eigen.phi(lambdas, 2.0), [2.0, 2.0, 2.0, -np.expm1(-2.0)], rtol=1e-7
    )


def test_impulse_response_one_comp():
    V, k = 10.0, 0.2
    modes = eigen.eigendecomposition([[-k]], [1 / V])
    t = np.linspace(0, 30, 31)
    np.testing.assert_allclose(
        eigen.calculate_impulse_response(modes, t), np.exp(-k * t) / V
    )


def test_stanpump_order_puts_ke0_last():
    S, B = three_comp_ce(np.random.default_rng(7))
    ke0 = -S[3, 3]
    modes = eigen.eigendecomposition(S, B)
    ordered = stanpump_order(modes, ke0)
    assert ordered.lambdas[-1] == pytest.approx(ke0)
    assert np.all(np.diff(ordered.lambdas[:-1]) < 0)  # PK modes, fastest first
    # Reordering the modes does not change the system.
    t = np.linspace(0, 60, 61)
    for state in range(4):
        np.testing.assert_allclose(
            eigen.calculate_impulse_response(ordered, t, state),
            eigen.calculate_impulse_response(modes, t, state),
        )


def test_rate_at():
    infu_time, infu_rate = [1.0, 3.0, 5.0], [10.0, 20.0, 0.0]
    t = np.array([0.0, 0.999, 1.0, 2.9, 3.0, 4.9, 5.0, 100.0])
    np.testing.assert_array_equal(
        rate_at(t, infu_time, infu_rate), [0, 0, 10, 10, 20, 20, 0, 0]
    )
    assert float(rate_at(2.0, infu_time, infu_rate)) == 10.0
