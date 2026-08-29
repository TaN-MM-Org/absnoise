"""Allan-variance tests: machine-precision drift check, closed form
against the defining integrals, both limits, and seeded Monte Carlo."""
import numpy as np
import pytest

from absnoise import telegraph_traces
from absnoise.allan import allan_variance, avar_exponential, avar_white


def test_linear_drift_exact():
    # x = c t: adjacent m-sample means always differ by c m dt exactly,
    # so AVAR = (c T)^2 / 2 to machine precision at every window
    c, dt = 0.37, 0.01
    t = np.arange(2000) * dt
    taus, avar = allan_variance(c * t, dt, m_list=[1, 5, 40, 250])
    assert np.allclose(avar, 0.5 * (c * taus) ** 2, rtol=1e-12)


def test_closed_form_matches_defining_integrals():
    # AVAR(T) = (E[A^2] - E[AB]) / T^2 for A, B adjacent T-integrals of
    # a process with C(t) = var exp(-|t|/tau); evaluate both double
    # integrals numerically and compare with avar_exponential
    var, tau = 0.21, 1.7
    for T in (0.3, 1.7, 6.0):
        u = np.linspace(0.0, T, 20001)
        C = var * np.exp(-u / tau)
        # E[A^2] = 2 int_0^T (T-u) C(u) du
        EA2 = 2.0 * np.trapezoid((T - u) * C, u)
        # E[AB] over s in [0,T], t in [T,2T]: substituting w = t - s
        # in [0, 2T] gives the triangular overlap kernel
        w = np.linspace(0.0, 2.0 * T, 40001)
        kern = np.where(w <= T, w, 2.0 * T - w)
        EAB = np.trapezoid(kern * var * np.exp(-w / tau), w)
        avar_num = (EA2 - EAB) / T ** 2
        assert avar_exponential(var, tau, T) == pytest.approx(
            avar_num, rel=1e-6)


def test_closed_form_limits():
    var, tau = 0.25, 1.0
    # T << tau: 2 var T / (3 tau)
    T = 1e-4
    assert avar_exponential(var, tau, T) == pytest.approx(
        2.0 * var * T / (3.0 * tau), rel=1e-3)
    # T >> tau: S(0)/(2T) with S(0) = 4 var tau
    T = 1e4
    assert avar_exponential(var, tau, T) == pytest.approx(
        avar_white(4.0 * var * tau, T), rel=1e-3)


def test_telegraph_monte_carlo_matches_closed_form():
    rng = np.random.default_rng(3)
    f, tauA, dt = 0.3, 1.0, 0.05
    tr = telegraph_traces(f, tauA, dt, 400_000, 16, rng)
    x = tr.astype(float).sum(axis=1)         # sum of 16 channels
    var = 16.0 * f * (1.0 - f)
    m_list = [20, 80, 320]                   # T = 1, 4, 16 tauA
    taus, avar = allan_variance(x, dt, m_list)
    expected = avar_exponential(var, tauA, taus)
    assert np.allclose(avar, expected, rtol=0.12)


def test_white_noise_matches_s0_over_2t():
    rng = np.random.default_rng(5)
    dt, sig = 0.01, 1.3
    x = rng.normal(0.0, sig, 500_000)
    S0 = 2.0 * sig * sig * dt                # single-sided PSD of the samples
    m_list = [4, 16, 64]
    taus, avar = allan_variance(x, dt, m_list)
    assert np.allclose(avar, avar_white(S0, taus), rtol=0.05)


def test_input_validation():
    with pytest.raises(ValueError):
        allan_variance([1.0, 2.0], 0.1)
    with pytest.raises(ValueError):
        allan_variance([1.0, np.nan, 2.0, 3.0], 0.1)
    with pytest.raises(ValueError):
        allan_variance(np.arange(10.0), 0.1, m_list=[6])
    with pytest.raises(ValueError):
        allan_variance(np.arange(10.0), -0.1)
    with pytest.raises(ValueError):
        avar_exponential(0.1, -1.0, 1.0)
    with pytest.raises(ValueError):
        avar_exponential(0.1, 1.0, 0.0)
    with pytest.raises(ValueError):
        avar_white(-1.0, 1.0)
