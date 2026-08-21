"""Pair-process master equation: exact limits and monotonicity."""
import numpy as np

from absnoise import channel_generator, noneq_penalty, sigma_spectrum


def test_singles_limit_is_exact():
    # Gp = 0 reproduces the independent-spin Lorentzian:
    # var = 2 f(1-f), tau_eff = 1/Gs
    for f in (0.05, 0.3, 0.5):
        _, S0, var, te = sigma_spectrum(f, Gs=2.0, Gp=0.0,
                                        omegas=np.array([0.0]))
        assert abs(var - 2 * f * (1 - f)) < 1e-12
        assert abs(te - 0.5) < 1e-12


def test_equilibrium_variance_invariant_under_pair_rate():
    for Gp in (0.0, 0.5, 3.0, 30.0):
        _, _, var, _ = sigma_spectrum(0.2, 1.0, Gp, np.array([0.0]))
        assert abs(var - 2 * 0.2 * 0.8) < 1e-12


def test_pair_processes_only_shorten_the_correlation_time():
    # tau_eff decreases monotonically with Gp, so the
    # single-quasiparticle bound is the worst case
    taus = []
    for Gp in (0.0, 0.3, 1.0, 3.0, 10.0):
        _, _, _, te = sigma_spectrum(0.25, 1.0, Gp, np.array([0.0]))
        taus.append(te)
    assert all(b <= a + 1e-12 for a, b in zip(taus, taus[1:]))


def test_generator_conserves_probability():
    W = channel_generator(0.3, 1.7, 0.9)
    assert np.max(np.abs(W.sum(axis=0))) < 1e-14
    off = W - np.diag(np.diag(W))
    assert np.min(off) >= 0.0


def test_spectrum_integrates_to_variance():
    # (1/2pi) int S(omega) domega over both signs = var; single-sided
    # S defined for omega >= 0 here, so int_0^inf S domega / (2 pi) =
    # var / 2 ... checked numerically on a wide grid
    f, Gs, Gp = 0.3, 1.0, 2.0
    om = np.linspace(0.0, 4000.0, 2000001)
    S, S0, var, _ = sigma_spectrum(f, Gs, Gp, om)
    integral = np.trapezoid(S, om) / (2 * np.pi)
    # remaining deficit is the analytic 1/omega tail beyond the grid
    assert abs(integral - var) / var < 5e-4


def test_noneq_penalty_limits():
    assert abs(noneq_penalty(0.2, 0.2) - 1.0) < 1e-12
    assert noneq_penalty(0.01, 0.3) > 1.0
