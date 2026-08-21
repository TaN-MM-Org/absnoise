"""Device-level budgets: exact spectral shapes and internal consistency."""
import numpy as np

from absnoise import RECIPES, SensorBudget


def _budget():
    return SensorBudget(RECIPES[1])


def test_frequency_noise_is_a_lorentzian_with_the_right_knee():
    b = _budget()
    T, tauA = 0.3 * b.recipe.Tc, 1e-6
    fk = 1.0 / (2 * np.pi * tauA)
    Sy, _ = b.freq_noise_spectrum(T, tauA, np.array([0.0, fk, 10 * fk]))
    # exactly half the plateau at the knee; -20 dB/decade beyond
    assert abs(Sy[1] / Sy[0] - 0.5) < 1e-12
    assert abs(Sy[2] / Sy[0] - 1.0 / 101.0) < 1e-12


def test_participation_and_inductance_are_physical():
    b = _budget()
    T = 0.3 * b.recipe.Tc
    assert b.LJ(T) > 0.0
    assert 0.0 < b.participation(T) < 1.0


def test_phonon_resolution_closed_form():
    b = _budget()
    T, t = 0.2, 1.0
    from absnoise.constants import KB
    expect = np.sqrt(2 * KB * T**2 * b.tau_th(T) / (b.Ce(T) * t))
    assert abs(b.dT_phonon(T, t) - expect) < 1e-18


def test_energy_resolution_finite_and_bounded_below_by_andreev_only():
    # the full matched filter faces Andreev + phonon noise, so it can
    # never resolve better than the Andreev-only closed form
    b = _budget()
    T, tauA = 0.3 * b.recipe.Tc, 1e-6
    full = b.energy_resolution(T, tauA)
    andreev_only = b.energy_resolution_analytic_A(T, tauA)
    assert np.isfinite(full) and full > 0.0
    assert full >= 0.99 * andreev_only


def test_andreev_bound_relation():
    # achieved resolution respects the Cauchy-Schwarz bound
    b = _budget()
    T, tauA, t = 0.3 * b.recipe.Tc, 1e-6, 1.0
    achieved, s = b.dT_andreev(T, tauA, t)
    from absnoise.constants import KB
    bound = np.sqrt(2 * KB * T**2 * tauA / (s["C_A"] * t))
    assert achieved >= bound - 1e-18
