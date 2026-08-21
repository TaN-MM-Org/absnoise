"""Matched-level design, click dynamics, and self-heating anchors."""
import numpy as np
import pytest

from absnoise import (SensorBudget, click_monte_carlo, click_template,
                      ep_power, gap_bcs, matched_Tc, matched_recipe,
                      steady_temperature)
from absnoise.constants import H_PLANCK, KB


def _matched_budget(T0=0.05):
    r = matched_recipe(T0, W=5.3e-6, L=0.5e-6)
    return SensorBudget(r)


def test_matched_condition_is_exact():
    # Delta*(T0) from the solved gap equation equals 2.3994 kB T0
    for T0 in (0.05, 0.1):
        Tc = matched_Tc(T0)
        D = gap_bcs(T0, Tc, 1.7639 * KB * Tc)
        assert abs(D / (KB * T0) - 2.3994) < 1e-9


def test_energy_conservation_of_the_exponential_integrator():
    # the exact peak-temperature identity (1/2) gamma (Tpk^2 - T0^2)
    # = E_gamma, to machine precision (the testbench criterion of the
    # source repository)
    T0 = 0.05
    b = _matched_budget(T0)
    E = H_PLANCK * 26e9
    tpl = click_template(b, E, tauA=1e-7, T0=T0)
    assert tpl["energy_residual"] < 1e-12


def test_template_dips_and_recovers():
    T0 = 0.05
    b = _matched_budget(T0)
    E = H_PLANCK * 26e9
    for scen in ("C", "T"):
        tpl = click_template(b, E, tauA=1e-7, T0=T0, scenario=scen)
        ms, m0 = tpl["ms"], tpl["m0"]
        assert ms.min() < m0                     # occupation dips
        assert abs(ms[-1] - m0) < 5e-3 * m0      # and recovers
        assert tpl["Te_peak"] > T0


def test_click_monte_carlo_detects_photons():
    # statistics-limited ordering relations, not tight values
    T0 = 0.05
    b = _matched_budget(T0)
    E = H_PLANCK * 26e9
    out = click_monte_carlo(b, E, tauA=1e-7, T0=T0, scenario="C",
                            n_trials=200, seed=3)
    assert out["snr"] > 1.0
    assert out["efficiency"] > out["dark_fraction"]
    assert 0.0 < out["m_dip"] < 1.0
    assert out["energy_residual"] < 1e-12


def test_click_template_rejects_unknown_scenario():
    b = _matched_budget()
    with pytest.raises(ValueError):
        click_template(b, 1e-23, tauA=1e-7, T0=0.05, scenario="X")


def test_selfheat_steady_state_closed_form():
    # exact round trip: ep_power(steady_temperature(P)) = P
    area, Tp = 1e-12, 0.05
    for P in (0.0, 1e-18, 1e-15, 1e-13):
        Te = steady_temperature(P, Tp, area)
        back = float(ep_power(Te, Tp, area))
        # exact up to floating point: the absolute floor 1e-30 W is the
        # roundoff of the T^3 cancellation at these scales
        assert abs(back - P) <= max(1e-12 * P, 1e-30)
    assert abs(steady_temperature(0.0, Tp, area) - Tp) < 1e-15
    # monotone in power
    Ps = np.logspace(-18, -13, 8)
    Tes = steady_temperature(Ps, Tp, area)
    assert np.all(np.diff(Tes) > 0)
    with pytest.raises(ValueError):
        steady_temperature(-1e-15, Tp, area)
