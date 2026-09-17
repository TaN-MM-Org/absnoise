"""Measurement-planning anchors: the planned error bars match
`fit_ic_curve`'s reported error bars on the same design (two code
paths of one matrix, through the package's own `ic_model`); seeded
Monte Carlo matches the planned sigmas; the low-temperature leverage
warning of the fit reappears in the plan as `covers_tc_knee`; the
greedy temperature design reproduces its own rule and never loses to
a random subset; and the PSD band planner emits exactly the bands
`fit_telegraph_psd` accepts, while a band violating its rules is
refused by the fit -- cross-validated in both directions."""
import numpy as np
import pytest

from absnoise import (RECIPES, design_ic_temperatures, fit_ic_curve,
                      fit_telegraph_psd, ic_model, plan_ic_measurement,
                      psd_band_for_tau, telegraph_psd_model)
from absnoise.icfit import _shape_zero
from absnoise.constants import E_CHARGE, HBAR

REC = RECIPES[0]
TAU = REC.tau
TC = REC.Tc


def _ic0():
    from absnoise import ShortJunction
    sj = ShortJunction(REC, n_phi=601)
    return sj.Nch * 2.0 * E_CHARGE / HBAR * _shape_zero(TC, TAU)


def test_plan_matches_fit_covariance():
    """Noise-free Ic(T) from the package's own model: the fit recovers
    the truth, and its sigmas equal the plan's prediction."""
    ic0 = _ic0()
    T = np.linspace(0.05, 0.85, 9) * TC
    ic = np.array([float(ic_model(t, ic0, TC, TAU, n_phi=601))
                   for t in T])
    sig = 0.01 * ic.max()
    fit = fit_ic_curve(T, ic, sigma_Ic=sig, n_phi=601)
    assert abs(fit.Ic0 - ic0) < 1e-4 * ic0
    plan = plan_ic_measurement(T, sig, ic0, TC, TAU, n_phi=601)
    assert plan["identifiable"] and plan["covers_tc_knee"]
    for name in ("Ic0", "Tc", "tau"):
        assert abs(plan["sigma"][name] - fit.sigma[name]) \
            < 0.05 * fit.sigma[name]


def test_monte_carlo_matches_planned_sigma():
    """250 seeded synthetic cooldowns: the scatter of the fitted Tc
    must match the planned error bar."""
    ic0 = _ic0()
    T = np.linspace(0.1, 0.8, 8) * TC
    ic = np.array([float(ic_model(t, ic0, TC, TAU, n_phi=301))
                   for t in T])
    sig = 0.02 * ic.max()
    plan = plan_ic_measurement(T, sig, ic0, TC, TAU, n_phi=301)
    rng = np.random.default_rng(31)
    tcs = []
    for _ in range(250):
        noisy = ic + sig * rng.standard_normal(ic.size)
        fit = fit_ic_curve(T, np.abs(noisy), sigma_Ic=sig, n_phi=301)
        tcs.append(fit.Tc)
    emp = np.std(tcs, ddof=1)
    assert np.isclose(emp, plan["sigma"]["Tc"], rtol=0.2)


def test_low_temperature_leverage_flagged():
    """Below 0.3 Tc the curve barely feels Tc -- the same situation
    `fit_ic_curve` warns about is flagged in the plan, and the planned
    Tc error bar there is much worse than with the knee covered."""
    ic0 = _ic0()
    T_low = np.linspace(0.05, 0.25, 8) * TC
    T_full = np.linspace(0.05, 0.85, 8) * TC
    sig = 0.01 * ic0
    low = plan_ic_measurement(T_low, sig, ic0, TC, TAU, n_phi=301)
    full = plan_ic_measurement(T_full, sig, ic0, TC, TAU, n_phi=301)
    assert not low["covers_tc_knee"]
    assert full["covers_tc_knee"]
    if low["sigma"] is not None:
        assert low["sigma"]["Tc"] > 10.0 * full["sigma"]["Tc"]


def test_design_greedy_invariant_and_quality():
    ic0 = _ic0()
    cand = np.linspace(0.05, 0.9, 12) * TC
    sig = 0.01 * ic0
    out = design_ic_temperatures(cand, 5, sig, ic0, TC, TAU, n_phi=301)
    idx = out["indices"]
    assert len(idx) == 5 and len(set(idx)) == 5
    assert out["sigma"] is not None

    def logdet_of(subset):
        plan = plan_ic_measurement(cand[list(subset)], sig, ic0, TC,
                                   TAU, n_phi=301)
        from absnoise.lab import _jacobian
        j = _jacobian(cand[list(subset)], ic0, TC, TAU, 301) / sig
        s, d = np.linalg.slogdet(j.T @ j)
        return d if s > 0 else -np.inf

    best = logdet_of(idx)
    rng = np.random.default_rng(3)
    for _ in range(20):
        assert best >= logdet_of(rng.choice(12, 5, replace=False)) \
            - 1e-9


def test_psd_band_cross_validated_with_fit():
    """A band the planner emits must be accepted by the fit (with the
    generating tau recovered); an all-below-the-knee band must be
    refused by the fit -- the two tools enforce one rule."""
    tau, s0, floor = 3e-4, 1e-3, 1e-6
    f_lo, f_hi, f_c = psd_band_for_tau(tau)
    assert abs(f_c - 1.0 / (2.0 * np.pi * tau)) < 1e-12 * f_c
    f = np.geomspace(f_lo, f_hi, 40)
    fit = fit_telegraph_psd(f, telegraph_psd_model(f, s0, tau, floor))
    assert abs(fit.tau_s - tau) < 1e-4 * tau
    assert abs(fit.S0 - s0) < 1e-4 * s0
    # a band that never sees the knee: refused by the fit
    f_bad = np.geomspace(f_c * 20.0, f_c * 40.0, 40)
    with pytest.raises(ValueError):
        fit_telegraph_psd(f_bad, telegraph_psd_model(f_bad, s0, tau,
                                                     floor))
    with pytest.raises(ValueError, match="margin"):
        psd_band_for_tau(tau, margin=2.0)


def test_input_refusals():
    ic0 = _ic0()
    with pytest.raises(ValueError, match="tau"):
        plan_ic_measurement([0.1], 1e-8, ic0, TC, 1.5)
    with pytest.raises(ValueError, match="positive"):
        plan_ic_measurement([-0.1], 1e-8, ic0, TC, TAU)
    with pytest.raises(ValueError, match="n_pick"):
        design_ic_temperatures([0.1, 0.2, 0.3], 2, 1e-8, ic0, TC, TAU)
    with pytest.raises(ValueError, match="positive"):
        psd_band_for_tau(-1.0)
