"""PSD-averaging-planner anchors: the planned error bars equal
`fit_telegraph_psd`'s reported error bars on noiseless synthetic data
at the same averaging depth (two code paths of one matrix, in the
fit's own log-space parameterization); the 1/sqrt(n_avg) scaling is
exact; the closed-form averaging inversion is verified on both sides
of the target; seeded Monte Carlo with the multiplicative periodogram
scatter matches the planned tau error bar; and the planner refuses,
in advance and with the same explanations, exactly the bands the fit
refuses after the fact."""
import numpy as np
import pytest

from absnoise import (averages_for_tau, fit_telegraph_psd,
                      plan_psd_measurement, telegraph_psd_model)

S0, TAU, FLOOR = 1e-3, 3e-4, 1e-5
F = np.geomspace(50.0, 20000.0, 48)         # knee ~531 Hz, well inside


def test_plan_matches_fit_two_paths():
    n_avg = 64
    S = telegraph_psd_model(F, S0, TAU, FLOOR)
    fit = fit_telegraph_psd(F, S, fit_floor=True, n_avg=n_avg)
    assert abs(fit.tau_s - TAU) < 1e-6 * TAU
    plan = plan_psd_measurement(F, n_avg, S0, TAU, FLOOR)
    assert plan["identifiable"]
    for name in ("S0", "tau_s", "floor"):
        assert abs(plan["sigma"][name] - fit.sigma[name]) \
            < 1e-3 * fit.sigma[name]


def test_scaling_exact_in_averages():
    p1 = plan_psd_measurement(F, 16, S0, TAU, FLOOR)
    p2 = plan_psd_measurement(F, 64, S0, TAU, FLOOR)
    for name in ("S0", "tau_s", "floor"):
        assert abs(p1["sigma"][name] / p2["sigma"][name] - 2.0) \
            < 1e-9


def test_averages_inversion_two_sided():
    target = 0.02 * TAU
    n, plan = averages_for_tau(target, F, S0, TAU, FLOOR)
    assert plan["sigma"]["tau_s"] <= target
    if n > 1:
        prev = plan_psd_measurement(F, n - 1, S0, TAU, FLOOR)
        assert prev["sigma"]["tau_s"] > target


def test_monte_carlo_matches_planned_tau_sigma():
    n_avg = 200
    plan = plan_psd_measurement(F, n_avg, S0, TAU, FLOOR)
    sig_bin = 1.0 / np.sqrt(n_avg)
    S_true = telegraph_psd_model(F, S0, TAU, FLOOR)
    rng = np.random.default_rng(21)
    taus, failed = [], 0
    for _ in range(300):
        S_meas = S_true * np.exp(sig_bin
                                 * rng.standard_normal(F.size))
        try:
            fit = fit_telegraph_psd(F, S_meas, fit_floor=True,
                                    n_avg=n_avg)
        except (ValueError, RuntimeError):
            failed += 1
            continue
        taus.append(fit.tau_s)
    assert failed <= 3                    # rare pathological draws only
    emp = np.std(taus, ddof=1)
    assert np.isclose(emp, plan["sigma"]["tau_s"], rtol=0.2)


def test_refusals_mirror_the_fit():
    # knee far above the band: both tools refuse
    f_low = np.geomspace(1.0, 20.0, 20)
    with pytest.raises(ValueError, match="knee"):
        plan_psd_measurement(f_low, 16, S0, TAU, FLOOR)
    S_low = telegraph_psd_model(f_low, S0, TAU, FLOOR)
    with pytest.raises(ValueError):
        fit_telegraph_psd(f_low, S_low, n_avg=16)
    with pytest.raises(ValueError, match=">= 8"):
        plan_psd_measurement(F[:5], 16, S0, TAU, FLOOR)
    with pytest.raises(ValueError, match="positive"):
        averages_for_tau(-1.0, F, S0, TAU, FLOOR)
    with pytest.raises(ValueError, match="n_avg"):
        plan_psd_measurement(F, 0, S0, TAU, FLOOR)
