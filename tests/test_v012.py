"""0.12.0 changes, each checked against an independent reference.

Fixes (each test failed on 0.11.0):

* `gap_bcs` near Tc: against direct solutions of the gap equation in
  two forms (the integral form of `_solve_bcs_u` and the Matsubara sum)
  and against the Ginzburg-Landau limit derived from the sum;
* `ic_model` and `ShortJunction.Ic` at and above Tc, and the planner
  with a planned temperature above Tc (information is additive);
* `avar_exponential` for T << tau, against the defining integral
  rewritten without cancellation and against its Taylor series;
* the sign of `continuum_free_energy`, against continuity of the total
  free energy where a bound state leaves for the continuum and against
  the `JunctionModel` code path;
* `plan_psd_measurement` with a non-integer n_avg, against the fit.

New features:

* `IcFit.propagate`, against the exact linear result, a quadratic
  closed form and Monte Carlo sampling of the fit covariance;
* `dwell_times` and `markov_dwell_test`, against hand-counted paths,
  the geometric-law identity, p-value calibration on Markov traces and
  detection of a two-time-scale (non-Markov) trace.
"""
import dataclasses

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma, zeta

from absnoise import (RECIPES, SensorBudget, ShortJunction, dwell_times,
                      fit_hmm, fit_ic_curve, fit_telegraph_psd, gap_bcs,
                      ic_model, markov_dwell_test, plan_ic_measurement,
                      plan_psd_measurement, telegraph_psd_model,
                      telegraph_traces, avar_exponential)
from absnoise.constants import KB
from absnoise.decode import _dispersion, _simulate_path
from absnoise.levels import (_BCS_A, _T1, _solve_bcs_u,
                             _solve_bcs_u_matsubara, abs_energies,
                             continuum_delta, continuum_free_energy,
                             continuum_phase_grid)
from absnoise._compat import trapezoid


# ---- BCS gap near Tc ---------------------------------------------------

def test_gap_near_tc_matches_direct_solutions_of_the_gap_equation():
    # 0.11.0: 0 for 0.9995 <= t < 1, -0.67 % at t = 0.999 (spline)
    for t in (0.99, 0.992, 0.995, 0.998, 0.999, 0.9994, 0.9995, 0.9997,
              0.9998):
        ref = _solve_bcs_u(t)                 # integral form, quad
        assert abs(gap_bcs(t, 1.0, 1.0) / ref - 1.0) < 1e-9
    for t in (0.9999, 0.99999, 0.999999):
        ref = _solve_bcs_u_matsubara(t)       # integral form gives up
        assert abs(gap_bcs(t, 1.0, 1.0) / ref - 1.0) < 1e-9
    # the old spline's errors, and the zero region, are gone
    assert gap_bcs(0.9996 * 0.75, 0.75, 1.0) > 0.03


def test_two_forms_of_the_gap_equation_agree():
    # integral form (quad over xi) and Matsubara-sum form: two codes
    for t in (0.1, 0.5, 0.9, 0.99, 0.999):
        assert abs(_solve_bcs_u_matsubara(t) / _solve_bcs_u(t) - 1) < 1e-9


def test_gap_near_tc_ginzburg_landau_limit():
    # small-y expansion of the Matsubara sum:
    # 2 sum [1/w - 1/sqrt(w^2 + y^2)] = y^2 sum 1/w^3 + O(y^4)
    #                                = (7 zeta(3) / 8) y^2 + O(y^4)
    # so u -> (pi t / A) sqrt(8 ln(t1/t) / (7 zeta(3))) as t -> t1
    for t in (0.99999, 0.999999):
        u_gl = (np.pi * t / _BCS_A) * np.sqrt(8.0 * np.log(_T1 / t)
                                              / (7.0 * zeta(3)))
        u = gap_bcs(t, 1.0, 1.0)
        assert abs(u / u_gl - 1.0) < 2e-5
    # the closing temperature t1 = A e^gamma / pi of this equation
    assert abs(_T1 - 1.7639 * np.exp(np.euler_gamma) / np.pi) < 1e-15
    assert gap_bcs(1.0, 1.0, 1.0) == 0.0
    assert gap_bcs(1.2, 1.0, 1.0) == 0.0


def test_gap_below_099_tc_is_unchanged_and_continuous():
    # the main table is untouched below t = 0.99 (values of 0.11.0;
    # 1e-12 allows for rounding differences between SciPy versions)
    assert gap_bcs(0.5, 1.0, 1.0) == pytest.approx(0.9568871998163413,
                                                   rel=1e-12)
    assert gap_bcs(0.95, 1.0, 1.0) == pytest.approx(0.38036207931323796,
                                                    rel=1e-12)
    lo = gap_bcs(0.99 - 1e-12, 1.0, 1.0)
    hi = gap_bcs(0.99, 1.0, 1.0)
    assert abs(hi / lo - 1.0) < 1e-6
    ts = np.linspace(0.95, 0.99999, 400)
    assert np.all(np.diff(gap_bcs(ts, 1.0, 1.0)) < 0.0)


def test_budget_accepts_temperatures_just_below_tc():
    b = SensorBudget(RECIPES[1])
    T = 0.9996 * b.recipe.Tc          # 0.11.0: refused as "at or above Tc"
    dT, _ = b.dT_andreev(T, 1e-6, 1.0)
    assert np.isfinite(dT) and dT > 0.0
    with pytest.raises(ValueError):
        b.dT_andreev(b.recipe.Tc, 1e-6, 1.0)


# ---- Ic at and above Tc ------------------------------------------------

def test_ic_is_zero_at_and_above_tc():
    out = ic_model(np.array([0.5, 0.75, 0.8, 2.0]), 2e-6, 0.75, 0.78)
    assert np.all(np.isfinite(out))          # 0.11.0: NaN above Tc
    assert out[0] > 0.0 and np.all(out[1:] == 0.0)
    assert ic_model(0.8, 2e-6, 0.75, 0.78) == 0.0
    sj = ShortJunction(RECIPES[1])
    sj.calibrate()
    assert sj.Ic(0.8) == 0.0                 # 0.11.0: NaN
    # just below Tc: the two implementations still agree
    T = 0.9997 * 0.75
    assert ic_model(T, 1.0, 0.75, 0.78) > 0.0


def test_plan_ignores_a_planned_point_above_tc():
    # a point above Tc carries no information (Ic = 0 whatever the
    # parameters), so the plan must equal the plan without it. 0.11.0:
    # NaN sensitivities made the whole plan "not identifiable".
    T = np.linspace(0.05, 0.65, 8)
    a = plan_ic_measurement(T, 0.02e-6, 2e-6, 0.75, 0.78)
    b = plan_ic_measurement(np.append(T, 0.8), 0.02e-6, 2e-6, 0.75, 0.78)
    assert a["identifiable"] and b["identifiable"]
    for k in ("Ic0", "Tc", "tau"):
        assert abs(b["sigma"][k] / a["sigma"][k] - 1.0) < 1e-12


# ---- Allan variance for T << tau ---------------------------------------

def _avar_reference(var, tau, T):
    """The defining integral AVAR = (1/T^2) int_0^2T k(w) C(w) dw with
    k = 2T - 3w (w <= T), -(2T - w) (T < w <= 2T), C = var e^(-w/tau).
    int k = 0 and int k w = -2T^3/3, so C can be replaced by
    var (e^(-x) - 1 + x) - var x, x = w/tau, which removes the
    cancellation: AVAR = 2 var T / (3 tau) + (var/T^2) int k g,
    g = e^(-x) - 1 + x."""
    def g(w):
        # e^-x - 1 + x; its Taylor series below x = 1e-3, where
        # expm1(-x) + x would lose digits
        x = w / tau
        if x < 1e-3:
            return x * x * (0.5 - x / 6 + x * x / 24 - x ** 3 / 120)
        return np.expm1(-x) + x

    a, _ = quad(lambda w: (2 * T - 3 * w) * g(w), 0.0, T,
                epsabs=0.0, epsrel=1e-12)
    b, _ = quad(lambda w: -(2 * T - w) * g(w), T, 2 * T,
                epsabs=0.0, epsrel=1e-12)
    return 2.0 * var * T / (3.0 * tau) + var * (a + b) / T ** 2


@pytest.mark.parametrize("r", [1e-8, 1e-6, 1e-4, 1e-2, 0.3, 0.7, 3.0])
def test_avar_exponential_against_the_defining_integral(r):
    # 0.11.0 at r = 1e-6: -1.1e-4 instead of +6.7e-7 (cancellation)
    var, tau = 0.21, 1.7
    ref = _avar_reference(var, tau, r * tau)
    assert avar_exponential(var, tau, r * tau) == pytest.approx(
        ref, rel=1e-9)


def test_avar_exponential_taylor_series_and_arrays():
    # bracket/tau = sum_{k>=3} (-1)^k (4 - 2^k) r^k / k!
    #             = 2r^3/3 - r^4/2 + 7r^5/30 - r^6/12 + ...
    r = 1e-3
    ser = 2 * r / 3 - r ** 2 / 2 + 7 * r ** 3 / 30 - r ** 4 / 12
    assert avar_exponential(1.0, 1.0, r) == pytest.approx(ser, rel=1e-12)
    T = np.array([1e-7, 0.1, 0.4999, 0.5, 2.0, 40.0])
    out = avar_exponential(0.3, 1.0, T)
    assert out.shape == T.shape and np.all(out > 0.0)
    for Ti, oi in zip(T, out):
        assert oi == avar_exponential(0.3, 1.0, float(Ti))
    # continuous where the evaluation switches (r = 0.5)
    lo = avar_exponential(1.0, 1.0, 0.5 * (1 - 1e-12))
    assert abs(lo / avar_exponential(1.0, 1.0, 0.5) - 1) < 1e-11
    with pytest.raises(ValueError):
        avar_exponential(np.nan, 1.0, 1.0)


# ---- sign of continuum_free_energy -------------------------------------

def _bound_free_energy(phi, tau, c, D, T):
    E = abs_energies(phi, tau, c, D)
    x = E / (2 * KB * T)
    return -np.sum(2 * KB * T * (x + np.log1p(np.exp(-2 * x))))


def test_continuum_free_energy_keeps_the_total_continuous():
    # c D = 1.5, tau = 0.7: the upper bound level reaches the gap edge
    # where arccos(1 - tau + tau cos phi) = 1.5 and leaves the spectrum
    D, tau = 1.5e-23, 0.7
    c = 1.5 / D
    T = 0.3 * D / (1.764 * KB)
    phi_x = np.arccos((np.cos(1.5) - 1 + tau) / tau)
    d = 0.03
    nb = [len(abs_energies(phi_x + s * d, tau, c, D)) for s in (-1, 1)]
    assert nb == [2, 1]
    tot = [_bound_free_energy(phi_x + s * d, tau, c, D, T)
           + continuum_free_energy(phi_x + s * d, tau, c, D, T)
           for s in (-1, 1)]
    jump_b = (_bound_free_energy(phi_x + d, tau, c, D, T)
              - _bound_free_energy(phi_x - d, tau, c, D, T))
    # the bound part jumps by the free energy of the departing level;
    # the total must not (0.11.0: it jumped by 2 jump_b)
    assert jump_b > 0.9 * 2 * KB * T * np.log(2 * np.cosh(
        D / (2 * KB * T)))
    assert abs(tot[1] - tot[0]) < 0.02 * jump_b


def test_continuum_free_energy_matches_the_junction_model_path():
    # free_energy_multi / free_energy_components evaluate the same
    # expression on another energy grid (from 1e-13 Delta above the gap)
    D, tau = 1.5e-23, 0.7
    c = 1.5 / D
    T = 0.3 * D / (1.764 * KB)
    Efac = continuum_phase_grid(3000)
    E = D * Efac
    xD = D / (2 * KB * T)
    l2 = 2 * KB * T * (xD + np.log1p(np.exp(-2 * xD)))
    for phi in (0.5, 1.0, 2.5):
        dph = continuum_delta(phi, tau, c, D, Efac)
        ref = l2 * dph[0] / np.pi + trapezoid(
            np.tanh(E / (2 * KB * T)) * dph, E) / np.pi
        val = continuum_free_energy(phi, tau, c, D, T)
        assert np.sign(val) == np.sign(ref)
        assert abs(val / ref - 1.0) < 5e-4


# ---- planner with non-integer n_avg ------------------------------------

def test_psd_planner_honors_a_non_integer_n_avg():
    # 0.11.0 truncated n_avg = 1.5 to 1: 33 % larger planned error bars
    # than the fit reports at n_avg = 1.5
    S0, tau, floor = 1e-3, 3e-4, 1e-5
    f = np.geomspace(50.0, 20000.0, 48)
    for n in (1.5, 2.5, 7.3):
        S = telegraph_psd_model(f, S0, tau, floor) \
            * np.exp(digamma(n) - np.log(n))
        fit = fit_telegraph_psd(f, S, n_avg=n)
        plan = plan_psd_measurement(f, n, S0, tau, floor)
        for k in ("S0", "tau_s", "floor"):
            assert abs(plan["sigma"][k] / fit.sigma[k] - 1.0) < 1e-3
    with pytest.raises(ValueError):
        plan_psd_measurement(f, 0.9, S0, tau, floor)
    with pytest.raises(ValueError):
        plan_psd_measurement(f, np.inf, S0, tau, floor)


# ---- click_template input checks ---------------------------------------

def test_click_template_input_checks():
    from absnoise import click_template, matched_recipe
    b = SensorBudget(matched_recipe(0.05, W=5.3e-6, L=0.5e-6))
    with pytest.raises(ValueError):
        click_template(b, -1e-23, 1e-7, 0.05)
    with pytest.raises(ValueError):
        click_template(b, 1e-23, 0.0, 0.05)
    out = click_template(b, 0.0, 1e-7, 0.05, t_end=2e-9, t_end2=4e-9)
    assert out["energy_residual"] == 0.0           # 0.11.0: ZeroDivision
    assert np.all(out["ms"] == out["m0"])


# ---- IcFit.propagate ---------------------------------------------------

GEO = dict(xi=5e-6, L=0.2e-6, W=2e-6, Vbg=30.0, Rn=50.0)


@pytest.fixture(scope="module")
def example_fit():
    # README Example 3
    T = np.linspace(0.05, 0.65, 10)
    rng = np.random.default_rng(1)
    sigma = 0.02e-6
    Ic = ic_model(T, 2.0e-6, 0.75, 0.78) + rng.normal(0.0, sigma, T.size)
    return fit_ic_curve(T, Ic, sigma_Ic=sigma)


def test_propagate_is_exact_for_linear_functions(example_fit):
    fit = example_fit
    out = fit.propagate(lambda r: 3.0 * r.Tc - 2.0 * r.tau, **GEO)
    g = np.array([0.0, 3.0, -2.0])
    ref = np.sqrt(g @ fit.cov @ g)
    assert abs(out["sigma"] / ref - 1.0) < 1e-10
    assert out["value"] == pytest.approx(3 * fit.Tc - 2 * fit.tau,
                                         rel=1e-14)
    assert out["nonlinearity"] < 1e-6
    # vector output: the fit's own error bars come back, with the
    # covariance block of (Tc, tau)
    v = fit.propagate(lambda r: np.array([r.Tc, r.tau]), **GEO)
    assert np.allclose(v["sigma"], [fit.sigma["Tc"], fit.sigma["tau"]],
                       rtol=1e-10)
    assert np.allclose(v["cov"], fit.cov[1:, 1:], rtol=1e-9)


def test_propagate_quadratic_closed_form(example_fit):
    fit = example_fit
    out = fit.propagate(lambda r: r.Tc ** 2, **GEO)
    # first order: sigma(Tc^2) = 2 Tc sigma(Tc), exactly for sigma points
    assert abs(out["sigma"] / (2 * fit.Tc * fit.sigma["Tc"]) - 1) < 1e-9
    # second-order term along axis k is d_k[Tc]^2 <= sigma(Tc)^2, and
    # the largest of the three is at least a third of their sum
    hi = fit.sigma["Tc"] / (2 * fit.Tc)
    assert hi / 3.0 * 0.999 <= out["nonlinearity"] <= hi * 1.001


def test_propagate_matches_monte_carlo(example_fit):
    fit = example_fit

    def lj(r):
        return SensorBudget(r).LJ(0.2)

    out = fit.propagate(lj, **GEO)
    rng = np.random.default_rng(7)
    P = rng.multivariate_normal([fit.Ic0, fit.Tc, fit.tau], fit.cov, 400)
    vals = [lj(dataclasses.replace(fit, Ic0=a, Tc=b, tau=c)
               .to_recipe("mc", "mc", **GEO)) for a, b, c in P]
    # standard error of a 400-sample standard deviation: 3.5 %
    assert abs(np.std(vals, ddof=1) / out["sigma"] - 1.0) < 0.12
    assert out["value"] == pytest.approx(lj(fit.to_recipe("f", "f", **GEO)),
                                         rel=1e-14)


def test_propagate_refusals(example_fit):
    fit = example_fit
    with pytest.raises(ValueError, match="nonlinear"):
        fit.propagate(lambda r: 1.0 / (r.Tc - 0.74), **GEO)
    near_one = dataclasses.replace(fit, tau=0.99)
    with pytest.raises(ValueError, match="range"):
        near_one.propagate(lambda r: r.tau, **GEO)
    with pytest.raises(ValueError, match="non-finite"):
        fit.propagate(lambda r: np.nan, **GEO)
    with pytest.raises(TypeError):
        fit.propagate(3.0, **GEO)


# ---- dwell times and the Markov check ----------------------------------

def test_dwell_times_hand_counted():
    path = [1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0]
    d = dwell_times(path, dt=0.5)
    # first stay (1,1) and last stay (0) are cut by the record
    assert list(d[0]) == [1.5, 1.0]
    assert list(d[1]) == [0.5, 2.0]
    d = dwell_times(np.array([0, 0, 0]))
    assert d[0].size == 0 and d[1].size == 0
    with pytest.raises(ValueError):
        dwell_times([0, 2, 1])
    with pytest.raises(ValueError):
        dwell_times([], 1.0)


def test_dispersion_index_of_a_geometric_law_is_one():
    rng = np.random.default_rng(2)
    for p in (0.02, 0.2, 0.5):
        k = rng.geometric(p, 200_000)
        assert abs(_dispersion(k) - 1.0) < 0.02


def test_simulated_reference_paths_have_the_model_statistics():
    # stationary occupation p_up/(p_up+p_dn) and flip rate
    # 2 p_up p_dn/(p_up+p_dn) per sample, both closed forms
    rng = np.random.default_rng(3)
    path = _simulate_path(0.02, 0.05, 400_000, rng)
    assert path.size == 400_000
    assert abs(path.mean() - 0.02 / 0.07) < 0.01
    flips = np.mean(path[1:] != path[:-1])
    assert abs(flips / (2 * 0.02 * 0.05 / 0.07) - 1) < 0.03


def test_markov_traces_pass_with_calibrated_p_values():
    # telegraph_traces is an exact Markov process: p-values should be
    # spread over [0, 1] (mean 1/2), with no false alarm at 0.01
    ps = []
    for seed in range(12):
        rng = np.random.default_rng(100 + seed)
        n = telegraph_traces(0.3, 20.0, 1.0, 8000, 1, rng)[:, 0]
        y = n + rng.normal(0.0, 0.25, n.size)
        model, _ = fit_hmm(y)
        res = markov_dwell_test(y, model, n_boot=99, seed=seed)
        assert res["n_dwells"][0] >= 30 and res["n_dwells"][1] >= 30
        ps.append(res["p_value"])
    ps = np.array(ps)
    assert np.all(ps > 0.01)
    assert 0.3 < ps.mean() < 0.75


def test_two_time_scale_trace_is_detected():
    # state 1 lasts 5 samples or 60 samples on average (equal odds):
    # over-dispersed stays, which no single rate produces
    rng = np.random.default_rng(0)
    parts, s, tot = [], 0, 0
    while tot < 8000:
        if s == 0:
            L = rng.geometric(1 / 40)
        else:
            L = rng.geometric(1 / 5) if rng.random() < 0.5 \
                else rng.geometric(1 / 60)
        parts.append(np.full(L, s, dtype=np.int8))
        tot += L
        s ^= 1
    n = np.concatenate(parts)[:8000]
    y = n + rng.normal(0.0, 0.25, n.size)
    model, _ = fit_hmm(y)
    res = markov_dwell_test(y, model, n_boot=399, seed=1)
    assert res["dispersion"][1] > 1.4
    assert res["p_state"][1] <= 0.005
    assert res["p_value"] <= 0.01


def test_markov_dwell_test_refusals():
    rng = np.random.default_rng(5)
    n = telegraph_traces(0.3, 20.0, 1.0, 600, 1, rng)[:, 0]
    y = n + rng.normal(0.0, 0.2, n.size)
    model, _ = fit_hmm(y)
    with pytest.raises(ValueError, match="complete stays"):
        markov_dwell_test(y, model, n_boot=19)      # ~15 stays per state
    # tau = 0.2 samples: stays in state 1 last 1.44 samples on average
    n = telegraph_traces(0.3, 0.2, 1.0, 4000, 1, rng)[:, 0]
    y = n + rng.normal(0.0, 0.05, n.size)
    model, _ = fit_hmm(y)
    with pytest.raises(ValueError, match="coarse"):
        markov_dwell_test(y, model, n_boot=19)
    with pytest.raises(TypeError):
        markov_dwell_test(y, None)
    with pytest.raises(ValueError):
        markov_dwell_test(y, model, n_boot=5)
