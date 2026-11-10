"""0.16.0 changes, each checked against an independent reference.

New:

* `telegraph_psd_model(..., dt)`, `fit_telegraph_psd(..., dt)` and the
  planner with `dt`: the spectrum of a sampled exponentially correlated
  process (aliasing included), against the transform of the sampled
  autocovariance, the explicit sum over the aliases of the Lorentzian,
  Parseval, the continuous limit, noise-free fits, telegraph Monte
  Carlo and the fit's own error bars;
* `avar_exponential(..., dt)`: the Allan variance of sampled data,
  against the defining double sum over the autocovariance in 50-digit
  arithmetic, its one-sample and white limits, the continuous limit
  and telegraph Monte Carlo.

Fixes (each test fails on 0.15.0):

* `continuum_share` and `occupation_heat_capacities` at a phase that
  is not on their phase grid (they used the nearest grid phase);
* `load_ic_csv` and `load_trace_csv` with nan or inf in the file;
* `ShortJunction.andreev_sums` and `temperature_bound` with an unknown
  `which` or a temperature outside (0, Tc);
* `two_temperature_click` with a non-positive tauA or non-finite
  inputs;
* `psd_single_sided`, `allan_variance` and `TelegraphHMM.rates` with a
  sample interval that is not finite and positive.
"""
import dataclasses
import os
from decimal import Decimal, localcontext

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma, polygamma

from absnoise import (RECIPES, FiniteLJunction, JunctionModel, SensorBudget,
                      ShortJunction, TelegraphHMM, allan_variance,
                      averages_for_tau, avar_exponential, continuum_share,
                      fit_telegraph_psd, gap_bcs, load_ic_csv,
                      load_trace_csv, occupation_heat_capacities,
                      plan_psd_measurement, psd_single_sided,
                      telegraph_psd_model, telegraph_traces,
                      two_temperature_click)
from absnoise.constants import E_CHARGE, HBAR
from absnoise.levels import free_energy_components

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- the spectrum of a sampled process --------------------------------------

def _wiener_khinchin(f, S0, tau, dt):
    """Single-sided spectrum of the samples, from the sampled
    autocovariance var r^|k| summed term by term (no closed form):
    2 dt var [1 + 2 sum_{k>=1} r^k cos(2 pi f k dt)], up to r^k < 1e-19."""
    var = S0 / (4.0 * tau)
    eps = dt / tau
    k = np.arange(1, int(np.ceil(44.0 / eps)) + 1, dtype=float)
    rk = np.exp(-eps * k)
    f = np.atleast_1d(np.asarray(f, dtype=float))
    out = np.empty(f.size)
    for i, fi in enumerate(f):
        out[i] = 2.0 * dt * var * (1.0 + 2.0 * np.sum(
            rk * np.cos(2.0 * np.pi * fi * dt * k)))
    return out


@pytest.mark.parametrize("ratio", [0.01, 0.25, 1.0, 4.0])
def test_sampled_spectrum_is_the_transform_of_the_autocovariance(ratio):
    S0, tau = 2.5e-3, 3e-4
    dt = ratio * tau
    f = np.linspace(0.0, 0.5 / dt, 23)
    model = telegraph_psd_model(f, S0, tau, dt=dt)
    ref = _wiener_khinchin(f, S0, tau, dt)
    assert np.allclose(model, ref, rtol=1e-9, atol=0.0)


@pytest.mark.parametrize("ratio", [0.05, 0.5, 3.0])
def test_sampled_spectrum_is_the_sum_over_the_aliases(ratio):
    """S(f) = sum_n S0 / (1 + (2 pi tau (f + n/dt))^2): the Lorentzian
    folded back from every multiple of the sampling rate. The sum runs
    over |n| <= N, and the rest is added in closed form,
    S0 / (2 pi tau fs)^2 psi'(N + 1 +- f/fs) (its own relative error is
    below 1e-12 here)."""
    S0, tau = 1.0, 1.0
    dt = ratio * tau
    fs = 1.0 / dt
    N = 200000
    n = np.arange(-N, N + 1, dtype=float)
    for f in (0.0, 0.1 * fs, 0.31 * fs, 0.5 * fs):
        direct = np.sum(S0 / (1.0 + (2.0 * np.pi * tau * (f + n * fs)) ** 2))
        tail = S0 / (2.0 * np.pi * tau * fs) ** 2 * (
            polygamma(1, N + 1 + f / fs) + polygamma(1, N + 1 - f / fs))
        model = float(telegraph_psd_model(f, S0, tau, dt=dt))
        assert model == pytest.approx(direct + tail, rel=1e-9, abs=0)


def test_sampled_spectrum_parseval_and_continuous_limit():
    S0, tau = 1e-3, 2e-4
    for ratio in (0.02, 0.3, 2.0):
        dt = ratio * tau
        # integrate in pieces around the knee: the spectrum is peaked
        f_ny = 0.5 / dt
        edges = np.unique(np.concatenate(
            [[0.0], np.geomspace(1e-3, 1e3, 13) / (2 * np.pi * tau), [f_ny]]))
        edges = edges[edges <= f_ny]
        total = sum(quad(lambda x: float(telegraph_psd_model(
            x, S0, tau, dt=dt)), a, b, epsabs=0, epsrel=1e-13,
            limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
        assert total == pytest.approx(S0 / (4 * tau), rel=1e-10, abs=0)
    # dt -> 0: the Lorentzian (the difference is of order (dt/tau)^2)
    dt = 1e-4 * tau
    f = np.geomspace(1e-3, 10.0, 40) / (2 * np.pi * tau)
    ratio = telegraph_psd_model(f, S0, tau, 1e-6, dt=dt) / \
        telegraph_psd_model(f, S0, tau, 1e-6)
    assert np.max(np.abs(ratio - 1.0)) < 1e-6
    # dt=None is the 0.15.0 expression itself
    old = S0 / (1.0 + (2.0 * np.pi * f * tau) ** 2) + 1e-6
    assert np.array_equal(telegraph_psd_model(f, S0, tau, 1e-6), old)


def _record_bins(n, dt):
    return np.fft.rfftfreq(n, dt)[1:-1]       # no zero, no Nyquist bin


def test_fit_with_dt_recovers_a_noise_free_sampled_spectrum():
    """Noise-free sampled spectra (from the autocovariance sum, not from
    the package) over every bin of a 4096-sample record: with dt the fit
    returns S0, tau and the floor; the continuous Lorentzian does not.
    The last two numbers are the ones the docstring quotes (tau 3.7 %
    long with a fitted floor, 27 % short without one, at dt = tau/40)."""
    S0, tau = 1e-3, 2e-4
    n_avg = 8
    off = np.exp(digamma(n_avg) - np.log(n_avg))
    for ratio, floor in ((0.025, 0.0), (0.25, 0.0), (0.25, 2e-6)):
        dt = ratio * tau
        f = _record_bins(4096, dt)
        S = (_wiener_khinchin(f, S0, tau, dt) + floor) * off
        fit = fit_telegraph_psd(f, S, fit_floor=floor > 0, n_avg=n_avg,
                                dt=dt)
        assert fit.dt == dt
        assert fit.tau_s == pytest.approx(tau, rel=1e-6, abs=0)
        assert fit.S0 == pytest.approx(S0, rel=1e-6, abs=0)
        assert fit.variance == pytest.approx(S0 / (4 * tau), rel=1e-6, abs=0)
        if floor > 0:
            assert fit.floor == pytest.approx(floor, rel=1e-4, abs=0)
        assert fit.chi2 < 1e-10
    dt = 0.025 * tau
    f = _record_bins(4096, dt)
    S = _wiener_khinchin(f, S0, tau, dt)
    with_floor = fit_telegraph_psd(f, S)
    no_floor = fit_telegraph_psd(f, S, fit_floor=False)
    assert with_floor.dt is None
    assert with_floor.tau_s / tau - 1 == pytest.approx(0.0369, abs=5e-4)
    assert no_floor.tau_s / tau - 1 == pytest.approx(-0.2748, abs=5e-4)


def _telegraph_spectrum(seed, f_occ, tau, dt, n_seg, n_len, n_ch):
    rng = np.random.default_rng(seed)
    x = telegraph_traces(f_occ, tau, dt, n_seg * n_len, n_ch,
                         rng).sum(axis=1).astype(float)
    S = np.mean([psd_single_sided(s, dt)[1]
                 for s in x.reshape(n_seg, n_len)], axis=0)
    return S[1:-1]


def test_fit_with_dt_on_telegraph_monte_carlo():
    """40 telegraph records sampled at dt = tau/4 (exact sampling of the
    continuous process), 8 channels summed, 32 averaged segments, every
    bin below the Nyquist frequency. With dt the errors of tau and S0
    divided by the fit's own error bars scatter with mean near 0 and
    standard deviation near 1, chi2/dof is 1 and the scatter of tau is
    the planned error bar. The continuous Lorentzian with a floor puts
    tau about 4 % too long (at least 2.5 % on average here)."""
    f_occ, tau, dt, n_seg, n_len, n_ch = 0.3, 1.0, 0.25, 32, 2048, 8
    S0 = 4 * n_ch * f_occ * (1 - f_occ) * tau
    f = _record_bins(n_len, dt)
    pull_t, pull_s, chi, taus, bias = [], [], [], [], []
    for seed in range(40):
        S = _telegraph_spectrum(100 + seed, f_occ, tau, dt, n_seg, n_len,
                                n_ch)
        a = fit_telegraph_psd(f, S, fit_floor=False, n_avg=n_seg, dt=dt)
        pull_t.append((a.tau_s - tau) / a.sigma["tau_s"])
        pull_s.append((a.S0 - S0) / a.sigma["S0"])
        chi.append(a.chi2 / a.chi2_dof)
        taus.append(a.tau_s)
        c = fit_telegraph_psd(f, S, fit_floor=True, n_avg=n_seg)
        bias.append(c.tau_s / tau - 1)
    for p in (pull_t, pull_s):
        assert abs(np.mean(p)) < 0.6                 # 3.5 standard errors
        assert 0.75 < np.std(p, ddof=1) < 1.3
    assert np.mean(chi) == pytest.approx(1.0, abs=0.025)
    plan = plan_psd_measurement(f, n_seg, S0, tau, 0.0, fit_floor=False,
                                dt=dt)
    assert np.std(taus, ddof=1) == pytest.approx(plan["sigma"]["tau_s"],
                                                 rel=0.2, abs=0)
    assert np.mean(bias) > 0.025


def test_plan_with_dt_equals_the_fit_error_bars():
    """Two code paths of one matrix, as for the continuous model: the
    planned error bars equal those the fit reports on noise-free data
    (the expected log-periodogram, offset included)."""
    S0, tau, floor, n_avg = 1e-3, 3e-4, 1e-5, 16
    dt = tau / 3.0
    f = np.geomspace(50.0, 0.5 / dt, 48)
    S = telegraph_psd_model(f, S0, tau, floor, dt=dt) \
        * np.exp(digamma(n_avg) - np.log(n_avg))
    fit = fit_telegraph_psd(f, S, n_avg=n_avg, dt=dt)
    plan = plan_psd_measurement(f, n_avg, S0, tau, floor, dt=dt)
    assert plan["identifiable"]
    for k in ("S0", "tau_s", "floor"):
        assert plan["sigma"][k] == pytest.approx(fit.sigma[k], rel=1e-3, abs=0)
    # and they differ from the plan of the continuous model
    cont = plan_psd_measurement(f, n_avg, S0, tau, floor)
    assert abs(cont["sigma"]["floor"] / plan["sigma"]["floor"] - 1) > 0.05
    target = 0.5 * plan["sigma"]["tau_s"]
    n, p = averages_for_tau(target, f, S0, tau, floor, dt=dt)
    assert p["sigma"]["tau_s"] <= target
    assert plan_psd_measurement(f, n - 1, S0, tau, floor,
                                dt=dt)["sigma"]["tau_s"] > target


def test_sampled_spectrum_refusals():
    S0, tau = 1e-3, 3e-4
    dt = tau / 3.0
    f = np.geomspace(50.0, 0.5 / dt, 48)
    S = telegraph_psd_model(f, S0, tau, 1e-5, dt=dt)
    for bad in (0.0, -dt, np.nan, np.inf):
        with pytest.raises(ValueError, match="dt"):
            telegraph_psd_model(f, S0, tau, dt=bad)
        with pytest.raises(ValueError, match="dt"):
            fit_telegraph_psd(f, S, dt=bad)
        with pytest.raises(ValueError, match="dt"):
            plan_psd_measurement(f, 16, S0, tau, 1e-5, dt=bad)
    with pytest.raises(ValueError, match="Nyquist"):
        fit_telegraph_psd(f, S, dt=1.5 * dt)
    with pytest.raises(ValueError, match="Nyquist"):
        plan_psd_measurement(f, 16, S0, tau, 1e-5, dt=1.5 * dt)
    # the top bin may be the Nyquist frequency itself
    fit_telegraph_psd(f, S, dt=dt)


# ---- the Allan variance of sampled data -------------------------------------

def _avar_double_sum(var, eps, m, digits=50):
    """(1/(2 m^2)) sum_i sum_j h_i h_j var r^|i-j| with h = (-1 x m,
    +1 x m) and r = exp(-eps): the expectation of one term of the
    overlapping estimator of `allan_variance`, in 50-digit arithmetic
    from the float eps itself."""
    with localcontext() as ctx:
        ctx.prec = digits
        r = (-Decimal(eps)).exp()
        h = [-1] * m + [1] * m
        powers = [Decimal(1)]
        for _ in range(2 * m):
            powers.append(powers[-1] * r)
        s = Decimal(0)
        for i in range(2 * m):
            for j in range(2 * m):
                s += h[i] * h[j] * powers[abs(i - j)]
        return float(Decimal(var) * s / (2 * m * m))


@pytest.mark.parametrize("eps", [1e-7, 1e-4, 0.01, 0.3, 2.0, 20.0])
def test_sampled_allan_variance_equals_the_defining_double_sum(eps):
    for m in (1, 2, 3, 7, 24, 60):
        got = avar_exponential(1.7, 1.0, m * eps, dt=eps)
        assert got == pytest.approx(_avar_double_sum(1.7, eps, m),
                                    rel=1e-13, abs=0)


def test_sampled_allan_variance_limits_and_shape():
    # one-sample windows: var (1 - r)
    for eps in (1e-9, 1e-3, 0.7, 5.0):
        assert avar_exponential(2.0, 1.0, eps, dt=eps) == pytest.approx(
            -2.0 * np.expm1(-eps), rel=1e-14, abs=0)
    # samples much farther apart than tau are uncorrelated: var / m
    for m in (1, 3, 10):
        assert avar_exponential(1.0, 1.0, m * 50.0, dt=50.0) == \
            pytest.approx(1.0 / m, rel=1e-12, abs=0)
    # dt -> 0 at fixed window length: the continuous closed form
    for T in (0.3, 2.0, 10.0):
        m = 20000
        d = avar_exponential(1.0, 1.0, T, dt=T / m)
        assert d == pytest.approx(avar_exponential(1.0, 1.0, T), rel=1e-4,
                                  abs=0)
    # the differences quoted in the docstring, at dt = tau/40
    for m, diff in ((1, 0.509), (5, 0.0219), (20, 0.00174)):
        ratio = avar_exponential(1.0, 1.0, m / 40.0, dt=1 / 40.0) / \
            avar_exponential(1.0, 1.0, m / 40.0)
        assert ratio - 1 == pytest.approx(diff, rel=3e-3, abs=0)
    # array input keeps its shape; the switch to the series is smooth
    T = np.array([[1.0, 2.0], [3.0, 4.0]]) * 0.1
    assert avar_exponential(1.0, 1.0, T, dt=0.1).shape == (2, 2)
    a = avar_exponential(1.0, 1.0, 5 * 0.1 * (1 - 1e-13), dt=0.1 * (1 - 1e-13))
    b = avar_exponential(1.0, 1.0, 5 * 0.1, dt=0.1)
    assert a == pytest.approx(b, rel=1e-11, abs=0)


def test_sampled_allan_variance_matches_telegraph_monte_carlo():
    """Exactly sampled telegraph noise at dt = tau/2 (64 channels): the
    estimator follows the sampled closed form within 3 %, while the
    continuous form is 69 % (m = 1) and 23 % (m = 2) away."""
    tau, dt, f_occ, n_ch = 1.0, 0.5, 0.3, 64
    rng = np.random.default_rng(7)
    x = telegraph_traces(f_occ, tau, dt, 400000, n_ch,
                         rng).astype(float).sum(axis=1)
    var = n_ch * f_occ * (1 - f_occ)
    taus, av = allan_variance(x, dt, [1, 2, 4])
    sampled = avar_exponential(var, tau, taus, dt=dt)
    cont = avar_exponential(var, tau, taus)
    assert np.allclose(av, sampled, rtol=0.03, atol=0)
    assert np.all(cont[:2] / sampled[:2] < 0.85)


def test_sampled_allan_variance_refusals():
    with pytest.raises(ValueError, match="whole number"):
        avar_exponential(1.0, 1.0, 0.25, dt=0.1)
    with pytest.raises(ValueError, match="whole number"):
        avar_exponential(1.0, 1.0, 0.04, dt=0.1)
    for bad in (0.0, -0.1, np.nan, np.inf):
        with pytest.raises(ValueError, match="dt"):
            avar_exponential(1.0, 1.0, 0.2, dt=bad)


# ---- fixes ------------------------------------------------------------------

def _narrow():
    return dataclasses.replace(RECIPES[1], W=0.3e-6)   # Ti/Al/Au, 6 modes


def test_occupation_heat_capacities_at_an_off_grid_phase():
    """The bound part equals the level sum C_A of `FiniteLJunction` at
    the same phase (two code paths), also off the old 81-phase grid.
    0.15.0 used the grid phase 0.982 for phi0 = 1.0: C_bound
    1.28775e-21 J/K against C_A = 1.30354e-21 J/K (1.2 % off)."""
    r = RECIPES[1]
    T = 0.3 * r.Tc
    for phi0 in (1.0, 2.2):
        out = occupation_heat_capacities(r, T, phi0=phi0)
        ca = FiniteLJunction(r).andreev_sums(phi0, T)["C_A"]
        assert out["C_bound"] == pytest.approx(ca, rel=1e-5, abs=0)
    for bad in (0.0, -0.1, r.Tc, 2 * r.Tc, np.nan):
        with pytest.raises(ValueError):
            occupation_heat_capacities(r, bad)
    for bad in (0.0, np.pi, -1.0):
        with pytest.raises(ValueError):
            occupation_heat_capacities(r, T, phi0=bad)
    with pytest.raises(ValueError):
        occupation_heat_capacities(r, T, dT_frac=1.5)


def _continuum_share_on_grid_0150(recipe, T, n_phi, j):
    """The 0.15.0 computation, kept here: shares at grid phase j from
    np.gradient over the whole phase grid."""
    def I_parts(TT):
        m = JunctionModel(recipe, n_phi=n_phi)
        m.Delta = float(gap_bcs(TT, recipe.Tc, recipe.Delta))
        Fb, Fc = free_energy_components(m, TT)
        dphi = m.phis[1] - m.phis[0]
        return [float((2.0 * E_CHARGE / HBAR) * 2.0
                      * np.gradient(F, dphi)[j]) for F in (Fb, Fc)]
    dT = min(2e-3 * recipe.Tc, 0.5 * T, 0.1 * (recipe.Tc - T))
    b0, c0 = I_parts(T)
    bp, cp = I_parts(T + dT)
    bm, cm = I_parts(T - dT)
    db, dc = (bp - bm) / (2 * dT), (cp - cm) / (2 * dT)
    return dict(share_I=c0 / (b0 + c0), share_dIdT=dc / (db + dc))


def test_continuum_share_at_and_between_grid_phases():
    """On a grid phase the result is that of 0.15.0 (to rounding). Off
    the grid it is evaluated at phi0 and is continuous in phi0: across
    the midpoint of two grid phases, 0.15.0 jumped from the value of
    one grid phase to that of the next (share_I -0.25436 at phi0 = 1.0,
    the value at 0.982; now -0.25109)."""
    r, T, n_phi = _narrow(), 0.225, 81
    grid = JunctionModel(r, n_phi=n_phi).phis
    for j in (25, 40):
        new = continuum_share(r, T, n_phi=n_phi, phi0=grid[j])
        old = _continuum_share_on_grid_0150(r, T, n_phi, j)
        for k in ("share_I", "share_dIdT"):
            assert new[k] == pytest.approx(old[k], rel=1e-9, abs=0)
    mid = 0.5 * (grid[25] + grid[26])
    lo = continuum_share(r, T, n_phi=n_phi, phi0=mid - 1e-7)
    hi = continuum_share(r, T, n_phi=n_phi, phi0=mid + 1e-7)
    assert abs(hi["share_I"] - lo["share_I"]) < 1e-5
    assert continuum_share(r, T, phi0=1.0)["share_I"] == pytest.approx(
        -0.25109, abs=1e-5)
    for bad in (0.0, 0.01, np.pi - 0.01, 4.0, np.nan):
        with pytest.raises(ValueError, match="phi0"):
            continuum_share(r, T, phi0=bad)


def test_loaders_refuse_non_finite_values(tmp_path):
    """0.15.0 read "nan" and "inf" (Python's float accepts them): a nan
    time gave dt = nan, and a nan temperature passed the "strictly
    increasing" check."""
    p = str(tmp_path / "f.csv")
    cases = [
        (load_trace_csv, "t_s,y\n0.0,1.0\nnan,2.0\n2.0,0.5\n3.0,0.1\n"),
        (load_trace_csv, "t_s,y\n0.0,1.0\n1.0,nan\n2.0,0.5\n3.0,0.1\n"),
        (load_trace_csv, "t_s,y\n0.0,1.0\n1.0,-inf\n2.0,0.5\n"),
        (load_ic_csv, "T_K,Ic_A\n0.1,1e-6\nnan,2e-6\n0.3,1e-6\n"),
        (load_ic_csv, "T_K,Ic_A\n0.1,1e-6\n0.2,inf\n0.3,1e-6\n"),
        (load_ic_csv, "T_K,Ic_A,sigma_Ic_A\n0.1,1e-6,1e-8\n0.2,1e-6,nan\n"),
    ]
    for loader, text in cases:
        with open(p, "w") as fh:
            fh.write(text)
        with pytest.raises(ValueError, match="non-finite"):
            loader(p)
    with open(p, "w") as fh:
        fh.write("t_s,y\n0.0,1.0\n1.0,2.0\n2.0,0.5\n")
    dt, y = load_trace_csv(p)
    assert dt == 1.0 and np.array_equal(y, [1.0, 2.0, 0.5])


def test_short_junction_refuses_unknown_readout_and_temperature():
    """0.15.0 took any `which` other than "I" as "L" in `andreev_sums`
    ("i" gave the inductance coupling) and as "I" for the phase of
    `temperature_bound`; T = Tc raised ZeroDivisionError and T < 0
    returned NaN."""
    sj = ShortJunction(RECIPES[1])
    sj.calibrate()
    T = 0.225
    for bad in ("i", "l", "X", ""):
        with pytest.raises(ValueError, match="which"):
            sj.andreev_sums(1.0, T, bad)
        with pytest.raises(ValueError, match="which"):
            sj.temperature_bound(T, 1e-6, 1.0, which=bad)
    for bad in (0.0, -0.1, RECIPES[1].Tc, 1.0, np.nan):
        with pytest.raises(ValueError):
            sj.andreev_sums(1.0, bad, "I")
        with pytest.raises(ValueError):
            sj.temperature_bound(bad, 1e-6, 1.0, which="L")
    for bad in (0.0, -1e-6, np.inf):
        with pytest.raises(ValueError, match="tauA"):
            sj.temperature_bound(T, bad, 1.0)
        with pytest.raises(ValueError, match="t_int"):
            sj.temperature_bound(T, 1e-6, bad)
    # valid calls are unchanged: the budget gives the same numbers
    b = SensorBudget(RECIPES[1])
    a, s = b.dT_andreev(T, 1e-6, 1.0)
    a2, _, s2 = sj.temperature_bound(T, 1e-6, 1.0, which="L")
    assert a == a2 and s == s2


def test_two_temperature_click_refuses_bad_inputs():
    """0.15.0 ran with a negative tauA (the occupation left its
    equilibrium: m = 1.0012 > 1 after 2 ns for Ti/Al/Au at 0.1 K) and
    returned NaN trajectories for a NaN deposit or phonon parameter."""
    b = SensorBudget(RECIPES[1])
    ok = dict(c_ph=1e-15, kappa_pb=1.0, delta_pb=4.0, t_end=2e-9,
              dt=1e-10)
    bad_calls = [
        dict(E_gamma=1e-22, tauA=-1e-7, T0=0.1),
        dict(E_gamma=1e-22, tauA=0.0, T0=0.1),
        dict(E_gamma=np.nan, tauA=1e-7, T0=0.1),
        dict(E_gamma=np.inf, tauA=1e-7, T0=0.1),
        dict(E_gamma=1e-22, tauA=np.nan, T0=0.1),
        dict(E_gamma=1e-22, tauA=1e-7, T0=np.nan),
    ]
    for kw in bad_calls:
        with pytest.raises(ValueError):
            two_temperature_click(b, kw["E_gamma"], kw["tauA"], kw["T0"],
                                  **ok)
    for name in ("c_ph", "kappa_pb", "delta_pb"):
        for bad in (np.nan, np.inf):
            kw = dict(ok, **{name: bad})
            with pytest.raises(ValueError, match="finite"):
                two_temperature_click(b, 1e-22, 1e-7, 0.1, **kw)
    out = two_temperature_click(b, 1e-22, 1e-7, 0.1, **ok)
    assert np.all(np.isfinite(out["ms"])) and out["ms"].max() <= 1.0


def test_sample_interval_must_be_finite_and_positive():
    """0.15.0: psd_single_sided(x, -1) returned a negative spectrum at
    negative frequencies; allan_variance accepted dt = inf or nan
    (infinite or nan averaging times); TelegraphHMM.rates(-dt) gave a
    negative tauA."""
    x = np.random.default_rng(0).normal(size=64)
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="dt"):
            psd_single_sided(x, bad)
        with pytest.raises(ValueError, match="dt"):
            allan_variance(x, bad, [1, 2])
        with pytest.raises(ValueError, match="dt"):
            TelegraphHMM(0.01, 0.02, (0.0, 1.0), 0.1).rates(bad)
    with pytest.raises(ValueError):
        psd_single_sided(np.array([1.0, np.nan, 2.0]), 1.0)
    f, S = psd_single_sided(x, 0.5)
    assert np.sum(S) * (f[1] - f[0]) == pytest.approx(np.var(x), rel=1e-12,
                                                      abs=0)
    f_hat, tau_hat = TelegraphHMM(0.01, 0.02, (0.0, 1.0), 0.1).rates(1e-3)
    assert tau_hat == pytest.approx(-1e-3 / np.log1p(-0.03), rel=1e-12, abs=0)


def test_version_is_0_16_0():
    import absnoise
    assert absnoise.__version__ == "0.16.0"
    path = os.path.join(ROOT, "CITATION.cff")
    if not os.path.isfile(path):
        pytest.skip("CITATION.cff is not part of this tree")
    with open(path) as fh:
        text = fh.read()
    assert "version: 0.16.0" in text
    assert "date-released: 2026-11-10" in text
