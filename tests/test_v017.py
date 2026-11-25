"""0.17.0 changes, each checked against an independent reference.

New:

* the Matsubara sums at low temperature (head summed term by term, the
  rest integrated): against the full sum of 0.16.0, kept here;
* `telegraph_psd_model(..., sampling="average")` and the fit and
  planner with it: against the sum of the Lorentzian times sinc^2 over
  all aliases, the transform of the autocovariance of the averaged
  samples (from numerical integration), Parseval, the continuous limit,
  noise-free fits and an exact simulation of the averaged telegraph
  signal; `avar_exponential(..., sampling="average")` against the
  double sum over that autocovariance in 50-digit arithmetic;
* `n_record` (the expected periodogram of a finite record): against
  the quadratic form of the full covariance matrix, the long-record
  limit, the mean of simulated periodograms and fits on telegraph
  Monte Carlo;
* `n_avg` with one value per point: against the scalar fit and
  planner, and on spectra with the exact Gamma scatter;
* `fit_hmm(..., equal_noise=False)`: the likelihood with two equal
  widths, the exact gradient against finite differences, the batched
  decoder against `viterbi`, and recovery and error bars on simulated
  traces;
* `markov_dwell_test` correlation of successive stays: counted by hand
  on a short path, and on traces with and without linked stays;
* `design_ic_temperatures(..., method="exhaustive")`: against a
  brute-force loop over all subsets;
* `fit_ep_coupling(..., power_scale_error=...)`: the exact scaling of
  Sigma with the power calibration, the added variance and a Monte
  Carlo over random calibrations;
* the error bars of `click_monte_carlo`: against the scatter over
  independent seeds.

Fixes (each test fails on 0.16.0): input checks of `allan_variance`
(non-integer windows), `ic_model`, `gap_bcs`, `noneq_penalty`,
`tau_activated`, `channel_generator`, `steady_temperature`,
`TelegraphHMM`, `avar_white`, `matched_Tc`, `click_template`,
`click_monte_carlo`, `two_temperature_click`, `telegraph_traces`,
`save_ic_csv` and `save_trace_csv`, and `freq_noise_spectrum` with a
list of frequencies.
"""
import os
import tempfile
from decimal import Decimal, localcontext
from itertools import combinations

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import digamma

from absnoise import (RECIPES, JunctionModel, SensorBudget, TelegraphHMM,
                      allan_variance, avar_exponential, avar_white,
                      channel_generator, click_monte_carlo, click_template,
                      design_ic_temperatures, ep_temperature,
                      fit_ep_coupling, fit_hmm, fit_telegraph_psd, gap_bcs,
                      ic_model, load_ic_csv, markov_dwell_test,
                      matched_Tc, noneq_penalty, plan_ic_measurement,
                      plan_psd_measurement, averages_for_tau,
                      psd_single_sided, save_ic_csv, save_trace_csv,
                      steady_temperature, tau_activated,
                      telegraph_psd_model, telegraph_traces,
                      two_temperature_click)
from absnoise.constants import E_CHARGE, HBAR, KB
from absnoise.decode import _successive_correlation, _viterbi_batch
from absnoise.lab import _jacobian
from absnoise.levels import (_chi2, _matsubara_terms,
                             matsubara_free_energy_terms)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ---- Matsubara sums at low temperature ----------------------------------------

def _full_sum_0160(phi, tau, c, Delta, T, omega_max_fac=2000.0):
    """The summation of 0.16.0: every term up to omega_max_fac Delta.
    Returns (dF at the phases, F2, F4)."""
    kT = KB * T
    n_terms = int(np.ceil(omega_max_fac * Delta / (2.0 * np.pi * kT))) + 1
    w = (2.0 * np.arange(n_terms) + 1.0) * np.pi * kT
    y = w / Delta
    e2s = 1.0 / (y + np.sqrt(1.0 + y * y)) ** 2
    decay = np.exp(-c * w)
    x = e2s * decay
    x_as = decay / (4.0 * y * y)
    sum_as = float(_chi2(np.exp(-c * np.pi * kT))) * \
        (Delta / (2.0 * np.pi * kT)) ** 2
    ph = np.atleast_1d(phi)
    dF = np.empty(ph.size)
    for i, p in enumerate(ph):
        R = 1.0 - tau + tau * np.cos(p)
        g = sum_as + np.sum(x / (1.0 + 2.0 * R * x + x * x) - x_as)
        dF[i] = 4.0 * kT * tau * np.sin(p) * g
    kap = tau * x / (1.0 + x) ** 2
    s1 = tau * sum_as + float(np.sum(kap - tau * x_as))
    s2 = float(np.sum(kap * kap))
    return dF, 2.0 * kT * s1, -2.0 * kT * (s1 / 12.0 - 0.5 * s2), \
        2.0 * kT * (abs(s1) / 12.0 + 0.5 * s2)


@pytest.mark.parametrize("t_red", [0.0027, 0.01, 0.05, 0.3])
def test_matsubara_tail_equals_the_full_sum(t_red):
    """The head-plus-integral sum equals the full sum of 0.16.0 for
    temperatures from 0.0027 Tc (MoRe at 20 mK) to 0.3 Tc, lengths
    c Delta from 0 to 20 and transparencies 0.05 to 1. The full sum
    itself carries rounding errors near 1e-12 (both differ from a sum in
    80-bit arithmetic by up to 2.1e-12 in a check during development).
    F4 is compared with the size of its two terms, which can cancel."""
    Delta = 1.0e-23
    T = t_red * Delta / (1.764 * KB)
    ph = np.array([1e-3, 0.3, 1.0, 1.9, 2.8, np.pi - 1e-3])
    worst = 0.0
    for cD in (0.0, 0.3, 1.0, 6.5, 20.0):
        for tau in (0.05, 0.78, 1.0):
            c = cD / Delta
            ref, F2r, F4r, F4s = _full_sum_0160(ph, tau, c, Delta, T)
            new = matsubara_free_energy_terms(ph, tau, c, Delta, T)
            e = max(np.max(np.abs(new["dF"] - ref)) / np.max(np.abs(ref)),
                    abs(new["F2"] / F2r - 1.0),
                    abs(new["F4"] - F4r) / F4s)
            worst = max(worst, e)
    assert worst < 3e-12      # measured 1.9e-12 (0.0027 Tc), 4e-16 (0.3 Tc)


def test_matsubara_tail_uses_few_points_at_low_temperature():
    r = RECIPES[5]                                # MoRe, 20 mK
    m = JunctionModel(r)
    x, _, _, wt = _matsubara_terms(m.cs[0], r.Delta, 0.02, 2000.0)
    n_full = int(np.ceil(2000.0 * r.Delta / (2 * np.pi * KB * 0.02))) + 1
    assert n_full > 200000 and x.size < 2300 and wt is not None
    # high temperature: all terms, weight 1 (the 0.16.0 summation)
    x, _, _, wt = _matsubara_terms(m.cs[0], r.Delta, 0.95 * r.Tc, 2000.0)
    assert wt is None


def test_matsubara_current_of_more_at_20_mK_equals_the_full_sum():
    """MoRe, every mode, scale 1, at five phases: the current of
    `cpr_matsubara` against the 0.16.0 full sum (2.1e5 terms per
    mode)."""
    r = RECIPES[5]
    m = JunctionModel(r)
    ph = np.array([0.4, 1.2, 1.67, 2.2, 3.0])
    ref = np.zeros(ph.size)
    for c in m.cs:
        ref += _full_sum_0160(ph, m.tau, c, m.Delta, 0.02)[0]
    new = np.zeros(ph.size)
    for c in m.cs:
        new += matsubara_free_energy_terms(ph, m.tau, c, m.Delta, 0.02)["dF"]
    assert np.max(np.abs(new / ref - 1.0)) < 1e-12   # measured 2.4e-14
    I = m.cpr_matsubara(0.02, ph)
    assert np.allclose(I, new * 4 * E_CHARGE / HBAR,
                       rtol=1e-13, atol=0)                # measured 2e-16


# ---- averaged samples ----------------------------------------------------------

def _avg_cov(k, dt, tau, var):
    """Autocovariance of the means over successive sample intervals,
    from numerical integration of var exp(-|t - s| / tau)."""
    if k == 0:
        return 2.0 * var / dt ** 2 * quad(
            lambda u: (dt - u) * np.exp(-u / tau), 0.0, dt,
            epsabs=0, epsrel=1e-13)[0]
    return var / dt ** 2 * quad(
        lambda u: (dt - abs(u - k * dt)) * np.exp(-u / tau),
        (k - 1) * dt, (k + 1) * dt, points=[k * dt], epsabs=0,
        epsrel=1e-13)[0]


@pytest.mark.parametrize("ratio", [0.01, 0.05, 0.5, 3.0, 20.0])
def test_averaged_spectrum_is_the_aliased_sinc_lorentzian(ratio):
    """S(f) = sum_n S0 sinc^2(pi (f + n/dt) dt) / (1 + (2 pi tau (f + n/dt))^2):
    the Lorentzian through the transfer function of the average, folded
    back from every multiple of the sampling rate (|n| <= 200 000; the
    rest falls as n^-4)."""
    S0, tau = 1.0, 1.0
    dt = ratio * tau
    fs = 1.0 / dt
    n = np.arange(-200000, 200001, dtype=float)
    for fr in (0.0, 0.1, 0.31, 0.5):
        ff = (fr + n) * fs
        ref = np.sum(S0 * np.sinc(ff * dt) ** 2
                     / (1.0 + (2.0 * np.pi * tau * ff) ** 2))
        model = float(telegraph_psd_model(fr * fs, S0, tau, dt=dt,
                                          sampling="average"))
        assert model == pytest.approx(ref, rel=1e-13, abs=0)  # measured 1e-15


@pytest.mark.parametrize("ratio", [0.05, 0.5, 2.0])
def test_averaged_spectrum_is_the_transform_of_the_averaged_covariance(ratio):
    S0, tau = 1.0, 1.0
    dt = ratio * tau
    var = S0 / (4 * tau)
    K = int(np.ceil(40.0 / ratio))
    C = np.array([_avg_cov(k, dt, tau, var) for k in range(K + 1)])
    k = np.arange(1, K + 1)
    for fr in (0.0, 0.13, 0.4, 0.5):
        ref = 2 * dt * (C[0] + 2 * np.sum(C[1:] * np.cos(2 * np.pi * fr * k)))
        model = float(telegraph_psd_model(fr / dt, S0, tau, dt=dt,
                                          sampling="average"))
        assert model == pytest.approx(ref, rel=1e-11, abs=0)  # measured 5e-13


def test_averaged_spectrum_parseval_and_limits():
    S0, tau = 1e-3, 2e-4
    var = S0 / (4 * tau)
    for ratio in (0.02, 0.3, 2.0):
        dt = ratio * tau
        f_ny = 0.5 / dt
        edges = np.unique(np.concatenate(
            [[0.0], np.geomspace(1e-3, 1e3, 13) / (2 * np.pi * tau), [f_ny]]))
        edges = edges[edges <= f_ny]
        total = sum(quad(lambda x: float(telegraph_psd_model(
            x, S0, tau, dt=dt, sampling="average")), a, b, epsabs=0,
            epsrel=1e-13, limit=200)[0] for a, b in zip(edges[:-1], edges[1:]))
        # the variance of the averaged samples; measured 2e-16
        assert total == pytest.approx(_avg_cov(0, dt, tau, var), rel=1e-10,
                                      abs=0)
    # dt -> 0: the Lorentzian
    dt = 1e-4 * tau
    f = np.geomspace(1e-3, 10.0, 40) / (2 * np.pi * tau)
    ratio = telegraph_psd_model(f, S0, tau, dt=dt, sampling="average") / \
        telegraph_psd_model(f, S0, tau)
    assert np.max(np.abs(ratio - 1.0)) < 1e-6           # measured 8.3e-8
    # the instantaneous default is unchanged
    assert np.array_equal(telegraph_psd_model(f, S0, tau, dt=dt),
                          telegraph_psd_model(f, S0, tau, dt=dt,
                                              sampling="instant"))


def _avar_avg_decimal(var, tau, dt, m):
    """Expected Allan variance of means over m averaged samples, from the
    double sum over their autocovariance (closed forms below, whose
    values the test above checks by numerical integration), in 50-digit
    arithmetic:
    AVAR = [sum_{|h|<m} (m - |h|) C(h) - sum_{h=1}^{2m-1} min(h, 2m-h) C(h)] / m^2."""
    with localcontext() as ctx:
        ctx.prec = 50
        e = Decimal(dt) / Decimal(tau)
        r = (-e).exp()
        v = Decimal(var)
        C0 = 2 * v * (e - (1 - r)) / (e * e)

        def C(h):
            return v * ((1 - r) / e) ** 2 * r ** (h - 1)
        s1 = m * C0 + 2 * sum((m - h) * C(h) for h in range(1, m))
        s2 = sum(min(h, 2 * m - h) * C(h) for h in range(1, 2 * m))
        return float((s1 - s2) / (m * m))


@pytest.mark.parametrize("ratio", [1e-3, 0.025, 0.5, 5.0])
def test_allan_variance_of_averaged_samples_is_the_continuous_form(ratio):
    var, tau = 0.21, 1.0
    dt = ratio * tau
    for m in (1, 2, 5, 20, 60):
        ref = _avar_avg_decimal(var, tau, dt, m)
        got = avar_exponential(var, tau, m * dt, dt=dt, sampling="average")
        assert got == pytest.approx(ref, rel=1e-12, abs=0)   # measured 3e-16
        assert got == avar_exponential(var, tau, m * dt)


def _averaged_telegraph(f, tau, dt, n, rng):
    """Exact means over each sample interval of a two-state process with
    occupation f and correlation time tau: exponential stays, and the
    time spent filled in each interval from the switch times."""
    T = n * dt
    mean0, mean1 = tau / f, tau / (1 - f)
    state = int(rng.random() < f)
    t, ts, states = 0.0, [0.0], [state]
    while t < T:
        m = int(2 * T / (mean0 + mean1)) + 16
        d0 = rng.exponential(mean0, m)
        d1 = rng.exponential(mean1, m)
        dur = np.column_stack([d0, d1] if state == 0 else [d1, d0]).ravel()
        ends = t + np.cumsum(dur)
        nxt = np.tile([1 - state, state], m)
        ts.extend(ends.tolist())
        states.extend(nxt.tolist())
        t = ends[-1]
        state = int(nxt[-1])
    ts = np.array(ts)
    states = np.array(states[:ts.size])
    filled = np.concatenate([[0.0], np.cumsum(np.diff(ts) * states[:-1])])
    U = np.interp(np.arange(n + 1) * dt, ts, filled)
    return np.diff(U) / dt


def test_fit_of_averaged_samples_on_exact_monte_carlo():
    """40 records of 8 channels, each sample the exact mean of the
    process over its interval, dt = tau/4, 32 segments of 1024 samples.
    With sampling="average" the errors of tau and S0 divided by their
    error bars have mean near 0 and spread near 1, chi2/dof is 1 and the
    scatter of tau is the planned error bar. The instantaneous model
    puts tau 57 % too long, with chi2/dof 4.3."""
    f_occ, tau, dt, n_seg, n_len, n_ch = 0.3, 1.0, 0.25, 32, 1024, 8
    S0 = 4 * n_ch * f_occ * (1 - f_occ) * tau
    f = np.fft.rfftfreq(n_len, dt)[1:-1]
    pt, ps, chi, taus, bias, chi_i = [], [], [], [], [], []
    for seed in range(40):
        rng = np.random.default_rng(700 + seed)
        y = sum(_averaged_telegraph(f_occ, tau, dt, n_seg * n_len, rng)
                for _ in range(n_ch))
        S = np.mean([psd_single_sided(s, dt)[1]
                     for s in y.reshape(n_seg, n_len)], axis=0)[1:-1]
        a = fit_telegraph_psd(f, S, fit_floor=False, n_avg=n_seg, dt=dt,
                              sampling="average")
        assert a.sampling == "average" and a.n_record is None
        pt.append((a.tau_s - tau) / a.sigma["tau_s"])
        ps.append((a.S0 - S0) / a.sigma["S0"])
        chi.append(a.chi2 / a.chi2_dof)
        taus.append(a.tau_s)
        b = fit_telegraph_psd(f, S, fit_floor=False, n_avg=n_seg, dt=dt)
        bias.append(b.tau_s / tau - 1)
        chi_i.append(b.chi2 / b.chi2_dof)
    for p in (pt, ps):
        assert abs(np.mean(p)) < 0.5          # measured -0.09 and 0.03
        assert 0.75 < np.std(p, ddof=1) < 1.3  # measured 1.03 and 0.93
    assert np.mean(chi) == pytest.approx(1.0, abs=0.03)   # measured 1.013
    plan = plan_psd_measurement(f, n_seg, S0, tau, 0.0, fit_floor=False,
                                dt=dt, sampling="average")
    assert np.std(taus, ddof=1) == pytest.approx(plan["sigma"]["tau_s"],
                                                 rel=0.2, abs=0)  # 2 %
    assert np.mean(bias) > 0.4                # measured 0.57
    assert np.mean(chi_i) > 3.0               # measured 4.3


def test_noise_free_averaged_spectrum_is_fitted_exactly():
    S0, tau, n_avg = 1e-3, 2e-4, 8
    off = np.exp(digamma(n_avg) - np.log(n_avg))
    for ratio, floor in ((0.025, 0.0), (0.25, 2e-6), (1.0, 0.0)):
        dt = ratio * tau
        f = np.fft.rfftfreq(4096, dt)[1:-1]
        var = S0 / (4 * tau)
        K = int(np.ceil(44.0 / ratio))
        C = np.array([_avg_cov(k, dt, tau, var) for k in range(K + 1)])
        k = np.arange(1, K + 1)
        S = np.array([2 * dt * (C[0] + 2 * np.sum(
            C[1:] * np.cos(2 * np.pi * fi * dt * k))) for fi in f])
        S = (S + floor) * off
        if S.max() / S.min() < 4 or 1 / (2 * np.pi * tau) > f.max() / 2:
            continue
        fit = fit_telegraph_psd(f, S, fit_floor=floor > 0, n_avg=n_avg,
                                dt=dt, sampling="average")
        # measured 8e-15 (tau, S0) and 1.1e-14 (floor)
        assert fit.tau_s == pytest.approx(tau, rel=1e-9, abs=0)
        assert fit.S0 == pytest.approx(S0, rel=1e-9, abs=0)
        if floor > 0:
            assert fit.floor == pytest.approx(floor, rel=1e-7, abs=0)


# ---- finite records ------------------------------------------------------------

def _quadratic_form(S0, tau, dt, N, sampling):
    """E[S_j] = 2 dt v_j^H C v_j / N from the full N x N covariance
    matrix C of the samples, for the bins 0 < j < N/2."""
    var = S0 / (4 * tau)
    eps = dt / tau
    r = np.exp(-eps)
    h = np.arange(N)
    if sampling == "instant":
        Ch = var * r ** h
    else:
        Ch = var * ((1 - r) / eps) ** 2 * r ** (h - 1.0)
        Ch[0] = 2 * var * (eps + np.expm1(-eps)) / eps ** 2
    Cm = Ch[np.abs(h[:, None] - h[None, :])]
    j = np.arange(1, N // 2 + 1)
    j = j[2 * j < N]
    V = np.exp(-2j * np.pi * np.outer(j, h) / N)
    EX2 = np.real(np.einsum("ik,kl,il->i", V, Cm, V.conj()))
    return j, 2 * dt * EX2 / N


@pytest.mark.parametrize("sampling", ["instant", "average"])
def test_finite_record_spectrum_is_the_quadratic_form(sampling):
    S0, tau = 1.0, 1.0
    worst = 0.0
    for ratio in (0.02, 0.3, 2.0):
        for N in (16, 64, 256, 257):
            if N * ratio < 1.0:
                continue                   # record shorter than tau: the
                # quadratic form itself loses digits there
            dt = ratio * tau
            j, ref = _quadratic_form(S0, tau, dt, N, sampling)
            got = telegraph_psd_model(j / (N * dt), S0, tau, dt=dt,
                                      sampling=sampling, n_record=N)
            worst = max(worst, np.max(np.abs(got / ref - 1)))
    # measured 1.7e-13 (instant) and 1.8e-12 (average, at dt = 0.02 tau,
    # where the double-precision quadratic form itself rounds; the same
    # sum in 80-bit arithmetic agrees with the model to 5e-15)
    assert worst < 3e-12


def test_finite_record_limits_and_documented_numbers():
    S0, tau = 1.0, 1.0
    # long records: the infinite-record spectrum, to order 1/N
    for sampling in ("instant", "average"):
        dt = 0.1
        N = 2 ** 20
        j = np.array([5, 300, 40000, N // 2 - 1])
        a = telegraph_psd_model(j / (N * dt), S0, tau, dt=dt,
                                sampling=sampling, n_record=N)
        b = telegraph_psd_model(j / (N * dt), S0, tau, dt=dt,
                                sampling=sampling)
        assert np.max(np.abs(a / b - 1)) < 1e-4   # measured 9.5e-6, 2.9e-5
    # the numbers quoted in the docstring (first bin, instantaneous)
    for dt, want in ((0.3, 0.95877), (0.02, 1.51912)):
        f1 = 1.0 / (64 * dt)
        ratio = telegraph_psd_model(f1, S0, tau, dt=dt, n_record=64) / \
            telegraph_psd_model(f1, S0, tau, dt=dt)
        assert ratio == pytest.approx(want, abs=1e-5)   # measured 4.3e-6
    with pytest.raises(ValueError):
        telegraph_psd_model(0.5 / (64 * 0.1), S0, tau, dt=0.1, n_record=64)
    with pytest.raises(ValueError):                    # the Nyquist bin
        telegraph_psd_model(32 / (64 * 0.1), S0, tau, dt=0.1, n_record=64)
    with pytest.raises(ValueError):
        telegraph_psd_model(1.0, S0, tau, n_record=64)  # no dt
    with pytest.raises(ValueError):
        telegraph_psd_model(1.0, S0, tau, sampling="average")
    with pytest.raises(ValueError):
        telegraph_psd_model(1.0, S0, tau, dt=0.1, sampling="mean")
    with pytest.raises(ValueError):
        telegraph_psd_model(1.0, S0, tau, dt=0.1, n_record=64.5)


def test_finite_record_fit_on_telegraph_monte_carlo():
    """30 records, 4 channels, 400 segments of 128 samples at
    dt = 0.12 tau (15 tau per segment). The mean periodogram follows the
    finite-record model (largest deviation 2.3 standard errors over 63
    bins) and not the infinite-record one (8.5). Fitted without
    n_record, tau is 5.9 % short (2.9 error bars); with n_record the
    bias is gone. The error bars are somewhat conservative here (the
    spread of the errors in units of the error bar is 0.76), because
    neighboring bins of a short record are correlated, which the fit
    does not model."""
    f_occ, tau, dt, n_seg, n_len, n_ch = 0.3, 1.0, 0.12, 400, 128, 4
    S0 = 4 * n_ch * f_occ * (1 - f_occ) * tau
    f = np.fft.rfftfreq(n_len, dt)[1:-1]
    pt, pt0, chi, means = [], [], [], []
    for seed in range(30):
        rng = np.random.default_rng(900 + seed)
        x = telegraph_traces(f_occ, tau, dt, n_seg * n_len, n_ch,
                             rng).sum(axis=1).astype(float)
        S = np.mean([psd_single_sided(s, dt)[1]
                     for s in x.reshape(n_seg, n_len)], axis=0)[1:-1]
        means.append(S)
        a = fit_telegraph_psd(f, S, fit_floor=False, n_avg=n_seg, dt=dt,
                              n_record=n_len)
        b = fit_telegraph_psd(f, S, fit_floor=False, n_avg=n_seg, dt=dt)
        assert a.n_record == n_len
        pt.append((a.tau_s - tau) / a.sigma["tau_s"])
        pt0.append((b.tau_s - tau) / b.sigma["tau_s"])
        chi.append(a.chi2 / a.chi2_dof)
    assert abs(np.mean(pt)) < 0.5             # measured 0.06
    assert 0.5 < np.std(pt, ddof=1) < 1.3      # measured 0.76
    assert np.mean(pt0) < -2.0                 # measured -2.9
    assert np.mean(chi) == pytest.approx(1.0, abs=0.03)   # measured 1.012
    Sm = np.mean(means, axis=0)
    se = Sm / np.sqrt(30 * n_seg)
    E = telegraph_psd_model(f, S0, tau, dt=dt, n_record=n_len)
    Einf = telegraph_psd_model(f, S0, tau, dt=dt)
    assert np.max(np.abs(Sm - E) / se) < 3.5   # measured 2.3
    assert np.max(np.abs(Sm - Einf) / se) > 6  # measured 8.5
    plan = plan_psd_measurement(f, n_seg, S0, tau, 0.0, fit_floor=False,
                                dt=dt, n_record=n_len)
    assert plan["identifiable"]


@pytest.mark.parametrize("sampling", ["instant", "average"])
def test_record_length_bias_on_exact_gamma_scatter(sampling):
    """Even 1024-sample segments of 256 tau leak enough to bias tau at
    32 averages: 300 spectra with the exact Gamma scatter around the
    finite-record expectation, dt = tau/4. Without n_record the errors of
    tau, in units of the error bar, have mean -0.22 (instant) and -0.40
    (average); with n_record -0.007 (standard error 0.06)."""
    S0, tau, dt, N = 1.68, 1.0, 0.25, 1024
    f = np.fft.rfftfreq(N, dt)[1:-1]
    E = telegraph_psd_model(f, S0, tau, dt=dt, sampling=sampling,
                            n_record=N)
    rng = np.random.default_rng(3)
    with_rec, without = [], []
    for _ in range(300):
        S = E * rng.gamma(32, 1 / 32, f.size)
        a = fit_telegraph_psd(f, S, fit_floor=False, n_avg=32, dt=dt,
                              sampling=sampling, n_record=N)
        b = fit_telegraph_psd(f, S, fit_floor=False, n_avg=32, dt=dt,
                              sampling=sampling)
        with_rec.append((a.tau_s - tau) / a.sigma["tau_s"])
        without.append((b.tau_s - tau) / b.sigma["tau_s"])
    se = 1.0 / np.sqrt(300)
    assert abs(np.mean(with_rec)) < 3 * se     # measured -0.007 (both)
    assert np.mean(without) < -0.15            # measured -0.22, -0.40


# ---- n_avg per point -------------------------------------------------------------

def test_per_point_n_avg_equals_the_scalar_fit_and_plan():
    rng = np.random.default_rng(3)
    S0, tau, fl = 1e-3, 3e-4, 1e-5
    f = np.geomspace(60, 6000, 48)
    S = telegraph_psd_model(f, S0, tau, fl) * rng.gamma(4.0, 0.25, f.size)
    a = fit_telegraph_psd(f, S, n_avg=4)
    b = fit_telegraph_psd(f, S, n_avg=np.full(f.size, 4.0))
    for k in ("S0", "tau_s", "floor", "chi2"):
        assert getattr(b, k) == pytest.approx(getattr(a, k), rel=1e-7,
                                              abs=0)   # measured 2e-8
    for k in a.sigma:
        assert b.sigma[k] == pytest.approx(a.sigma[k], rel=1e-7, abs=0)
    pa = plan_psd_measurement(f, 4, S0, tau, fl)
    pb = plan_psd_measurement(f, np.full(f.size, 4.0), S0, tau, fl)
    for k in pa["sigma"]:
        assert pb["sigma"][k] == pytest.approx(pa["sigma"][k], rel=1e-12,
                                               abs=0)  # measured 7e-16
    with pytest.raises(ValueError):
        fit_telegraph_psd(f, S, n_avg=np.full(f.size - 1, 4.0))
    with pytest.raises(ValueError):
        fit_telegraph_psd(f, S, n_avg=np.full(f.size, 0.5))


def test_per_point_n_avg_on_exact_gamma_scatter():
    """300 spectra whose points average from 1 to 40 periodogram values
    (exact Gamma scatter). With the array the fit is unbiased (mean of
    ln(S0_fit / S0) within 3 standard errors of 0), chi2/dof is 1 and
    the scatter of tau is the planned error bar; one mean number of
    averages for all points biases S0 by -25 %."""
    rng = np.random.default_rng(11)
    S0, tau, fl = 1e-3, 3e-4, 1e-5
    f = np.geomspace(60, 6000, 48)
    n_i = np.round(np.geomspace(1, 40, f.size))
    true = telegraph_psd_model(f, S0, tau, fl)
    ls, chi, taus, ls_scalar = [], [], [], []
    for _ in range(300):
        S = true * rng.gamma(n_i, 1.0 / n_i)
        a = fit_telegraph_psd(f, S, n_avg=n_i)
        ls.append(np.log(a.S0 / S0))
        chi.append(a.chi2 / a.chi2_dof)
        taus.append(a.tau_s)
        ls_scalar.append(np.log(fit_telegraph_psd(
            f, S, n_avg=float(np.mean(n_i))).S0 / S0))
    assert abs(np.mean(ls)) < 3 * np.std(ls, ddof=1) / np.sqrt(300)  # 1.8
    assert np.mean(chi) == pytest.approx(1.0, abs=0.03)   # measured 0.993
    plan = plan_psd_measurement(f, n_i, S0, tau, fl)
    assert np.std(taus, ddof=1) == pytest.approx(plan["sigma"]["tau_s"],
                                                 rel=0.15, abs=0)  # 0.6 %
    assert np.mean(ls_scalar) < -0.2          # measured -0.28


def test_averages_for_tau_with_the_new_options():
    S0, tau, dt, N = 1.0, 1.0, 0.12, 128
    f = np.fft.rfftfreq(N, dt)[1:-1]
    for kw in (dict(sampling="average"), dict(n_record=N)):
        n, plan = averages_for_tau(0.01, f, S0, tau, 0.0, fit_floor=False,
                                   dt=dt, **kw)
        assert plan["sigma"]["tau_s"] <= 0.01
        prev = plan_psd_measurement(f, n - 1, S0, tau, 0.0,
                                    fit_floor=False, dt=dt, **kw)
        assert n == 1 or prev["sigma"]["tau_s"] > 0.01


# ---- decoder with one noise width per state --------------------------------------

def _unequal_trace(seed, n=20000, s0=0.2, s1=0.45):
    rng = np.random.default_rng(seed)
    occ = telegraph_traces(0.3, 1e-3, 5e-5, n, 1, rng)[:, 0]
    y = occ + np.where(occ == 1, s1, s0) * rng.standard_normal(n)
    return occ, y


def test_two_equal_widths_are_the_one_width_model():
    occ, y = _unequal_trace(1)
    one = TelegraphHMM(0.004, 0.009, (0.0, 1.0), 0.3)
    two = TelegraphHMM(0.004, 0.009, (0.0, 1.0), (0.3, 0.3))
    assert one.equal_noise and not two.equal_noise
    # measured: equal to the last bit
    assert two.loglik(y) == pytest.approx(one.loglik(y), rel=1e-14, abs=0)
    assert np.array_equal(two.viterbi(y), one.viterbi(y))
    assert np.allclose(two.posterior(y), one.posterior(y), rtol=0,
                       atol=1e-14)
    assert isinstance(one.sigma, float) and two.sigma.shape == (2,)
    with pytest.raises(ValueError):
        TelegraphHMM(0.004, 0.009, (0.0, 1.0), (0.3, 0.3, 0.3))
    with pytest.raises(ValueError):
        TelegraphHMM(0.004, 0.009, (0.0, 1.0), (0.3, -0.1))


def test_score_with_two_widths_matches_finite_differences():
    _, y = _unequal_trace(2, n=4000)
    m = TelegraphHMM(0.004, 0.011, (0.05, 0.93), (0.25, 0.4))
    g = m.score(y)
    p = m._params()
    for k in range(6):
        h = 1e-6 * abs(p[k])
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        fd = (TelegraphHMM._from_params(pp).loglik(y)
              - TelegraphHMM._from_params(pm).loglik(y)) / (2 * h)
        assert g[k] == pytest.approx(fd, rel=1e-5, abs=0)   # measured 3e-9


def test_batched_viterbi_with_two_widths():
    _, y = _unequal_trace(3, n=3000)
    m = TelegraphHMM(0.004, 0.011, (0.0, 1.0), (0.2, 0.45))
    Y = np.stack([y, y[::-1], y + 0.3])
    paths = _viterbi_batch(m, Y)
    for row, p in zip(Y, paths):
        assert np.array_equal(p, m.viterbi(row))


def test_fit_with_two_widths_recovers_them():
    """16 simulated traces with widths 0.2 (empty) and 0.45 (filled):
    the errors of tau, f and both widths divided by their error bars
    scatter with mean near 0 and spread near 1 (measured: means 0.07,
    -0.12, 0.39, -0.16 and spreads 0.95, 1.05, 1.04, 0.65; over 60 other
    traces the tau errors had mean -0.03 and spread 1.03). The one-width
    model on the same traces puts tau 37 % short."""
    dt = 5e-5
    pulls = {k: [] for k in ("tauA", "f", "sigma0", "sigma1")}
    short = []
    for seed in range(16):
        occ, y = _unequal_trace(200 + seed)
        m, ll = fit_hmm(y, equal_noise=False)
        assert np.all(np.diff(ll) > -1e-6 * np.abs(ll[1:]))
        f, tA = m.rates(dt)
        err = m.uncertainties(y, dt)["sigma"]
        pulls["tauA"].append((tA - 1e-3) / err["tauA"])
        pulls["f"].append((f - 0.3) / err["f"])
        pulls["sigma0"].append((m.sigmas[0] - 0.2) / err["sigma0"])
        pulls["sigma1"].append((m.sigmas[1] - 0.45) / err["sigma1"])
        one, _ = fit_hmm(y)
        short.append(one.rates(dt)[1] / 1e-3 - 1)
    for k, p in pulls.items():
        assert abs(np.mean(p)) < 0.8, k        # measured: see above
        assert 0.5 < np.std(p, ddof=1) < 1.6, k
    assert np.mean(short) < -0.25             # measured -0.37
    # the default is the one-width model of earlier releases
    _, y = _unequal_trace(5)
    a, _ = fit_hmm(y)
    assert a.equal_noise and isinstance(a.sigma, float)


def test_dwell_test_runs_with_two_widths():
    occ, y = _unequal_trace(7)
    m, _ = fit_hmm(y, equal_noise=False)
    res = markov_dwell_test(y, m, n_boot=60, seed=1)
    assert 0.0 < res["p_value"] <= 1.0
    assert np.isfinite(res["correlation"])


# ---- successive stays -------------------------------------------------------------

def test_successive_correlation_by_hand():
    # complete stays (first and last cut): 0:2, 1:3, 0:4, 1:1, 0:6, 1:2
    path = [1, 1] + [0] * 2 + [1] * 3 + [0] * 4 + [1] + [0] * 6 + [1] * 2 \
        + [0] * 5
    u0 = (np.array([1, 2, 3]) - 0.5) / 3      # ranks of 2, 4, 6 among 0s
    u1 = (np.array([3, 1, 2]) - 0.5) / 3      # ranks of 3, 1, 2 among 1s
    u = np.array([u0[0], u1[0], u0[1], u1[1], u0[2], u1[2]])
    ref = np.corrcoef(u[:-1], u[1:])[0, 1]
    assert _successive_correlation(path) == pytest.approx(ref, rel=1e-12,
                                                          abs=1e-15)  # 0
    assert np.isnan(_successive_correlation([0, 1, 0, 1]))


def test_correlation_of_successive_stays():
    """Linked stays: each empty stay and the filled stay after it come
    from the same uniform number, so each state alone has exactly
    geometric stays (the dispersion test sees nothing) while successive
    stays are rank-correlated. On Markov traces the p-values spread
    over (0, 1) as they should (over 40 traces during development:
    median 0.56, 2.5 % below 0.1); the 10 checked here give mean 0.55
    and smallest 0.12."""
    rng = np.random.default_rng(2)
    parts, n = [], 20000
    while sum(len(p) for p in parts) < n:
        u = rng.random()
        k0 = int(np.ceil(np.log1p(-u) / np.log1p(-1 / 47.0)))
        k1 = int(np.ceil(np.log1p(-u) / np.log1p(-1 / 20.0)))
        parts += [np.zeros(max(k0, 1), int), np.ones(max(k1, 1), int)]
    occ = np.concatenate(parts)[:n]
    y = occ + rng.normal(0.0, 0.3, n)
    m, _ = fit_hmm(y)
    res = markov_dwell_test(y, m, n_boot=200, seed=0)
    assert res["p_value"] > 0.1               # measured 1.0
    assert res["correlation"] > 0.3           # measured 0.51
    assert res["p_correlation"] <= 0.01       # measured 0.00995 (2/201)
    pc = []
    for seed in range(10):
        r = np.random.default_rng(1000 + seed)
        occ = telegraph_traces(0.3, 1e-3, 5e-5, n, 1, r)[:, 0]
        y = occ + r.normal(0.0, 0.3, n)
        m, _ = fit_hmm(y)
        pc.append(markov_dwell_test(y, m, n_boot=100, seed=seed)
                  ["p_correlation"])
    assert min(pc) > 0.05 and 0.3 < np.mean(pc) < 0.75


# ---- exhaustive design ------------------------------------------------------------

def test_exhaustive_design_is_the_best_subset():
    Ic0, Tc, tau, sig = 2.0e-6, 0.75, 0.78, 0.02e-6
    cand = np.linspace(0.03, 0.70, 11)
    J = _jacobian(cand, Ic0, Tc, tau, 601) / sig
    best, best_ld = None, -np.inf
    for s in combinations(range(cand.size), 4):
        sign, ld = np.linalg.slogdet(J[list(s)].T @ J[list(s)])
        if sign > 0 and ld > best_ld:
            best, best_ld = list(s), ld
    ex = design_ic_temperatures(cand, 4, sig, Ic0, Tc, tau,
                                method="exhaustive")
    assert ex["indices"] == best
    gr = design_ic_temperatures(cand, 4, sig, Ic0, Tc, tau)
    ld_gr = np.linalg.slogdet(J[gr["indices"]].T @ J[gr["indices"]])[1]
    assert best_ld >= ld_gr
    # measured: equal to the last bit
    assert ex["log_det"] == pytest.approx(best_ld, rel=1e-12, abs=0)
    assert gr["log_det"] == pytest.approx(ld_gr, rel=1e-12, abs=0)
    plan = plan_ic_measurement(cand[best], sig, Ic0, Tc, tau)
    assert ex["sigma"] == plan["sigma"]
    with pytest.raises(ValueError):
        design_ic_temperatures(np.linspace(0.03, 0.7, 60), 10, sig, Ic0,
                               Tc, tau, method="exhaustive")
    with pytest.raises(ValueError):
        design_ic_temperatures(cand, 4, sig, Ic0, Tc, tau, method="best")


# ---- heater calibration -------------------------------------------------------------

def _heating_data(seed, k=1.0):
    r = RECIPES[1]
    P = np.concatenate([[0.0, 0.0], np.geomspace(1e-17, 3e-14, 12)])
    rng = np.random.default_rng(seed)
    Te = ep_temperature(k * P, 0.03, r.area, 2.5, 3.0) \
        + rng.normal(0.0, 1e-3, P.size)
    return P, Te, r.area


def test_power_calibration_scales_sigma_exactly():
    P, Te, A = _heating_data(5)
    a = fit_ep_coupling(P, Te, A, 0.03, sigma_Te=1e-3, fit_Tp=True)
    b = fit_ep_coupling(1.3 * P, Te, A, 0.03, sigma_Te=1e-3, fit_Tp=True)
    # measured 3.1e-9, 6.7e-10 and 6.6e-10 (the optimizer's tolerance)
    assert b.Sigma == pytest.approx(1.3 * a.Sigma, rel=1e-8, abs=0)
    assert b.delta == pytest.approx(a.delta, rel=1e-8, abs=0)
    assert b.Tp == pytest.approx(a.Tp, rel=1e-8, abs=0)
    c = fit_ep_coupling(P, Te, A, 0.03, sigma_Te=1e-3, fit_Tp=True,
                        power_scale_error=0.05)
    assert c.sigma["Sigma"] ** 2 == pytest.approx(
        a.sigma["Sigma"] ** 2 + (0.05 * a.Sigma) ** 2, rel=1e-12,
        abs=0)                                            # measured 2e-16
    for k in ("delta", "Tp"):
        assert c.sigma[k] == a.sigma[k]
    assert c.power_scale_error == 0.05 and a.power_scale_error == 0.0
    with pytest.raises(ValueError):
        fit_ep_coupling(P, Te, A, 0.03, sigma_Te=1e-3, power_scale_error=-1)


def test_power_calibration_error_matches_monte_carlo():
    """150 heating curves, each with a heater calibration drawn from
    1 +- 5 %: the scatter of Sigma matches the error bar that includes
    the calibration (and is far above the one without it)."""
    rng = np.random.default_rng(8)
    S, err, err0 = [], [], []
    for i in range(150):
        k = 1.0 + 0.05 * rng.standard_normal()
        P, Te, A = _heating_data(1000 + i, k)
        fit = fit_ep_coupling(P, Te, A, 0.03, sigma_Te=1e-3, delta=3,
                              power_scale_error=0.05)
        S.append(fit.Sigma)
        err.append(fit.sigma["Sigma"])
        err0.append(np.sqrt(fit.sigma["Sigma"] ** 2 - (0.05 * fit.Sigma) ** 2))
    assert np.std(S, ddof=1) == pytest.approx(np.mean(err), rel=0.15,
                                              abs=0)   # measured 4.1 %
    assert np.std(S, ddof=1) > 3 * np.mean(err0)   # measured 9.0 times


# ---- error bars of the click Monte Carlo ------------------------------------------

def test_click_monte_carlo_error_bars_match_the_seed_scatter():
    """24 independent seeds of 60 records each: the scatter of the
    efficiency, the dark fraction and the SNR matches the mean of their
    stated standard errors within 35 % (the standard error of a spread
    from 24 values is about 15 %). Measured: -0.2 %, -3.7 % and +7.9 %."""
    from absnoise import matched_recipe
    b = SensorBudget(matched_recipe(0.05, W=5.3e-6, L=0.5e-6))
    from absnoise import H_PLANCK
    E = H_PLANCK * 26e9
    runs = [click_monte_carlo(b, E, tauA=1e-7, T0=0.05, n_trials=60,
                              seed=s) for s in range(24)]
    for key in ("efficiency", "dark_fraction", "snr"):
        vals = np.array([r[key] for r in runs])
        errs = np.array([r[key + "_err"] for r in runs])
        assert np.std(vals, ddof=1) == pytest.approx(np.mean(errs),
                                                     rel=0.35, abs=0), key


# ---- fixes (each fails on 0.16.0) ---------------------------------------------------

def test_allan_variance_refuses_a_window_that_is_not_whole():
    x = np.random.default_rng(0).normal(size=100)
    with pytest.raises(ValueError):           # 0.16.0: used m = 2
        allan_variance(x, 1.0, m_list=[2.5])
    taus, a = allan_variance(x, 1.0, m_list=[2.0, 4])
    assert np.array_equal(taus, [2.0, 4.0])


def test_avar_white_refuses_non_finite_input():
    for S0, T in ((np.nan, 1.0), (1.0, np.nan), (np.inf, 1.0)):
        with pytest.raises(ValueError):       # 0.16.0: NaN or inf
            avar_white(S0, T)


def test_ic_model_refuses_parameters_out_of_range():
    for tau in (1.2, 0.0, -0.3, np.nan):
        with pytest.raises(ValueError):       # 0.16.0: NaN
            ic_model(0.1, 1e-6, 0.75, tau)
    for Tc in (0.0, -1.0, np.inf):
        with pytest.raises(ValueError):
            ic_model(0.1, 1e-6, Tc, 0.5)
    with pytest.raises(ValueError):
        ic_model(-0.1, 1e-6, 0.75, 0.5)       # 0.16.0: NaN
    assert ic_model(0.8, 1e-6, 0.75, 1.0) == 0.0


def test_gap_refuses_negative_or_non_finite_temperature():
    for T in (-0.1, np.nan, np.array([0.1, -0.1])):
        with pytest.raises(ValueError):       # 0.16.0: NaN
            gap_bcs(T, 1.0, 1.0)
    with pytest.raises(ValueError):
        gap_bcs(0.1, 0.0, 1.0)
    assert gap_bcs(0.0, 1.0, 2.0) == 2.0


def test_master_equation_helpers_check_their_inputs():
    with pytest.raises(ValueError):           # 0.16.0: about 4.6e149
        noneq_penalty(0.0, 0.3)
    with pytest.raises(ValueError):           # 0.16.0: 0
        noneq_penalty(0.2, 1.3)
    assert noneq_penalty(0.2, 0.3) == pytest.approx(
        np.sqrt(0.21 / 0.16), rel=1e-15, abs=0)           # measured 0
    with pytest.raises(ValueError):           # 0.16.0: ZeroDivisionError
        tau_activated(1e-23, 2e-23, 0.0, 1e-6)
    with pytest.raises(ValueError):
        tau_activated(1e-23, 2e-23, 0.1, -1e-6)
    assert tau_activated(3e-23, 2e-23, 0.1, 1e-6) == 1e-6
    with pytest.raises(ValueError):           # 0.16.0: negative rates
        channel_generator(0.3, -1.0, 0.0)
    W = channel_generator(0.3, 2.0, 0.5)
    assert np.allclose(W.sum(axis=0), 0.0, rtol=0, atol=1e-15)  # 1.1e-16


def test_steady_temperature_checks_its_inputs():
    with pytest.raises(ValueError):           # 0.16.0: NaN
        steady_temperature(1e-15, -0.1, 1e-12)
    with pytest.raises(ValueError):
        steady_temperature(1e-15, 0.1, -1e-12)
    with pytest.raises(ValueError):
        steady_temperature(np.nan, 0.1, 1e-12)
    T = steady_temperature(1e-15, 0.1, 1e-12)
    assert T == pytest.approx((0.1 ** 3 + 1e-15 / 2e-12) ** (1 / 3),
                              rel=1e-15, abs=0)           # measured 0


def test_hmm_refuses_non_finite_parameters():
    with pytest.raises(ValueError, match="sigma"):   # 0.16.0: accepted
        TelegraphHMM(0.1, 0.1, (0.0, 1.0), np.nan)
    with pytest.raises(ValueError, match="levels"):
        TelegraphHMM(0.1, 0.1, (0.0, np.nan), 0.3)


def test_freq_noise_spectrum_takes_a_list():
    b = SensorBudget(RECIPES[1])
    a = b.freq_noise_spectrum(0.2, 1e-6, [0.0, 1e6])   # 0.16.0: TypeError
    c = b.freq_noise_spectrum(0.2, 1e-6, np.array([0.0, 1e6]))
    assert np.array_equal(a[0], c[0]) and np.array_equal(a[1], c[1])
    with pytest.raises(ValueError):
        b.freq_noise_spectrum(0.2, 1e-6, [np.nan])


def test_matched_tc_refuses_an_unreachable_ratio():
    for ratio in (20.0, -1.0, np.nan):
        with pytest.raises(ValueError, match="must lie between"):
            matched_Tc(0.05, ratio=ratio)
    # the README value 0.0776059 K; measured 0.07760594
    assert matched_Tc(0.05) == pytest.approx(0.0776059, abs=1e-7)


def test_click_functions_check_the_time_grid():
    from absnoise import matched_recipe
    b = SensorBudget(matched_recipe(0.05, W=5.3e-6, L=0.5e-6))
    for kw in (dict(dt=-1e-10), dict(dt=0.0), dict(dt2=np.nan),
               dict(t_end=-1.0)):
        with pytest.raises(ValueError):       # 0.16.0: ran, or divided by 0
            click_template(b, 1e-22, 1e-7, 0.05, **kw)
    with pytest.raises(ValueError):           # 0.16.0: snr = inf
        click_monte_carlo(b, 1e-22, 1e-7, 0.05, n_trials=1)
    for kw in (dict(dt=0.0), dict(dt=-1e-10), dict(t_end=np.nan)):
        with pytest.raises(ValueError):       # 0.16.0: ZeroDivisionError
            two_temperature_click(b, 1e-22, 1e-7, 0.05, 1e-20, 1.0, 4.0,
                                  **kw)


def test_telegraph_traces_refuse_a_count_that_is_not_whole():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):           # 0.16.0: 2 steps
        telegraph_traces(0.3, 1.0, 0.1, 2.7, 1, rng)
    assert telegraph_traces(0.3, 1.0, 0.1, 3.0, 2, rng).shape == (3, 2)


def test_save_functions_refuse_what_load_refuses():
    path = os.path.join(tempfile.mkdtemp(), "x.csv")
    with pytest.raises(ValueError):           # 0.16.0: wrote "nan"
        save_trace_csv(path, 1.0, [1.0, np.nan])
    with pytest.raises(ValueError):
        save_ic_csv(path, [0.1, 0.05], [1e-6, 2e-6])
    with pytest.raises(ValueError):
        save_ic_csv(path, [0.05, 0.1], [1e-6, np.inf])
    save_ic_csv(path, [0.05, 0.1], [2e-6, 1e-6], [1e-8, 1e-8])
    T, Ic, s = load_ic_csv(path)
    assert np.array_equal(T, [0.05, 0.1]) and np.array_equal(s, [1e-8, 1e-8])


def test_version_is_0_17_0():
    import absnoise
    assert absnoise.__version__ == "0.17.0"
    path = os.path.join(ROOT, "CITATION.cff")
    if not os.path.isfile(path):
        pytest.skip("CITATION.cff is not part of this tree")
    with open(path) as fh:
        text = fh.read()
    assert "version: 0.17.0" in text
    assert "date-released: 2026-11-25" in text
