"""0.14.0 changes, each checked against an independent reference.

Fix (fails on 0.13.0):

* `FiniteLJunction.level_data` for |phi| < dphi, where the two levels of
  an Andreev doublet cross: against the level slopes and curvatures
  from implicit differentiation of the level condition along each
  branch, coded here (its own root finding, analytic derivatives).

New:

* `JunctionModel.Ic_matsubara`, `calibrate(method="matsubara")` and
  `junction_properties(method="matsubara")`: against the closed-form
  short-junction critical current (zero length) and the independent
  level-plus-continuum path (finite length);
* `matched_filter_sigma(..., f_max)`, `energy_resolution(..., f_max)`
  and `SensorBudget.bandwidth_for_resolution`: the band integral
  against adaptive quadrature (every branch of the closed form), the
  budget against quadrature of 4 / NEP^2 from `nep_spectrum`, the
  low-bandwidth limit, the 0.10.x grid integral over its own band, and
  the closed-form bandwidth without a readout floor;
* `TelegraphPSDFit.cov` against the scatter of 400 simulated spectra.

Speed (result checked against the 0.13.0 code, kept here):

* `markov_dwell_test` decodes its simulated traces in batches; the
  paths are integers and are compared exactly, and the test result
  equals that of the 0.13.0 loop exactly.

Tutorial: `examples/tutorial_readout_bandwidth.py` is run and its
printed numbers checked.
"""
import dataclasses
import os

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq

from absnoise import (RECIPES, FiniteLJunction, JunctionModel, SensorBudget,
                      abs_energies, fit_hmm, fit_telegraph_psd, gap_bcs,
                      junction_properties, markov_dwell_test,
                      matched_filter_sigma, telegraph_psd_model,
                      telegraph_traces)
from absnoise.budgets import _quartic_band_integral
from absnoise.constants import E_CHARGE, HBAR, KB
from absnoise.decode import (_dispersion, _simulate_path, _viterbi_batch,
                             dwell_times)

from test_v013 import _NUM, _docstring_output, _run_example, _same_print

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _narrow(idx, W, L=None):
    r = dataclasses.replace(RECIPES[idx], W=W)
    return r if L is None else dataclasses.replace(r, L=L)


# ---- 0.13.0 code paths, kept as references ---------------------------------

def _level_data_0130(fl, phi, T, dphi=1e-3):
    D = float(gap_bcs(T, fl.Tc, fl.Delta0))
    E0, E1, E2 = [], [], []
    for c in fl.cs:
        Em = abs_energies(phi - dphi, fl.tau, c, D)
        Ec = abs_energies(phi, fl.tau, c, D)
        Ep = abs_energies(phi + dphi, fl.tau, c, D)
        n = min(len(Em), len(Ec), len(Ep))
        for i in range(n):
            E0.append(Ec[i])
            E1.append((Ep[i] - Em[i]) / (2 * dphi))
            E2.append((Ep[i] - 2 * Ec[i] + Em[i]) / dphi ** 2)
    return np.array(E0), np.array(E1), np.array(E2), D


def _ratio_from_levels(E, E1, E2, T, which):
    """The sums of `FiniteLJunction.andreev_sums` and the saturation
    ratio, written out here from level data."""
    v = 1.0 / (2.0 * np.cosh(E / (2 * KB * T)) ** 2)
    g = (2 * E_CHARGE / HBAR) * (-(E1 if which == "I" else E2))
    R = -2.0 * np.sum(g * v * E) / (KB * T ** 2)
    S0 = 8.0 * np.sum(g ** 2 * v)
    CA = 2.0 * np.sum(v * E ** 2) / (KB * T ** 2)
    with np.errstate(divide="ignore"):         # R = 0 for "I" at phi = 0
        ratio = float(np.sqrt(S0 / (2 * R ** 2) / (2 * KB * T ** 2 / CA)))
    return R, S0, CA, ratio


def _markov_dwell_test_0130(y, model, n_boot, seed):
    """The bootstrap loop of 0.13.0 (one `viterbi` call per trace)."""
    def stats(path):
        d = dwell_times(path)
        return {s: (_dispersion(d[s]) if d[s].size >= 2
                    and d[s].mean() > 1.0 else np.nan) for s in (0, 1)}
    disp = stats(model.viterbi(y))
    rng = np.random.default_rng(seed)
    boot = {0: [], 1: []}
    for _ in range(n_boot):
        path = _simulate_path(model.p_up, model.p_dn, y.size, rng)
        yb = model.means[path] + model.sigma * rng.standard_normal(y.size)
        db = stats(model.viterbi(yb))
        for s in (0, 1):
            boot[s].append(db[s])
    p_state = {}
    for s in (0, 1):
        b = np.asarray(boot[s])
        b = b[np.isfinite(b)]
        hi = (1.0 + np.sum(b >= disp[s])) / (b.size + 1.0)
        lo = (1.0 + np.sum(b <= disp[s])) / (b.size + 1.0)
        p_state[s] = float(min(1.0, 2.0 * min(hi, lo)))
    return float(min(1.0, 2.0 * min(p_state.values()))), p_state, disp


# ---- finite-length sums near phi = 0 ----------------------------------------

def _analytic_branch_levels(phi, tau, c, D):
    """Levels E, dE/dphi and d2E/dphi2 of one channel by implicit
    differentiation of theta(E) = s a_s(phi) - 2 pi k along each branch,
    theta(E) = 2 arccos(E/D) - c E, a_s = 2 asin(u), u = sqrt(tau)
    sin(phi/2):
      theta'(E) = -2 / sqrt(D^2 - E^2) - c,
      theta''(E) = -2 E / (D^2 - E^2)^(3/2),
      a_s' = 2 u' / sqrt(1 - u^2),  u' = sqrt(tau) cos(phi/2) / 2,
      a_s'' = 2 (u'' (1 - u^2) + u u'^2) / (1 - u^2)^(3/2), u'' = -u/4,
      E' = s a_s' / theta',  E'' = (s a_s'' - theta'' E'^2) / theta'."""
    u = np.sqrt(tau) * np.sin(phi / 2)
    up = np.sqrt(tau) * np.cos(phi / 2) / 2
    upp = -u / 4
    a = 2 * np.arcsin(u)
    a1 = 2 * up / np.sqrt(1 - u * u)
    a2 = 2 * (upp * (1 - u * u) + u * up * up) / (1 - u * u) ** 1.5
    out = []
    for k in range(50):
        for s in (1, -1):
            y = s * a - 2 * np.pi * k
            if not (-c * D < y < np.pi):
                continue
            E = brentq(lambda e: 2 * np.arccos(min(e / D, 1.0)) - c * e - y,
                       0.0, D, xtol=1e-17 * D, rtol=1e-15)
            t1 = -2 / np.sqrt(D * D - E * E) - c
            t2 = -2 * E / (D * D - E * E) ** 1.5
            e1 = s * a1 / t1
            e2 = (s * a2 - t2 * e1 * e1) / t1
            out.append((E, e1, e2))
    return out


@pytest.mark.parametrize("phi", [0.0, 1e-7, 1e-4, 9e-4])
def test_finite_length_sums_follow_the_branches_through_phi_zero(phi):
    """Against the analytic level derivatives. The tolerance 1e-6 holds
    the central-difference error of the stencil, about E'''' dphi^2 / 12
    = 1e-7 relative for levels that vary on a phase scale of 1 rad, and
    the root error (1e-15 Delta, amplified by 1/dphi^2 = 1e6 in E'' to
    1e-9 Delta). 0.13.0 matched the levels by energy order across the
    crossing at phi = 0 and was off by factors of 100 to 1000."""
    r = _narrow(3, 0.6e-6)
    fl = FiniteLJunction(r)
    T = 0.3 * r.Tc
    D = float(gap_bcs(T, r.Tc, r.Delta))
    lev = [x for c in fl.cs for x in _analytic_branch_levels(phi, r.tau, c, D)]
    E, E1, E2 = (np.array(z) for z in zip(*lev))
    for which in ("L", "I"):
        R, S0, CA, ratio = _ratio_from_levels(E, E1, E2, T, which)
        s = fl.andreev_sums(phi, T, which)
        if which == "L":
            assert s["R_occ"] == pytest.approx(R, rel=1e-6)
            assert fl.saturation_ratio(phi, T, "L") == pytest.approx(ratio,
                                                                     rel=1e-6)
            old = _ratio_from_levels(*_level_data_0130(fl, phi, T)[:3],
                                     T, "L")[3]
            assert old > 10 * ratio             # 0.13.0
        else:
            # the current responsivity vanishes at phi = 0 by symmetry;
            # the noise sum does not
            assert abs(s["R_occ"] - R) <= 1e-6 * np.sum(np.abs(
                (2 * E_CHARGE / HBAR) * E1 * E)) / (KB * T ** 2)
            assert s["S0_over_tau"] == pytest.approx(S0, rel=1e-6)
        assert s["C_A"] == pytest.approx(CA, rel=1e-12)


def test_finite_length_full_recipe_near_zero_phase():
    # the numbers of the changelog: Ti/Al(thick) at 0.3 Tc
    r = RECIPES[3]
    fl = FiniteLJunction(r)
    T = 0.3 * r.Tc
    new = fl.saturation_ratio(1e-4, T, "L")
    old = _ratio_from_levels(*_level_data_0130(fl, 1e-4, T)[:3], T, "L")[3]
    assert old == pytest.approx(853.79, rel=1e-4)
    assert new == pytest.approx(1.00747, abs=1e-5)
    # continuous with the old (correct) value just outside the stencil
    # (the ratio itself changes by 2e-8 between the two phases)
    assert new == pytest.approx(fl.saturation_ratio(1.1e-3, T, "L"),
                                rel=1e-7)


@pytest.mark.parametrize("phi", [1e-3, 2e-3, 1.0, 3.0, -0.5])
def test_finite_length_unchanged_away_from_zero_phase(phi):
    fl = FiniteLJunction(_narrow(3, 0.6e-6))
    T = 0.3 * fl.Tc
    new = fl.level_data(phi, T)
    old = _level_data_0130(fl, phi, T)
    for a, b in zip(new[:3], old[:3]):
        assert np.array_equal(a, b)


def test_finite_length_refusals():
    fl = FiniteLJunction(_narrow(3, 0.6e-6))
    with pytest.raises(ValueError):
        fl.andreev_sums(1.0, 0.3, which="inductance")
    with pytest.raises(ValueError):
        fl.saturation_ratio(1.0, fl.Tc)
    with pytest.raises(ValueError):
        fl.andreev_sums(1.0, -0.1)
    with pytest.raises(ValueError):
        fl.level_data(np.nan, 0.3)


# ---- exact critical current and calibration ---------------------------------

def test_ic_matsubara_equals_the_closed_form_at_zero_length():
    r = _narrow(1, 0.3e-6, L=0.0)
    m = JunctionModel(r, n_phi=61)
    K = 2 * m.N * (2 * E_CHARGE / HBAR)
    # 20 mK = 0.027 Tc: tanh(E/2kT) = 1 to 1e-26 at every phase, so the
    # T = 0 closed form of the maximum holds: sin^2(phi*/2) = u*,
    # u* = (1 - sqrt(1 - tau)) / tau
    D, tau = m.Delta, r.tau
    u = (1 - np.sqrt(1 - tau)) / tau
    I0 = K * D * tau / 4 * 2 * np.sqrt(u * (1 - u)) / np.sqrt(1 - tau * u)
    Ic, phc = m.Ic_matsubara(0.02, return_phase=True)
    assert Ic == pytest.approx(I0, rel=1e-9)
    assert phc == pytest.approx(2 * np.arcsin(np.sqrt(u)), abs=1e-6)
    # 0.5 Tc: the maximum of the closed-form current on 200001 phases
    # (grid error below 1e-10 relative)
    T = 0.5 * r.Tc
    m.Delta = D = float(gap_bcs(T, r.Tc, r.Delta))
    ph = np.linspace(0.0, np.pi, 200001)
    E = D * np.sqrt(1 - tau * np.sin(ph / 2) ** 2)
    I = K * D * D * tau * np.sin(ph) / (4 * E) * np.tanh(E / (2 * KB * T))
    assert m.Ic_matsubara(T) == pytest.approx(I.max(), rel=1e-9)


def test_ic_matsubara_agrees_with_the_level_path_and_calibrates():
    r = _narrow(1, 0.3e-6)                         # Ti/Al/Au, 0.3 um wide
    m = JunctionModel(r)
    Ic_lev = m.Ic(0.02)
    Ic_mat, phc = m.Ic_matsubara(0.02, return_phase=True)
    # two independent calculations of the same model (the level path
    # carries its energy and phase grid errors, about 5e-4)
    assert Ic_mat == pytest.approx(Ic_lev, rel=2e-3)
    assert abs(phc - m.phis[int(np.argmax(m.current_phase(0.02)))]) < 0.05
    # the exact maximum is at least as large as the Matsubara current
    # anywhere on a fine grid
    assert Ic_mat >= np.max(m.cpr_matsubara(0.02, np.linspace(0.01, 3.13,
                                                               40)))
    s_lev = m.calibrate()
    assert s_lev == r.Ic20 / Ic_lev                # default unchanged
    s_mat = m.calibrate(method="matsubara")
    assert s_mat == pytest.approx(r.Ic20 / Ic_mat, rel=1e-12)
    assert m.Ic_matsubara(0.02) == pytest.approx(r.Ic20, rel=1e-12)
    with pytest.raises(ValueError):
        m.calibrate(method="exact")
    with pytest.raises(ValueError):
        m.Ic_matsubara(0.0)
    with pytest.raises(ValueError):
        m.Ic_matsubara(0.02, n_grid=3)


def test_junction_properties_matsubara_method():
    r = _narrow(1, 0.3e-6)
    T = np.array([0.225])
    lev = junction_properties(r, T)
    mat = junction_properties(r, T, method="matsubara")
    # the series terms differ only by the calibration scale
    k = mat["scale"] / lev["scale"]
    assert mat["dIdphi0"][0] == pytest.approx(k * lev["dIdphi0"][0], rel=1e-12)
    assert mat["a3"][0] == pytest.approx(k * lev["a3"][0], rel=1e-12)
    m = JunctionModel(r)
    m.Delta = float(gap_bcs(0.225, r.Tc, r.Delta))
    m.scale = mat["scale"]
    assert mat["Ic"][0] == pytest.approx(m.Ic_matsubara(0.225), rel=1e-12)
    assert mat["Ic"][0] == pytest.approx(lev["Ic"][0], rel=3e-3)
    with pytest.raises(ValueError):
        junction_properties(r, T, method="grid")


# ---- band-limited energy resolution -----------------------------------------

def _quad_band(a, b, c, X):
    return quad(lambda x: 1.0 / (a + b * x * x + c * x ** 4), 0.0, X,
                epsabs=0.0, epsrel=1e-13, limit=500)[0]


@pytest.mark.parametrize("abc", [
    (1.0, 3.0, 1.0),                    # real, distinct p, q
    (1.0, 0.5, 1.0),                    # complex p, q
    (1.0, 0.0, 1.0),                    # purely imaginary p, q
    (1.0, 2.0, 1.0),                    # p = q exactly
    (1.0, 2.0 * (1 + 1e-12), 1.0),      # |q - p| below the series switch
    (1.0, 2.0 * (1 - 1e-12), 1.0),
    (1.0, 2.0 * (1 + 2e-10), 1.0),      # just above it
    (2.0, 1.0, 0.0),                    # no x^4 term
    (3.0, 0.0, 0.0),                    # constant
    (1e-30, 1e-48, 1e-66),              # budget-like magnitudes
])
@pytest.mark.parametrize("X", [1e-3, 0.7, 5.0, 300.0])
def test_band_integral_matches_quadrature(abc, X):
    a, b, c = abc
    s = 1e9 if a < 1e-20 else 1.0            # x in units of 1e-9
    assert _quartic_band_integral(a, b, c, X * s) == pytest.approx(
        _quad_band(a, b, c, X * s), rel=1e-10)


def test_band_integral_unlimited_limit():
    for a, b, c in [(1.0, 3.0, 1.0), (1.0, 0.5, 1.0), (2.0, 1.0, 0.0)]:
        lim = _quartic_band_integral(a, b, c, np.inf)
        assert lim == pytest.approx(np.pi / (2 * np.sqrt(a) * np.sqrt(
            b + 2 * np.sqrt(a * c))), rel=1e-15)
        assert _quartic_band_integral(a, b, c, 1e9) == pytest.approx(
            lim, rel=1e-8)


@pytest.mark.parametrize("which", ["L", "I"])
@pytest.mark.parametrize("S_ro", [0.0, 3e-15])
def test_band_limited_resolution_matches_the_nep_integral(which, S_ro):
    b = SensorBudget(RECIPES[1])
    T, tauA = 0.225, 1e-6
    nep2 = lambda f: b.nep_spectrum(T, tauA, np.array([f]), S_ro_y=S_ro,
                                    which=which)["total"][0] ** 2
    prev = np.inf
    for fm in (1e4, 1e6, 3e7, 1e9):
        # quadrature in log f, split at the two corner frequencies
        edges = sorted({1e-6 * fm, 1.0 / (2 * np.pi * b.tau_th(T)),
                        1.0 / (2 * np.pi * tauA), fm})
        edges = [e for e in edges if e <= fm]
        integ = 4 * fm * 1e-6 / nep2(0.0) + sum(
            quad(lambda lf: 4 * np.exp(lf) / nep2(np.exp(lf)), np.log(lo),
                 np.log(hi), epsabs=0, epsrel=1e-12, limit=200)[0]
            for lo, hi in zip(edges[:-1], edges[1:]))
        ref = integ ** -0.5
        got = b.energy_resolution(T, tauA, S_ro_y=S_ro, which=which,
                                  f_max=fm)
        assert got == pytest.approx(ref, rel=1e-8)
        assert got < prev
        prev = got
    assert prev > b.energy_resolution(T, tauA, S_ro_y=S_ro, which=which)


def test_band_limited_resolution_low_bandwidth_limit_and_grid_band():
    b = SensorBudget(RECIPES[1])
    T, tauA = 0.225, 1e-6
    nep0 = b.nep_spectrum(T, tauA, np.array([0.0]))["total"][0]
    # sigma_E^-2 = 4 f_max / NEP(0)^2 for f_max far below every corner
    assert b.energy_resolution(T, tauA, f_max=1.0) == pytest.approx(
        nep0 / np.sqrt(4.0), rel=1e-12)
    # the 0.10.x "grid" method integrates 4 / NEP^2 by the trapezoid
    # rule over [f_lo, f_hi]; the closed form over the same band is
    # sigma(f_hi)^-2 - sigma(f_lo)^-2 (trapezoid error on 6000 log
    # points, about 1e-7)
    tth = b.tau_th(T)
    f_lo = 1.0 / (200 * 2 * np.pi * max(tauA, tth))
    f_hi = 10.0 / (2 * np.pi * min(tauA, tth))
    band = (b.energy_resolution(T, tauA, f_max=f_hi) ** -2
            - b.energy_resolution(T, tauA, f_max=f_lo) ** -2) ** -0.5
    assert b.energy_resolution(T, tauA, method="grid") == pytest.approx(
        band, rel=1e-6)


def test_bandwidth_for_resolution():
    b = SensorBudget(RECIPES[1])
    T, tauA = 0.225, 1e-6
    # no readout floor: closed form f = sqrt(a/b) tan(pi/(2 F^2)) / 2 pi
    P = b.nep_spectrum(T, tauA, np.array([0.0]))["phonon"][0] ** 2
    Q = b.nep_spectrum(T, tauA, np.array([0.0]))["andreev"][0] ** 2
    a_, b_ = P + Q, Q * b.tau_th(T) ** 2
    for F in (1.01, 1.1, 2.0):
        ref = np.sqrt(a_ / b_) * np.tan(np.pi / (2 * F * F)) / (2 * np.pi)
        assert b.bandwidth_for_resolution(T, tauA, F) == pytest.approx(
            ref, rel=1e-9)
    # with a floor: the band-limited resolution at the answer, from
    # quadrature of the NEP, is F times the unlimited value
    S_ro = 3e-15
    fb = b.bandwidth_for_resolution(T, tauA, 1.1, S_ro_y=S_ro)
    assert b.energy_resolution(T, tauA, S_ro_y=S_ro, f_max=fb) == \
        pytest.approx(1.1 * b.energy_resolution(T, tauA, S_ro_y=S_ro),
                      rel=1e-10)
    for bad in (1.0, 0.5, np.nan):
        with pytest.raises(ValueError):
            b.bandwidth_for_resolution(T, tauA, bad)


def test_band_limited_refusals_and_budget_checks():
    b = SensorBudget(RECIPES[1])
    for fm in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError):
            b.energy_resolution(0.225, 1e-6, f_max=fm)
    with pytest.raises(ValueError):
        b.energy_resolution(0.225, 1e-6, method="grid", f_max=1e6)
    # only phonon noise: unbounded with an unlimited band, finite with
    # a limited one (sigma^-2 = 4 f_max / P)
    with pytest.raises(ValueError):
        matched_filter_sigma(1.0, 0.0, 0.0, 1e-9, 1e-6)
    assert matched_filter_sigma(1.0, 0.0, 0.0, 1e-9, 1e-6, f_max=4.0) == \
        pytest.approx(0.25, rel=1e-14)
    for bad in [dict(sigma=0.0), dict(delta=-3), dict(L_r=0.0),
                dict(nu_r=np.nan)]:
        with pytest.raises(ValueError):
            SensorBudget(RECIPES[1], **bad)


# ---- covariance of the spectrum fit ------------------------------------------

def test_psd_fit_covariance_matches_monte_carlo():
    f = np.geomspace(20.0, 8000.0, 120)
    S0, tau, fl, n = 1e-3, 3e-4, 2e-5, 4
    model = telegraph_psd_model(f, S0, tau, fl)
    rng = np.random.default_rng(3)
    est, covs = [], []
    for _ in range(400):
        S = model * rng.gamma(n, 1.0 / n, f.size)
        fit = fit_telegraph_psd(f, S, n_avg=n)
        est.append([fit.S0, fit.tau_s, fit.floor])
        covs.append(fit.cov)
        assert np.allclose(np.sqrt(np.diag(fit.cov)),
                           [fit.sigma["S0"], fit.sigma["tau_s"],
                            fit.sigma["floor"]], rtol=1e-12, atol=0)
    est = np.array(est)
    C_mc = np.cov(est.T)
    C_fit = np.mean(covs, axis=0)
    corr = lambda C, i, j: C[i, j] / np.sqrt(C[i, i] * C[j, j])
    # standard error of a correlation from 400 samples: below 0.05
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert abs(corr(C_mc, i, j) - corr(C_fit, i, j)) < 0.12
    for i in range(3):
        assert np.sqrt(C_mc[i, i] / C_fit[i, i]) == pytest.approx(1, abs=0.15)
    plain = fit_telegraph_psd(f, model * rng.gamma(n, 1.0 / n, f.size),
                              fit_floor=False)
    assert plain.cov.shape == (2, 2)


# ---- batched decoding in markov_dwell_test -----------------------------------

def test_batched_viterbi_returns_the_same_paths():
    rng = np.random.default_rng(5)
    n = telegraph_traces(0.3, 1e-3, 5e-5, 6000, 1, rng)[:, 0]
    model, _ = fit_hmm(n + rng.normal(0.0, 0.3, n.size))
    Y = np.array([model.means[_simulate_path(model.p_up, model.p_dn, 3000,
                                             rng)]
                  + model.sigma * rng.standard_normal(3000)
                  for _ in range(12)])
    Y[2, 50] = 1e4                               # a far outlier
    Y[3] = 0.5 * (model.means[0] + model.means[1])   # ties everywhere
    Y[4] = model.means[0]
    P = _viterbi_batch(model, Y)
    for i in range(len(Y)):
        assert np.array_equal(P[i], model.viterbi(Y[i]))


def test_markov_dwell_test_unchanged():
    rng = np.random.default_rng(2)
    n = telegraph_traces(0.3, 1e-3, 5e-5, 8000, 1, rng)[:, 0]
    y = n + rng.normal(0.0, 0.3, n.size)
    model, _ = fit_hmm(y)
    res = markov_dwell_test(y, model, n_boot=40, seed=1)
    p, p_state, disp = _markov_dwell_test_0130(y, model, 40, 1)
    assert res["p_value"] == p and res["p_state"] == p_state
    assert res["dispersion"] == disp


# ---- tutorial -----------------------------------------------------------------

def test_tutorial_readout_bandwidth_prints_its_documented_result():
    out = _run_example("tutorial_readout_bandwidth.py").splitlines()
    _same_print(out, _docstring_output("tutorial_readout_bandwidth.py"))
    tau = [float(x) for x in _NUM.findall(out[0])]
    assert tau[0] == pytest.approx(1.023, abs=0.002)
    bw = [float(x) for x in _NUM.findall(out[-1])]
    assert bw[1] == pytest.approx(3.08, abs=0.02)


def test_version_matches_pyproject():
    path = os.path.join(ROOT, "pyproject.toml")
    if not os.path.isfile(path):
        pytest.skip("pyproject.toml is not part of this tree")
    with open(path) as fh:
        line = next(ln for ln in fh if ln.startswith("version = "))
    import absnoise
    assert line.split('"')[1] == absnoise.__version__ == "0.14.0"
