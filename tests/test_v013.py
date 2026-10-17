"""0.13.0 changes, each checked against an independent reference.

Fixes (each test fails on 0.12.1):

* `abs_energies` at and near phi = 0: degenerate doublets kept, and the
  branch angle computed without cancellation (residual of the level
  condition with an exactly computed angle; the first-order doublet
  splitting; C_A of `FiniteLJunction` continuous at phi = 0);
* `JunctionModel.dIdphi0` and the dIdphi0 / a3 of `junction_properties`,
  against the closed-form short-junction slope and phi^3 coefficient;
* `ShortJunction.dlnIc_dT` / `dlnI1_dT` far below and just below Tc,
  against `ic_model` (an independent implementation of Ic(T));
* `sigma_spectrum` for tiny occupations and small rates, against the
  closed form of independent spins and scale covariance;
* the readout-floor refusal of `nep_spectrum`, and `continuum_share`
  at low temperature.

New:

* the Matsubara-sum current-phase relation and its Taylor coefficients,
  against the short-junction closed forms (exact), against the
  bound-plus-continuum path of `JunctionModel.current_phase` (an
  independent calculation) and against the Legendre chi function;
* `fit_ep_coupling`, against noise-free data, an independent optimizer
  on an independently coded chi-square, the closed-form error bar of a
  one-parameter fit, Monte Carlo scatter, and the conductance of
  `SensorBudget`; `EPCouplingFit.propagate` against exact linear and
  quadratic results.

Speed (results checked against the 0.12.1 code, kept here):

* `telegraph_traces` (identical states), `free_energy_multi` and
  `free_energy_components` (1e-13), the scalar path of the level
  solver (bit-identical).
"""
import dataclasses
import math
import os
import re
import subprocess
import sys
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq, minimize

import absnoise
from absnoise import (RECIPES, FiniteLJunction, JunctionModel, SensorBudget,
                      ShortJunction, abs_energies, continuum_share,
                      ep_power, ep_temperature, fit_ep_coupling,
                      flip_probabilities, gap_bcs, ic_model,
                      junction_properties, sigma_spectrum,
                      telegraph_traces)
from absnoise._compat import trapezoid
from absnoise.constants import E_CHARGE, HBAR, KB
from absnoise.levels import (_chi2, _theta, continuum_delta,
                             continuum_phase_grid, free_energy_components,
                             matsubara_free_energy_terms)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DELTA = 1.5e-23


# ---- the 0.12.1 code paths, kept as references -----------------------------

def _abs_energies_0121(phi, tau, c, Delta):
    R = np.clip(1.0 - tau + tau * np.cos(phi), -1.0, 1.0)
    a = np.arccos(R)
    lo, hi = -c * Delta, np.pi
    targets, k = [], 0
    while True:
        added = False
        for y in (a - 2.0 * np.pi * k, -a - 2.0 * np.pi * k):
            if lo < y < hi:
                targets.append(y)
                added = True
        if not added and k > 0:
            break
        k += 1
        if k > 200:
            break
    roots = []
    for y in targets:
        try:
            roots.append(brentq(
                lambda E: 2.0 * np.arccos(np.clip(E / Delta, 0.0, 1.0))
                - c * E - y, 0.0, Delta, xtol=1e-16 * Delta + 1e-40,
                rtol=8.9e-16))
        except ValueError:
            pass
    return np.array(sorted(set(roots)))


def _telegraph_0121(f, tauA, dt, n_steps, n_channels, rng):
    p_up, p_dn = flip_probabilities(f, tauA, dt)
    n = (rng.random(n_channels) < f).astype(np.int8)
    out = np.empty((n_steps, n_channels), dtype=np.int8)
    for i in range(n_steps):
        r = rng.random(n_channels)
        n = n ^ (((n == 0) & (r < p_up)) | ((n == 1) & (r < p_dn)))
        out[i] = n
    return out


def _sigma_spectrum_0121(f, Gs, Gp, omegas):
    W = absnoise.channel_generator(f, Gs, Gp)
    pi = np.array([(1 - f) ** 2, f * (1 - f), f * (1 - f), f * f])
    sig = np.array([1.0, 0.0, 0.0, -1.0])
    d = np.sqrt(pi)
    Ws = (W * d[None, :]) / d[:, None]
    assert np.max(np.abs(Ws - Ws.T)) < 1e-12 * max(Gs, Gp, 1.0)
    Ws = 0.5 * (Ws + Ws.T)
    lam, U = np.linalg.eigh(Ws)
    x = d * (sig - np.dot(sig, pi))
    w = (U.T @ x) ** 2
    var = float(np.sum(w))
    S = np.zeros_like(np.asarray(omegas, dtype=float))
    S0 = 0.0
    for wk, lk in zip(w, lam):
        if lk < -1e-14:
            S += 4.0 * wk * (-lk) / (lk ** 2 + np.asarray(omegas) ** 2)
            S0 += 4.0 * wk / (-lk)
    return S, float(S0), var, float(S0 / (4.0 * var)) if var > 0 else 0.0


def _free_energy_multi_0121(m, T_list, n_E=3000):
    T_list = np.atleast_1d(np.asarray(T_list, dtype=float))
    F = np.zeros((len(T_list), len(m.phis)))

    def log2cosh(x):
        x = np.abs(x)
        return x + np.log1p(np.exp(-2.0 * x))

    Efac = continuum_phase_grid(n_E)
    E = m.Delta * Efac
    tanhw = np.tanh(E[None, :] / (2.0 * KB * T_list[:, None]))
    xD = m.Delta / (2.0 * KB * T_list)
    l2cD = 2.0 * KB * T_list * (xD + np.log1p(np.exp(-2.0 * xD)))
    for k in range(m.N):
        for j, p in enumerate(m.phis):
            Eb = m._levels[k][j]
            if len(Eb):
                xb = Eb[None, :] / (2.0 * KB * T_list[:, None])
                F[:, j] += -np.sum(2.0 * KB * T_list[:, None] *
                                   log2cosh(xb), axis=1)
            dph = continuum_delta(p, m.tau, m.cs[k], m.Delta, Efac)
            F[:, j] += (l2cD * dph[0] / np.pi
                        + trapezoid(tanhw * dph[None, :], E,
                                    axis=1) / np.pi)
    return F


def _narrow(idx, W, L=None):
    r = dataclasses.replace(RECIPES[idx], W=W)
    return r if L is None else dataclasses.replace(r, L=L)


def _short_series(N, tau, D, T):
    """Closed-form a1, a3 of the short-junction current
    I = K (tau D^2 / 4) sin(phi) / E * tanh(E / 2kT), K = 2N (2e/hbar),
    E = D sqrt(1 - tau sin^2(phi/2)). With sin(phi) = phi - phi^3/6,
    1/E = (1 + tau phi^2/8)/D and tanh(E/2kT) = th - x0 sech^2(x0)
    tau phi^2 / 8 (x0 = D/2kT, th = tanh x0) to this order:
    a1 = K tau D th / 4,
    a3 = (K tau D / 4) [th (tau/8 - 1/6) - x0 tau sech^2(x0) / 8]."""
    K = 2 * N * (2 * E_CHARGE / HBAR)
    x0 = D / (2 * KB * T)
    th = math.tanh(x0)
    s2 = 1.0 / math.cosh(x0) ** 2
    a1 = K * tau * D * th / 4
    a3 = K * tau * D / 4 * (th * (tau / 8 - 1 / 6) - x0 * tau * s2 / 8)
    return a1, a3


# ---- levels at and near phi = 0 ----------------------------------------------

def test_degenerate_doublets_are_kept_at_phi_zero():
    c, tau = 3.0 / DELTA, 0.7
    E0 = abs_energies(0.0, tau, c, DELTA)
    E1 = abs_energies(1e-6, tau, c, DELTA)
    # 0.12.1: one level at phi = 0 (and for |phi| < 2e-8): the pair merged
    assert len(E0) == len(E1) == 2
    assert np.allclose(E0, E1, rtol=1e-6, atol=0.0)
    assert E0[0] == E0[1]
    assert len(abs_energies(1e-9, tau, c, DELTA)) == 2


@pytest.mark.parametrize("phi", [1e-9, 1e-7, 1e-5])
def test_levels_near_phi_zero_satisfy_the_exact_level_condition(phi):
    # theta(E) = +-a - 2 pi k with the branch angle computed exactly in
    # the test: a = 2 asin(sqrt(tau) sin(phi/2)) = arccos(R)
    c, tau = 3.0 / DELTA, 0.7
    a = 2.0 * math.asin(math.sqrt(tau) * math.sin(phi / 2))
    E = abs_energies(phi, tau, c, DELTA)
    assert len(E) == 2
    th = [2.0 * math.acos(e / DELTA) - c * e for e in E]
    # lower level has the larger theta (+a branch); theta is a
    # difference of two numbers near 1.8, so its rounding is ~1e-15
    # (0.12.1 missed a by 3.5e-13 at phi = 1e-5, by 0.8 % at 1e-7, and
    # merged the pair below phi = 2e-8)
    assert abs(th[0] - a) < 1e-12 * a + 2e-15
    assert abs(th[1] + a) < 1e-12 * a + 2e-15
    # first-order splitting 2 a / |theta'(E)|, theta' = -2/sqrt(D^2-E^2) - c
    Em = 0.5 * (E[0] + E[1])
    dth = 2.0 / math.sqrt(DELTA ** 2 - Em ** 2) + c
    split = 2.0 * a / dth
    assert abs((E[1] - E[0]) / split - 1.0) < 1e-5


def test_levels_away_from_degeneracy_are_unchanged():
    for tau in (0.3, 0.78, 1.0):
        for phi in (0.4, 1.0, 2.0, 3.0):
            for c in (0.0, 1.0 / DELTA, 4.0 / DELTA):
                R = 1 - tau + tau * np.cos(phi)
                if 1 - abs(R) < 1e-3:
                    continue
                assert np.array_equal(abs_energies(phi, tau, c, DELTA),
                                      _abs_energies_0121(phi, tau, c,
                                                         DELTA))


def test_transparent_level_near_pi_is_found():
    # tau = 1 near phi = pi: the level close to zero energy,
    # E = (pi - a) / (2/D + c) to first order, was dropped by 0.12.1
    c = 3.0 / DELTA
    d = 1e-9
    E = abs_energies(np.pi - d, 1.0, c, DELTA)
    assert len(E) == 1
    assert abs(E[0] / (d / (2.0 / DELTA + c)) - 1.0) < 1e-6
    # 0.12.1 found no level here, or (with NumPy 1.24) one at 4e-32 Delta
    old = _abs_energies_0121(np.pi - d, 1.0, c, DELTA)
    assert len(old) == 0 or abs(old[0] / E[0] - 1.0) > 0.5


def test_occupation_heat_capacity_is_continuous_at_phi_zero():
    fl = FiniteLJunction(_narrow(3, 0.6e-6))
    T = 0.3 * fl.Tc
    c0 = fl.andreev_sums(0.0, T)["C_A"]
    c1 = fl.andreev_sums(1e-7, T)["C_A"]
    # 0.12.1: c0 was half of c1
    assert abs(c0 / c1 - 1.0) < 1e-9


def test_theta_scalar_path_is_bit_identical():
    c = 2.0 / DELTA
    for e in np.linspace(-0.1, 1.1, 37) * DELTA:
        assert _theta(float(e), c, DELTA) == (
            2.0 * np.arccos(np.clip(e / DELTA, 0.0, 1.0)) - c * e)


# ---- Matsubara-sum current-phase relation ---------------------------------------

def test_chi2_function():
    assert _chi2(1.0) == pytest.approx(np.pi ** 2 / 8, rel=1e-14)
    assert _chi2(0.0) == 0.0
    for z in (0.1, 0.5, 0.9):
        n = np.arange(400)
        ser = np.sum(z ** (2 * n + 1) / (2 * n + 1) ** 2)
        assert _chi2(z) == pytest.approx(ser, rel=1e-13)


@pytest.mark.parametrize("t", [0.027, 0.3, 0.9])
def test_matsubara_cpr_equals_short_junction_closed_form(t):
    r = _narrow(1, 0.3e-6, L=0.0)
    m = JunctionModel(r, n_phi=61)
    T = t * r.Tc
    D = float(gap_bcs(T, r.Tc, r.Delta))
    m.Delta = D
    K = 2 * m.N * (2 * E_CHARGE / HBAR)
    E = D * np.sqrt(1 - r.tau * np.sin(m.phis / 2) ** 2)
    I_ref = K * (D ** 2 * r.tau * np.sin(m.phis) / (4 * E)) \
        * np.tanh(E / (2 * KB * T))
    I = m.cpr_matsubara(T)
    assert np.max(np.abs(I - I_ref)) < 1e-9 * np.max(I_ref)
    a1, a3 = m.cpr_series(T)
    r1, r3 = _short_series(m.N, r.tau, D, T)
    assert a1 == pytest.approx(r1, rel=1e-9)
    assert a3 == pytest.approx(r3, rel=1e-8)


def test_matsubara_cpr_matches_bound_plus_continuum_path():
    # finite length: two independent calculations of the same model
    r = _narrow(3, 0.4e-6)                      # Ti/Al(thick), L/xi = 0.08
    m = JunctionModel(r, n_phi=61)
    T = 0.3 * r.Tc
    m.Delta = float(gap_bcs(T, r.Tc, r.Delta))
    I_old = m.current_phase(T)
    I_new = m.cpr_matsubara(T)
    sel = (m.phis > 0.5) & (m.phis < 3.0)
    assert np.max(np.abs(I_old - I_new)[sel]) < 2e-3 * np.max(I_new)
    # the slope from the series against an odd polynomial through the
    # bound-plus-continuum current away from phi = 0
    a1, a3 = m.cpr_series(T)
    ph = m.phis[(m.phis > 0.25) & (m.phis < 1.2)]
    Iv = I_old[(m.phis > 0.25) & (m.phis < 1.2)]
    A = np.vstack([ph, ph ** 3, ph ** 5, ph ** 7]).T
    co = np.linalg.lstsq(A, Iv, rcond=None)[0]
    assert co[0] == pytest.approx(a1, rel=3e-3)
    # the series also reproduces the Matsubara current itself at small phi
    p = 1e-3
    assert m.cpr_matsubara(T, p) == pytest.approx(a1 * p + a3 * p ** 3,
                                                  rel=1e-9)


def test_matsubara_terms_refusals():
    for bad in [dict(T=0.0), dict(T=np.nan), dict(Delta=-1.0),
                dict(tau=1.5), dict(c=-1.0)]:
        kw = dict(phi=1.0, tau=0.5, c=0.0, Delta=DELTA, T=0.1)
        kw.update(bad)
        with pytest.raises(ValueError):
            matsubara_free_energy_terms(**kw)


def test_dIdphi0_is_exact_and_old_fit_was_wrong():
    r = _narrow(1, 0.3e-6, L=0.0)
    m = JunctionModel(r, n_phi=121)
    T = 0.3 * r.Tc
    D = float(gap_bcs(T, r.Tc, r.Delta))
    m.Delta = D
    ref, _ = _short_series(m.N, r.tau, D, T)
    assert m.dIdphi0(T) == pytest.approx(ref, rel=1e-9)
    # the 0.12.1 computation: cubic through the first 8 points
    I = m.current_phase(T)
    p = np.polyfit(m.phis[:8], I[:8], 3)
    old = float(np.polyval(np.polyder(p), 0.0))
    assert abs(old / ref - 1.0) > 0.4          # it was 44 % low
    with pytest.warns(DeprecationWarning):
        m.dIdphi0(T, k=8)


def test_junction_properties_uses_exact_series():
    r = _narrow(1, 0.3e-6, L=0.0)
    T = 0.3 * r.Tc
    out = junction_properties(r, [0.1, T], n_phi=61)
    D = float(gap_bcs(T, r.Tc, r.Delta))
    N = JunctionModel(r).N
    a1, a3 = _short_series(N, r.tau, D, T)
    assert out["dIdphi0"][1] == pytest.approx(out["scale"] * a1, rel=1e-9)
    assert out["a3"][1] == pytest.approx(out["scale"] * a3, rel=1e-8)
    # the 0.12.1 values (two-term fit over 0 < phi < 0.76 rad of the
    # 121-point current): a3 off by more than 1 %
    m = JunctionModel(r, n_phi=121)
    m.Delta = D
    m.scale = out["scale"]
    assert abs(m.cpr_coeffs(T)[1] / (out["scale"] * a3) - 1.0) > 0.01


# ---- ShortJunction log slopes ---------------------------------------------------

def _ref_dlnIc(r, T, h):
    """d ln Ic / dT from ic_model (independent code), 5-point stencil."""
    Ic0 = 1.0
    f = [np.log(ic_model(T + k * h, Ic0, r.Tc, r.tau, n_phi=2001))
         for k in (-2, -1, 1, 2)]
    return (f[0] - 8 * f[1] + 8 * f[2] - f[3]) / (12 * h)


def test_log_slopes_far_below_and_just_below_tc():
    sj = ShortJunction(RECIPES[5])               # MoRe, Tc = 7.4 K
    sj.calibrate()
    with warnings.catch_warnings():
        warnings.simplefilter("error")           # 0.12.1: inf / NaN + warnings
        v1, v2 = sj.dlnIc_dT(0.01), sj.dlnI1_dT(0.01)
    assert np.isfinite(v1) and np.isfinite(v2)
    assert abs(v1) < 1e-12 and abs(v2) < 1e-12  # exp(-Delta/kT) ~ e^-1300
    r = RECIPES[1]
    sj = ShortJunction(r)
    sj.calibrate()
    for t in (0.99, 0.999, 0.9999):
        T = t * r.Tc
        v = sj.dlnIc_dT(T)                       # 0.12.1: -inf above 0.998 Tc
        ref = _ref_dlnIc(r, T, 1e-3 * (r.Tc - T))
        assert abs(v / ref - 1.0) < 5e-3
    # unchanged between 2 rel Tc and 0.98 Tc (the 0.12.1 step)
    T = 0.5 * r.Tc
    dT = 2e-3 * r.Tc
    old = (np.log(sj.Ic(T + dT)) - np.log(sj.Ic(T - dT))) / (2 * dT)
    assert sj.dlnIc_dT(T) == old
    # (the grid maximum of Ic makes the slope of a tiny step differ by
    # about 2e-4 from that of the 2e-3 Tc step)
    assert sj.dlnIc_dT(T) == pytest.approx(_ref_dlnIc(r, T, 1e-4), rel=1e-3)
    for bad in (0.0, -0.1, r.Tc, 1.2 * r.Tc, np.nan):
        with pytest.raises(ValueError):
            sj.dlnIc_dT(bad)


# ---- master equation --------------------------------------------------------

def test_sigma_spectrum_tiny_occupation_and_closed_form():
    for f in (1e-200, 1e-170, 0.0, 1.0):
        # 0.12.1: AssertionError (f^2 underflows, or f = 0, 1)
        _, S0, var, te = sigma_spectrum(f, 2.0, 0.0, np.array([0.0]))
        assert var == pytest.approx(2 * f * (1 - f), rel=1e-12, abs=0.0)
        assert S0 == pytest.approx(8 * f * (1 - f) / 2.0, rel=1e-12,
                                   abs=0.0)
        if 0 < f < 1:
            assert te == pytest.approx(0.5, rel=1e-12)
    with pytest.raises(AssertionError), np.errstate(all="ignore"):
        _sigma_spectrum_0121(1e-200, 2.0, 0.0, np.array([0.0]))


def test_sigma_spectrum_is_scale_covariant():
    om = np.array([0.0, 0.3, 2.0])
    S1, S01, v1, t1 = sigma_spectrum(0.25, 1.0, 1.0, om)
    for lam in (1e-20, 1e-15, 1e15):
        S, S0, v, t = sigma_spectrum(0.25, lam, lam, lam * om)
        assert np.allclose(S * lam, S1, rtol=1e-12, atol=0.0)
        assert S0 * lam == pytest.approx(S01, rel=1e-12)
        assert t * lam == pytest.approx(t1, rel=1e-12)
    assert _sigma_spectrum_0121(0.25, 1e-15, 1e-15, om)[1] == 0.0  # 0.12.1


def test_sigma_spectrum_agrees_with_0121_on_ordinary_inputs():
    om = np.linspace(0.0, 5.0, 11)
    for f in (0.01, 0.2, 0.5, 0.9):
        for Gs, Gp in ((1.0, 0.0), (2.0, 0.7), (1.0, 10.0)):
            a = sigma_spectrum(f, Gs, Gp, om)
            b = _sigma_spectrum_0121(f, Gs, Gp, om)
            assert np.allclose(a[0], b[0], rtol=1e-12, atol=0.0)
            for x, y in zip(a[1:], b[1:]):
                assert x == pytest.approx(y, rel=1e-12)
    for bad in [(0.2, 0.0, 1.0), (0.2, -1.0, 0.0), (0.2, 1.0, -1.0),
                (1.2, 1.0, 0.0), (0.2, np.inf, 0.0)]:
        with pytest.raises(ValueError):
            sigma_spectrum(*bad, om)


# ---- budget and continuum share refusals --------------------------------------

def test_nep_spectrum_refuses_a_negative_readout_floor():
    b = SensorBudget(RECIPES[1])
    T = 0.3 * b.recipe.Tc
    for bad in (-1e-18, np.nan, np.inf):
        with pytest.raises(ValueError):         # 0.12.1: NaN readout channel
            b.nep_spectrum(T, 1e-6, np.array([0.0, 1e7]), S_ro_y=bad)


def test_continuum_share_at_low_temperature():
    r = _narrow(1, 0.2e-6)
    # 0.12.1 raised ZeroDivisionError at 0.027 Tc (20 mK for Ti/Al/Au):
    # dI/dT is below rounding there
    out = continuum_share(r, 0.027 * r.Tc, n_phi=41)
    assert np.isfinite(out["share_I"]) and np.isnan(out["share_dIdT"])
    for bad in (0.0, -0.01, r.Tc):
        with pytest.raises(ValueError):
            continuum_share(r, bad, n_phi=41)


# ---- speed-ups against the 0.12.1 code -----------------------------------------

@pytest.mark.parametrize("f,x,ns,nc", [
    (0.3, 0.05, 1, 1), (0.3, 0.05, 700, 1000), (0.02, 3.0, 5000, 3),
    (0.9, 0.5, 2000, 64), (0.5, 7.0, 300, 7), (0.0, 1.0, 50, 4),
    (1.0, 1.0, 50, 4), (0.3, 1e-4, 4000, 2)])
def test_telegraph_traces_identical_to_the_step_loop(f, x, ns, nc):
    a = _telegraph_0121(f, 1.0, x, ns, nc, np.random.default_rng(7))
    b = telegraph_traces(f, 1.0, x, ns, nc, np.random.default_rng(7))
    assert a.dtype == b.dtype and np.array_equal(a, b)


def test_vectorized_free_energy_matches_the_phase_loop():
    r = _narrow(1, 0.3e-6)
    m = JunctionModel(r, n_phi=31)
    m.compute_levels()
    T = 0.3 * r.Tc
    new = m.free_energy_multi([0.02, T])
    old = _free_energy_multi_0121(m, [0.02, T])
    assert np.max(np.abs(new - old) / np.abs(old)) < 1e-13
    Fb, Fc = free_energy_components(m, T)
    ref = _free_energy_multi_0121(m, [T])[0]
    assert np.max(np.abs(Fb + Fc - ref) / np.abs(ref)) < 1e-13


# ---- electron-phonon fit ---------------------------------------------------------

AREA = 4e-13                                     # m^2


def _heating(seed=0, noise=1e-3, Sigma=2.4, delta=3.0, Tp=0.05):
    P = np.concatenate([[0.0, 0.0], np.geomspace(1e-16, 3e-13, 14)])
    Te0 = ep_temperature(P, Tp, AREA, Sigma, delta)
    rng = np.random.default_rng(seed)
    return P, Te0, Te0 + rng.normal(0.0, noise, P.size)


def test_ep_fit_recovers_noise_free_parameters():
    P, Te0, _ = _heating()
    f = fit_ep_coupling(P, Te0, AREA, 0.05, sigma_Te=1e-3)
    assert f.Sigma == pytest.approx(2.4, rel=1e-8)
    assert f.delta == pytest.approx(3.0, rel=1e-8)
    g = fit_ep_coupling(P, Te0, AREA, 0.047, sigma_Te=1e-3, fit_Tp=True)
    assert (g.Sigma, g.delta, g.Tp) == pytest.approx((2.4, 3.0, 0.05),
                                                     rel=1e-7)
    h = fit_ep_coupling(P, Te0, AREA, 0.05, sigma_Te=1e-3, delta=3.0)
    assert h.names == ("Sigma",) and h.Sigma == pytest.approx(2.4,
                                                              rel=1e-10)
    # the model is the closed form of steady_temperature / ep_power
    assert np.allclose(ep_power(f.Te_fit, 0.05, AREA, f.Sigma, f.delta),
                       P, rtol=1e-6, atol=1e-30)


def test_ep_fit_matches_an_independent_optimizer():
    P, _, Te = _heating(seed=3)
    s = 1e-3
    fit = fit_ep_coupling(P, Te, AREA, 0.05, sigma_Te=s)

    def chi2(p):                                 # coded here, not imported
        Sg, d = p
        model = np.power(0.05 ** d + P / (Sg * AREA), 1.0 / d)
        return float(np.sum(((model - Te) / s) ** 2))
    res = minimize(chi2, [2.0, 2.8], method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
    assert res.x[0] == pytest.approx(fit.Sigma, rel=1e-6)
    assert res.x[1] == pytest.approx(fit.delta, rel=1e-6)
    assert chi2(res.x) == pytest.approx(fit.chi2, rel=1e-8)


def test_ep_fit_error_bar_closed_form_for_one_parameter():
    # delta and Tp fixed: dTe/dSigma = -(1/delta) Te^(1-delta) P/(Sigma^2 A)
    P, _, Te = _heating(seed=5)
    s = 1e-3
    fit = fit_ep_coupling(P, Te, AREA, 0.05, sigma_Te=s, delta=3.0)
    Tm = ep_temperature(P, 0.05, AREA, fit.Sigma, 3.0)
    dT = -(1 / 3.0) * Tm ** (-2.0) * P / (fit.Sigma ** 2 * AREA)
    ref = 1.0 / np.sqrt(np.sum((dT / s) ** 2))
    assert fit.sigma["Sigma"] == pytest.approx(ref, rel=1e-6)


def test_ep_fit_error_bars_match_monte_carlo():
    P, Te0, _ = _heating()
    rng = np.random.default_rng(42)
    S, D, sS, sD = [], [], [], []
    for _ in range(150):
        y = Te0 + rng.normal(0.0, 1e-3, Te0.size)
        g = fit_ep_coupling(P, y, AREA, 0.05, sigma_Te=1e-3)
        S.append(g.Sigma)
        D.append(g.delta)
        sS.append(g.sigma["Sigma"])
        sD.append(g.sigma["delta"])
    # standard error of a standard deviation from 150 draws: 5.8 %
    assert 0.8 < np.std(S) / np.mean(sS) < 1.2
    assert 0.8 < np.std(D) / np.mean(sD) < 1.2
    assert abs(np.mean(S) - 2.4) < 4 * np.mean(sS) / np.sqrt(150)
    assert abs(np.mean(D) - 3.0) < 4 * np.mean(sD) / np.sqrt(150)


def test_ep_fit_conductance_matches_the_budget():
    P, _, Te = _heating(seed=1)
    fit = fit_ep_coupling(P, Te, AREA, 0.05, sigma_Te=1e-3)
    r = dataclasses.replace(RECIPES[1], L=0.2e-6, W=2e-6)
    b = SensorBudget(r, sigma=fit.Sigma, delta=fit.delta)
    T = 0.2
    assert fit.G(T) == pytest.approx(b.Gep(T), rel=1e-14)
    h = 1e-6
    num = (ep_power(T + h, 0.05, AREA, fit.Sigma, fit.delta)
           - ep_power(T - h, 0.05, AREA, fit.Sigma, fit.delta)) / (2 * h)
    assert fit.G(T) == pytest.approx(float(num), rel=1e-8)


def test_ep_propagate_linear_and_quadratic():
    P, _, Te = _heating(seed=2)
    fit = fit_ep_coupling(P, Te, AREA, 0.05, sigma_Te=1e-3)
    C = fit.cov
    out = fit.propagate(lambda s, d: 2.0 * s - 3.0 * d)
    ref = np.sqrt(4 * C[0, 0] + 9 * C[1, 1] - 12 * C[0, 1])
    assert out["sigma"] == pytest.approx(ref, rel=1e-10)
    assert out["nonlinearity"] < 1e-6
    q = fit.propagate(lambda s, d: s * s)
    assert q["sigma"] == pytest.approx(2 * fit.Sigma * np.sqrt(C[0, 0]),
                                       rel=1e-9)
    # with Tp fitted, the marginal (Sigma, delta) block is used
    g = fit_ep_coupling(P, Te, AREA, 0.05, sigma_Te=1e-3, fit_Tp=True)
    o = g.propagate(lambda s, d: d)
    assert o["sigma"] == pytest.approx(g.sigma["delta"], rel=1e-10)
    with pytest.raises(ValueError, match="nonlinear"):
        g.propagate(lambda s, d: np.exp(100 * (d - 3.0)))


def test_ep_fit_refusals_and_warning():
    P, Te0, Te = _heating()
    with pytest.raises(ValueError):
        fit_ep_coupling(P, Te[:-1], AREA, 0.05)
    with pytest.raises(ValueError):
        fit_ep_coupling(-P, Te, AREA, 0.05)
    with pytest.raises(ValueError):
        fit_ep_coupling(P, Te, -AREA, 0.05)
    with pytest.raises(ValueError):
        fit_ep_coupling(P, Te, AREA, 0.05, sigma_Te=0.0)
    with pytest.raises(ValueError):
        fit_ep_coupling(P[:3], Te[:3], AREA, 0.05)
    with pytest.raises(ValueError):
        fit_ep_coupling(np.zeros(6), np.full(6, 0.05), AREA, 0.05)
    with pytest.raises(ValueError):
        fit_ep_coupling(P, np.full(P.size, 0.04), AREA, 0.05)   # no heating
    # a delta outside the searched range is refused, not clipped
    Pd = np.geomspace(1e-16, 3e-13, 12)
    with pytest.raises(ValueError):
        fit_ep_coupling(Pd, ep_temperature(Pd, 0.05, AREA, 2.4, 9.5),
                        AREA, 0.05, sigma_Te=1e-4)
    # near-linear data (Te^3 at most 0.8 % above the bath): warns that
    # only G is determined; with noise the exponent is not determined
    # at all and the fit runs to the end of the range (refused)
    Pl = np.geomspace(1e-19, 1e-18, 10)
    Tl = ep_temperature(Pl, 0.05, AREA, 2.4, 3.0)
    with pytest.warns(UserWarning):
        fl = fit_ep_coupling(Pl, Tl, AREA, 0.05, sigma_Te=1e-6)
    assert fl.sigma["delta"] > 100 * fit_ep_coupling(
        P, Te0, AREA, 0.05, sigma_Te=1e-6).sigma["delta"]
    with pytest.raises(ValueError):
        fit_ep_coupling(Pl, Tl + np.random.default_rng(0).normal(
            0, 1e-6, Pl.size), AREA, 0.05, sigma_Te=1e-6)
    with pytest.raises(ValueError):
        ep_temperature(-1.0, 0.05, AREA, 2.4, 3.0)


# ---- tutorials --------------------------------------------------------------------

def _run_example(name):
    if not os.path.isfile(os.path.join(ROOT, "examples", name)):
        pytest.skip("examples/ is not part of this source tree (sdist)")
    env = dict(os.environ, OMP_NUM_THREADS="1",
               PYTHONPATH=os.path.join(ROOT, "src") + os.pathsep
               + os.environ.get("PYTHONPATH", ""))
    out = subprocess.run([sys.executable, os.path.join(ROOT, "examples",
                                                       name)],
                         capture_output=True, text=True, env=env,
                         timeout=300, check=True)
    return out.stdout


def _docstring_output(name):
    """The printed lines pasted in the script's docstring."""
    with open(os.path.join(ROOT, "examples", name)) as fh:
        text = fh.read()
    block = text.split('"""')[1]
    lines = block.split("SciPy 1.18.1):\n\n")[1].rstrip().splitlines()
    return [ln[4:] for ln in lines]


_NUM = re.compile(r"[-+]?\d+(?:\.\d+)?(?:e[-+]?\d+)?")


def _same_print(got, want):
    """Lines equal except that each printed number may differ by two
    units in its last printed digit (rounding differences between
    NumPy / SciPy versions); the text around the numbers must match."""
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert _NUM.sub("#", g) == _NUM.sub("#", w), (g, w)
        for a, b in zip(_NUM.findall(g), _NUM.findall(w)):
            mant = b.lower().split("e")[0]
            dec = len(mant.split(".")[1]) if "." in mant else 0
            exp = int(b.lower().split("e")[1]) if "e" in b.lower() else 0
            assert abs(float(a) - float(b)) <= 2.0 * 10.0 ** (exp - dec) \
                * (1 + 1e-9), (g, w)


def test_tutorial_detector_budget_prints_its_documented_result():
    out = _run_example("tutorial_detector_budget.py").splitlines()
    _same_print(out, _docstring_output("tutorial_detector_budget.py"))
    # key result: sigma_E with both error contributions
    nums = [float(x) for x in _NUM.findall(out[-1])]
    assert nums[0] == pytest.approx(1.998e-22, rel=1e-3)
    assert nums[1] == pytest.approx(0.016e-22, rel=0.07)


def test_tutorial_readout_trace_prints_its_documented_result():
    out = _run_example("tutorial_readout_trace.py").splitlines()
    _same_print(out, _docstring_output("tutorial_readout_trace.py"))
    tau = [float(x) for x in _NUM.findall(out[0])]
    assert tau[2] == pytest.approx(0.998, abs=0.002)
