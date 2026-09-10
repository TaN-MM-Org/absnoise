"""Anchors for the Ic(T) junction fit: the closed-form T = 0 phase
maximum against the grid maximum, the internal forward model against
the package's independent `ShortJunction` implementation, exact
noise-free parameter recovery, Monte-Carlo compatibility of the
reported sigmas, refusals on malformed or unconstraining data, and
the fitted-recipe round trip back into the device pipeline."""
import numpy as np
import pytest

from absnoise import (RECIPES, ShortJunction, fit_ic_curve, ic_model)
from absnoise.constants import E_CHARGE, HBAR
from absnoise.icfit import _shape_max, _shape_zero
from absnoise.materials import n_modes

RECIPE = RECIPES[1]                       # Ti/Al/Au: Tc 0.75 K, tau 0.78


def test_zero_temperature_closed_form_equals_grid_maximum():
    for tau in (0.05, 0.3, 0.78, 0.99):
        Tc = 1.0
        grid = _shape_max(1e-5 * Tc, Tc, tau, n_phi=200001)[0]
        exact = _shape_zero(Tc, tau)
        assert abs(grid - exact) / exact < 1e-8


def test_forward_model_matches_shortjunction_implementation():
    """ic_model and ShortJunction.Ic are two implementations of the
    same ensemble; with the same phase grid they agree to machine
    precision at every recipe and temperature."""
    for rec in RECIPES:
        sj = ShortJunction(rec, n_phi=2001)          # scale = 1
        pref = sj.Nch * 2.0 * E_CHARGE / HBAR
        Ic0 = pref * _shape_zero(rec.Tc, rec.tau)
        for T in (0.05 * rec.Tc, 0.3 * rec.Tc, 0.7 * rec.Tc):
            mine = float(ic_model(T, Ic0, rec.Tc, rec.tau, n_phi=2001))
            ref = sj.Ic(T)
            assert abs(mine - ref) / ref < 1e-12


def test_noise_free_fit_recovers_generating_parameters():
    Ic0, Tc, tau = 2.0e-6, 0.75, 0.78
    T = np.linspace(0.05, 0.65, 10)
    y = ic_model(T, Ic0, Tc, tau)
    fit = fit_ic_curve(T, y, sigma_Ic=1e-9)
    assert abs(fit.Ic0 - Ic0) / Ic0 < 1e-6
    assert abs(fit.Tc - Tc) / Tc < 1e-6
    assert abs(fit.tau - tau) / tau < 1e-5
    assert fit.chi2 / max(fit.chi2_dof, 1) < 1e-6    # exact data


def test_monte_carlo_scatter_matches_reported_sigma():
    Ic0, Tc, tau = 2.0e-6, 0.75, 0.78
    T = np.linspace(0.05, 0.65, 12)
    y0 = ic_model(T, Ic0, Tc, tau)
    sig = 0.01 * Ic0
    rng = np.random.default_rng(2)
    fits = np.array([
        [f.Ic0, f.Tc, f.tau] for f in
        (fit_ic_curve(T, y0 + rng.normal(0, sig, T.size), sigma_Ic=sig)
         for _ in range(60))])
    rep = fit_ic_curve(T, y0 + rng.normal(0, sig, T.size),
                       sigma_Ic=sig)
    reported = np.array([rep.sigma["Ic0"], rep.sigma["Tc"],
                         rep.sigma["tau"]])
    emp = fits.std(axis=0)
    assert np.all(emp < 1.5 * reported)
    assert np.all(emp > 0.5 * reported)
    # and the parameters themselves are unbiased at this noise level
    assert abs(fits[:, 1].mean() - Tc) < 3.0 * reported[1]


def test_refusals_and_weak_constraint_warning():
    T = np.linspace(0.05, 0.65, 10)
    y = ic_model(T, 2e-6, 0.75, 0.78)
    with pytest.raises(ValueError):                  # too few points
        fit_ic_curve(T[:3], y[:3], sigma_Ic=1e-9)
    with pytest.raises(ValueError):                  # no sigmas, n = 4
        fit_ic_curve(T[:4], y[:4])
    with pytest.raises(ValueError):                  # negative current
        fit_ic_curve(T, -y, sigma_Ic=1e-9)
    bad = y.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError):
        fit_ic_curve(T, bad, sigma_Ic=1e-9)
    with pytest.raises(ValueError):                  # bad sigma
        fit_ic_curve(T, y, sigma_Ic=0.0)
    # plateau-only data: fit succeeds but warns that Tc is weak
    Tlow = np.linspace(0.02, 0.10, 8)                # << Tc = 0.75
    ylow = ic_model(Tlow, 2e-6, 0.75, 0.78)
    with pytest.warns(UserWarning, match="weakly determined"):
        fit_ic_curve(Tlow, ylow, sigma_Ic=1e-9)


def test_fitted_recipe_reenters_the_package_pipeline():
    """Fit synthetic data, package as a Recipe, and let the package's
    own calibrated ShortJunction reproduce the input curve."""
    Ic0, Tc, tau = 2.0e-6, 0.75, 0.78
    T = np.linspace(0.05, 0.65, 10)
    y = ic_model(T, Ic0, Tc, tau)
    fit = fit_ic_curve(T, y, sigma_Ic=1e-9)
    rec = fit.to_recipe(name="my junction", label="mine", xi=5e-6,
                        L=0.2e-6, W=2e-6, Vbg=30.0, Rn=50.0)
    assert rec.tau == fit.tau and rec.Tc == fit.Tc
    sj = ShortJunction(rec, n_phi=2001)
    sj.calibrate(T0=0.02)                            # match Ic20
    for k in (0, 4, 9):
        assert abs(sj.Ic(T[k]) - y[k]) / y[k] < 1e-4
    assert n_modes(rec) >= 1
