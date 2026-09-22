"""0.10.1 fixes, each with the check that failed before the fix:
integrals that run on NumPy 1.x (no `numpy.trapezoid` there), the
current-readout energy resolution at the operating phase its sibling
methods use, the Ic(T) model at full transparency tau = 1, the PSD
fit's refusal of n_avg < 1, and `__all__` listing every public name."""
import dataclasses
import types

import numpy as np
import pytest

import absnoise
from absnoise import (RECIPES, SensorBudget, ShortJunction,
                      fit_telegraph_psd, ic_model, plan_ic_measurement,
                      telegraph_psd_model)
from absnoise.constants import E_CHARGE, HBAR, KB
from absnoise.icfit import _shape_max, _shape_zero
from absnoise.levels import continuum_free_energy
from absnoise._compat import trapezoid

B = SensorBudget(RECIPES[1])
T = 0.3 * B.recipe.Tc
TAUA = 1e-6


def test_integrals_work_without_numpy_trapezoid(monkeypatch):
    """NumPy 1.x has `trapz` and no `trapezoid`; the package allows
    NumPy >= 1.24, so its integrals must not need `trapezoid`."""
    D = RECIPES[1].Delta
    ref_E = B.energy_resolution(T, TAUA)
    ref_F = continuum_free_energy(1.0, 0.5, 1.0 / D, D, T)
    fn = getattr(np, "trapezoid", None) or getattr(np, "trapz")
    monkeypatch.delattr(np, "trapezoid", raising=False)
    monkeypatch.setattr(np, "trapz", fn, raising=False)
    assert B.energy_resolution(T, TAUA) == ref_E
    assert continuum_free_energy(1.0, 0.5, 1.0 / D, D, T) == ref_F


def test_energy_resolution_current_readout_matches_nep():
    """which="I": the matched-filter identity built from `nep_spectrum`
    must reproduce `energy_resolution`, as it does for "L"."""
    tth = B.tau_th(T)
    fmax = 10.0 / (2 * np.pi * min(TAUA, tth))
    f = np.logspace(np.log10(1.0 / (200 * 2 * np.pi * max(TAUA, tth))),
                    np.log10(fmax), 6000)
    for S_ro in (0.0, 1e-22):
        nep = B.nep_spectrum(T, TAUA, f, S_ro_y=S_ro, which="I")
        sig_from_nep = 1.0 / np.sqrt(
            4.0 * trapezoid(1.0 / nep["total"] ** 2, f))
        sig_direct = B.energy_resolution(T, TAUA, S_ro_y=S_ro,
                                         which="I")
        assert abs(sig_from_nep - sig_direct) / sig_direct < 1e-10


def test_ic_model_is_finite_at_full_transparency():
    """tau = 1 is inside the documented range (0, 1]; the T = 0
    normalization used to be 0/0 = NaN there."""
    Tc = 1.0
    exact = _shape_zero(Tc, 1.0)
    assert exact == 1.764 * KB * Tc / 2.0
    grid = _shape_max(1e-5 * Tc, Tc, 1.0, n_phi=200001)[0]
    assert abs(grid - exact) / exact < 1e-8
    # against the independent ShortJunction implementation
    rec = dataclasses.replace(RECIPES[1], tau=1.0)
    sj = ShortJunction(rec, n_phi=2001)
    Ic0 = sj.Nch * 2.0 * E_CHARGE / HBAR * _shape_zero(rec.Tc, 1.0)
    for Tk in (0.05 * rec.Tc, 0.3 * rec.Tc, 0.7 * rec.Tc):
        mine = float(ic_model(Tk, Ic0, rec.Tc, 1.0, n_phi=2001))
        assert abs(mine - sj.Ic(Tk)) / sj.Ic(Tk) < 1e-12
    Ts = np.linspace(0.05, 0.85, 9)
    plan = plan_ic_measurement(Ts, 1e-8, 1e-6, 1.0, 1.0)
    assert plan["identifiable"]
    assert all(np.isfinite(v) for v in plan["sigma"].values())


def test_psd_fit_refuses_n_avg_below_one():
    f = np.geomspace(50.0, 20000.0, 48)
    S = telegraph_psd_model(f, 1e-3, 3e-4, 1e-5)
    for bad in (0, -4, np.nan):
        with pytest.raises(ValueError, match="n_avg"):
            fit_telegraph_psd(f, S, n_avg=bad)
    assert fit_telegraph_psd(f, S, n_avg=1).chi2 is not None


def test_all_lists_every_public_name():
    public = {n for n, v in vars(absnoise).items()
              if not n.startswith("_")
              and not isinstance(v, types.ModuleType)}
    assert public == set(absnoise.__all__)
