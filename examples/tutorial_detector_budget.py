"""Tutorial: from two characterization measurements to a detector energy
resolution with an error bar.

A lab that has built a graphene Josephson junction usually measures two
curves before it asks how good a detector the junction is:

1. the critical current Ic versus temperature (gives Ic0, Tc, tau), and
2. the electron temperature Te versus absorbed (heater) power P at a
   fixed bath temperature (gives the electron-phonon cooling law
   P = Sigma A (Te^delta - Tp^delta)).

This script makes both "measurements" synthetically (fixed seeds,
illustrative true values), fits them with `fit_ic_curve` and
`fit_ep_coupling`, builds the detector budget from the fitted numbers,
and carries the error bars of both fits into the energy resolution and
the phonon-noise NEP. The two curves are independent measurements, so
their contributions add in quadrature (first order).

Run it with ``python examples/tutorial_detector_budget.py`` (a few seconds).
It printed, with absnoise 0.13.0 (Python 3.13, NumPy 2.5.3,
SciPy 1.18.1):

    Ic(T) fit:   Ic0 = 2.006 +- 0.006 uA, Tc = 0.7514 +- 0.0022 K, tau = 0.793 +- 0.013
    heating fit: Sigma = 2.51 +- 0.06 W m^-2 K^-delta, delta = 2.991 +- 0.020, chi2 = 6.6 for 13 dof
    budget at T = 0.200 K, tauA = 1 us (illustrative):
      thermal time tau_th = 0.69 +- 0.01 ns
      phonon NEP = 5.19e-19 +- 0.02e-19 W/sqrt(Hz)
      energy resolution = 1.998e-22 J +- 0.014e-22 (Ic fit) +- 0.009e-22 (heating fit)
                        = 1.998e-22 +- 0.016e-22 J, the energy of a 301 +- 2 GHz photon
"""
import numpy as np

from absnoise import (H_PLANCK, SensorBudget, ep_temperature, fit_ep_coupling,
                      fit_ic_curve, ic_model)

# ---- 1. the Ic(T) curve: 12 temperatures, 10 nA error per point -----------
T_ic = np.linspace(0.04, 0.68, 12)                       # K
rng = np.random.default_rng(11)
sig_ic = 0.01e-6                                         # A
Ic = ic_model(T_ic, 2.0e-6, 0.75, 0.78) + rng.normal(0.0, sig_ic, T_ic.size)
icfit = fit_ic_curve(T_ic, Ic, sigma_Ic=sig_ic)

# your measured geometry (not determined by an Ic(T) curve)
geometry = dict(xi=5e-6, L=0.2e-6, W=2e-6, Vbg=30.0, Rn=50.0)
recipe = icfit.to_recipe(name="my junction", label="mine", **geometry)
area = recipe.area                                       # m^2

# ---- 2. the heating curve: Te(P) at a 50 mK bath, 2 mK thermometer error -----
P = np.concatenate([[0.0, 0.0], np.geomspace(3e-17, 1e-13, 13)])   # W
sig_Te = 2e-3                                                      # K
Te = ep_temperature(P, 0.05, area, 2.5, 3.0) + rng.normal(0.0, sig_Te, P.size)
epfit = fit_ep_coupling(P, Te, area, Tp_K=0.05, sigma_Te=sig_Te)

print(f"Ic(T) fit:   Ic0 = {icfit.Ic0 * 1e6:.3f} +- {icfit.sigma['Ic0'] * 1e6:.3f} uA, "
      f"Tc = {icfit.Tc:.4f} +- {icfit.sigma['Tc']:.4f} K, "
      f"tau = {icfit.tau:.3f} +- {icfit.sigma['tau']:.3f}")
print(f"heating fit: Sigma = {epfit.Sigma:.2f} +- {epfit.sigma['Sigma']:.2f} "
      f"W m^-2 K^-delta, delta = {epfit.delta:.3f} +- {epfit.sigma['delta']:.3f}, "
      f"chi2 = {epfit.chi2:.1f} for {epfit.chi2_dof} dof")

# ---- 3. the detector budget from the fitted numbers -------------------------
T_op, tauA = 0.2, 1e-6          # operating temperature (K), tauA (s): illustrative


def with_ic(func):
    """Error from the Ic(T) fit, cooling law held at its fitted values."""
    return icfit.propagate(lambda r: func(SensorBudget(r, sigma=epfit.Sigma,
                                                       delta=epfit.delta)),
                           **geometry)


def with_ep(func):
    """Error from the heating fit, junction held at its fitted values."""
    return epfit.propagate(lambda s, d: func(SensorBudget(recipe, sigma=s,
                                                          delta=d)))


def combined(func):
    a, b = with_ic(func), with_ep(func)
    return a["value"], a["sigma"], b["sigma"], float(np.hypot(a["sigma"], b["sigma"]))


v, s1, s2, s = combined(lambda b: b.tau_th(T_op))
print(f"budget at T = {T_op:.3f} K, tauA = 1 us (illustrative):")
print(f"  thermal time tau_th = {v * 1e9:.2f} +- {s * 1e9:.2f} ns")
v, s1, s2, s = combined(lambda b: b.nep_spectrum(T_op, tauA, np.array([0.0]))["phonon"][0])
print(f"  phonon NEP = {v / 1e-19:.2f}e-19 +- {s / 1e-19:.2f}e-19 W/sqrt(Hz)")
v, s1, s2, s = combined(lambda b: b.energy_resolution(T_op, tauA))
nu, snu = v / H_PLANCK / 1e9, s / H_PLANCK / 1e9
print(f"  energy resolution = {v / 1e-22:.3f}e-22 J +- {s1 / 1e-22:.3f}e-22 (Ic fit) "
      f"+- {s2 / 1e-22:.3f}e-22 (heating fit)")
print(f"                    = {v / 1e-22:.3f}e-22 +- {s / 1e-22:.3f}e-22 J, "
      f"the energy of a {nu:.0f} +- {snu:.0f} GHz photon")
