"""Device-level sensitivity budget: Andreev occupation noise versus
phonon thermal-fluctuation noise and readout imprecision, and the
resulting detector metrics.

Conventions (single-sided PSDs; variance of a t-second average of a
white process with PSD S is S/(2t), validated by Monte Carlo):

 * Andreev bound:      var(T)_A  >= 2 kB T^2 tauA / (C_A t)
 * Phonon TFN:         var(T)_ph  = 2 kB T^2 tau_th / (C_e t),
                       tau_th = C_e / G_ep   (S_T = 4 kB T^2/G_ep)
 * Crossover:          Andreev dominates when tauA/C_A > tau_th/C_e.

Readout conversion for an inductively read junction terminating a
resonator of inductance L_r (participation p = L_J/(L_J+L_r)):
    dnu_r / nu_r = -(p/2) dL_J/L_J = +(p/2) dI'(0)/I'(0),
so the fractional-frequency noise PSD is
    S_y(f) = (p/2)^2 S_{I'}(f) / I'(0)^2 .
"""

import numpy as np

from .constants import KB, HBAR, E_CHARGE, PHI0, H_PLANCK
from .materials import (Recipe, carrier_density, heat_capacity, gth,
                       ep_power)
from .shortjunction import ShortJunction


class SensorBudget:
    def __init__(self, recipe: Recipe, sigma=2.0, delta=3, L_r=2e-9,
                 nu_r=6.0e9):
        self.recipe = recipe
        self.sj = ShortJunction(recipe)
        self.sj.calibrate()
        self.sigma, self.delta = sigma, delta
        self.L_r, self.nu_r = L_r, nu_r
        self.n = carrier_density(recipe.Vbg)
        self.area = recipe.area

    # ---- thermal subsystem ----------------------------------------------
    def Ce(self, T):
        return float(heat_capacity(T, self.n, self.area))

    def Gep(self, T):
        return float(gth(T, self.area, self.sigma, self.delta))

    def tau_th(self, T):
        return self.Ce(T) / self.Gep(T)

    # ---- temperature resolutions (K, for integration time t) ------------
    def dT_andreev(self, T, tauA, t, which="L"):
        a, b, s = self.sj.temperature_bound(T, tauA, t, which=which)
        return a, s

    def dT_phonon(self, T, t):
        return np.sqrt(2.0 * KB * T**2 * self.tau_th(T) /
                       (self.Ce(T) * t))

    # ---- readout observables --------------------------------------------
    def LJ(self, T):
        """Josephson inductance at phi=0: L_J = (hbar/2e)/I'(0)."""
        return (HBAR / (2 * E_CHARGE)) / self.sj.dIdphi0(T)

    def participation(self, T):
        LJ = self.LJ(T)
        return LJ / (LJ + self.L_r)

    def freq_noise_spectrum(self, T, tauA, freqs, which="L"):
        """Predicted single-sided fractional-frequency PSD S_y(f) (1/Hz)
        and absolute S_nu (Hz^2/Hz) of the readout resonator from
        Andreev occupation noise."""
        phi = 1e-4 if which == "L" else self.sj.phi_max(T)
        s = self.sj.andreev_sums(phi, T, which)
        I1 = self.sj.dIdphi0(T)
        SI = s["S0_over_tau"] * tauA / (1.0 + (2 * np.pi * freqs *
                                               tauA)**2)
        p = self.participation(T)
        Sy = (p / 2.0)**2 * SI / I1**2
        return Sy, Sy * self.nu_r**2

    # ---- calorimetric energy resolution ---------------------------------
    def energy_resolution(self, T, tauA, S_ro_y=0.0, which="L",
                          n_f=6000):
        """Matched-filter energy resolution (J) for a delta heat deposit.

        Signal chain: deposit E -> dTe(t) = (E/Ce) exp(-t/tau_th)
        -> occupations respond with lag tauA -> fractional frequency
        y(t). Noise: Andreev Lorentzian + phonon TFN (filtered by the
        same occupation lag) + white readout floor S_ro_y (1/Hz).
        sigma_E^-2 = 2 int_0^inf df |Y(f)|^2 / S_y(f), Y = signal
        transform per unit deposited energy.
        """
        Ce, G = self.Ce(T), self.Gep(T)
        tth = Ce / G
        phi = 1e-4 if which == "L" else None
        s = self.sj.andreev_sums(1e-4 if phi is None else phi, T, which)
        I1 = self.sj.dIdphi0(T)
        p = self.participation(T)
        conv = (p / 2.0) / I1              # dO -> fractional frequency
        R = s["R_occ"] * conv              # (1/K): dy/dT
        fmax = 10.0 / (2 * np.pi * min(tauA, tth))
        f = np.logspace(np.log10(1.0 / (200 * 2 * np.pi * max(tauA,
                        tth))), np.log10(fmax), n_f)
        w = 2 * np.pi * f
        Hth = 1.0 / (1.0 + 1j * w * tth)
        HA = 1.0 / (1.0 + 1j * w * tauA)
        Y = np.abs(R * (1.0 / Ce) * tth * Hth * HA)   # per unit energy
        S_A = (conv**2) * s["S0_over_tau"] * tauA / (1 + (w * tauA)**2)
        S_ph = (R**2) * (4 * KB * T**2 / G) * np.abs(Hth)**2 * \
            np.abs(HA)**2
        S_tot = S_A + S_ph + S_ro_y
        integ = 4.0 * np.trapezoid(Y**2 / S_tot, f)
        return 1.0 / np.sqrt(integ)

    def energy_resolution_analytic_A(self, T, tauA, which="L"):
        """Closed form for the Andreev-noise-only matched-filter
        resolution (J):  sigma_E^2 = Ce^2 S_T(0) / tau_th, with
        S_T(0) = S0 tauA / R_occ^2  (= 4 kB T^2 tauA / C_A when the
        bound is saturated). Derivation in Supplement; used as an
        analytic anchor for energy_resolution()."""
        phi = 1e-4 if which == "L" else self.sj.phi_max(T)
        s = self.sj.andreev_sums(phi, T, which)
        ST0 = s["S0_over_tau"] * tauA / s["R_occ"]**2
        return self.Ce(T) * np.sqrt(ST0 / self.tau_th(T))
