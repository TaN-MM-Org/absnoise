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
from ._compat import trapezoid
from .materials import (Recipe, carrier_density, heat_capacity, gth,
                       ep_power)
from .shortjunction import ShortJunction


def matched_filter_sigma(P, Q, W, tau_th, tauA):
    """Exact matched-filter energy resolution (J) for the budget's
    noise model, in closed form (new in 0.11.0).

    Referred to the input power, the three noise channels of
    `SensorBudget.nep_spectrum` add as

        NEP^2(f) = P + Q u + W u v,
        u = 1 + (2 pi f tau_th)^2,   v = 1 + (2 pi f tauA)^2,

    with P = 4 kB T^2 G (phonon), Q = S_T_A(0) G^2 (occupation) and
    W = S_ro_y G^2 / R^2 (readout floor), all in W^2/Hz. With
    omega = 2 pi f the matched-filter integral
    sigma_E^-2 = 4 int_0^inf df / NEP^2 becomes
    (2/pi) int_0^inf d omega / (a + b omega^2 + c omega^4) with

        a = P + Q + W,  b = Q tau_th^2 + W (tau_th^2 + tauA^2),
        c = W tau_th^2 tauA^2,

    and the standard integral
    int_0^inf dx / (a + b x^2 + c x^4) = pi / (2 sqrt(a) sqrt(b + 2 sqrt(a c)))
    (valid for a > 0, c >= 0, b + 2 sqrt(a c) > 0) gives

        sigma_E^2 = sqrt(a) sqrt(b + 2 sqrt(a c)).

    The tests compare it with adaptive numerical quadrature of the
    same integrand and with the occupation-only closed form
    sigma_E^2 = Q tau_th (P = W = 0).
    """
    P, Q, W = float(P), float(Q), float(W)
    tth, tA = float(tau_th), float(tauA)
    if min(P, Q, W) < 0.0 or not np.isfinite(P + Q + W):
        raise ValueError("noise terms must be finite and non-negative")
    if not (tth > 0.0 and tA > 0.0):
        raise ValueError("tau_th and tauA must be positive")
    a = P + Q + W
    b = Q * tth ** 2 + W * (tth ** 2 + tA ** 2)
    c = W * tth ** 2 * tA ** 2
    if not (a > 0.0 and b + 2.0 * np.sqrt(a * c) > 0.0):
        raise ValueError("the noise model has no occupation or readout "
                         "term: the matched-filter bandwidth, and so "
                         "the resolution, is unbounded")
    return float(np.sqrt(np.sqrt(a) * np.sqrt(b + 2.0 * np.sqrt(a * c))))


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

    # ---- input checks (new in 0.11.0) -----------------------------------
    def _check_T(self, T, need_gap=True):
        """Refuse a temperature the model cannot evaluate. Before
        0.11.0, T >= Tc or T = 0 raised ZeroDivisionError from deep
        inside the level sums, and a negative T gave NaN."""
        T = float(T)
        if not (np.isfinite(T) and T > 0.0):
            raise ValueError("temperature must be finite and positive (K)")
        if need_gap and not self.sj.Delta(T) > 0.0:
            raise ValueError(
                f"T = {T:.4g} K is at or above the junction's Tc = "
                f"{self.recipe.Tc:.4g} K (the gap has closed): there "
                "are no Andreev levels to read out")
        return T

    @staticmethod
    def _check_pos(value, name):
        v = float(value)
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be finite and positive")
        return v

    # ---- thermal subsystem ----------------------------------------------
    def Ce(self, T):
        T = self._check_T(T, need_gap=False)
        return float(heat_capacity(T, self.n, self.area))

    def Gep(self, T):
        T = self._check_T(T, need_gap=False)
        return float(gth(T, self.area, self.sigma, self.delta))

    def tau_th(self, T):
        return self.Ce(T) / self.Gep(T)

    # ---- temperature resolutions (K, for integration time t) ------------
    def dT_andreev(self, T, tauA, t, which="L"):
        T = self._check_T(T)
        tauA = self._check_pos(tauA, "tauA")
        t = self._check_pos(t, "averaging time t")
        if which not in ("L", "I"):
            raise ValueError("which must be 'L' or 'I'")
        a, b, s = self.sj.temperature_bound(T, tauA, t, which=which)
        return a, s

    def dT_phonon(self, T, t):
        T = self._check_T(T, need_gap=False)
        t = self._check_pos(t, "averaging time t")
        return np.sqrt(2.0 * KB * T**2 * self.tau_th(T) /
                       (self.Ce(T) * t))

    # ---- readout observables --------------------------------------------
    def LJ(self, T):
        """Josephson inductance at phi=0: L_J = (hbar/2e)/I'(0)."""
        T = self._check_T(T)
        return (HBAR / (2 * E_CHARGE)) / self.sj.dIdphi0(T)

    def participation(self, T):
        LJ = self.LJ(T)
        return LJ / (LJ + self.L_r)

    def freq_noise_spectrum(self, T, tauA, freqs, which="L"):
        """Predicted single-sided fractional-frequency PSD S_y(f) (1/Hz)
        and absolute S_nu (Hz^2/Hz) of the readout resonator from
        Andreev occupation noise."""
        T = self._check_T(T)
        tauA = self._check_pos(tauA, "tauA")
        if which not in ("L", "I"):
            raise ValueError("which must be 'L' or 'I'")
        phi = 1e-4 if which == "L" else self.sj.phi_max(T)
        s = self.sj.andreev_sums(phi, T, which)
        I1 = self.sj.dIdphi0(T)
        SI = s["S0_over_tau"] * tauA / (1.0 + (2 * np.pi * freqs *
                                               tauA)**2)
        p = self.participation(T)
        Sy = (p / 2.0)**2 * SI / I1**2
        return Sy, Sy * self.nu_r**2

    # ---- calorimetric energy resolution ---------------------------------
    def _chain(self, T, tauA, which):
        """Shared signal-chain quantities of `energy_resolution`,
        `nep_spectrum` and `energy_resolution_analytic_A`."""
        T = self._check_T(T)
        tauA = self._check_pos(tauA, "tauA")
        if which not in ("L", "I"):
            raise ValueError("which must be 'L' or 'I'")
        Ce, G = self.Ce(T), self.Gep(T)
        # operating phase as in freq_noise_spectrum (before 0.10.1 the
        # "I" readout of energy_resolution was evaluated at phi = 1e-4)
        phi = 1e-4 if which == "L" else self.sj.phi_max(T)
        s = self.sj.andreev_sums(phi, T, which)
        I1 = self.sj.dIdphi0(T)
        p = self.participation(T)
        conv = (p / 2.0) / I1              # dO -> fractional frequency
        R = s["R_occ"] * conv              # (1/K): dy/dT
        ST_A0 = s["S0_over_tau"] * tauA / s["R_occ"] ** 2
        return dict(T=T, tauA=tauA, Ce=Ce, G=G, tth=Ce / G, s=s,
                    conv=conv, R=R, ST_A0=ST_A0)

    def energy_resolution(self, T, tauA, S_ro_y=0.0, which="L",
                          n_f=6000, method="exact"):
        """Matched-filter energy resolution (J) for a delta heat deposit.

        Signal chain: deposit E -> dTe(t) = (E/Ce) exp(-t/tau_th)
        -> occupations respond with lag tauA -> fractional frequency
        y(t). Noise: Andreev Lorentzian + phonon TFN (filtered by the
        same occupation lag) + white readout floor S_ro_y (1/Hz).
        sigma_E^-2 = 4 int_0^inf df |Y(f)|^2 / S_y(f), Y = signal
        transform per unit deposited energy.

        method="exact" (default from 0.11.0) evaluates the integral
        over all frequencies in closed form (`matched_filter_sigma`).
        method="grid" is the 0.10.x computation, kept to reproduce old
        numbers: a trapezoid rule on n_f log-spaced points between
        1/(400 pi max(tauA, tau_th)) and 10/(2 pi min(tauA, tau_th)).
        Cutting the band there drops part of the integral, so "grid"
        overestimates sigma_E, by 3.3 % for Ti/Al/Au at 0.3 Tc with
        tauA = 1 us and by 10.7 % with tauA = 1 ns (and more when
        phonon noise dominates).
        """
        c = self._chain(T, tauA, which)
        S_ro_y = float(S_ro_y)
        if not (np.isfinite(S_ro_y) and S_ro_y >= 0.0):
            raise ValueError("S_ro_y must be finite and >= 0")
        T, tauA, G, tth = c["T"], c["tauA"], c["G"], c["tth"]
        if method == "exact":
            return matched_filter_sigma(
                4.0 * KB * T ** 2 * G, c["ST_A0"] * G ** 2,
                S_ro_y * G ** 2 / c["R"] ** 2, tth, tauA)
        if method != "grid":
            raise ValueError("method must be 'exact' or 'grid'")
        Ce, s, conv, R = c["Ce"], c["s"], c["conv"], c["R"]
        fmax = 10.0 / (2 * np.pi * min(tauA, tth))
        f = np.logspace(np.log10(1.0 / (200 * 2 * np.pi * max(tauA,
                        tth))), np.log10(fmax), int(n_f))
        w = 2 * np.pi * f
        Hth = 1.0 / (1.0 + 1j * w * tth)
        HA = 1.0 / (1.0 + 1j * w * tauA)
        Y = np.abs(R * (1.0 / Ce) * tth * Hth * HA)   # per unit energy
        S_A = (conv**2) * s["S0_over_tau"] * tauA / (1 + (w * tauA)**2)
        S_ph = (R**2) * (4 * KB * T**2 / G) * np.abs(Hth)**2 * \
            np.abs(HA)**2
        S_tot = S_A + S_ph + S_ro_y
        integ = 4.0 * trapezoid(Y**2 / S_tot, f)
        return 1.0 / np.sqrt(integ)

    def nep_spectrum(self, T, tauA, freqs, S_ro_y=0.0, which="L"):
        """Noise-equivalent power spectrum NEP(f), the bolometric
        figure of merit every detector paper quotes (new in v0.8).

        Each noise channel is referred to the input power through the
        full signal chain dy/dP(f) = R (1/G) H_th(f) H_A(f) -- the
        same chain `energy_resolution` uses -- giving closed forms:

         * phonon TFN:  NEP_ph^2(f) = 4 kB T^2 G_ep, exactly flat at
           every frequency: the Lorentzian rolloff of the temperature
           fluctuations cancels against the responsivity rolloff
           (J. C. Mather, Appl. Opt. 21, 1125 (1982), the classic
           nonequilibrium-bolometer result).
         * Andreev occupation noise: NEP_A^2(f) =
           S_T_A(0) G^2 (1 + (2 pi f tau_th)^2) with S_T_A(0) =
           S0 tauA / R_occ^2 -- the occupation Lorentzian cancels
           exactly against the occupation lag in the response,
           leaving only the thermal rolloff to undo.
         * readout imprecision: NEP_ro^2(f) = S_ro_y G^2
           (1 + (2 pi f tau_th)^2)(1 + (2 pi f tauA)^2) / R^2.

        Returns dict(total, andreev, phonon, readout), each in
        W/sqrt(Hz). Anchors asserted in the tests rather than stated:
        the phonon channel is flat and equals Mather's 4 kB T^2 G to
        machine precision; the matched-filter identity
        sigma_E = [4 integral df / NEP_total^2]^(-1/2), integrated
        numerically, reproduces the closed form of `energy_resolution`
        (two independent code paths); the DC
        crossover NEP_A(0) > NEP_ph(0) is exactly the documented
        tauA/C_A > tau_th/C_e criterion when the occupation bound is
        saturated; and the Andreev channel's rise is exactly
        sqrt(1 + (2 pi f tau_th)^2).
        """
        f = np.asarray(freqs, dtype=float)
        if np.any(f < 0) or not np.all(np.isfinite(f)):
            raise ValueError("frequencies must be finite and >= 0")
        c = self._chain(T, tauA, which)
        T, tauA, G, tth = c["T"], c["tauA"], c["G"], c["tth"]
        R, ST_A0 = c["R"], c["ST_A0"]
        w = 2 * np.pi * f
        th2 = 1.0 + (w * tth) ** 2
        A2 = 1.0 + (w * tauA) ** 2
        nep2_ph = np.full_like(f, 4.0 * KB * T ** 2 * G)
        nep2_A = ST_A0 * G ** 2 * th2
        nep2_ro = float(S_ro_y) * G ** 2 * th2 * A2 / R ** 2
        total = np.sqrt(nep2_ph + nep2_A + nep2_ro)
        return dict(total=total, andreev=np.sqrt(nep2_A),
                    phonon=np.sqrt(nep2_ph),
                    readout=np.sqrt(nep2_ro + np.zeros_like(f)))

    def energy_resolution_analytic_A(self, T, tauA, which="L"):
        """Closed form for the Andreev-noise-only matched-filter
        resolution (J):  sigma_E^2 = Ce^2 S_T(0) / tau_th, with
        S_T(0) = S0 tauA / R_occ^2  (= 4 kB T^2 tauA / C_A when the
        bound is saturated). Derivation in Supplement; used as an
        analytic anchor for energy_resolution()."""
        c = self._chain(T, tauA, which)
        return c["Ce"] * np.sqrt(c["ST_A0"] / c["tth"])
