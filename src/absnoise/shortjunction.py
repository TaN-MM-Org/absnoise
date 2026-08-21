"""Closed-form short-junction ABS ensemble of a proximity junction.

Each of the N_ch = 2 (valley) x N_orb spin-degenerate channels carries a
single Andreev doublet

    E_m(phi, T) = Delta(T) sqrt(1 - tau_m sin^2(phi/2)),

with supercurrent operator per channel

    I_m = (2e/hbar) * (-dE_m/dphi) * sigma_m,
    sigma_m = 1 - n_up - n_down,  <sigma_m> = tanh(E_m / 2 kB T).

All observables (Ic, I'(0), responsivities, occupation-noise sums,
Andreev heat capacity C_A) follow from the level table. Temperature
enters through both the occupation and the BCS suppression Delta(T);
the two channels are kept separate because only the occupation channel
fluctuates (Supplement).
"""

import numpy as np

from .constants import HBAR, KB, E_CHARGE
from .levels import gap_bcs
from .materials import Recipe, n_modes


def _sech2(x):
    """Numerically safe sech^2."""
    x = np.abs(x)
    return np.where(x > 350.0, 0.0, 1.0 / np.cosh(np.minimum(x, 350.0))**2)


class ShortJunction:
    """Uniform-transparency short-junction ensemble for one recipe."""

    def __init__(self, recipe: Recipe, n_phi=2001):
        self.recipe = recipe
        self.tau = recipe.tau
        self.Delta0 = recipe.Delta
        self.Tc = recipe.Tc
        self.Nch = 2 * n_modes(recipe)      # valley x orbital, spin-degen
        self.phis = np.linspace(1e-6, np.pi - 1e-6, n_phi)
        self.scale = 1.0

    # ---- level structure -------------------------------------------------
    def Delta(self, T):
        return float(gap_bcs(T, self.Tc, self.Delta0))

    def E(self, phi, T):
        return self.Delta(T) * np.sqrt(1.0 - self.tau * np.sin(phi / 2)**2)

    def dEdphi(self, phi, T):
        D = self.Delta(T)
        E = self.E(phi, T)
        return -D**2 * self.tau * np.sin(phi) / (4.0 * E)

    def d2Edphi2_0(self, T):
        """Curvature at phi = 0: E'' = -Delta tau / 4."""
        return -self.Delta(T) * self.tau / 4.0

    # ---- mean observables ------------------------------------------------
    def current_phase(self, T):
        E = self.E(self.phis, T)
        occ = np.tanh(E / (2.0 * KB * T))
        I1 = (2.0 * E_CHARGE / HBAR) * (-self.dEdphi(self.phis, T)) * occ
        return self.scale * self.Nch * I1

    def Ic(self, T):
        return float(np.max(self.current_phase(T)))

    def phi_max(self, T):
        I = self.current_phase(T)
        return float(self.phis[int(np.argmax(I))])

    def dIdphi0(self, T):
        """I'(0) = Nch (2e/hbar)(Delta tau/4) tanh(Delta/2kBT)."""
        D = self.Delta(T)
        return self.scale * self.Nch * (2.0 * E_CHARGE / HBAR) * \
            (D * self.tau / 4.0) * np.tanh(D / (2.0 * KB * T))

    def calibrate(self, T0=0.02):
        self.scale = 1.0
        self.scale = self.recipe.Ic20 / self.Ic(T0)
        return self.scale

    # ---- responsivities (numerical, include gap channel) ----------------
    def dlnIc_dT(self, T, rel=2e-3):
        dT = rel * self.Tc
        return (np.log(self.Ic(T + dT)) - np.log(self.Ic(T - dT))) / (2 * dT)

    def dlnI1_dT(self, T, rel=2e-3):
        dT = rel * self.Tc
        return (np.log(self.dIdphi0(T + dT)) -
                np.log(self.dIdphi0(T - dT))) / (2 * dT)

    # ---- occupation (Andreev) channel at a phase bias --------------------
    def level_tables(self, phi, T):
        """Per-channel level data at operating phase phi.

        Returns dict with E, coupling g for the two observables:
          g_I  = (2e/hbar)(-dE/dphi)      (current readout at phase phi)
          g_L  = (2e/hbar)(-d2E/dphi2)    (inductance readout, phi -> 0)
        plus occupation factor v = 2 f(1-f) per channel (two spins).
        """
        D = self.Delta(T)
        E = self.E(phi, T)
        x = E / (2.0 * KB * T)
        v = _sech2(x) / 2.0                 # 2 f(1-f)
        gI = (2.0 * E_CHARGE / HBAR) * (-self.dEdphi(phi, T))
        # d2E/dphi2 general phi:
        s2 = np.sin(phi / 2.0)**2
        u = 1.0 - self.tau * s2
        d2E = -D * self.tau / 4.0 * (np.cos(phi) / np.sqrt(u)
                                     + self.tau * np.sin(phi)**2 /
                                     (4.0 * u**1.5))
        gL = (2.0 * E_CHARGE / HBAR) * (-d2E)
        return dict(E=E, v=v, gI=gI, gL=gL, Delta=D)

    def andreev_sums(self, phi, T, which="I"):
        """Occupation-channel responsivity and zero-frequency noise sums.

        R_occ  = dO/dT (occupation part) = sum_ch g * v * E / (kB T^2)
        S0/tauA = sum_ch g^2 * v * 4         (single-sided, per tauA)
        C_A    = sum_ch,spins E^2 f(1-f)/(kB T^2) = sum_ch v E^2/(kB T^2)
        All sums carry the calibration scale on couplings.
        """
        t = self.level_tables(phi, T)
        g = float(self.scale * (t["gI"] if which == "I" else t["gL"]))
        v, E = float(t["v"]), float(t["E"])
        Nch = self.Nch
        R = -Nch * g * v * E / (KB * T**2)   # occupancy falls with T
        S0_over_tau = Nch * 4.0 * g**2 * v
        CA = Nch * v * E**2 / (KB * T**2)
        return dict(R_occ=R, S0_over_tau=S0_over_tau, C_A=CA)

    def temperature_bound(self, T, tauA, t_int, phi=None, which="I"):
        """Achieved and bound temperature resolution (K) for integration
        time t_int: var(T) = S_T(0)/(2 t);
        bound: var(T) >= 2 kB T^2 tauA / (C_A t)  (Supplement, Cauchy-
        Schwarz over channels; saturated by uniform-tau short junctions)."""
        if phi is None:
            phi = 1e-4 if which == "L" else self.phi_max(T)
        s = self.andreev_sums(phi, T, which)
        S_T0 = s["S0_over_tau"] * tauA / s["R_occ"]**2
        achieved = np.sqrt(S_T0 / (2.0 * t_int))
        bound = np.sqrt(2.0 * KB * T**2 * tauA / (s["C_A"] * t_int))
        return achieved, bound, s
