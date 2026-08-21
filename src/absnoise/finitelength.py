"""Finite-length corrections: bound-saturation deficit and universality
breaking, from the exact per-mode ABS solver (abs_model).

For a finite-length ballistic channel the Andreev levels disperse with
the longitudinal velocity v_x of each transverse mode, so the couplings
are no longer proportional to the energies and the Cauchy-Schwarz bound
is not saturated. This module computes level-resolved sums
(responsivity, noise, Andreev heat capacity) from the exact levels.
Continuum contributions to the occupation channel are neglected
(bound levels dominate for L < xi; stated in Limitations).
"""

import numpy as np

from .constants import HBAR, KB, E_CHARGE
from .materials import Recipe
from .levels import JunctionModel, abs_energies, gap_bcs


class FiniteLJunction:
    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        m = JunctionModel(recipe)
        self.cs = m.cs                  # 2L/(hbar v_x) per orbital mode
        self.tau = recipe.tau
        self.Tc = recipe.Tc
        self.Delta0 = recipe.Delta

    def level_data(self, phi, T, dphi=1e-3):
        """Level-resolved E, dE/dphi, d2E/dphi2 at (phi, T)."""
        D = float(gap_bcs(T, self.Tc, self.Delta0))
        E0, E1, E2 = [], [], []
        for c in self.cs:
            Em = abs_energies(phi - dphi, self.tau, c, D)
            Ec = abs_energies(phi, self.tau, c, D)
            Ep = abs_energies(phi + dphi, self.tau, c, D)
            n = min(len(Em), len(Ec), len(Ep))
            for i in range(n):
                E0.append(Ec[i])
                E1.append((Ep[i] - Em[i]) / (2 * dphi))
                E2.append((Ep[i] - 2 * Ec[i] + Em[i]) / dphi**2)
        return (np.array(E0), np.array(E1), np.array(E2), D)

    def andreev_sums(self, phi, T, which="I"):
        """Same sums as ShortJunction.andreev_sums, level-resolved.
        Factor 2 for valley degeneracy (cs lists orbital modes of one
        valley-degenerate set; couplings per spin-degenerate channel)."""
        E, E1, E2, D = self.level_data(phi, T)
        v = 1.0 / (2.0 * np.cosh(np.minimum(E / (2 * KB * T), 350.0))**2)
        g = (2 * E_CHARGE / HBAR) * (-(E1 if which == "I" else E2))
        R = -2.0 * np.sum(g * v * E) / (KB * T**2)
        S0_over_tau = 2.0 * 4.0 * np.sum(g**2 * v)
        CA = 2.0 * np.sum(v * E**2) / (KB * T**2)
        return dict(R_occ=R, S0_over_tau=S0_over_tau, C_A=CA)

    def saturation_ratio(self, phi, T, which="I"):
        """(achieved dT_min) / (bound dT_min); = 1 when saturated."""
        s = self.andreev_sums(phi, T, which)
        achieved2 = s["S0_over_tau"] / (2.0 * s["R_occ"]**2)
        bound2 = 2.0 * KB * T**2 / s["C_A"]
        return float(np.sqrt(achieved2 / bound2))
