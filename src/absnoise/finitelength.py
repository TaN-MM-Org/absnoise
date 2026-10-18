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
from scipy.optimize import brentq

from .constants import HBAR, KB, E_CHARGE
from .materials import Recipe
from .levels import JunctionModel, _theta, abs_energies, gap_bcs


def _branch_levels(phi, tau, c, Delta):
    """Levels of one channel labelled by their branch (new in 0.14.0).

    The level condition theta(E) = +-a - 2 pi k (see `abs_energies`)
    is solved with the signed angle a_s(phi) = 2 asin(sqrt(tau)
    sin(phi/2)) (phi reduced to (-pi, pi]), which is an analytic
    function of phi through phi = 0, where a = |a_s|. A level is then
    followed along its branch (k, s), target s a_s - 2 pi k, also where
    two levels cross. Returns {(k, s): E} for the targets inside
    (-c Delta, pi)."""
    ph = phi - 2.0 * np.pi * np.round(phi / (2.0 * np.pi))
    a_s = 2.0 * np.arcsin(np.sqrt(tau) * np.sin(0.5 * ph))
    lo, hi = -c * Delta, np.pi
    out = {}
    for k in range(0, 201):
        added = False
        for s in (1, -1):
            y = s * a_s - 2.0 * np.pi * k
            if lo < y < hi:
                added = True
                try:
                    out[(k, s)] = brentq(
                        lambda E: _theta(E, c, Delta) - y, 0.0, Delta,
                        xtol=1e-16 * Delta + 1e-40, rtol=8.9e-16)
                except ValueError:
                    pass
        if not added and k > 0:
            break
    return out


class FiniteLJunction:
    def __init__(self, recipe: Recipe):
        self.recipe = recipe
        m = JunctionModel(recipe)
        self.cs = m.cs                  # 2L/(hbar v_x) per orbital mode
        self.tau = recipe.tau
        self.Tc = recipe.Tc
        self.Delta0 = recipe.Delta

    def level_data(self, phi: float, T: float, dphi: float = 1e-3):
        """Level-resolved E, dE/dphi, d2E/dphi2 at (phi, T), by central
        differences with step dphi; returns (E, E1, E2, Delta).

        Changed in 0.14.0: where the stencil phi - dphi, phi, phi + dphi
        crosses phi = 0 (mod 2 pi), each level is followed along its own
        branch (`_branch_levels`). The two levels of an Andreev doublet
        cross there, and up to 0.13.0 the levels were matched by their
        order in energy, so for |phi| < dphi the stencil switched branch
        in the middle: dE/dphi of a doublet level came out as
        -alpha phi / dphi instead of -alpha, and d2E/dphi2 as about
        -2 alpha / dphi (alpha: the slope of the doublet splitting). For
        Ti/Al(thick) at 0.3 Tc, `saturation_ratio(1e-4, T, "L")` was
        853.79 instead of 1.00747. Elsewhere the result is unchanged
        (the same code as 0.13.0)."""
        phi, T, dphi = float(phi), float(T), float(dphi)
        if not (np.isfinite(phi) and np.isfinite(dphi) and dphi > 0.0):
            raise ValueError("phi and dphi must be finite, dphi > 0")
        D = float(gap_bcs(T, self.Tc, self.Delta0))
        E0, E1, E2 = [], [], []
        k0 = np.round(phi / (2.0 * np.pi))
        if abs(phi - 2.0 * np.pi * k0) < dphi:
            for c in self.cs:
                Em = _branch_levels(phi - dphi, self.tau, c, D)
                Ec = _branch_levels(phi, self.tau, c, D)
                Ep = _branch_levels(phi + dphi, self.tau, c, D)
                for key in sorted(Ec, key=Ec.get):
                    if key in Em and key in Ep:
                        E0.append(Ec[key])
                        E1.append((Ep[key] - Em[key]) / (2 * dphi))
                        E2.append((Ep[key] - 2 * Ec[key] + Em[key])
                                  / dphi**2)
            return (np.array(E0), np.array(E1), np.array(E2), D)
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

    def _check(self, T, which):
        """Refuse inputs the sums cannot answer (new in 0.14.0; up to
        0.13.0 an unknown `which` was taken as "L", and T at or above
        Tc gave no levels and a NaN ratio)."""
        if which not in ("I", "L"):
            raise ValueError("which must be 'I' or 'L'")
        T = float(T)
        if not (np.isfinite(T) and T > 0.0):
            raise ValueError("temperature must be finite and positive (K)")
        if not gap_bcs(T, self.Tc, self.Delta0) > 0.0:
            raise ValueError(f"T = {T:.4g} K is at or above Tc = "
                             f"{self.Tc:.4g} K: there are no Andreev "
                             "levels")
        return T

    def andreev_sums(self, phi: float, T: float, which: str = "I") -> dict:
        """Same sums as ShortJunction.andreev_sums, level-resolved.
        Factor 2 for valley degeneracy (cs lists orbital modes of one
        valley-degenerate set; couplings per spin-degenerate channel)."""
        T = self._check(T, which)
        E, E1, E2, D = self.level_data(phi, T)
        v = 1.0 / (2.0 * np.cosh(np.minimum(E / (2 * KB * T), 350.0))**2)
        g = (2 * E_CHARGE / HBAR) * (-(E1 if which == "I" else E2))
        R = -2.0 * np.sum(g * v * E) / (KB * T**2)
        S0_over_tau = 2.0 * 4.0 * np.sum(g**2 * v)
        CA = 2.0 * np.sum(v * E**2) / (KB * T**2)
        return dict(R_occ=R, S0_over_tau=S0_over_tau, C_A=CA)

    def saturation_ratio(self, phi: float, T: float,
                         which: str = "I") -> float:
        """(achieved dT_min) / (bound dT_min); = 1 when saturated."""
        s = self.andreev_sums(phi, T, which)
        achieved2 = s["S0_over_tau"] / (2.0 * s["R_occ"]**2)
        bound2 = 2.0 * KB * T**2 / s["C_A"]
        return float(np.sqrt(achieved2 / bound2))
