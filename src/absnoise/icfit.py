"""Fit junction parameters from a measured Ic(T) curve.

The shipped `RECIPES` carry the measured parameters of Jung et al.'s
six contact stacks; every other junction in the world needs its own
(Tc*, tau, Ic) before any budget in this package applies to it.  The
standard characterization measurement -- critical current versus
temperature -- determines exactly those parameters, and this module
implements the fit.

The forward model is the package's own short-junction ensemble: a
uniform-transparency Andreev doublet per channel,

    Ic(T) = Ic0 * M(T; Tc, tau) / M(0; tau),
    M(T)  = max_phi [ (Delta(T)^2 tau sin phi / 4 E) tanh(E / 2 kB T) ],
    E     = Delta(T) sqrt(1 - tau sin^2(phi/2)),

with Delta(T) the numerically solved BCS gap (`gap_bcs`) at
Delta0 = 1.764 kB Tc.  The three fitted parameters are the physical
ones: Ic0 (the T -> 0 critical current, which absorbs channel count
and any series factors), Tc (the proximity critical temperature) and
tau (the transparency, which sets the SHAPE of the suppression through
the gap sqrt(1 - tau sin^2(phi/2)) -- transparent junctions keep their
supercurrent to higher T/Tc than tunnel junctions, so the normalized
curve identifies tau).  The zero-temperature normalization uses the
closed form of the maximizing phase, sin^2(phi*/2) = (1 - sqrt(1-tau))
/ tau (the root of tau u^2 - 2u + 1), asserted against the grid
maximum in the tests.

The optimizer is `scipy.optimize.least_squares` (trust-region
reflective, bounded 0 < tau <= 1, Tc > 0, Ic0 > 0) on whitened
residuals; the parameter covariance is the standard linearized
estimate (J^T J)^{-1} at the solution -- exact-noise when sigmas are
given (with the chi-square consistency check that entails), scaled by
the residual variance otherwise (which is why that path needs at
least one residual degree of freedom).  A measured curve that never
climbs a meaningful fraction of the way to Tc constrains Tc and tau
only weakly; the fit WARNS when max(T) < 0.3 Tc_fit instead of
letting a well-formatted result imply a well-determined one.

Exact facts the test suite asserts rather than states: the closed-form
zero-temperature maximum equals the phi-grid maximum; the internal
forward model agrees with `ShortJunction.Ic` (an independent
implementation in this package) across recipes and temperatures;
noise-free synthetic data returns the generating (Ic0, Tc, tau);
seeded Monte-Carlo scatter is compatible with the reported sigmas;
degenerate inputs raise; and a fitted result drops back into the
package as a `Recipe` whose calibrated `ShortJunction` reproduces the
input curve.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.optimize import least_squares

from .constants import KB, BCS_RATIO
from .levels import gap_bcs
from .materials import Recipe


def _shape_max(T, Tc, tau, n_phi=601):
    """max_phi of the CPR magnitude at each T (J units of Delta)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    D = np.atleast_1d(gap_bcs(T, Tc, BCS_RATIO * KB * Tc))
    phi = np.linspace(1e-6, np.pi - 1e-6, n_phi)
    s2 = np.sin(phi / 2.0) ** 2
    E = D[:, None] * np.sqrt(1.0 - tau * s2[None, :])   # (nT, nphi)
    I = (D[:, None] ** 2 * tau * np.sin(phi)[None, :] / (4.0 * E)) \
        * np.tanh(E / (2.0 * KB * T[:, None]))
    return I.max(axis=1)


def _shape_zero(Tc, tau):
    """Closed-form T = 0 maximum: the maximizing u = sin^2(phi/2) is
    the root of tau u^2 - 2u + 1, u* = (1 - sqrt(1-tau))/tau."""
    D0 = BCS_RATIO * KB * Tc
    if tau >= 1.0:
        u = 1.0
    else:
        u = (1.0 - np.sqrt(1.0 - tau)) / tau
    return D0 * tau / 4.0 * 2.0 * np.sqrt(u * (1.0 - u)) \
        / np.sqrt(1.0 - tau * u)


def ic_model(T, Ic0, Tc, tau, n_phi=601):
    """Short-junction Ic(T) with unit-normalized zero-T value Ic0."""
    out = Ic0 * _shape_max(T, Tc, tau, n_phi) / _shape_zero(Tc, tau)
    return float(out[0]) if np.ndim(T) == 0 else out


@dataclasses.dataclass
class IcFit:
    """Fitted junction parameters with uncertainties and diagnostics.

    Ic0, Tc, tau : the fitted zero-temperature critical current (A),
        proximity critical temperature (K) and transparency.
    sigma : dict of one-standard-deviation uncertainties keyed
        'Ic0', 'Tc', 'tau' (linearized covariance at the solution).
    cov : (3, 3) parameter covariance, order (Ic0, Tc, tau).
    chi2, chi2_dof : weighted residual sum of squares and n - 3, when
        measurement sigmas were provided (chi2/dof near 1 says model
        and stated sigmas are consistent); None otherwise.
    Ic_fit : the model evaluated at the data temperatures.
    n_points : number of measurements used.
    """

    Ic0: float
    Tc: float
    tau: float
    sigma: dict
    cov: np.ndarray
    chi2: float | None
    chi2_dof: int | None
    Ic_fit: np.ndarray
    n_points: int

    def to_recipe(self, name, label, xi, L, W, Vbg, Rn) -> Recipe:
        """Package the fit as a `Recipe` for the rest of the package.

        The geometry and gating fields (xi, L, W, Vbg, Rn) are the
        user's measured values -- a critical-current curve does not
        determine them, so this method does not pretend to.  Ic20 is
        the fitted model at 20 mK, matching the calibration convention
        of `ShortJunction.calibrate`.
        """
        ic20 = float(ic_model(0.020, self.Ic0, self.Tc, self.tau))
        return Recipe(name=name, label=label, Tc=self.Tc, xi=xi, L=L,
                      W=W, Vbg=Vbg, tau=self.tau, Ic20=ic20, Rn=Rn)


def fit_ic_curve(T, Ic, sigma_Ic=None, p0=None, n_phi=601) -> IcFit:
    """Fit (Ic0, Tc, tau) to a measured critical-current curve.

    Parameters
    ----------
    T : (n,) temperatures (K), all positive.
    Ic : (n,) measured critical currents (A), all positive.
    sigma_Ic : optional (n,) or scalar one-standard-deviation current
        uncertainties.  With sigmas: exact known-noise covariance and
        a chi-square consistency check.  Without: residual-variance
        covariance, requiring n >= 5 (n - 3 >= 2 residual degrees of
        freedom, so the noise scale is estimated from more than one
        number).
    p0 : optional (Ic0, Tc, tau) starting point; by default Ic0
        starts at max(Ic), Tc at 1.3 max(T) and tau at 0.5.
    n_phi : phase-grid resolution of the forward model.

    Raises ValueError on malformed input and RuntimeError if the
    optimizer fails to converge; warns if max(T) < 0.3 Tc_fit (the
    data barely leave the plateau, so Tc and tau are weakly
    determined -- inspect `cov`).
    """
    T = np.asarray(T, dtype=float).ravel()
    y = np.asarray(Ic, dtype=float).ravel()
    if T.shape != y.shape:
        raise ValueError("T and Ic must have the same length")
    n = T.size
    if n < 4:
        raise ValueError("need at least 4 (T, Ic) points for 3 "
                         "parameters")
    if not (np.all(np.isfinite(T)) and np.all(np.isfinite(y))):
        raise ValueError("T and Ic must be finite")
    if np.any(T <= 0.0) or np.any(y <= 0.0):
        raise ValueError("temperatures and critical currents must be "
                         "positive")
    if sigma_Ic is None:
        if n < 5:
            raise ValueError(
                "without sigma_Ic the noise scale is estimated from "
                "the residuals, which needs n >= 5 points; provide "
                "sigma_Ic or more temperatures")
        w = np.ones(n)
    else:
        sig = np.broadcast_to(np.asarray(sigma_Ic, dtype=float),
                              (n,)).copy()
        if not np.all(np.isfinite(sig)) or np.any(sig <= 0.0):
            raise ValueError("sigma_Ic must be finite and positive")
        w = 1.0 / sig

    if p0 is None:
        p0 = (float(y.max()), 1.3 * float(T.max()), 0.5)
    lo = [0.0, 1e-6, 1e-3]
    hi = [np.inf, np.inf, 1.0]
    p0 = np.clip(np.asarray(p0, dtype=float), lo, hi)

    def resid(p):
        return w * (ic_model(T, p[0], p[1], p[2], n_phi) - y)

    res = least_squares(resid, p0, bounds=(lo, hi), xtol=1e-12,
                        ftol=1e-12, gtol=1e-12)
    if not res.success:
        raise RuntimeError(f"Ic(T) fit did not converge: {res.message}")
    J = res.jac
    A = J.T @ J
    try:
        cov = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        raise RuntimeError(
            "singular information matrix at the solution: the data do "
            "not determine (Ic0, Tc, tau); widen the temperature range")
    rss = float(2.0 * res.cost)                      # sum of resid^2
    dof = n - 3
    if sigma_Ic is None:
        cov = cov * (rss / dof)
        chi2 = chi2_dof = None
    else:
        chi2, chi2_dof = rss, dof
    p = res.x
    if float(T.max()) < 0.3 * p[1]:
        warnings.warn(
            f"data reach only T = {T.max():.3g} K against a fitted "
            f"Tc = {p[1]:.3g} K; Ic(T) is nearly flat there, so Tc and "
            "tau are weakly determined -- inspect the covariance",
            stacklevel=2)
    s = np.sqrt(np.diag(cov))
    return IcFit(Ic0=float(p[0]), Tc=float(p[1]), tau=float(p[2]),
                 sigma=dict(Ic0=float(s[0]), Tc=float(s[1]),
                            tau=float(s[2])),
                 cov=cov, chi2=chi2, chi2_dof=chi2_dof,
                 Ic_fit=ic_model(T, *p, n_phi), n_points=n)
