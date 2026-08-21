"""Andreev-bound-state model of a ballistic graphene Josephson junction.

Each transverse orbital mode is a 1D two-lead ballistic channel of length
L with a single centered barrier of transparency tau, terminated by ideal
Andreev mirrors with proximity gap Delta*. Starting from Beenakker's
determinant condition

    det[ 1 - alpha(E)^2 r_A(phi) s_N(E) r_A(-phi) s_N*(-E) ] = 0,

with alpha(E) = exp(-i arccos(E/Delta)) and s_N the normal-region
scattering matrix including the ballistic electron-hole phase
eta(E) = 2 E L / (hbar v_x), the 2x2 determinant reduces exactly to the
closed-form secular equation (derivation in Supplement)

    cos( 2 arccos(E/Delta) - eta(E) ) = 1 - tau + tau cos(phi).

Analytic limits (testbench anchors):
  * L -> 0:   E = Delta sqrt(1 - tau sin^2(phi/2))
  * tau = 1:  Kulik levels 2 arccos(E/Delta) - eta(E) = +-phi mod 2 pi
  * tau = 1, L = 0, T = 0: I = (e Delta/hbar) sin(phi/2) per
    spin-degenerate channel  =>  e Ic Rn = pi Delta.

Junction current (bound states only, valid for L << xi; largest
L/xi = 0.43 in the recipe set):
    I(phi,T) = (2e/hbar) * 2_valley * sum_modes dF_m/dphi,
    F_m(phi) = - sum_{E_i>0} 2 kB T ln[2 cosh(E_i / 2 kB T)]   (spin incl.)
"""

import numpy as np
from scipy.optimize import brentq

from .constants import HBAR, KB, E_CHARGE, V_F
from .materials import Recipe, carrier_density, n_modes

COS_TH_MIN = 0.05   # discard grazing modes (negligible current carriers)


def _theta(E, c, Delta):
    """theta(E) = 2 arccos(E/Delta) - c E, monotonically decreasing."""
    return 2.0 * np.arccos(np.clip(E / Delta, 0.0, 1.0)) - c * E


def abs_energies(phi, tau, c, Delta):
    """Positive ABS energies (J) for one channel.

    c = 2 L / (hbar v_x). Solve theta(E) = +- arccos(R) - 2 pi k with
    R = 1 - tau + tau cos(phi); theta decreases monotonically from pi
    (at E=0) to -c*Delta (at E=Delta), so each admissible target has a
    unique root.
    """
    R = np.clip(1.0 - tau + tau * np.cos(phi), -1.0, 1.0)
    a = np.arccos(R)                     # in [0, pi]
    lo, hi = -c * Delta, np.pi
    targets = []
    k = 0
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
            r0 = brentq(lambda E: _theta(E, c, Delta) - y, 0.0, Delta,
                        xtol=1e-16 * Delta + 1e-40, rtol=8.9e-16)
            roots.append(r0)
        except ValueError:
            pass
    return np.array(sorted(set(roots)))


def continuum_phase_grid(n_E=3000, Emax_fac=60.0):
    """Log-spaced energy grid factor above the gap, dense near threshold."""
    return 1.0 + np.logspace(-13, np.log10(Emax_fac), n_E)


def continuum_delta(phi, tau, c, Delta, Efac):
    """Scattering-phase function delta(E) = -arg D(E) above the gap."""
    E = Delta * Efac
    gam = np.arccosh(Efac)
    w = np.exp(-2.0 * gam) * np.exp(1j * c * E)
    R = np.clip(1.0 - tau + tau * np.cos(phi), -1.0, 1.0)
    D = 1.0 - 2.0 * w * R + w * w
    return -np.angle(D)


def continuum_free_energy(phi, tau, c, Delta, T, n_E=800, Emax_fac=60.0):
    """Continuum (E > Delta) contribution to the phase-dependent free
    energy of one channel (J), from the scattering-phase density of
    states. Above the gap the secular function continues to

        D(E) = 1 - 2 w R + w^2,   w = exp(-2 gamma) exp(i eta),
        gamma = arccosh(E/Delta),  R = 1 - tau + tau cos(phi),

    and D factorizes as (1 - w e^{i sigma})(1 - w e^{-i sigma}) with
    cos(sigma) = R, so |arg D| < pi (no winding). The free energy is
        F_cont = (1/pi) int_Delta^inf tanh(E/2kBT) arg D(E) dE ,
    with the sign fixed by continuity of F_bound + F_cont across the
    phase where a bound state exits into the continuum (verified in the
    testbench).
    """
    E = Delta * (1.0 + np.logspace(-9, np.log10(Emax_fac), n_E))
    gam = np.arccosh(E / Delta)
    eta = c * E
    w = np.exp(-2.0 * gam) * np.exp(1j * eta)
    R = np.clip(1.0 - tau + tau * np.cos(phi), -1.0, 1.0)
    D = 1.0 - 2.0 * w * R + w * w
    delta_ph = np.angle(D)
    x = Delta / (2.0 * KB * T)
    log2cosh_D = x + np.log1p(np.exp(-2.0 * x))
    boundary = (2.0 * KB * T / np.pi) * log2cosh_D * delta_ph[0]
    integral = (1.0 / np.pi) * np.trapezoid(
        np.tanh(E / (2.0 * KB * T)) * delta_ph, E)
    return boundary + integral


class JunctionModel:
    """Mode-summed ABS model of one recipe, calibrated by one overall
    scale factor so that Ic(20 mK) matches the measured value."""

    def __init__(self, recipe: Recipe, n_phi=121):
        self.recipe = recipe
        self.Delta = recipe.Delta
        self.tau = recipe.tau
        n = carrier_density(recipe.Vbg)
        kf = np.sqrt(np.pi * n)
        N = n_modes(recipe)
        qn = (np.arange(N) + 0.5) * np.pi / recipe.W
        cos_th = np.sqrt(np.clip(1.0 - (qn / kf) ** 2, 0.0, 1.0))
        keep = cos_th >= COS_TH_MIN
        self.cos_th = cos_th[keep]
        self.N = int(np.sum(keep))
        self.cs = 2.0 * recipe.L / (HBAR * V_F * self.cos_th)
        self.phis = np.linspace(1e-3, np.pi - 1e-3, n_phi)
        self._levels = None
        self.scale = 1.0

    def compute_levels(self):
        """levels[m][j] = array of ABS energies of mode m at phi_j."""
        self._levels = [
            [abs_energies(p, self.tau, c, self.Delta) for p in self.phis]
            for c in self.cs]
        return self._levels

    def free_energy_multi(self, T_list, n_E=3000):
        """F(phi) for several temperatures at once (list of arrays).

        Bound-state part from the root table; continuum part from the
        scattering phase delta(E) = -arg D(E) including the threshold
        boundary term, cached per (mode, phi) and contracted against the
        tanh weight of each temperature (sign convention validated by
        continuity across bound-state exit in the testbench).
        """
        if self._levels is None:
            self.compute_levels()
        T_list = np.atleast_1d(np.asarray(T_list, dtype=float))
        nT, nP = len(T_list), len(self.phis)
        F = np.zeros((nT, nP))

        def log2cosh(x):
            x = np.abs(x)
            return x + np.log1p(np.exp(-2.0 * x))

        Efac = continuum_phase_grid(n_E)
        E = self.Delta * Efac
        tanhw = np.tanh(E[None, :] / (2.0 * KB * T_list[:, None]))
        xD = self.Delta / (2.0 * KB * T_list)
        l2cD = 2.0 * KB * T_list * (xD + np.log1p(np.exp(-2.0 * xD)))

        for m in range(self.N):
            for j, p in enumerate(self.phis):
                Eb = self._levels[m][j]
                if len(Eb):
                    xb = Eb[None, :] / (2.0 * KB * T_list[:, None])
                    F[:, j] += -np.sum(2.0 * KB * T_list[:, None] *
                                       log2cosh(xb), axis=1)
                dph = continuum_delta(p, self.tau, self.cs[m],
                                      self.Delta, Efac)
                F[:, j] += (l2cD * dph[0] / np.pi
                            + np.trapezoid(tanhw * dph[None, :], E,
                                           axis=1) / np.pi)
        return F

    def free_energy(self, T):
        """Phase-dependent free energy at a single temperature (J)."""
        return self.free_energy_multi([T])[0]

    def cpr_multi(self, T_list):
        """CPR I(phi) (A) for several temperatures, shape (nT, n_phi):
        I = scale * (4e/hbar) dF/dphi (spin inside F, valley factor 2)."""
        F = self.free_energy_multi(T_list)
        dphi = self.phis[1] - self.phis[0]
        return self.scale * (2.0 * E_CHARGE / HBAR) * 2.0 * \
            np.gradient(F, dphi, axis=1)

    def current_phase(self, T):
        return self.cpr_multi([T])[0]

    def Ic(self, T):
        return float(np.max(self.current_phase(T)))

    def dIdphi0(self, T, k=8):
        """dI/dphi at phi=0 (A/rad); Josephson inductance
        L_J = Phi0/(2 pi I'(0)) at zero phase bias."""
        I = self.current_phase(T)
        p = np.polyfit(self.phis[:k], I[:k], 3)
        return float(np.polyval(np.polyder(p), 0.0))

    def cpr_coeffs(self, T, kfit=30):
        """Fit I(phi) ~ a1 phi + a3 phi^3 near phi=0; returns (a1, a3)
        for inductance and Kerr calculations."""
        I = self.current_phase(T)
        ph = self.phis[:kfit]
        A = np.vstack([ph, ph**3]).T
        coef, *_ = np.linalg.lstsq(A, I[:kfit], rcond=None)
        return float(coef[0]), float(coef[1])

    def calibrate(self, T0=0.02):
        self.scale = 1.0
        self.scale = self.recipe.Ic20 / self.Ic(T0)
        return self.scale


_BCS_A = 1.7639          # Delta0 / kB Tc (weak-coupling BCS)


def _solve_bcs_u(t):
    """Solve the weak-coupling BCS gap equation for u = Delta/Delta0 at
    reduced temperature t = T/Tc (the universal function)."""
    from scipy.integrate import quad

    if t >= 0.9999:
        return 0.0
    if t < 0.08:
        return 1.0 - np.sqrt(2 * np.pi * t / _BCS_A) * np.exp(-_BCS_A / t)

    def rhs(u):
        def integrand(x):
            E = np.sqrt(x * x + u * u)
            a = _BCS_A * E / t
            return 1.0 / E / (np.exp(min(a, 500)) + 1.0)
        v, _ = quad(integrand, 0, 60, limit=300)
        return 2.0 * v

    return brentq(lambda u: np.log(1.0 / u) - rhs(u), 1e-6, 1.0 - 1e-13,
                  xtol=1e-15)


def _build_bcs_table():
    """Tabulate u(t) once per process and return a cubic spline."""
    from scipy.interpolate import CubicSpline
    ts = np.concatenate([np.linspace(0.002, 0.08, 40, endpoint=False),
                         np.linspace(0.08, 0.95, 180),
                         np.linspace(0.9505, 0.9995, 50)])
    us = np.array([_solve_bcs_u(t) for t in ts])
    return CubicSpline(ts, us)


_BCS_SPLINE = None


def gap_bcs(T, Tc, Delta0):
    """Delta(T) from the numerically solved weak-coupling BCS gap
    equation (universal u(t), t = T/Tc), with the exact low-T asymptote
    u = 1 - sqrt(2 pi t / A) exp(-A/t) below t = 0.08. The universal
    table is computed once per process from the gap equation itself (no
    data file needed). Validated against both asymptotes in the
    testbench (the widely used tanh interpolation misrepresents
    dDelta/dT at low T by an order of magnitude)."""
    global _BCS_SPLINE
    t = np.asarray(T, dtype=float) / Tc
    scalar = (t.ndim == 0)
    t = np.atleast_1d(t)
    u = np.zeros_like(t)
    lo = t < 0.08
    u[lo] = 1.0 - np.sqrt(2 * np.pi * t[lo] / _BCS_A) * \
        np.exp(-_BCS_A / np.maximum(t[lo], 1e-6))
    mid = (~lo) & (t < 0.9995)
    if np.any(mid):
        if _BCS_SPLINE is None:
            _BCS_SPLINE = _build_bcs_table()
        u[mid] = np.clip(_BCS_SPLINE(t[mid]), 0.0, 1.0)
    out = Delta0 * u
    return float(out[0]) if scalar else out


def junction_properties(recipe, T_grid, n_phi=121, verbose=False):
    """Temperature table of junction observables for one recipe.

    Includes the BCS suppression of the proximity gap Delta*(T).
    Returns dict with Ic (A), dIdphi0 (A/rad), a3 (A/rad^3), and the
    calibration scale (fixed at 20 mK).
    """
    base = JunctionModel(recipe, n_phi=n_phi)
    # calibration at 20 mK with Delta*(20 mK) ~ Delta0
    base.calibrate(0.02)
    scale = base.scale
    Ic = np.zeros_like(T_grid)
    dI0 = np.zeros_like(T_grid)
    a3 = np.zeros_like(T_grid)
    for i, T in enumerate(T_grid):
        m = JunctionModel(recipe, n_phi=n_phi)
        m.Delta = float(gap_bcs(T, recipe.Tc, recipe.Delta))
        m.scale = scale
        if m.Delta <= 1e-4 * recipe.Delta:
            continue
        Ic[i] = m.Ic(T)
        c1, c3 = m.cpr_coeffs(T)
        dI0[i] = c1
        a3[i] = c3
        if verbose:
            print(f"  T={T*1e3:6.1f} mK  Ic={Ic[i]*1e6:8.4f} uA  "
                  f"I'(0)={dI0[i]*1e6:8.4f} uA/rad")
    return dict(T=T_grid, Ic=Ic, dIdphi0=dI0, a3=a3, scale=scale)


def free_energy_components(model, T, n_E=3000):
    """Bound-state and continuum parts of F(phi) separately (J), for a
    JunctionModel with its current model.Delta. Used to quantify the
    continuum share of the inductive response."""
    if model._levels is None:
        model.compute_levels()
    Fb = np.zeros_like(model.phis)
    Fc = np.zeros_like(model.phis)

    def log2cosh(x):
        x = np.abs(x)
        return x + np.log1p(np.exp(-2.0 * x))

    Efac = continuum_phase_grid(n_E)
    E = model.Delta * Efac
    tanhw = np.tanh(E / (2.0 * KB * T))
    xD = model.Delta / (2.0 * KB * T)
    l2cD = 2.0 * KB * T * (xD + np.log1p(np.exp(-2.0 * xD)))
    for m in range(model.N):
        for j, p in enumerate(model.phis):
            Eb = model._levels[m][j]
            if len(Eb):
                Fb[j] += -np.sum(2.0 * KB * T *
                                 log2cosh(Eb / (2.0 * KB * T)))
            dph = continuum_delta(p, model.tau, model.cs[m],
                                  model.Delta, Efac)
            Fc[j] += (l2cD * dph[0] / np.pi
                      + np.trapezoid(tanhw * dph, E) / np.pi)
    return Fb, Fc


def continuum_share(recipe, T, n_phi=81, phi0=np.pi / 2):
    """Fraction of the supercurrent I(phi0) and of its temperature
    derivative carried by the continuum (E > Delta*) part of the free
    energy, at phi0 = pi/2 where the decomposition is regular (at
    phi -> 0 the bound state sits at the gap edge and the two parts
    individually diverge while their sum stays finite). This quantifies
    the error of the bound-state-only occupation sums at each recipe's
    finite L/xi; it vanishes identically in the short-junction limit.
    Returns dict(share_I, share_dIdT)."""

    def I_parts(TT):
        m = JunctionModel(recipe, n_phi=n_phi)
        m.Delta = float(gap_bcs(TT, recipe.Tc, recipe.Delta))
        Fb, Fc = free_energy_components(m, TT)
        dphi = m.phis[1] - m.phis[0]
        j = int(np.argmin(np.abs(m.phis - phi0)))
        out = []
        for F in (Fb, Fc):
            I = (2.0 * E_CHARGE / HBAR) * 2.0 * np.gradient(F, dphi)
            out.append(float(I[j]))
        return out

    dT = 2e-3 * recipe.Tc
    b0, c0 = I_parts(T)
    bp, cp = I_parts(T + dT)
    bm, cm = I_parts(T - dT)
    db, dc = (bp - bm) / (2 * dT), (cp - cm) / (2 * dT)
    return dict(share_I=c0 / (b0 + c0), share_dIdT=dc / (db + dc))
