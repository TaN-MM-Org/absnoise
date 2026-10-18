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

Junction current (largest L/xi = 0.43 in the recipe set):
    I(phi,T) = (2e/hbar) * 2_valley * sum_modes dF_m/dphi,
    F_m(phi) = - sum_{E_i>0} 2 kB T ln[2 cosh(E_i / 2 kB T)]   (spin incl.)
               + F_cont,m(phi),
the bound-state term plus the continuum term of `continuum_free_energy`
(`JunctionModel.free_energy_multi`; this docstring said "bound states
only" up to 0.13.0, although the code has included the continuum
since 0.1.0).

New in 0.13.0: the same free energy, bound states and continuum
together, as a sum over Matsubara frequencies (Brouwer and Beenakker,
Chaos, Solitons & Fractals 8, 1249 (1997), Eqs. (6) and (8)):
`matsubara_free_energy_terms`, `JunctionModel.cpr_matsubara` and the
exact Taylor coefficients `JunctionModel.cpr_series`. New in 0.14.0:
the exact critical current `JunctionModel.Ic_matsubara`, and
`calibrate(method="matsubara")`.
"""

import numpy as np
from scipy.optimize import brentq

from .constants import HBAR, KB, E_CHARGE, V_F
from ._compat import trapezoid
from .materials import Recipe, carrier_density, n_modes

COS_TH_MIN = 0.05   # discard grazing modes (negligible current carriers)


def _theta(E, c, Delta):
    """theta(E) = 2 arccos(E/Delta) - c E, monotonically decreasing."""
    if isinstance(E, float):
        # scalar path for the root finder (0.13.0): the same operations
        # as below without the overhead of numpy.clip on a scalar, so
        # the result is bit-identical
        x = E / Delta
        if x < 0.0:
            x = 0.0
        elif x > 1.0:
            x = 1.0
        return 2.0 * np.arccos(x) - c * E
    return 2.0 * np.arccos(np.clip(E / Delta, 0.0, 1.0)) - c * E


def _branch_angle(phi, tau):
    """a = arccos(R), R = 1 - tau + tau cos(phi), in [0, pi].

    Where R is within 1e-3 of +1 or -1 the direct arccos(R) loses
    accuracy (R is rounded before arccos sees it: for tau = 0.7 every
    |phi| below about 2e-8 gave a = 0 exactly). There the exact
    identities arccos(1 - 2 s^2) = 2 arcsin(s), with
    s^2 = tau sin^2(phi/2) = (1 - R)/2, and
    pi - arccos(R) = 2 arcsin(sqrt(1 - s^2)), with
    1 - s^2 = (1 - tau) + tau cos^2(phi/2), are used instead (new in
    0.13.0; elsewhere the value is unchanged)."""
    R = min(max(1.0 - tau + tau * np.cos(phi), -1.0), 1.0)
    if 1.0 - abs(R) >= 1e-3:
        return float(np.arccos(R))
    if R > 0.0:
        s = np.sqrt(tau) * abs(np.sin(0.5 * phi))
        return float(2.0 * np.arcsin(min(s, 1.0)))
    q = (1.0 - tau) + tau * np.cos(0.5 * phi) ** 2
    return float(np.pi - 2.0 * np.arcsin(min(np.sqrt(max(q, 0.0)), 1.0)))


def abs_energies(phi: float, tau: float, c: float,
                 Delta: float) -> np.ndarray:
    """Positive ABS energies (J) for one channel.

    c = 2 L / (hbar v_x). Solve theta(E) = +- arccos(R) - 2 pi k with
    R = 1 - tau + tau cos(phi); theta decreases monotonically from pi
    (at E=0) to -c*Delta (at E=Delta), so each admissible target has a
    unique root.

    A degenerate pair of levels is returned as two equal values (fixed
    in 0.13.0). At phi = 0 (and, for tau = 1, at phi = pi) the two branches
    +-arccos(R) give the same target, and the two levels of a doublet
    coincide. Up to 0.12.1 such a pair was merged into one level, and,
    because arccos(R) was taken of a rounded R, the same happened for
    every |phi| below about 2e-8 / sqrt(tau): the free energy and the
    occupation sums then lost one level of each doublet (C_A of
    `FiniteLJunction` was halved at phi = 0).
    """
    a = _branch_angle(phi, tau)          # arccos(R), in [0, pi]
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
    return np.array(sorted(roots))


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


def _continuum_delta_rows(phis, tau, c, Delta, Efac):
    """`continuum_delta` for many phases at once, shape (n_phi, n_E),
    with the same arithmetic per element (new in 0.13.0)."""
    E = Delta * Efac
    gam = np.arccosh(Efac)
    w = np.exp(-2.0 * gam) * np.exp(1j * c * E)
    R = np.clip(1.0 - tau + tau * np.cos(np.asarray(phis, dtype=float)),
                -1.0, 1.0)
    D = 1.0 - (2.0 * w)[None, :] * R[:, None] + (w * w)[None, :]
    return -np.angle(D)


def continuum_free_energy(phi: float, tau: float, c: float, Delta: float,
                          T: float, n_E: int = 800,
                          Emax_fac: float = 60.0) -> float:
    """Continuum (E > Delta) contribution to the phase-dependent free
    energy of one channel (J), from the scattering-phase density of
    states. Above the gap the secular function continues to

        D(E) = 1 - 2 w R + w^2,   w = exp(-2 gamma) exp(i eta),
        gamma = arccosh(E/Delta),  R = 1 - tau + tau cos(phi),

    and D factorizes as (1 - w e^{i sigma})(1 - w e^{-i sigma}) with
    cos(sigma) = R, so |arg D| < pi (no winding). With the scattering
    phase delta(E) = -arg D(E) (as in `continuum_delta`), the free
    energy is
        F_cont = (2 kB T / pi) ln[2 cosh(Delta / 2 kB T)] delta(Delta+)
                 + (1/pi) int_Delta^inf tanh(E/2kBT) delta(E) dE ,
    the same expression `JunctionModel.free_energy_multi` and
    `free_energy_components` evaluate. The sign is fixed by continuity
    of F_bound + F_cont across the phase where a bound state exits into
    the continuum, which the tests check (fixed in 0.12.0: up to 0.11.0
    this function returned the opposite sign, so F_bound + F_cont
    jumped there by twice the free energy of the departing level).
    The energy grid starts 1e-9 Delta above the gap, so within a few
    1e-3 rad of a bound-state exit the threshold is not fully resolved.
    """
    E = Delta * (1.0 + np.logspace(-9, np.log10(Emax_fac), n_E))
    gam = np.arccosh(E / Delta)
    eta = c * E
    w = np.exp(-2.0 * gam) * np.exp(1j * eta)
    R = np.clip(1.0 - tau + tau * np.cos(phi), -1.0, 1.0)
    D = 1.0 - 2.0 * w * R + w * w
    delta_ph = -np.angle(D)
    x = Delta / (2.0 * KB * T)
    log2cosh_D = x + np.log1p(np.exp(-2.0 * x))
    boundary = (2.0 * KB * T / np.pi) * log2cosh_D * delta_ph[0]
    integral = (1.0 / np.pi) * trapezoid(
        np.tanh(E / (2.0 * KB * T)) * delta_ph, E)
    return boundary + integral


def _chi2(z):
    """Legendre chi function chi_2(z) = sum_{n>=0} z^(2n+1) / (2n+1)^2
    = (Li2(z) - Li2(-z)) / 2 for 0 <= z <= 1 (Li2 from
    scipy.special.spence: Li2(x) = spence(1 - x))."""
    from scipy.special import spence
    z = np.asarray(z, dtype=float)
    return 0.5 * (spence(1.0 - z) - spence(1.0 + z))


def _matsubara_terms(c, Delta, T, omega_max_fac):
    """Matsubara frequencies and x_n = exp(-2 s_n - c w_n) (s_n =
    arcsinh(w_n / Delta)) for one channel, together with the analytic
    sum of the large-frequency form x_as = Delta^2 e^(-c w) / (4 w^2)
    over all n >= 0."""
    kT = KB * T
    n_terms = int(np.ceil(omega_max_fac * Delta / (2.0 * np.pi * kT))) + 1
    w = (2.0 * np.arange(n_terms) + 1.0) * np.pi * kT
    y = w / Delta
    e2s = 1.0 / (y + np.sqrt(1.0 + y * y)) ** 2          # exp(-2 s)
    decay = np.exp(-c * w)
    x = e2s * decay
    x_as = decay / (4.0 * y * y)
    sum_as = float(_chi2(np.exp(-c * np.pi * kT))) * \
        (Delta / (2.0 * np.pi * kT)) ** 2
    return x, x_as, sum_as


def _matsubara_dF(ph, tau, kT, x, x_as, sum_as):
    """dF/dphi of one channel at the phases ph (1-D array) from the
    Matsubara terms of `_matsubara_terms` (the summation of
    `matsubara_free_energy_terms`, moved here unchanged in 0.14.0 so
    that `JunctionModel.Ic_matsubara` can reuse the terms)."""
    R = 1.0 - tau + tau * np.cos(ph)
    # sum x / (1 + 2 R x + x^2) = sum_as + sum (x/(...) - x_as), in
    # blocks of phases so that memory stays bounded
    g = np.empty(ph.size)
    x2 = x * x
    step = max(1, int(2 ** 20 // x.size))
    for i in range(0, ph.size, step):
        r = R[i:i + step, None]
        g[i:i + step] = sum_as + np.sum(
            x[None, :] / (1.0 + 2.0 * r * x[None, :] + x2[None, :])
            - x_as[None, :], axis=1)
    return 4.0 * kT * tau * np.sin(ph) * g


def matsubara_free_energy_terms(phi, tau, c, Delta, T,
                                omega_max_fac=2000.0):
    """Phase derivative and Taylor coefficients of the free energy of
    one channel (bound states and continuum together), from the
    Matsubara-sum (Brouwer-Beenakker) form (new in 0.13.0).

    With the secular function of this module continued to imaginary
    energy E = i w_n, w_n = (2n + 1) pi kB T,

        D(i w) = 1 + 2 R x + x^2,   x = exp(-2 s - c w),
        s = arcsinh(w / Delta),     R = 1 - tau + tau cos(phi),

    the phase-dependent free energy of one channel (spin included, the
    normalization of `JunctionModel`) is
    F(phi) = -2 kB T sum_{n>=0} ln D(i w_n) + const (P. W. Brouwer and
    C. W. J. Beenakker, Chaos, Solitons & Fractals 8, 1249 (1997),
    Eqs. (6) and (8)). D(i w) is real and positive, so the sum has no bound /
    continuum split and no root finding. For c = 0 it reduces exactly
    to -2 kB T ln 2cosh(E_A / 2 kB T) with
    E_A = Delta sqrt(1 - tau sin^2(phi/2)) (asserted in the tests).

    Returns dict(dF, F2, F4): dF = dF/dphi at each phi (J/rad, array
    like phi), and the coefficients of F = F0 + F2 phi^2 + F4 phi^4 +
    ..., F2 = 2 kB T sum kappa and F4 = -2 kB T sum (kappa/12 -
    kappa^2/2) with kappa = tau x / (1 + x)^2.

    The sums run to w = omega_max_fac * Delta; the large-w form
    x_as = Delta^2 e^(-c w) / (4 w^2) is summed to infinity in closed
    form (Legendre chi function), so the neglected rest falls as
    (Delta / w_max)^3 (about 1e-10 relative for the default).
    """
    T = float(T)
    if not (np.isfinite(T) and T > 0.0):
        raise ValueError("T must be finite and positive (K)")
    if not (np.isfinite(Delta) and Delta > 0.0):
        raise ValueError("Delta must be finite and positive (J)")
    if not (0.0 <= tau <= 1.0):
        raise ValueError("tau must lie in [0, 1]")
    if not (np.isfinite(c) and c >= 0.0):
        raise ValueError("c = 2L/(hbar v_x) must be finite and >= 0")
    x, x_as, sum_as = _matsubara_terms(c, Delta, T, omega_max_fac)
    kT = KB * T
    ph = np.atleast_1d(np.asarray(phi, dtype=float)).ravel()
    dF = _matsubara_dF(ph, tau, kT, x, x_as, sum_as)
    kap = tau * x / (1.0 + x) ** 2
    s_kap = tau * sum_as + float(np.sum(kap - tau * x_as))
    s_kap2 = float(np.sum(kap * kap))
    F2 = 2.0 * kT * s_kap
    F4 = -2.0 * kT * (s_kap / 12.0 - 0.5 * s_kap2)
    dF = float(dF[0]) if np.ndim(phi) == 0 else dF
    return dict(dF=dF, F2=F2, F4=F4)


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
        tanh weight of each temperature (sign convention: see
        `continuum_free_energy`).
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

        # (changed in 0.13.0) the continuum part is evaluated for all
        # phases of a mode at once; the arithmetic per element is the
        # one of the per-phase loop of 0.12.1 (checked in the tests)
        for m in range(self.N):
            for j in range(nP):
                Eb = self._levels[m][j]
                if len(Eb):
                    xb = Eb[None, :] / (2.0 * KB * T_list[:, None])
                    F[:, j] += -np.sum(2.0 * KB * T_list[:, None] *
                                       log2cosh(xb), axis=1)
            dph = _continuum_delta_rows(self.phis, self.tau, self.cs[m],
                                        self.Delta, Efac)
            F += (l2cD[:, None] * dph[None, :, 0] / np.pi
                  + trapezoid(tanhw[:, None, :] * dph[None, :, :], E,
                              axis=-1) / np.pi)
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

    def cpr_matsubara(self, T, phis=None, omega_max_fac=2000.0):
        """Current-phase relation I(phi) (A) from the Matsubara-sum
        free energy (new in 0.13.0): bound states and continuum
        together, exact for this model up to the truncation stated in
        `matsubara_free_energy_terms` (about 1e-10 relative). Uses
        `self.Delta` and the calibration `self.scale`, like
        `current_phase`; phis defaults to `self.phis`.

        Unlike `current_phase` (bound levels plus scattering-phase
        continuum on an energy grid, differentiated on the phase
        grid), it needs no level solving and no finite differences, so
        it stays accurate near phi = 0, where bound levels leave for
        the continuum and the two parts of `current_phase` nearly
        cancel."""
        ph = self.phis if phis is None else np.asarray(phis, dtype=float)
        dF = np.zeros(np.size(ph))
        for c in self.cs:
            dF += matsubara_free_energy_terms(
                np.ravel(ph), self.tau, c, self.Delta, T,
                omega_max_fac)["dF"]
        out = self.scale * (2.0 * E_CHARGE / HBAR) * 2.0 * dF
        return float(out[0]) if np.ndim(ph) == 0 else out.reshape(
            np.shape(ph))

    def cpr_series(self, T, omega_max_fac=2000.0):
        """Exact Taylor coefficients (a1, a3) of I(phi) = a1 phi +
        a3 phi^3 + ... at phi = 0 (A/rad, A/rad^3), from the
        Matsubara-sum free energy (new in 0.13.0). a1 = I'(0) gives
        the Josephson inductance L_J = (hbar/2e) / a1 and a3 the Kerr
        (phase-cubic) term. With F = F0 + F2 phi^2 + F4 phi^4 per
        channel, I = scale (4e/hbar) sum_channels dF/dphi, so
        a1 = scale (4e/hbar) sum 2 F2 and a3 = scale (4e/hbar)
        sum 4 F4. Uses `self.Delta` and `self.scale`."""
        F2 = F4 = 0.0
        for c in self.cs:
            t = matsubara_free_energy_terms(0.0, self.tau, c, self.Delta,
                                            T, omega_max_fac)
            F2 += t["F2"]
            F4 += t["F4"]
        k = self.scale * (2.0 * E_CHARGE / HBAR) * 2.0
        return float(k * 2.0 * F2), float(k * 4.0 * F4)

    def dIdphi0(self, T, k=None):
        """dI/dphi at phi = 0 (A/rad); Josephson inductance
        L_J = (hbar/2e) / I'(0) at zero phase bias.

        Changed in 0.13.0: the exact value from the Matsubara-sum free
        energy (`cpr_series`). Up to 0.12.1 a cubic was fitted to the
        first k = 8 points of `current_phase`, whose first point is a
        one-sided difference about 14 times too large and whose next
        points scatter by up to 17 % where bound levels leave for the
        continuum; the result was 44 % low for a zero-length junction
        (where the exact value is known in closed form) and 57 % low
        for Ti/Al/Au at 0.3 Tc. `k` is ignored and deprecated."""
        if k is not None:
            import warnings
            warnings.warn("JunctionModel.dIdphi0(k=...) is ignored since "
                          "0.13.0 (the slope is exact) and will be "
                          "removed", DeprecationWarning, stacklevel=2)
        return self.cpr_series(T)[0]

    def cpr_coeffs(self, T, kfit=30):
        """Fit I(phi) ~ a1 phi + a3 phi^3 to the first kfit points of
        `current_phase`; returns (a1, a3).

        This is a two-term least-squares fit over phi up to about
        0.76 rad (kfit = 30, n_phi = 121), so it carries the bias of
        the neglected phi^5 term (about 2 % in a3 for a zero-length
        junction) and the scatter of `current_phase` near phi = 0. For
        the exact coefficients use `cpr_series` (new in 0.13.0), which
        `junction_properties` uses since 0.13.0."""
        I = self.current_phase(T)
        ph = self.phis[:kfit]
        A = np.vstack([ph, ph**3]).T
        coef, *_ = np.linalg.lstsq(A, I[:kfit], rcond=None)
        return float(coef[0]), float(coef[1])

    def Ic_matsubara(self, T: float, n_grid: int = 31,
                     xtol: float = 1e-9, omega_max_fac: float = 2000.0,
                     return_phase: bool = False):
        """Critical current (A) of the full model from the Matsubara-sum
        current-phase relation (new in 0.14.0): bound states and
        continuum together, without the energy and phase grids of
        `Ic`. At 20 mK (gap Delta0, scale 1) `Ic` is 4.2e-4 to 1.8e-3
        below this value for four of the shipped recipes, 1.5e-4 above
        it for Ti/Nb/NbN, and 1.7 % above it for MoRe, whose
        level-path current scatters by up to 3 % between neighboring
        grid phases near its maximum.

        The current is evaluated on n_grid phases inside (0, pi); every
        local maximum of that grid is refined by a bounded Brent search
        (scipy.optimize.minimize_scalar) over its two neighboring
        intervals, to a phase tolerance xtol (rad), and the largest is
        returned. An error delta in the phase changes the maximum by
        only about |I''| delta^2 / 2 (I'' is of the order of I). Uses `self.Delta` and `self.scale`,
        like `cpr_matsubara`. With return_phase=True, returns
        (Ic, phi_c)."""
        from scipy.optimize import minimize_scalar
        T = float(T)
        if not (np.isfinite(T) and T > 0.0):
            raise ValueError("T must be finite and positive (K)")
        if not (np.isfinite(self.Delta) and self.Delta > 0.0):
            raise ValueError("self.Delta must be finite and positive: "
                             "there is no supercurrent without a gap")
        n_grid = int(n_grid)
        if n_grid < 5:
            raise ValueError("n_grid must be at least 5")
        xtol = float(xtol)
        if not (np.isfinite(xtol) and 0.0 < xtol < 0.1):
            raise ValueError("xtol must lie in (0, 0.1) rad")
        kT = KB * T
        terms = [_matsubara_terms(c, self.Delta, T, omega_max_fac)
                 for c in self.cs]
        k = self.scale * (2.0 * E_CHARGE / HBAR) * 2.0

        def current(ph):
            ph = np.atleast_1d(np.asarray(ph, dtype=float))
            dF = np.zeros(ph.size)
            for x, x_as, sum_as in terms:
                dF += _matsubara_dF(ph, self.tau, kT, x, x_as, sum_as)
            return k * dF

        edges = np.linspace(0.0, np.pi, n_grid + 2)
        grid = edges[1:-1]
        Ig = current(grid)
        best_I, best_phi = -np.inf, float("nan")
        for j in range(n_grid):
            left = Ig[j - 1] if j > 0 else -np.inf
            right = Ig[j + 1] if j < n_grid - 1 else -np.inf
            if not (Ig[j] >= left and Ig[j] >= right):
                continue
            res = minimize_scalar(lambda p: -float(current(p)[0]),
                                  bounds=(edges[j], edges[j + 2]),
                                  method="bounded",
                                  options=dict(xatol=xtol))
            cand = [(-float(res.fun), float(res.x)), (float(Ig[j]),
                                                     float(grid[j]))]
            for Iv, pv in cand:
                if Iv > best_I:
                    best_I, best_phi = Iv, pv
        return (best_I, best_phi) if return_phase else best_I

    def calibrate(self, T0: float = 0.02, method: str = "levels") -> float:
        """Set `scale` so that the critical current at T0 equals the
        measured `recipe.Ic20`; returns the scale.

        method="levels" (the default, unchanged) uses `Ic`, the maximum
        of `current_phase` on the phase grid. method="matsubara" (new in
        0.14.0) uses `Ic_matsubara`, the exact maximum of the
        Matsubara-sum current; the scales differ by the grid error of
        `Ic` (4.2e-4 to 1.8e-3 for five shipped recipes, 1.7 % for
        MoRe; see `Ic_matsubara`). The gap is
        `self.Delta` (by default the zero-temperature gap of the
        recipe, the convention of `junction_properties`)."""
        if method not in ("levels", "matsubara"):
            raise ValueError("method must be 'levels' or 'matsubara'")
        self.scale = 1.0
        Ic = self.Ic(T0) if method == "levels" else self.Ic_matsubara(T0)
        self.scale = self.recipe.Ic20 / Ic
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

# Near Tc (new in 0.12.0). The gap vanishes as a square root, which a
# cubic spline in t cannot follow: up to 0.11.0 the spline above was
# off by 0.13 % at t = 0.998 and 0.67 % at t = 0.999, and gap_bcs
# returned 0 for 0.9995 <= t < 1. Above _T_NEAR the gap is now taken
# from a second table of the smooth function h(t) = u(t) / sqrt(1 - t/t1),
# where t1 is the reduced temperature at which this gap equation closes.
_T_NEAR = 0.99
# With Delta0 = A kB Tc the weak-coupling gap equation closes at
# kB T1 = Delta0 e^gamma / pi (gamma: Euler's constant), i.e. at
# t1 = A e^gamma / pi. For the exact BCS ratio A = pi e^-gamma this is
# 1; for the rounded A = 1.7639 used here it is 1.0000130.
_T1 = _BCS_A * np.exp(np.euler_gamma) / np.pi


def _matsubara_lhs(y, n_terms=4000):
    """2 sum_{n>=0} [1/(2n+1) - 1/sqrt((2n+1)^2 + y^2)], with the tail
    beyond n_terms added as its integral (midpoint rule; error below
    1e-15 for the y < 1 used here)."""
    w = 2.0 * np.arange(n_terms) + 1.0
    s = float(np.sum(1.0 / w - 1.0 / np.sqrt(w * w + y * y)))
    X = 2.0 * n_terms
    tail = 0.5 * (np.log(0.5 * y) - np.log(X) + np.arcsinh(X / y))
    return 2.0 * (s + tail)


def _solve_bcs_u_matsubara(t):
    """u = Delta/Delta0 at reduced temperature t from the Matsubara-sum
    form of the same weak-coupling gap equation,

        ln(t1 / t) = 2 sum_{n>=0} [1/(2n+1) - 1/sqrt((2n+1)^2 + y^2)],
        y = Delta / (pi kB T) = u A / (pi t),

    which follows from the integral form used by `_solve_bcs_u` through
    the partial-fraction expansion
    tanh(x/2) / x = sum_{n>=0} 4 / (x^2 + (2n+1)^2 pi^2).
    It stays well conditioned as u -> 0, where the integral form loses
    accuracy. The tests check that the two forms agree."""
    rhs = np.log(_T1 / t)
    return brentq(lambda u: _matsubara_lhs(u * _BCS_A / (np.pi * t)) - rhs,
                  1e-12, 1.5, xtol=1e-16, rtol=1e-15)


def _build_near_tc_table():
    """Cubic spline of h(t) = u(t) / sqrt(1 - t/t1) on [0.985, 1]."""
    from scipy.interpolate import CubicSpline
    ts = np.linspace(0.985, 1.0, 31)
    hs = np.array([_solve_bcs_u_matsubara(t) / np.sqrt(1.0 - t / _T1)
                   for t in ts])
    return CubicSpline(ts, hs)


_NEAR_TC_SPLINE = None


def gap_bcs(T, Tc: float, Delta0: float):
    """Delta(T) from the numerically solved weak-coupling BCS gap
    equation (universal u(t), t = T/Tc), with the exact low-T asymptote
    u = 1 - sqrt(2 pi t / A) exp(-A/t) below t = 0.08. The universal
    table is computed once per process from the gap equation itself (no
    data file needed). Validated against both asymptotes in the
    testbench (the widely used tanh interpolation misrepresents
    dDelta/dT at low T by an order of magnitude).

    From t = 0.99 to Tc (changed in 0.12.0) the gap comes from a table
    of u / sqrt(1 - t/t1) built from the Matsubara-sum form of the same
    equation, which follows the square-root closing exactly (agreement
    with direct solutions better than 1e-9 relative, tested). Up to
    0.11.0 the result was 0 for 0.9995 <= t < 1 and up to 0.7 % off
    between t = 0.995 and 0.9995. The result is 0 for t >= 1. The
    equation uses A = Delta0/(kB Tc) = 1.7639 (the BCS value
    pi e^-gamma = 1.76388, rounded), so its solution would close at
    t1 = 1.000013 rather than at t = 1; the step from u(1) = 0.0063 to
    0 at Tc is that rounding, kept so that results below 0.99 Tc stay
    identical to earlier releases."""
    global _BCS_SPLINE, _NEAR_TC_SPLINE
    t = np.asarray(T, dtype=float) / Tc
    scalar = (t.ndim == 0)
    t = np.atleast_1d(t)
    u = np.zeros_like(t)
    lo = t < 0.08
    u[lo] = 1.0 - np.sqrt(2 * np.pi * t[lo] / _BCS_A) * \
        np.exp(-_BCS_A / np.maximum(t[lo], 1e-6))
    mid = (~lo) & (t < _T_NEAR)
    if np.any(mid):
        if _BCS_SPLINE is None:
            _BCS_SPLINE = _build_bcs_table()
        u[mid] = np.clip(_BCS_SPLINE(t[mid]), 0.0, 1.0)
    near = (t >= _T_NEAR) & (t < 1.0)
    if np.any(near):
        if _NEAR_TC_SPLINE is None:
            _NEAR_TC_SPLINE = _build_near_tc_table()
        tn = t[near]
        u[near] = _NEAR_TC_SPLINE(tn) * np.sqrt(1.0 - tn / _T1)
    out = Delta0 * u
    return float(out[0]) if scalar else out


def junction_properties(recipe: Recipe, T_grid, n_phi: int = 121,
                        verbose: bool = False,
                        method: str = "levels") -> dict:
    """Temperature table of junction observables for one recipe.

    Includes the BCS suppression of the proximity gap Delta*(T).
    Returns dict with Ic (A), dIdphi0 (A/rad), a3 (A/rad^3), and the
    calibration scale (fixed at 20 mK).

    Changed in 0.13.0: dIdphi0 and a3 are the exact Taylor
    coefficients of the current-phase relation at phi = 0
    (`JunctionModel.cpr_series`, Matsubara-sum free energy). Up to
    0.12.1 they came from `cpr_coeffs`, a two-term fit over
    0 < phi < 0.76 rad of a current that scatters near phi = 0: for
    Ti/Al/Au at 0.3 Tc a3 was 6 % too large in magnitude (a1 within
    5e-4). Ic is unchanged.

    method="levels" (default) takes Ic and the calibration from the
    level-plus-continuum `current_phase` on its phase grid, as before.
    method="matsubara" (new in 0.14.0) takes both from
    `JunctionModel.Ic_matsubara`, the exact maximum of the
    Matsubara-sum current, so that Ic, dIdphi0 and a3 all come from the
    same exact free energy (n_phi is then not used).
    """
    if method not in ("levels", "matsubara"):
        raise ValueError("method must be 'levels' or 'matsubara'")
    T_grid = np.asarray(T_grid, dtype=float)
    base = JunctionModel(recipe, n_phi=n_phi)
    # calibration at 20 mK with Delta*(20 mK) ~ Delta0
    base.calibrate(0.02, method=method)
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
        Ic[i] = m.Ic(T) if method == "levels" else m.Ic_matsubara(T)
        c1, c3 = m.cpr_series(T)
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
        for j in range(len(model.phis)):
            Eb = model._levels[m][j]
            if len(Eb):
                Fb[j] += -np.sum(2.0 * KB * T *
                                 log2cosh(Eb / (2.0 * KB * T)))
        # all phases of a mode at once (0.13.0; same arithmetic per
        # element as the per-phase loop of 0.12.1)
        dph = _continuum_delta_rows(model.phis, model.tau, model.cs[m],
                                    model.Delta, Efac)
        Fc += (l2cD * dph[:, 0] / np.pi
               + trapezoid(tanhw[None, :] * dph, E, axis=-1) / np.pi)
    return Fb, Fc


def continuum_share(recipe: Recipe, T: float, n_phi: int = 81,
                    phi0: float = np.pi / 2) -> dict:
    """Fraction of the supercurrent I(phi0) and of its temperature
    derivative carried by the continuum (E > Delta*) part of the free
    energy, at phi0 = pi/2 where the decomposition is regular (at
    phi -> 0 the bound state sits at the gap edge and the two parts
    individually diverge while their sum stays finite). This quantifies
    the error of the bound-state-only occupation sums at each recipe's
    finite L/xi; it vanishes identically in the short-junction limit.
    Returns dict(share_I, share_dIdT).

    Changed in 0.13.0: T must lie in (0, Tc), and the temperature step
    of the derivative is 2e-3 Tc but at most T/2 and (Tc - T)/10. Up
    to 0.12.1 the step was always 2e-3 Tc, so below T = 2e-3 Tc the
    model was evaluated at a negative temperature (share_dIdT NaN).
    Far below Tc dI/dT falls as exp(-Delta/kB T) and vanishes to
    machine precision (below about 0.03 Tc for Ti/Al/Au); share_dIdT
    is then 0/0 and is returned as NaN (0.12.1 raised
    ZeroDivisionError, for example at 20 mK for Ti/Al/Au). Up to about
    0.06 Tc dI/dT is still at the rounding level, so share_dIdT there
    carries no information.
    """
    T = float(T)
    if not (np.isfinite(T) and 0.0 < T < recipe.Tc):
        raise ValueError("T must lie between 0 and the recipe's Tc (K)")

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

    dT = min(2e-3 * recipe.Tc, 0.5 * T, 0.1 * (recipe.Tc - T))
    b0, c0 = I_parts(T)
    bp, cp = I_parts(T + dT)
    bm, cm = I_parts(T - dT)
    db, dc = (bp - bm) / (2 * dT), (cp - cm) / (2 * dT)
    share_dIdT = dc / (db + dc) if db + dc != 0.0 else float("nan")
    return dict(share_I=c0 / (b0 + c0), share_dIdT=share_dIdT)


def occupation_heat_capacities(recipe, T, phi0=np.pi / 2, n_phi=81,
                               dT_frac=1e-3, n_E=3000):
    """Bound and continuum contributions to the occupation-channel heat
    capacity at fixed phase, C = -T d^2F/dT^2 with the gap frozen --
    the thermodynamic weight of occupation fluctuations alone, with
    Delta(T) held at its value at T so that only the occupations vary.

    The bound part of this quantity is *identically* the level sum
    C_A = 2 sum_levels E^2 / (4 kB T^2 cosh^2(E/2kBT)) used by the
    Cauchy-Schwarz temperature bound: differentiating
    F_b = -2 kB T sum log 2cosh(E/2kBT) twice at fixed E gives exactly
    that sum. The test suite asserts the two independent code paths
    agree, rather than trusting the algebra. The continuum part is the
    same second derivative applied to the validated scattering-phase
    continuum free energy (the Krein spectral-shift contribution of
    the junction to the quasiparticle heat capacity) -- the piece the
    bound-only occupation sums neglect, now quantified instead of
    waved at. It vanishes identically in the short-junction limit,
    where the continuum carries no phase dependence (asserted).

    Returns dict(C_bound, C_cont, share_CA) in J/K; share_CA is the
    continuum fraction of the total.
    """
    m = JunctionModel(recipe, n_phi=n_phi)
    m.Delta = float(gap_bcs(T, recipe.Tc, recipe.Delta))
    m.compute_levels()
    j = int(np.argmin(np.abs(m.phis - phi0)))
    dT = dT_frac * T
    vals_b, vals_c = [], []
    for TT in (T - dT, T, T + dT):
        Fb, Fc = free_energy_components(m, TT, n_E=n_E)
        vals_b.append(Fb[j])
        vals_c.append(Fc[j])
    # x2: valley degeneracy, applied externally to the free-energy
    # components exactly as in `continuum_share`
    Cb = -2.0 * T * (vals_b[0] - 2.0 * vals_b[1] + vals_b[2]) / dT ** 2
    Cc = -2.0 * T * (vals_c[0] - 2.0 * vals_c[1] + vals_c[2]) / dT ** 2
    return dict(C_bound=float(Cb), C_cont=float(Cc),
                share_CA=float(Cc / (Cb + Cc)))
