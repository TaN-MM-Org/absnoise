"""Electron-phonon coupling from a measured self-heating curve (new in
0.13.0).

Every detector number of this package (`SensorBudget`: C_e, G_ep,
tau_th, the phonon noise, the NEP and the energy resolution) depends on
the electron-phonon cooling law

    P = Sigma A (Te^delta - Tp^delta),

whose coefficient Sigma and exponent delta are inputs (default
Sigma = 2.0 W m^-2 K^-3, delta = 3, after G.-H. Lee et al., Nature 586,
42 (2020)). They differ between devices, and Lee et al. obtained theirs
the way this module does (with delta fixed at 3): by fitting this law to the measured
electron temperature Te of the graphene as a function of the absorbed
power P at a fixed bath temperature (their Fig. 4c and Table I:
delta = 3 and Sigma = 2.04 to 3.30 W m^-2 K^-3 for four gate voltages).

`fit_ep_coupling` fits Sigma, delta and, optionally, the bath
temperature Tp to (P, Te) pairs. The model is solved for the measured
quantity,

    Te(P) = (Tp^delta + P / (Sigma A))^(1/delta)

(the closed form of `steady_temperature`), and fitted by weighted least
squares in Te, so the errors you give are those of your thermometer.
The covariance is the usual linearized one, (J^T W J)^-1, exact-noise
when the Te errors are given (with a chi-square check) and scaled by
the residual variance otherwise. `EPCouplingFit.propagate` carries it
into any number computed from (Sigma, delta), with the sigma-point rule
of `IcFit.propagate`.

What the data can and cannot determine, stated rather than hidden: if
every point stays close to the bath (Te - Tp << Tp), the law is linear,
P = G (Te - Tp) with G = delta Sigma A Tp^(delta-1), and only G is
determined, not Sigma and delta separately. The fit then warns, and
refuses when the information matrix is numerically singular. Fitting Tp
as well needs points at small power.
"""
from __future__ import annotations

import dataclasses
import warnings

import numpy as np
from scipy.optimize import least_squares

__all__ = ["EPCouplingFit", "fit_ep_coupling", "ep_temperature"]

_DELTA_RANGE = (1.0, 8.0)        # search range of the exponent
_COND_MAX = 1e10


def ep_temperature(P, Tp, area, sigma, delta):
    """Electron temperature (K) at absorbed power P (W) for the cooling
    law P = sigma A (Te^delta - Tp^delta): (Tp^delta + P/(sigma A))^(1/delta).
    The same closed form as `steady_temperature`, for array P and float
    delta, with checked inputs."""
    P = np.asarray(P, dtype=float)
    Tp, area, sigma, delta = float(Tp), float(area), float(sigma), \
        float(delta)
    if not (np.isfinite(Tp) and Tp >= 0.0):
        raise ValueError("Tp must be finite and >= 0 (K)")
    if not (np.isfinite(area) and area > 0.0 and np.isfinite(sigma)
            and sigma > 0.0 and np.isfinite(delta) and delta > 0.0):
        raise ValueError("area, sigma and delta must be finite and "
                         "positive")
    if np.any(P < 0.0) or not np.all(np.isfinite(P)):
        raise ValueError("absorbed power must be finite and >= 0 (W)")
    return (Tp ** delta + P / (sigma * area)) ** (1.0 / delta)


@dataclasses.dataclass
class EPCouplingFit:
    """Fitted electron-phonon cooling law P = Sigma A (Te^delta - Tp^delta).

    Sigma : coefficient (W m^-2 K^-delta); delta : exponent;
    Tp : bath temperature (K), fitted or as given.
    sigma : one-standard-deviation errors of the fitted parameters
        ('Sigma', 'delta' and, if fitted, 'Tp').
    cov : covariance of the fitted parameters in the order of `names`.
    names : the fitted parameters, a subset of ('Sigma', 'delta', 'Tp').
    chi2, chi2_dof : chi-square and degrees of freedom when the Te
        errors were given; None otherwise.
    condition_number : of the correlation-scaled information matrix
        (above about 1e8 the parameters are nearly degenerate).
    Te_fit : the model at the measured powers; area : as given (m^2);
    n_points : number of points.
    """

    Sigma: float
    delta: float
    Tp: float
    sigma: dict
    cov: np.ndarray
    names: tuple
    chi2: float | None
    chi2_dof: int | None
    condition_number: float
    Te_fit: np.ndarray
    area: float
    n_points: int

    def G(self, T) -> np.ndarray:
        """Thermal conductance dP/dTe = delta Sigma A Te^(delta-1) (W/K)
        of the fitted law at electron temperature T, the quantity
        `SensorBudget.Gep` computes from (sigma, delta)."""
        T = np.asarray(T, dtype=float)
        out = self.delta * self.Sigma * self.area * T ** (self.delta - 1.0)
        return float(out) if out.ndim == 0 else out

    def propagate(self, func, max_nonlinearity: float = 0.1) -> dict:
        """Error bar of any number computed from the fitted law.

        func : callable func(Sigma, delta) returning a number or an
            array, for example
            ``lambda s, d: SensorBudget(rec, sigma=s, delta=d).energy_resolution(T, tauA)``.

        Sigma points, as in `IcFit.propagate`: along each principal
        axis k of the (Sigma, delta) covariance (the marginal block when
        Tp was fitted; a fixed delta contributes nothing) the
        parameters move by d_k = +-sqrt(l_k) v_k, and
        cov_f = sum_k g_k g_k^T with g_k = (f(p + d_k) - f(p - d_k))/2,
        exact for a function linear in (Sigma, delta). Refused when the
        second-order change over one standard deviation exceeds
        max_nonlinearity times the error bar, when Sigma or delta would
        leave (0, inf) within one standard deviation, or when func
        returns a non-finite value.

        Returns dict(value, sigma, cov, nonlinearity).
        """
        if not callable(func):
            raise TypeError("func must be callable: func(Sigma, delta)")
        thr = float(max_nonlinearity)
        if not (np.isfinite(thr) and thr > 0.0):
            raise ValueError("max_nonlinearity must be finite and positive")
        p0 = np.array([self.Sigma, self.delta], dtype=float)
        cov2 = np.zeros((2, 2))
        idx = [i for i, n in enumerate(("Sigma", "delta"))
               if n in self.names]
        pos = [self.names.index(n) for n in ("Sigma", "delta")
               if n in self.names]
        for a, pa in zip(idx, pos):
            for b, pb in zip(idx, pos):
                cov2[a, b] = self.cov[pa, pb]
        lam, vec = np.linalg.eigh(0.5 * (cov2 + cov2.T))
        if np.any(lam < -1e-12 * max(1.0, float(np.max(np.abs(lam))))):
            raise ValueError("the fit covariance is not positive "
                             "semi-definite")
        lam = np.clip(lam, 0.0, None)

        def evaluate(p):
            if not (p[0] > 0.0 and p[1] > 0.0):
                raise ValueError(
                    "within one standard deviation the parameters leave "
                    f"their range (Sigma = {p[0]:.4g}, delta = "
                    f"{p[1]:.4g}; need both > 0): the error bars are "
                    "too large for first-order propagation")
            val = np.asarray(func(float(p[0]), float(p[1])), dtype=float)
            if not np.all(np.isfinite(val)):
                raise ValueError("func returned a non-finite value at "
                                 f"Sigma = {p[0]:.4g}, delta = {p[1]:.4g}")
            return val

        f0 = evaluate(p0)
        g, c = [], []
        for k in range(2):
            d = np.sqrt(lam[k]) * vec[:, k]
            if not np.any(d):
                continue
            fp, fm = evaluate(p0 + d), evaluate(p0 - d)
            g.append(0.5 * (fp - fm))
            c.append(0.5 * (fp + fm - 2.0 * f0))
        if not g:
            g, c = [np.zeros_like(f0)], [np.zeros_like(f0)]
        g = np.array(g).reshape(len(g), -1)
        c = np.array(c).reshape(len(c), -1)
        cov_f = g.T @ g
        sig = np.sqrt(np.diag(cov_f))
        curv = np.max(np.abs(c), axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(curv == 0.0, 0.0, curv / sig)
        nonlin = float(np.max(ratio))
        if nonlin > thr:
            raise ValueError(
                f"func is too nonlinear over the fit's error bars: the "
                f"second-order change over one standard deviation is "
                f"{nonlin:.3g} times the first-order error bar (limit "
                f"{thr:g}); sample (Sigma, delta) from `cov` instead")
        return dict(value=float(f0) if f0.ndim == 0 else f0,
                    sigma=float(sig[0]) if f0.ndim == 0
                    else sig.reshape(f0.shape),
                    cov=cov_f, nonlinearity=nonlin)


def fit_ep_coupling(P_W, Te_K, area, Tp_K, sigma_Te=None, delta=None,
                    fit_Tp=False, Sigma0=None) -> EPCouplingFit:
    """Fit the electron-phonon cooling law to a measured self-heating
    curve.

    P_W : (n,) absorbed power (W), >= 0 (include P = 0 points if you
        have them; they pin Tp).
    Te_K : (n,) measured electron temperature (K) at each power.
    area : the area A of the electron system (m^2) in P = Sigma A (...).
    Tp_K : bath (phonon) temperature (K): the value used when
        fit_Tp is False, the starting value otherwise.
    sigma_Te : optional one-standard-deviation errors of Te (K), scalar
        or (n,). With them: exact-noise covariance and a chi-square
        check. Without: residual-variance covariance (needs at least
        two more points than fitted parameters).
    delta : None to fit the exponent (searched in 1 <= delta <= 8), or
        a number to hold it fixed (for example 3, the value of Lee et
        al.).
    fit_Tp : also fit the bath temperature.
    Sigma0 : optional starting value of Sigma (W m^-2 K^-delta).

    Returns an `EPCouplingFit`. Raises ValueError on malformed input,
    on a fit that ends at an end of the delta range (the data do not
    determine delta there), and when the information matrix is
    singular (the data cannot separate the parameters); RuntimeError
    if the optimizer fails. Warns when every point stays within 10 %
    of the bath in Te^delta (only G is then well determined).
    """
    P = np.asarray(P_W, dtype=float).ravel()
    Te = np.asarray(Te_K, dtype=float).ravel()
    if P.shape != Te.shape:
        raise ValueError("P_W and Te_K must have the same length")
    if not (np.all(np.isfinite(P)) and np.all(np.isfinite(Te))):
        raise ValueError("P_W and Te_K must be finite")
    if np.any(P < 0.0):
        raise ValueError("absorbed powers must be >= 0 (W)")
    if np.any(Te <= 0.0):
        raise ValueError("electron temperatures must be positive (K)")
    area = float(area)
    if not (np.isfinite(area) and area > 0.0):
        raise ValueError("area must be finite and positive (m^2)")
    Tp0 = float(Tp_K)
    if not (np.isfinite(Tp0) and Tp0 > 0.0):
        raise ValueError("Tp_K must be finite and positive (K)")
    fit_delta = delta is None
    if not fit_delta:
        delta = float(delta)
        if not (np.isfinite(delta) and delta > 0.0):
            raise ValueError("a fixed delta must be finite and positive")
    names = ("Sigma",) + (("delta",) if fit_delta else ()) \
        + (("Tp",) if fit_Tp else ())
    k = len(names)
    n = P.size
    if sigma_Te is None:
        if n < k + 2:
            raise ValueError(
                f"without sigma_Te the noise level is estimated from the "
                f"residuals, which needs at least {k + 2} points for "
                f"{k} fitted parameters")
        w = np.ones(n)
    else:
        if n < k + 1:
            raise ValueError(f"need at least {k + 1} points for {k} "
                             "fitted parameters")
        s = np.broadcast_to(np.asarray(sigma_Te, dtype=float),
                            (n,)).copy()
        if not np.all(np.isfinite(s)) or np.any(s <= 0.0):
            raise ValueError("sigma_Te must be finite and positive")
        w = 1.0 / s
    if not np.any(P > 0.0):
        raise ValueError("no point has P > 0: the cooling law needs "
                         "heated points")

    d0 = 3.0 if fit_delta else delta
    if Sigma0 is None:
        hot = P > 0.0
        excess = Te[hot] ** d0 - Tp0 ** d0
        ok = excess > 0.0
        if not np.any(ok):
            raise ValueError("no heated point lies above the bath "
                             "temperature: the data show no heating")
        Sigma0 = float(np.median(P[hot][ok] / (area * excess[ok])))
    Sigma0 = float(Sigma0)
    if not (np.isfinite(Sigma0) and Sigma0 > 0.0):
        raise ValueError("Sigma0 must be finite and positive")

    # parameters: ln Sigma, [delta], [Tp]
    p0 = [np.log(Sigma0)]
    lo, hi = [-np.inf], [np.inf]
    if fit_delta:
        p0.append(d0)
        lo.append(_DELTA_RANGE[0])
        hi.append(_DELTA_RANGE[1])
    if fit_Tp:
        p0.append(Tp0)
        lo.append(1e-6 * Tp0)
        hi.append(np.inf)

    def unpack(p):
        Sg = np.exp(p[0])
        i = 1
        dl = delta
        if fit_delta:
            dl = p[i]
            i += 1
        tp = p[i] if fit_Tp else Tp0
        return Sg, dl, tp

    def resid(p):
        Sg, dl, tp = unpack(p)
        return w * ((tp ** dl + P / (Sg * area)) ** (1.0 / dl) - Te)

    res = least_squares(resid, np.array(p0), bounds=(lo, hi),
                        xtol=1e-14, ftol=1e-14, gtol=1e-14,
                        x_scale="jac")
    if not res.success:
        raise RuntimeError(f"electron-phonon fit did not converge: "
                           f"{res.message}")
    Sg, dl, tp = unpack(res.x)
    if fit_delta:
        span = _DELTA_RANGE[1] - _DELTA_RANGE[0]
        if min(dl - _DELTA_RANGE[0], _DELTA_RANGE[1] - dl) < 1e-6 * span:
            raise ValueError(
                f"the fitted exponent ends at the edge of the searched "
                f"range {_DELTA_RANGE} (delta = {dl:.4g}): the data do "
                "not determine delta; fix it (delta=...) or extend the "
                "heating range")

    # Jacobian in (Sigma, delta, Tp) itself: chain rule from ln Sigma
    J = res.jac.copy()
    J[:, 0] = J[:, 0] / Sg
    A = J.T @ J
    dg = np.sqrt(np.diag(A))
    if np.any(dg <= 0.0) or not np.all(np.isfinite(dg)):
        raise ValueError("the data carry no information on "
                         + ", ".join(nm for nm, v in zip(names, dg)
                                     if not v > 0.0))
    As = A / np.outer(dg, dg)
    sv = np.linalg.svd(As, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0.0 else np.inf
    if not (np.isfinite(cond) and cond <= _COND_MAX):
        raise ValueError(
            "the data cannot separate " + ", ".join(names) + " (the "
            f"information matrix has condition number {cond:.3g}): "
            "heat further above the bath, or fix delta"
            + (" or Tp" if fit_Tp else ""))
    cov = np.linalg.inv(As) / np.outer(dg, dg)
    rss = float(2.0 * res.cost)
    dof = n - k
    if sigma_Te is None:
        cov = cov * (rss / dof)
        chi2 = chi2_dof = None
    else:
        chi2, chi2_dof = rss, dof
    lin = float(np.max((Te[P > 0.0] ** dl - tp ** dl) / tp ** dl))
    if fit_delta and lin < 0.1:
        warnings.warn(
            f"every point stays within {lin * 100:.3g} % of the bath in "
            "Te^delta: the law is nearly linear there, so Sigma and "
            "delta are strongly correlated (only G = delta Sigma A "
            "Tp^(delta-1) is well determined) -- inspect `cov`",
            stacklevel=2)
    err = np.sqrt(np.diag(cov))
    return EPCouplingFit(
        Sigma=float(Sg), delta=float(dl), Tp=float(tp),
        sigma={nm: float(e) for nm, e in zip(names, err)},
        cov=cov, names=names, chi2=chi2, chi2_dof=chi2_dof,
        condition_number=cond,
        Te_fit=(tp ** dl + P / (Sg * area)) ** (1.0 / dl),
        area=area, n_points=n)
