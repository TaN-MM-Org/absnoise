"""Plan the measurements this package later fits.

`absnoise.icfit` calibrates a junction from a measured Ic(T) curve,
and `absnoise.psdfit` extracts occupation noise from a measured
spectrum -- both after the data exist. This module answers the
planning questions that come first: which temperatures are worth
measuring, how good will the calibration be, and which frequency band
can determine the correlation time at all.

Statistics, stated plainly: predicted error bars are the standard
weighted-least-squares covariance (J^T W J)^-1 -- the same matrix
`fit_ic_curve` reports -- evaluated around your expected junction
before any data exist (the Fisher information of independent Gaussian
measurements; any statistics text, under "Cramer-Rao bound"). The
temperature-design tool maximizes its determinant (D-optimal design;
F. Pukelsheim, Optimal Design of Experiments, SIAM (2006)). The PSD
band planner contains no statistics at all: it applies, in advance,
exactly the acceptance rules `fit_telegraph_psd` enforces after the
fact, so a band it approves is one the fit will accept.
"""
from __future__ import annotations

import numpy as np

from .icfit import ic_model

__all__ = ["plan_ic_measurement", "design_ic_temperatures",
           "psd_band_for_tau"]

_COND_MAX = 1e10
_NAMES = ("Ic0", "Tc", "tau")


def _check_point(Ic0, Tc, tau):
    if not (np.isfinite(Ic0) and Ic0 > 0.0):
        raise ValueError("Ic0 must be finite and positive (A)")
    if not (np.isfinite(Tc) and Tc > 0.0):
        raise ValueError("Tc must be finite and positive (K)")
    if not (0.0 < tau <= 1.0):
        raise ValueError("tau must lie in (0, 1]")


def _jacobian(T, Ic0, Tc, tau, n_phi):
    """d Ic / d(Ic0, Tc, tau) by central differences through the
    package's own `ic_model` -- the exact curve `fit_ic_curve` fits."""
    T = np.asarray(T, dtype=float).ravel()
    x0 = dict(Ic0=float(Ic0), Tc=float(Tc), tau=float(tau))
    jac = np.empty((T.size, 3))
    for j, name in enumerate(_NAMES):
        h = 1e-6 * abs(x0[name])
        xp, xm = dict(x0), dict(x0)
        xp[name] += h
        xm[name] = max(xm[name] - h, 1e-12)
        if name == "tau" and xp[name] > 1.0:
            xp[name] = 1.0
        fp = np.asarray(ic_model(T, xp["Ic0"], xp["Tc"], xp["tau"],
                                 n_phi=n_phi))
        fm = np.asarray(ic_model(T, xm["Ic0"], xm["Tc"], xm["tau"],
                                 n_phi=n_phi))
        jac[:, j] = (fp - fm) / (xp[name] - xm[name])
    return jac


def _invert_information(fisher, names):
    """Scale-invariant inversion: parameters carry different units
    (amperes, kelvin, dimensionless), so identifiability is judged on
    the correlation-scaled matrix D^-1 F D^-1, D = sqrt(diag F) --
    exact functional degeneracies survive the scaling, unit mismatches
    do not."""
    d = np.sqrt(np.diag(fisher))
    if np.any(d <= 0.0) or not np.all(np.isfinite(d)):
        return False, np.inf, None
    fs = fisher / np.outer(d, d)
    sv = np.linalg.svd(fs, compute_uv=False)
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf
    if not (np.isfinite(cond) and cond <= _COND_MAX):
        return False, cond, None
    cov = np.linalg.inv(fs) / np.outer(d, d)
    err = np.sqrt(np.diag(cov))
    return True, cond, {n: float(s) for n, s in zip(names, err)}


def plan_ic_measurement(T_K, sigma_Ic_A, Ic0, Tc, tau, n_phi=601):
    """Predicted calibration error bars for a planned Ic(T) run.

    T_K : (n,) the temperatures you intend to measure (K).
    sigma_Ic_A : expected 1-sigma error of each Ic point (scalar or
        (n,), amperes).
    Ic0, Tc, tau : your expected junction (from a previous cooldown,
        the shipped recipes, or design values) -- the working point
        the sensitivities are computed around.

    Returns dict(identifiable, condition_number, sigma,
    covers_tc_knee). `sigma` maps Ic0 (A), Tc (K) and tau to the
    error bars `fit_ic_curve` would report on data with these sigmas,
    or is None when the design cannot tell the parameters apart.
    `covers_tc_knee` is False when max(T) < 0.3 Tc -- the same
    leverage warning the fit itself gives: far below Tc the curve
    barely feels Tc, so its error bar balloons.
    """
    _check_point(Ic0, Tc, tau)
    T = np.asarray(T_K, dtype=float).ravel()
    if T.size < 1 or np.any(T <= 0.0) or not np.all(np.isfinite(T)):
        raise ValueError("temperatures must be positive and finite (K)")
    sig = np.broadcast_to(np.asarray(sigma_Ic_A, dtype=float),
                          T.shape).copy()
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigma_Ic_A must be finite and positive")
    jac = _jacobian(T, Ic0, Tc, tau, n_phi) / sig[:, None]
    fisher = jac.T @ jac
    identifiable, cond, sigma = _invert_information(fisher, _NAMES)
    if T.size < 3:
        identifiable, sigma = False, None
    return {"identifiable": identifiable, "condition_number": cond,
            "sigma": sigma,
            "covers_tc_knee": bool(float(T.max()) >= 0.3 * float(Tc))}


def design_ic_temperatures(candidates_K, n_pick, sigma_Ic_A, Ic0, Tc,
                           tau, n_phi=601):
    """Pick the most informative temperatures to measure.

    Greedy D-optimal selection from the reachable candidate
    temperatures: each pick most increases the determinant of the
    information matrix. Transparent and monotone, but a good-practice
    heuristic, not a proof of the globally best subset.

    Returns dict(indices, condition_number, sigma) with the chosen
    candidate indices in pick order and the planned error bars of the
    chosen set. Refuses when even the full candidate list cannot
    identify (Ic0, Tc, tau).
    """
    _check_point(Ic0, Tc, tau)
    T = np.asarray(candidates_K, dtype=float).ravel()
    if np.any(T <= 0.0) or not np.all(np.isfinite(T)):
        raise ValueError("temperatures must be positive and finite (K)")
    n = T.size
    n_pick = int(n_pick)
    if not 3 <= n_pick <= n:
        raise ValueError(f"n_pick must be between 3 (the number of "
                         f"fitted parameters) and {n}")
    sig = np.broadcast_to(np.asarray(sigma_Ic_A, dtype=float),
                          T.shape).copy()
    if np.any(sig <= 0.0) or not np.all(np.isfinite(sig)):
        raise ValueError("sigma_Ic_A must be finite and positive")
    full = plan_ic_measurement(T, sig, Ic0, Tc, tau, n_phi)
    if not full["identifiable"]:
        raise ValueError(
            "even the full candidate list cannot determine (Ic0, Tc, "
            "tau); extend the temperature range toward Tc, where the "
            "curve actually bends")
    rows = _jacobian(T, Ic0, Tc, tau, n_phi) / sig[:, None]
    # column-scaled (unit-free) greedy; identical scaling leaves the
    # choices unchanged but keeps the start-up regularizer meaningful
    scale = np.sqrt(np.mean(rows * rows, axis=0))
    rs = rows / scale
    eps = 1e-12 * float(np.max(np.sum(rs * rs, axis=1)))
    fs = eps * np.eye(3)
    chosen = []
    for _ in range(n_pick):
        best_j, best_det = -1, -np.inf
        for j in range(n):
            if j in chosen:
                continue
            det = float(np.linalg.slogdet(fs + np.outer(rs[j],
                                                        rs[j]))[1])
            if det > best_det:
                best_j, best_det = j, det
        fs = fs + np.outer(rs[best_j], rs[best_j])
        chosen.append(best_j)
    sub = plan_ic_measurement(T[chosen], sig[chosen], Ic0, Tc, tau,
                              n_phi)
    return {"indices": list(chosen),
            "condition_number": sub["condition_number"],
            "sigma": sub["sigma"]}


def psd_band_for_tau(tau_expected_s, margin=4.0):
    """A frequency band from which `fit_telegraph_psd` can work.

    The fit refuses, by design, a band that never sees the knee
    f_c = 1/(2 pi tau): the fitted knee must land inside
    [2 f_min, f_max/2] and the spectrum must vary by at least a factor
    of 4 across the band. This planner applies those same rules in
    advance: it returns (f_lo, f_hi) = (f_c/margin, f_c*margin), which
    satisfies both with room to spare for any margin >= 4 (at
    f = f_c * margin the Lorentzian has dropped by 1 + margin^2 >= 17,
    comfortably past the factor-4 rule) -- checked in the tests by
    running the fit on a synthetic spectrum over exactly this band.

    tau_expected_s : your rough expectation of the correlation time
        (an order of magnitude is enough; the band spans margin^2 in
        frequency).
    margin : half-decade factor on each side of the knee (>= 4).

    Returns (f_lo_hz, f_hi_hz, f_knee_hz).
    """
    if not (np.isfinite(tau_expected_s) and tau_expected_s > 0.0):
        raise ValueError("tau_expected_s must be finite and positive")
    m = float(margin)
    if not (np.isfinite(m) and m >= 4.0):
        raise ValueError("margin must be >= 4: smaller margins cannot "
                         "guarantee the knee-visibility rules the fit "
                         "enforces")
    f_c = 1.0 / (2.0 * np.pi * float(tau_expected_s))
    return f_c / m, f_c * m, f_c
