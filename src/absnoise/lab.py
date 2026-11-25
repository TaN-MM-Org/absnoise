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
F. Pukelsheim, Optimal Design of Experiments, SIAM (2006)), greedily
or (new in 0.17.0) by checking every subset. The PSD
band planner contains no statistics at all: it applies, in advance,
exactly the acceptance rules `fit_telegraph_psd` enforces after the
fact, so a band it approves is one the fit will accept.
"""
from __future__ import annotations

import numpy as np

from .icfit import ic_model

__all__ = ["plan_ic_measurement", "design_ic_temperatures",
           "psd_band_for_tau", "plan_psd_measurement",
           "averages_for_tau"]

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


_MAX_SUBSETS = 2_000_000


def design_ic_temperatures(candidates_K, n_pick, sigma_Ic_A, Ic0, Tc,
                           tau, n_phi=601, method="greedy"):
    """Pick the most informative temperatures to measure.

    method="greedy" (the default): greedy D-optimal selection from the
    reachable candidate temperatures: each pick most increases the
    determinant of the information matrix. Transparent and monotone,
    but a good-practice heuristic, not a proof of the globally best
    subset.

    method="exhaustive" (new in 0.17.0): the determinant is computed
    for every subset of n_pick candidates, and the largest is taken,
    so the result is the best subset (the D-optimal one) for this
    model and these errors. It is refused when there are more than
    2 000 000 subsets (n choose n_pick); the 15 candidates and 5 picks
    of README Example 4 make 3003. The indices are then in increasing
    order.

    Returns dict(indices, condition_number, sigma, log_det) with the
    chosen candidate indices in pick order, the planned error bars of
    the chosen set and (new in 0.17.0) the natural log of the
    determinant of its information matrix J^T W J in SI units, the
    quantity both methods maximize (compare two designs by the
    difference). Refuses when even the full candidate list cannot
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
    if method not in ("greedy", "exhaustive"):
        raise ValueError("method must be 'greedy' or 'exhaustive'")
    if method == "exhaustive":
        from math import comb
        n_sub = comb(n, n_pick)
        if n_sub > _MAX_SUBSETS:
            raise ValueError(
                f"{n_sub} subsets of {n_pick} among {n} candidates is "
                f"more than {_MAX_SUBSETS} to check one by one; use "
                "method='greedy' or fewer candidates")
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
    if method == "exhaustive":
        chosen = _best_subset(rs, n_pick)
        sub = plan_ic_measurement(T[chosen], sig[chosen], Ic0, Tc, tau,
                                  n_phi)
        return {"indices": list(chosen),
                "condition_number": sub["condition_number"],
                "sigma": sub["sigma"],
                "log_det": _log_det(rows[chosen])}
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
            "sigma": sub["sigma"],
            "log_det": _log_det(rows[chosen])}


def _log_det(rows):
    """ln det of sum r r^T over the rows (the information matrix of
    the chosen temperatures, SI units)."""
    sign, ld = np.linalg.slogdet(rows.T @ rows)
    return float(ld) if sign > 0 else float("-inf")


def _best_subset(rs, k):
    """Indices of the k rows of rs (n x 3) whose information matrix
    sum r r^T has the largest determinant, by checking every subset
    (new in 0.17.0). The subsets are taken in blocks of 100 000, and
    the 3 x 3 determinants of a block are computed at once
    (numpy.linalg.det on the stack of summed matrices)."""
    from itertools import combinations, islice
    outer = np.einsum("ni,nj->nij", rs, rs).reshape(rs.shape[0], 9)
    best_det, best = -np.inf, None
    it = combinations(range(rs.shape[0]), k)
    while True:
        block = np.array(list(islice(it, 100_000)), dtype=np.intp)
        if block.size == 0:
            break
        M = outer[block].sum(axis=1).reshape(-1, 3, 3)
        det = np.linalg.det(M)
        j = int(np.argmax(det))
        if det[j] > best_det:
            best_det, best = float(det[j]), block[j]
    return [int(i) for i in best]


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


def _psd_log_jacobian(f, S0, tau_s, floor, fit_floor, dt=None,
                      sampling="instant", n_record=None):
    """d log(model) / d log(parameter) at the working point -- the
    same log-space parameterization `fit_telegraph_psd` uses (and the
    same model: with dt, the spectrum of the sampled process, with the
    sampling and record length of the fit)."""
    from .psdfit import telegraph_psd_model
    f = np.asarray(f, dtype=float).ravel()
    p0 = [np.log(S0), np.log(tau_s)] \
        + ([np.log(max(floor, 1e-300))] if fit_floor else [])

    def logmodel(p):
        s0, tau = np.exp(p[0]), np.exp(p[1])
        fl = np.exp(p[2]) if fit_floor else floor
        return np.log(telegraph_psd_model(f, s0, tau, fl, dt, sampling,
                                          n_record))

    jac = np.empty((f.size, len(p0)))
    for j in range(len(p0)):
        h = 1e-6
        pp, pm = list(p0), list(p0)
        pp[j] += h
        pm[j] -= h
        jac[:, j] = (logmodel(pp) - logmodel(pm)) / (2.0 * h)
    return jac


def plan_psd_measurement(f_hz, n_avg: float, S0: float, tau_s: float,
                         floor: float = 0.0, fit_floor: bool = True,
                         dt: float | None = None, sampling: str = "instant",
                         n_record: int | None = None) -> dict:
    """Predicted PSD-fit error bars for a planned averaging depth.

    f_hz : the frequency bins you will record (must satisfy the same
        knee-visibility rules `fit_telegraph_psd` enforces -- checked
        here in advance, with the same explanations).
    n_avg : number of independent periodogram averages per bin (>= 1;
        an effective, non-integer number is accepted, as in the fit).
    S0, tau_s, floor : your expected spectrum (plateau, correlation
        time, white floor -- the working point).

    Returns dict(sigma, identifiable, condition_number): `sigma` maps
    S0 (input PSD units), tau_s (s) and -- when fitted -- floor to
    the error bars the fit would report at this averaging depth. The
    per-bin log-variance is psi'(n_avg) (trigamma), the value the fit
    uses (changed in 0.11.0; up to 0.10.x it was 1/n_avg), so every
    error bar scales exactly as sqrt(psi'(n_avg)), which approaches
    1/sqrt(n_avg) for many averages (0.4 % larger at n_avg = 64, 28 %
    larger at n_avg = 1).

    dt (s, new in 0.16.0): plan the fit `fit_telegraph_psd(...,
    dt=dt)` of a spectrum of a trace sampled every dt, with the
    sampled-process model (aliasing included); the frequencies must
    then lie in (0, 1/(2 dt)], as the fit requires.

    sampling, n_record (new in 0.17.0): plan the fit with the same
    options of `fit_telegraph_psd` (a readout that averages over each
    sample interval; the expected periodogram of segments of n_record
    samples). n_avg may also be an array with one value per frequency,
    as in the fit (new in 0.17.0); each point then has log-variance
    psi'(n_i).
    """
    dt, sampling, n_record = _check_psd_point(f_hz, S0, tau_s, floor, dt,
                                              sampling, n_record)
    f = np.asarray(f_hz, dtype=float).ravel()
    if np.ndim(n_avg) > 0:
        n_arr = np.asarray(n_avg, dtype=float).ravel()
        if n_arr.size != f.size:
            raise ValueError("an array n_avg needs one value per frequency")
        if not (np.all(np.isfinite(n_arr)) and np.all(n_arr >= 1.0)):
            raise ValueError("every n_avg must be finite and >= 1")
        from scipy.special import polygamma
        inv_var = 1.0 / polygamma(1, n_arr)
        jac = _psd_log_jacobian(f, S0, tau_s, floor, fit_floor, dt,
                                sampling, n_record)
        fisher = (jac * inv_var[:, None]).T @ jac
    else:
        # an effective, non-integer n_avg is accepted, as
        # `fit_telegraph_psd` accepts it (fixed in 0.12.0: int()
        # truncated it, so n_avg = 1.5 was planned as 1)
        n_avg = float(n_avg)
        if not (np.isfinite(n_avg) and n_avg >= 1.0):
            raise ValueError("n_avg must be finite and >= 1 (the number "
                             "of averaged periodograms behind each bin)")
        jac = _psd_log_jacobian(f, S0, tau_s, floor, fit_floor, dt,
                                sampling, n_record)
        fisher = jac.T @ jac / _log_var(n_avg)
    names = ["S0", "tau_s"] + (["floor"] if fit_floor else [])
    identifiable, cond, sig_frac = _invert_information(fisher, names)
    sigma = None
    if identifiable:
        vals = {"S0": S0, "tau_s": tau_s, "floor": floor}
        sigma = {k: sig_frac[k] * vals[k] for k in names}
    return {"sigma": sigma, "identifiable": identifiable,
            "condition_number": cond}


def _log_var(n_avg):
    """Variance of the logarithm of an average of n_avg independent
    periodogram bins (Gamma(n, 1/n) scatter): the trigamma function
    psi'(n_avg). Same value as `fit_telegraph_psd` uses."""
    from scipy.special import polygamma
    return float(polygamma(1, float(n_avg)))


def averages_for_tau(target_sigma_tau_s, f_hz, S0, tau_s, floor=0.0,
                     fit_floor=True, dt=None, sampling="instant",
                     n_record=None):
    """How many periodogram averages does a target tau error bar
    cost? Every error bar scales as sqrt(psi'(n_avg)) (see
    `plan_psd_measurement`), so this returns the smallest integer
    n_avg with psi'(n_avg) <= (target / sigma_unit)^2, where
    sigma_unit is the error bar for unit log-variance per bin. Up to
    0.10.x the scaling was taken as 1/sqrt(n_avg), which gives the
    same number or one fewer.

    Returns (n_avg, plan) with `plan` the `plan_psd_measurement`
    result at the returned depth. Refuses bands the fit itself would
    refuse, with the same explanation. dt (new in 0.16.0), sampling and
    n_record (new in 0.17.0): as in `plan_psd_measurement`.
    """
    t = float(target_sigma_tau_s)
    if not (np.isfinite(t) and t > 0.0):
        raise ValueError("target_sigma_tau_s must be positive (s)")
    base = plan_psd_measurement(f_hz, 1, S0, tau_s, floor, fit_floor,
                                dt, sampling, n_record)
    if not base["identifiable"]:
        raise ValueError("the band cannot determine the parameters "
                         "at any averaging depth; change the band "
                         "(see psd_band_for_tau)")
    sigma_unit = base["sigma"]["tau_s"] / np.sqrt(_log_var(1))
    q = (t / sigma_unit) ** 2           # allowed per-bin log-variance
    # psi'(n) > 1/n, so no n below 1/q qualifies; psi'(n) is
    # decreasing and exceeds 1/n by about 1/(2 n^2), so the search
    # below stops after one or two steps
    n = max(1, int(np.floor(1.0 / q)))
    while _log_var(n) > q:
        n += 1
    return n, plan_psd_measurement(f_hz, n, S0, tau_s, floor,
                                   fit_floor, dt, sampling, n_record)


def _check_psd_point(f_hz, S0, tau_s, floor, dt=None, sampling="instant",
                     n_record=None):
    from .psdfit import _bin_index, _check_sampling, telegraph_psd_model
    f = np.asarray(f_hz, dtype=float).ravel()
    if f.size < 8 or np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("need >= 8 positive, finite frequencies "
                         "(the fit's own rule)")
    if not (np.isfinite(S0) and S0 > 0.0 and np.isfinite(tau_s)
            and tau_s > 0.0 and np.isfinite(floor) and floor >= 0.0):
        raise ValueError("S0 and tau_s must be positive and floor "
                         ">= 0")
    if dt is not None:
        dt = float(dt)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive (s), or None")
        if float(f.max()) > 0.5 / dt * (1.0 + 1e-9):
            raise ValueError(
                f"frequency {f.max():.6g} Hz lies above the Nyquist "
                f"frequency 1/(2 dt) = {0.5 / dt:.6g} Hz: the fit "
                "would refuse it")
    dt, sampling, n_record = _check_sampling(dt, sampling, n_record)
    if n_record is not None:
        _bin_index(f, dt, n_record)
    f_knee = 1.0 / (2.0 * np.pi * tau_s)
    if f_knee < 2.0 * f.min() or f_knee > 0.5 * f.max():
        raise ValueError(
            f"the expected knee {f_knee:.3g} Hz lies outside the "
            f"planned band [{f.min():.3g}, {f.max():.3g}] Hz with "
            "the fit's factor-2 margin: `fit_telegraph_psd` would "
            "refuse this measurement, so it is refused here, before "
            "the fridge time (see psd_band_for_tau)")
    S = telegraph_psd_model(f, S0, tau_s, floor, dt, sampling, n_record)
    if float(S.max() / S.min()) < 4.0:
        raise ValueError(
            "the expected spectrum varies by less than a factor of 4 "
            "across the band: the fit's knee-visibility rule would "
            "refuse it, so it is refused here, in advance")
    return dt, sampling, n_record
