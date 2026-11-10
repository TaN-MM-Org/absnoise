"""Allan variance: when does averaging an Andreev thermometer stop helping?

Occupation noise is exponentially correlated (correlation time tauA),
and the practical question for a thermometer or calorimeter readout is
not the PSD but the stability versus integration time: average longer
and the reading improves, until slow noise or drift takes over. The
two-sample (Allan) variance is the standard language for that question
(D. W. Allan, Proc. IEEE 54, 221 (1966)); this module computes it from
data with the overlapping estimator (W. J. Riley, Handbook of
Frequency Stability Analysis, NIST Special Publication 1065, 2008) and
gives the closed forms this package's own noise processes obey.

For a stationary process with autocovariance C(t) = var * exp(-|t|/tau)
(the occupation noise of a single Andreev channel, and the telegraph
traces of `telegraph_traces`), integrating the defining double
integrals of the two-sample variance gives, with averaging time T,

    AVAR(T) = var * (tau / T^2) *
              (2 T - 3 tau + 4 tau e^(-T/tau) - tau e^(-2 T/tau)),

which limits to var * (2 T / (3 tau)) for T << tau (adjacent windows
see nearly the same noise state) and to 2 var tau / T = S(0) / (2 T)
for T >> tau (the white floor of the Lorentzian, S(0) = 4 var tau in
this package's single-sided convention). The derivation is elementary
but error-prone, so the test suite checks the closed form against a
direct numerical evaluation of the defining integrals, both limits,
and seeded telegraph Monte Carlo.

Sampled data (new in 0.16.0): `allan_variance` averages a whole number
m of samples, not a continuous window, and for windows of a few samples
its expectation differs from the formula above (by 51 % for m = 1 at
dt = tau/40). `avar_exponential(..., dt=dt)` gives the exact
expectation for samples taken every dt.

Exact facts the test suite asserts, rather than states:

* A pure linear drift x = c t has Allan deviation c T / sqrt(2),
  reproduced to machine precision by the estimator.
* The closed form matches numerical integration of the defining double
  integrals to better than 1e-6 relative.
* Seeded telegraph traces reproduce the closed form, and seeded white
  noise reproduces S0 / (2 T).
"""
from __future__ import annotations

import numpy as np


def allan_variance(x, dt, m_list=None):
    """Overlapping Allan variance of a uniformly sampled series.

    x : 1D array, samples of the quantity whose stability is asked
        about (an occupation, a temperature estimate, a fractional
        frequency); uniform sampling interval dt.
    m_list : averaging windows in samples (each >= 1, at most
        len(x) // 2); by default an octave ladder 1, 2, 4, ... .

    Returns (taus, avar): averaging times m * dt and the overlapping
    two-sample variance

        AVAR(m dt) = < ( mean(x[i+m : i+2m]) - mean(x[i : i+m]) )^2 > / 2

    over all overlapping index positions i. The overlapping estimator
    uses the data more efficiently than the non-overlapping one at
    identical expectation value (Riley, NIST SP 1065).
    """
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 4:
        raise ValueError("need at least 4 samples")
    if not np.all(np.isfinite(x)):
        raise ValueError("series contains non-finite samples")
    dt = float(dt)
    # (fixed in 0.16.0) an infinite or NaN dt was accepted and gave
    # infinite or NaN averaging times
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    n = x.size
    if m_list is None:
        m_list = []
        m = 1
        while m <= n // 2:
            m_list.append(m)
            m *= 2
    ms = np.asarray(m_list, dtype=int)
    if ms.size == 0 or np.any(ms < 1) or np.any(ms > n // 2):
        raise ValueError("each window must satisfy 1 <= m <= len(x)//2")
    csum = np.concatenate([[0.0], np.cumsum(x)])
    taus = ms * dt
    avar = np.empty(ms.size)
    for k, m in enumerate(ms):
        means = (csum[m:] - csum[:-m]) / m       # all overlapping windows
        d = means[m:] - means[:-m]               # gap m: adjacent windows
        avar[k] = 0.5 * np.mean(d * d)
    return taus, avar


def _avar_sampled_bracket(m, eps):
    """N(m, eps) = m (1 - r^2) - r (3 - 4 r^m + r^(2m)), r = exp(-eps),
    for integer m >= 1 and eps = dt/tau > 0 (arrays of equal shape),
    without cancellation (new in 0.16.0).

    For m eps >= 0.5 the closed form is used directly. Below, its terms
    cancel to order eps^3, and N is summed as its Taylor series in eps:
    expanding every exponential gives
    N = sum_{n>=3} (-1)^n c_n eps^n / n!,
    c_n = 4 (m+1)^n - (2m+1)^n - m 2^n - 3 (the coefficients of n = 0,
    1 and 2 vanish identically), evaluated as
    4 x1^n - x2^n - m x3^n - 3 eps^n with x1 = (m+1) eps,
    x2 = (2m+1) eps, x3 = 2 eps, all below 1.5. 40 terms reach machine
    precision there (the terms fall like 1.5^n / n!)."""
    shape = np.shape(m)
    m = np.atleast_1d(np.asarray(m, dtype=float)).ravel()
    eps = np.atleast_1d(np.asarray(eps, dtype=float)).ravel()
    r = np.exp(-eps)
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        out = (m * (-np.expm1(-2.0 * eps))
               - r * (3.0 - 4.0 * np.exp(-m * eps)
                      + np.exp(-2.0 * m * eps)))
    small = m * eps < 0.5
    if np.any(small):
        ms, es = m[small], eps[small]
        x1, x2, x3 = (ms + 1.0) * es, (2.0 * ms + 1.0) * es, 2.0 * es
        p1 = p2 = p3 = p0 = np.ones_like(es)
        fact = 1.0
        series = np.zeros_like(es)
        for n in range(1, 41):
            p1, p2, p3, p0 = p1 * x1, p2 * x2, p3 * x3, p0 * es
            fact *= n
            if n >= 3:
                series = series + (-1.0) ** n * (
                    4.0 * p1 - p2 - ms * p3 - 3.0 * p0) / fact
        out[small] = series
    return out.reshape(shape)


def avar_exponential(var: float, tau_c: float, T, dt: float | None = None):
    """Closed-form Allan variance of exponentially correlated noise.

    var : process variance (for a telegraph occupation, f (1 - f)).
    tau_c : correlation time (for a single Andreev channel, tauA).
    T : averaging time(s), scalar or array.

    Returns var * (tau/T^2) * (2T - 3 tau + 4 tau e^(-T/tau)
    - tau e^(-2T/tau)), the two-sample variance of a process with
    autocovariance var * exp(-|t|/tau). Limits: var * 2T/(3 tau) for
    T << tau, and S(0)/(2T) with S(0) = 4 var tau for T >> tau.

    For T < tau/2 (changed in 0.12.0) the bracket is summed as its
    Taylor series in r = T/tau,
    sum_{k>=3} (-1)^k (4 - 2^k) r^k / k! = 2r^3/3 - r^4/2 + ...,
    because the four terms of the closed form cancel to order r^3:
    up to 0.11.0 the result was 6e-5 too low at T = 1e-4 tau, 17 % too
    high at 1e-5 tau and negative at 1e-6 tau.

    dt (s, new in 0.16.0): the sample interval of the data. The formula
    above is the Allan variance of the continuous-time process, averaged
    over windows of length T. `allan_variance` averages m = T/dt
    samples instead, and for a sampled process (autocovariance
    var r^|k|, r = exp(-dt/tau)) the expectation of that estimator is

        AVAR(m) = var [m (1 - r^2) - r (3 - 4 r^m + r^(2m))]
                  / (m^2 (1 - r)^2),

    which follows from summing the autocovariance over the two
    adjacent windows (checked in the tests against that double sum in
    50-digit arithmetic and against telegraph Monte Carlo). It is
    var (1 - r) for one-sample windows and tends to the continuous
    form as dt/tau -> 0 at fixed T; for few samples per window the two
    differ: by 51 % at m = 1 and 2.2 % at m = 5 for dt = tau/40, and by
    0.2 % at m = 20. With dt, every T must be a whole number m >= 1 of
    sample intervals (within 1e-9 relative), as the windows of
    `allan_variance` are.
    """
    v = float(var)
    tau = float(tau_c)
    if not (np.isfinite(v) and v >= 0.0 and np.isfinite(tau)
            and tau > 0.0):
        raise ValueError("var must be finite and >= 0, and tau_c finite "
                         "and > 0")
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0) or not np.all(np.isfinite(T)):
        raise ValueError("averaging times must be finite and positive")
    if dt is not None:
        dt = float(dt)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive (s), or None")
        m = np.round(T / dt)
        if np.any(m < 1.0) or np.any(np.abs(T / dt - m) > 1e-9 * m):
            raise ValueError("with dt, every averaging time must be a "
                             "whole number m >= 1 of sample intervals "
                             "(T = m dt), as in allan_variance")
        a = -np.expm1(-dt / tau)                 # 1 - r
        eps = np.full_like(m, dt / tau)
        out = v * _avar_sampled_bracket(m, eps) / (m ** 2 * a ** 2)
        return float(out) if out.ndim == 0 else out
    r = T / tau
    with np.errstate(over="ignore", invalid="ignore"):
        out = v * (tau / T ** 2) * (2.0 * T - 3.0 * tau
                                    + 4.0 * tau * np.exp(-r)
                                    - tau * np.exp(-2.0 * r))
    small = r < 0.5
    if np.any(small):
        rs = r[small] if out.ndim else r
        # bracket / tau = sum_{k>=3} c_k r^k; terms fall like 1/k!, so
        # 24 terms reach machine precision for r < 0.5
        term = np.ones_like(rs)
        series = np.zeros_like(rs)
        for k in range(1, 27):
            term = term * rs / k                 # r^k / k!
            if k >= 3:
                series = series + (-1.0) ** k * (4.0 - 2.0 ** k) * term
        val = v * series / rs ** 2
        if out.ndim:
            out[small] = val
        else:
            out = np.asarray(val)
    return float(out) if out.ndim == 0 else out


def avar_white(S0, T):
    """Allan variance of white noise with single-sided PSD S0: S0/(2T).

    The white-noise floor every counting measurement eventually reaches;
    for the package's Lorentzian occupation noise it is the T >> tauA
    limit with S0 = 4 var tauA.
    """
    S0 = float(S0)
    if S0 < 0.0:
        raise ValueError("a power spectral density cannot be negative")
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("averaging times must be positive")
    out = S0 / (2.0 * T)
    return float(out) if out.ndim == 0 else out
