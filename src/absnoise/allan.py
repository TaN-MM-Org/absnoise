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
    if dt <= 0.0:
        raise ValueError("dt must be positive")
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


def avar_exponential(var, tau_c, T):
    """Closed-form Allan variance of exponentially correlated noise.

    var : process variance (for a telegraph occupation, f (1 - f)).
    tau_c : correlation time (for a single Andreev channel, tauA).
    T : averaging time(s), scalar or array.

    Returns var * (tau/T^2) * (2T - 3 tau + 4 tau e^(-T/tau)
    - tau e^(-2T/tau)), the two-sample variance of a process with
    autocovariance var * exp(-|t|/tau). Limits: var * 2T/(3 tau) for
    T << tau, and S(0)/(2T) with S(0) = 4 var tau for T >> tau.
    """
    v = float(var)
    tau = float(tau_c)
    if v < 0.0 or tau <= 0.0:
        raise ValueError("var must be >= 0 and tau_c > 0")
    T = np.asarray(T, dtype=float)
    if np.any(T <= 0.0):
        raise ValueError("averaging times must be positive")
    r = T / tau
    out = v * (tau / T ** 2) * (2.0 * T - 3.0 * tau
                                + 4.0 * tau * np.exp(-r)
                                - tau * np.exp(-2.0 * r))
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
