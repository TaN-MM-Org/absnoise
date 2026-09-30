"""Stochastic validation of the Andreev occupation-noise model.

Simulates M independent spin-resolved two-state occupations n(t) with
detailed-balance rates
    up:   gamma_in  = f(E) / tauA
    down: gamma_out = (1 - f(E)) / tauA
so that <n> = f, var(n) = f(1-f), correlation time tauA, and the
single-sided PSD of each n is  S_n(omega) = 4 f(1-f) tauA/(1+omega^2
tauA^2).  The observable O = g sum_ch (1 - n_up - n_down) then has
    S_O(omega) = g^2 M_ch * 2 * 4 f(1-f) tauA / (1 + omega^2 tauA^2),
which the tests compare against this closed form (plateau and knee),
together with the variance convention var(mean over t) = S(0) / (2 t).

Sampling (changed in 0.11.0): the traces are the continuous-time
process read at the sample times, with no time-step error. Over one
step dt the exact flip probabilities are

    p_up = f (1 - exp(-dt/tauA)),   p_dn = (1 - f)(1 - exp(-dt/tauA)),

the matrix exponential of the two-state rate matrix (asserted in the
tests against scipy.linalg.expm), so the sampled autocorrelation is
exactly exp(-k dt/tauA) at lag k and any dt > 0 is allowed. Up to
0.10.x the probabilities were the first-order values dt f/tauA and
dt (1-f)/tauA (autocorrelation (1 - dt/tauA)^k), which required
dt small (flip probability below 0.12, checked by an `assert`).
"""

import numpy as np


def flip_probabilities(f, tauA, dt):
    """Exact per-sample flip probabilities (p_up, p_dn) of a two-state
    occupation with equilibrium occupation f and correlation time
    tauA, sampled every dt: p_up = f (1 - e^(-dt/tauA)) for 0 -> 1 and
    p_dn = (1 - f)(1 - e^(-dt/tauA)) for 1 -> 0."""
    f, tauA, dt = float(f), float(tauA), float(dt)
    if not (0.0 <= f <= 1.0):
        raise ValueError("occupation f must lie in [0, 1]")
    if not (np.isfinite(tauA) and tauA > 0.0):
        raise ValueError("tauA must be finite and positive")
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    s = -np.expm1(-dt / tauA)           # 1 - exp(-dt/tauA), accurate
    return f * s, (1.0 - f) * s


def telegraph_traces(f, tauA, dt, n_steps, n_channels, rng):
    """Markov two-state occupations, shape (n_steps, n_channels),
    sampled exactly every dt (see the module docstring)."""
    p_up, p_dn = flip_probabilities(f, tauA, dt)
    n_steps, n_channels = int(n_steps), int(n_channels)
    if n_steps < 1 or n_channels < 1:
        raise ValueError("n_steps and n_channels must be >= 1")
    n = (rng.random(n_channels) < f).astype(np.int8)
    out = np.empty((n_steps, n_channels), dtype=np.int8)
    for i in range(n_steps):
        r = rng.random(n_channels)
        flip_up = (n == 0) & (r < p_up)
        flip_dn = (n == 1) & (r < p_dn)
        n = n ^ (flip_up | flip_dn)
        out[i] = n
    return out


def psd_single_sided(x, dt):
    """Periodogram single-sided PSD of x(t).

    Bins 1 .. n/2 - 1 carry twice the two-sided density; the zero and
    (for even n) Nyquist bins appear once in the two-sided spectrum
    and are not doubled (fixed in 0.11.0; before, the Nyquist bin was
    doubled). With this convention sum(S) * df equals the variance of
    x exactly (Parseval), df = 1 / (n dt)."""
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)
    n = len(x)
    X = np.fft.rfft(x)
    S = 2.0 * dt * np.abs(X)**2 / n
    S[0] *= 0.5
    if n % 2 == 0:
        S[-1] *= 0.5
    freqs = np.fft.rfftfreq(n, dt)
    return freqs, S
