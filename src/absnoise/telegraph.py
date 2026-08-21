"""Stochastic validation of the Andreev occupation-noise model.

Simulates M independent spin-resolved two-state occupations n(t) with
detailed-balance rates
    up:   gamma_in  = f(E) / tauA
    down: gamma_out = (1 - f(E)) / tauA
so that <n> = f, var(n) = f(1-f), correlation time tauA, and the
single-sided PSD of each n is  S_n(omega) = 4 f(1-f) tauA/(1+omega^2
tauA^2).  The observable O = g sum_ch (1 - n_up - n_down) then has
    S_O(omega) = g^2 M_ch * 2 * 4 f(1-f) tauA / (1 + omega^2 tauA^2),
which the testbench compares against the analytic sums of
short_junction.andreev_sums, together with the variance convention
var(mean over t) = S(0) / (2 t).
"""

import numpy as np


def telegraph_traces(f, tauA, dt, n_steps, n_channels, rng):
    """Markov two-state occupations, shape (n_steps, n_channels)."""
    p_up = (dt / tauA) * f          # 0 -> 1
    p_dn = (dt / tauA) * (1.0 - f)  # 1 -> 0
    assert max(p_up, p_dn) < 0.12, "dt too coarse for the rates"
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
    """Periodogram single-sided PSD of x(t)."""
    x = x - np.mean(x)
    n = len(x)
    X = np.fft.rfft(x)
    S = 2.0 * dt * np.abs(X)**2 / n
    freqs = np.fft.rfftfreq(n, dt)
    return freqs, S
