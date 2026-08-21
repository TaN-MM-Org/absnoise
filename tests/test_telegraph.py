"""Monte Carlo telegraph noise against the analytic Lorentzian and the
variance convention var(t-average) = S(0)/(2t)."""
import numpy as np

from absnoise import psd_single_sided, telegraph_traces


def test_telegraph_psd_and_variance():
    rng = np.random.default_rng(7)
    f, tauA = 0.3, 1.0
    dt, n_steps, M = 0.05, 400_000, 24
    tr = telegraph_traces(f, tauA, dt, n_steps, M, rng)
    x = np.sum(1.0 - 2.0 * tr, axis=1)
    freqs, S = psd_single_sided(x, dt)
    # var(1-2n) = 4 f(1-f); telegraph S(0) = 4 var tau
    S0_an = M * 4.0 * 4.0 * f * (1 - f) * tauA
    m = (freqs > 0.005) & (freqs < 0.03)
    assert abs(np.mean(S[m]) - S0_an) / S0_an < 0.10
    # knee: S at f_k = 1/(2 pi tauA) is S0/2
    fk = 1.0 / (2 * np.pi * tauA)
    mk = (freqs > 0.8 * fk) & (freqs < 1.25 * fk)
    assert abs(np.mean(S[mk]) / S0_an - 0.5) < 0.12
    # variance of window averages
    t_int = 50.0
    k = int(t_int / dt)
    nwin = n_steps // k
    means = x[:nwin * k].reshape(nwin, k).mean(axis=1)
    assert abs(np.var(means) - S0_an / (2 * t_int)) / \
        (S0_an / (2 * t_int)) < 0.25
