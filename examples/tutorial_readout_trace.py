"""Tutorial: from a noisy readout trace to the occupation lifetime, with
error bars, a model check, a second estimate by another method, and
the temperature resolution it implies.

A dispersive readout of an Andreev level records a noisy two-level
signal. This script makes such a record synthetically (fixed seed,
illustrative values: occupation f = 0.3, correlation time 1 ms, 50 us
samples, Gaussian readout noise 0.35 between levels 0 and 1) and then
works only from the record:

1. `fit_hmm` and `uncertainties`: f and tauA with error bars;
2. `markov_dwell_test`: are the stays consistent with one Markov rate?
3. a second estimate of tauA, from the noise spectrum of the same
   record (8 averaged periodograms, `fit_telegraph_psd` with n_avg = 8),
   and the difference of the two in units of the two error bars added
   in quadrature. Both estimates use the same record, so their errors
   are correlated and this is a loose consistency check, not a
   statistical test;
4. the occupation-noise temperature resolution of the Ti/Al/Au recipe
   at 0.3 Tc for this tauA. It scales exactly as sqrt(tauA), so its
   relative error is half that of tauA.

Run it with ``python examples/tutorial_readout_trace.py`` (about 10 s).
It printed, with absnoise 0.13.0 (Python 3.13, NumPy 2.5.3,
SciPy 1.18.1):

    decoder:  f = 0.301 +- 0.011, tauA = 0.998 +- 0.031 ms (EM iterations: 6)
    model check: p = 0.92 (dispersion 0.90 empty, 0.93 filled; 637 and 638 stays)
    spectrum: tauA = 0.974 +- 0.033 ms (chi2 = 1993 for 2045 dof)
    difference decoder - spectrum: 0.5 error bars
    Ti/Al/Au at 0.225 K: dT_A in 1 s = 1279.01 +- 20.03 uK for tauA = 0.998 ms
"""
import numpy as np

from absnoise import (RECIPES, SensorBudget, fit_hmm, fit_telegraph_psd,
                      markov_dwell_test, psd_single_sided, telegraph_traces)

# ---- the "measurement" -------------------------------------------------------
rng = np.random.default_rng(21)
dt, n_samples = 5e-5, 2 ** 16                       # 50 us samples, 3.3 s
n = telegraph_traces(0.3, 1e-3, dt, n_samples, 1, rng)[:, 0]
y = n + rng.normal(0.0, 0.35, n_samples)            # what the readout records

# ---- 1. decoder with error bars ----------------------------------------------
model, logliks = fit_hmm(y)
f_hat, tau_hat = model.rates(dt)
err = model.uncertainties(y, dt)["sigma"]
print(f"decoder:  f = {f_hat:.3f} +- {err['f']:.3f}, tauA = {tau_hat * 1e3:.3f} "
      f"+- {err['tauA'] * 1e3:.3f} ms (EM iterations: {len(logliks)})")

# ---- 2. is it one Markov rate? -----------------------------------------------
chk = markov_dwell_test(y, model, n_boot=200, seed=0)
print(f"model check: p = {chk['p_value']:.2f} (dispersion "
      f"{chk['dispersion'][0]:.2f} empty, {chk['dispersion'][1]:.2f} filled; "
      f"{chk['n_dwells'][0]} and {chk['n_dwells'][1]} stays)")

# ---- 3. independent estimate from the noise spectrum ---------------------------
segs = y.reshape(8, -1)                             # 8 independent segments
fr = np.fft.rfftfreq(segs.shape[1], dt)
S = np.mean([psd_single_sided(s, dt)[1] for s in segs], axis=0)
keep = slice(1, 2049)                               # every bin up to 1.25 kHz
psd = fit_telegraph_psd(fr[keep], S[keep], n_avg=8)
print(f"spectrum: tauA = {psd.tau_s * 1e3:.3f} +- {psd.sigma['tau_s'] * 1e3:.3f} ms "
      f"(chi2 = {psd.chi2:.0f} for {psd.chi2_dof} dof)")
z = (tau_hat - psd.tau_s) / np.hypot(err["tauA"], psd.sigma["tau_s"])
print(f"difference decoder - spectrum: {z:.1f} error bars")

# ---- 4. what this tauA means for a thermometer ---------------------------------
b = SensorBudget(RECIPES[1])
T = 0.3 * b.recipe.Tc
dT, _ = b.dT_andreev(T, tau_hat, t=1.0)
s_dT = dT * 0.5 * err["tauA"] / tau_hat             # dT ~ sqrt(tauA), exactly
print(f"Ti/Al/Au at {T:.3f} K: dT_A in 1 s = {dT * 1e6:.2f} +- {s_dT * 1e6:.2f} uK "
      f"for tauA = {tau_hat * 1e3:.3f} ms")
