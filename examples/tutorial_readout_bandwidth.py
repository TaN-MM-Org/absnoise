"""Tutorial: how much readout bandwidth does a junction calorimeter need?

The energy resolution of `SensorBudget.energy_resolution` assumes a
matched filter that sees every frequency. A real readout passes only a
band (the resonator linewidth, the amplifier, the digitizer), and the
filter can use only what passes. This script measures the two inputs
that decide how much bandwidth is enough, from one noise record of the
resonator, and then answers the question with error bars:

1. a synthetic record of the fractional frequency shift y(t) of the
   readout resonator (fixed seed, illustrative true values): the
   occupation noise of the Ti/Al/Au junction at 0.3 Tc with a
   correlation time tauA = 1 us, plus a white readout floor 30 times
   below the occupation plateau;
2. its averaged periodogram (32 segments), fitted with
   `fit_telegraph_psd(..., n_avg=32)`: tauA, the floor, and the
   plateau S0, which is checked against the budget's own prediction;
3. the energy resolution for an unlimited band and for readout
   bandwidths of 1, 3 and 10 MHz (`energy_resolution(..., f_max=...)`),
   and the bandwidth that brings it within 10 % of the unlimited value
   (`bandwidth_for_resolution`). The error bars come from the fit's
   covariance of (tauA, floor) (`TelegraphPSDFit.cov`), carried through
   with sigma points (one standard deviation along each principal axis,
   exact for a linear function).

The occupation part of the record is the exactly sampled Gaussian
process with the correlation exp(-|t|/tauA) (the many-channel limit of
the telegraph occupations), so its spectrum is the Lorentzian of the
package (with the folded-back part above the Nyquist frequency, which is
negligible in the fitted band).

Run it with ``python examples/tutorial_readout_bandwidth.py`` (a few
seconds). It printed, with absnoise 0.14.0 (Python 3.13, NumPy 2.5.3,
SciPy 1.18.1):

    spectrum fit: tauA = 1.023 +- 0.021 us, floor = 5.58e-15 +- 0.02e-15 1/Hz, chi2 = 2820 for 2946 dof
    plateau S0 = 1.747e-13 +- 0.056e-13 1/Hz, budget prediction 1.682e-13 (1.2 error bars apart)
    energy resolution, unlimited band: 2.751e-21 +- 0.043e-21 J
    readout bandwidth  1 MHz: 3.735e-21 +- 0.046e-21 J (1.36 times the unlimited value)
    readout bandwidth  3 MHz: 3.035e-21 +- 0.044e-21 J (1.10 times the unlimited value)
    readout bandwidth 10 MHz: 2.826e-21 +- 0.043e-21 J (1.03 times the unlimited value)
    bandwidth for 1.1 times the unlimited value: 3.08 +- 0.03 MHz
"""
import numpy as np
from scipy.signal import lfilter

from absnoise import (RECIPES, SensorBudget, fit_telegraph_psd,
                      psd_single_sided)

budget = SensorBudget(RECIPES[1])            # Ti/Al/Au
T = 0.3 * budget.recipe.Tc                   # 0.225 K
tauA_true = 1e-6                             # s, illustrative
S0_true = budget.freq_noise_spectrum(T, tauA_true, np.array([0.0]))[0][0]
floor_true = S0_true / 30.0                  # white readout floor, 1/Hz

# ---- 1. the "measured" record: 32 segments of 2^14 samples, 20 ns apart ----
dt, n_seg, n_avg = 2e-8, 2 ** 14, 32
rng = np.random.default_rng(7)
rho = np.exp(-dt / tauA_true)
var = S0_true / (4.0 * tauA_true)            # S(0) = 4 var tau
drive = np.sqrt(var * (1.0 - rho ** 2)) * rng.standard_normal(n_seg * n_avg)
x0 = np.sqrt(var) * rng.standard_normal()
occ = lfilter([1.0], [1.0, -rho], drive, zi=[rho * x0])[0]
white = np.sqrt(floor_true / (2.0 * dt)) * rng.standard_normal(occ.size)
y = occ + white

# ---- 2. averaged periodogram and the spectrum fit ---------------------------
segs = y.reshape(n_avg, n_seg)
f = np.fft.rfftfreq(n_seg, dt)
S = np.mean([psd_single_sided(s, dt)[1] for s in segs], axis=0)
band = (f > 0.0) & (f <= 9e6)
fit = fit_telegraph_psd(f[band], S[band], n_avg=n_avg)
print(f"spectrum fit: tauA = {fit.tau_s * 1e6:.3f} +- {fit.sigma['tau_s'] * 1e6:.3f} us, "
      f"floor = {fit.floor / 1e-15:.2f}e-15 +- {fit.sigma['floor'] / 1e-15:.2f}e-15 1/Hz, "
      f"chi2 = {fit.chi2:.0f} for {fit.chi2_dof} dof")
print(f"plateau S0 = {fit.S0 / 1e-13:.3f}e-13 +- {fit.sigma['S0'] / 1e-13:.3f}e-13 1/Hz, "
      f"budget prediction {S0_true / 1e-13:.3f}e-13 "
      f"({abs(fit.S0 - S0_true) / fit.sigma['S0']:.1f} error bars apart)")

# ---- 3. energy resolution versus readout bandwidth --------------------------
p0 = np.array([fit.tau_s, fit.floor])
cov = fit.cov[1:, 1:]                        # (tau_s, floor) block
lam, vec = np.linalg.eigh(cov)


def with_error(func):
    """func(tauA, floor) at the fit, and its error by sigma points."""
    g = [0.5 * (func(*(p0 + np.sqrt(lk) * vec[:, k]))
                - func(*(p0 - np.sqrt(lk) * vec[:, k])))
         for k, lk in enumerate(lam)]
    return func(*p0), float(np.sqrt(np.sum(np.square(g))))


def sigma_E(f_max):
    return lambda tau, fl: budget.energy_resolution(T, tau, S_ro_y=fl,
                                                    f_max=f_max)


s_inf, e_inf = with_error(sigma_E(None))
print(f"energy resolution, unlimited band: {s_inf / 1e-21:.3f}e-21 "
      f"+- {e_inf / 1e-21:.3f}e-21 J")
for bw in (1e6, 3e6, 1e7):
    s, e = with_error(sigma_E(bw))
    print(f"readout bandwidth {bw / 1e6:2.0f} MHz: {s / 1e-21:.3f}e-21 "
          f"+- {e / 1e-21:.3f}e-21 J ({s / s_inf:.2f} times the unlimited value)")
bw, ebw = with_error(lambda tau, fl: budget.bandwidth_for_resolution(
    T, tau, factor=1.1, S_ro_y=fl))
print(f"bandwidth for 1.1 times the unlimited value: {bw / 1e6:.2f} "
      f"+- {ebw / 1e6:.2f} MHz")
