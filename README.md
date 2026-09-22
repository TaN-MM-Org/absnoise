# absnoise

[![PyPI](https://img.shields.io/pypi/v/absnoise)](https://pypi.org/project/absnoise/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22048608-blue)](https://doi.org/10.5281/zenodo.22048608) [![tests](https://github.com/TaN-MM-Org/absnoise/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/absnoise/actions)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

`absnoise` is a Python package about one question: **how well can a
superconducting junction measure temperature, or detect a single
photon?** The junctions in question are proximity Josephson junctions:
two superconductors joined by a short piece of normal (not
superconducting) conductor, in the shipped examples graphene. Such a
junction carries its supercurrent (a current that flows with zero
resistance) through a few discrete energy levels, the Andreev levels. Each level is
randomly filled and emptied, even in perfect thermal equilibrium. That
flicker -- not the readout electronics -- sets the final limit on how
well the junction can sense temperature.

The package follows the whole chain, from the levels to a detector:

- Where are the Andreev levels, and how do they move with phase and
  temperature?
- How large and how fast is the occupation flicker, and what
  temperature resolution does it allow?
- For a real device read out by a microwave resonator: what are the
  frequency-noise spectrum, the noise-equivalent power and the energy
  resolution for a single photon?

It also works backward from measured data: a critical-current curve, a
readout time trace, or a noise spectrum. And it helps plan those
measurements before the fridge time is spent.

The main results are checked by automated tests against exact formulas
or a second, independent calculation (see
[How the results are checked](#how-the-results-are-checked)). When a
question cannot be answered from the data or parameters given, the
package stops with an error that says why, instead of returning a
number that looks fine but is not.

## Contents

- [A short guide to the words used here](#a-short-guide-to-the-words-used-here)
- [Install](#install)
- [Units and conventions](#units-and-conventions)
- [Examples](#examples) (each with the output it prints)
- [What is in the package](#what-is-in-the-package)
- [When it refuses, and why](#when-it-refuses-and-why)
- [How the results are checked](#how-the-results-are-checked)
- [Corrections in earlier versions](#corrections-in-earlier-versions)
- [Limits](#limits)
- [Where it comes from](#where-it-comes-from)
- [Citing, support and license](#citing-support-and-license)

## A short guide to the words used here

**The junction**

- **Superconducting gap** (`Delta`) -- the energy a superconductor
  needs to break a pair of electrons. It shrinks as the temperature
  rises and closes at the **critical temperature** `Tc`. Here the gap
  is the one the superconductor induces in the junction (the
  "proximity gap"), taken as `Delta = 1.764 kB Tc` at zero temperature.
- **Phase** (`phi`) -- the phase difference across the junction, in
  radians. The supercurrent depends on it (the **current-phase
  relation**); the largest supercurrent is the **critical current**
  `Ic`.
- **Andreev levels** -- discrete energy levels inside the gap that
  carry the supercurrent. Their energies depend on the phase.
- **Transparency** (`tau`, between 0 and 1) -- how easily electrons
  pass the contacts. `tau = 1` is a perfectly clean contact.
- **Channel** -- one of the separate paths (quantum modes) that
  electrons can take across the junction; a wide junction has many.
- **Short junction** -- a junction much shorter than the
  superconducting **coherence length** `xi` (roughly, how far the
  electron pairing of the superconductors reaches into the normal
  conductor). Then each channel has one level, `E = Delta sqrt(1 - tau sin^2(phi/2))`. For a longer junction
  the package solves the exact levels.
- **Recipe** -- one junction design: materials, `Tc`, size, gate
  voltage, transparency, measured critical current and resistance.

**The noise**

- **Occupation** -- whether a level holds an extra electron-like
  excitation (a quasiparticle). It flips between filled and empty at
  random: a **telegraph** signal.
- **Correlation time** (`tauA`) -- how long an occupation typically
  lasts before it changes. The package does not predict `tauA`; you
  supply it, or measure it with the fitting tools.
- **Power spectral density (PSD)** -- how the noise power is spread over
  frequency. A telegraph signal has a **Lorentzian** spectrum: flat up
  to a **knee** at `1/(2 pi tau)`, then falling. **White noise** has
  the same density at all frequencies. A **periodogram** is the PSD
  estimated from one recorded trace; averaging many periodograms
  reduces their scatter.
- **Monte Carlo** -- a simulation driven by random numbers, used here
  to generate example noise traces and check the formulas against
  them.
- **Heat capacity** -- energy needed to warm something by 1 K. `C_e`
  is that of the junction's electrons; `C_A` is the part that comes
  from the Andreev occupations.
- **Thermal conductance** `G` and **thermal time** `tau_th = C_e / G`
  -- how fast the electrons lose heat to the crystal lattice (the
  **phonons**). Random heat exchange with the phonons causes
  **phonon thermal-fluctuation noise** (TFN), the classic noise limit
  of a thermal detector.
- **Cauchy-Schwarz bound** -- the package's lower limit on the
  temperature error from occupation noise,
  `var(T) >= 2 kB T^2 tauA / (C_A t)` for averaging time `t`. A
  uniform short junction reaches it exactly.

**The detector**

- **Readout** -- the junction ends a microwave resonator. A change in
  temperature changes the junction's **Josephson inductance** `L_J`,
  which moves the resonance frequency. `y` is the fractional frequency
  shift. Two readouts are named by `which`: `"L"` (the inductance at
  phase near 0; the default of the device budget) and `"I"` (the
  supercurrent at the phase where it is largest).
- **Noise-equivalent power (NEP)** -- the input power that would give a
  signal as large as the noise in a 1 Hz bandwidth (half a second of
  averaging, with the convention below), in W/sqrt(Hz). Smaller is
  better.
- **Energy resolution** -- the uncertainty (one standard deviation)
  with which a deposited energy, for example one photon, can be
  measured, using the best ("matched") filter. Smaller is better.
- **Allan variance** -- a measure of how much a reading changes from
  one averaging window to the next. It shows how long averaging still
  helps.
- **Hidden Markov model (HMM)** -- a statistical model of a hidden
  two-state signal seen through noise. Used here to recover the
  occupation from a noisy readout trace.

## Install

```
pip install absnoise
```

It needs Python 3.9 or newer, NumPy 1.24 or newer and SciPy 1.10 or
newer. The optional extra `absnoise[plot]` installs matplotlib; no part
of the package imports it. For development: clone the repository and
run `pip install -e .[test]`, then `pytest`.

## Units and conventions

- SI units throughout: kelvin, joules, seconds, amperes, hertz, metres.
  Phases are in radians.
- Spectra are **single-sided** (positive frequencies only). With this
  convention, the variance of a `t`-second average of white noise with
  spectral density `S` is `S / (2 t)`, and a Lorentzian with variance
  `var` and correlation time `tau` has plateau `S(0) = 4 var tau`.
- The shipped constants (`KB`, `HBAR`, ...) are CODATA 2018 values.
- Graphene thermal parameters follow Lee et al. (see
  [Where it comes from](#where-it-comes-from)): by default the
  electron-phonon cooling power is `P = Sigma A (Te^3 - Tp^3)` with
  `Sigma = 2.0 W m^-2 K^-3`.

## Examples

Each example below runs as written, and the output shown is what it
printed with absnoise 0.10.1 (Python 3.11, NumPy 2.4.6, SciPy 1.17.1).
Values not taken from the shipped recipes -- correlation times,
geometries, noise levels -- are illustrative, not measurements.

### 1. A detector budget for a shipped junction

```python
import numpy as np
from absnoise import RECIPES, SensorBudget, H_PLANCK

budget = SensorBudget(RECIPES[1])        # the Ti/Al/Au junction of Jung et al.
T = 0.3 * budget.recipe.Tc               # operating temperature (K)
tauA = 1e-6                              # occupation correlation time (s), illustrative

dT_A, sums = budget.dT_andreev(T, tauA, t=1.0)    # 1 s of averaging
dT_ph = budget.dT_phonon(T, t=1.0)
print(f"T = {T:.3f} K, thermal time tau_th = {budget.tau_th(T) * 1e9:.2f} ns")
print(f"temperature resolution in 1 s: Andreev {dT_A * 1e6:.2f} uK, "
      f"phonon {dT_ph * 1e6:.2f} uK")
print(f"tauA/C_A = {tauA / sums['C_A']:.3e} s K/J, "
      f"tau_th/C_e = {budget.tau_th(T) / budget.Ce(T):.3e} s K/J")

sigE = budget.energy_resolution(T, tauA)
print(f"energy resolution: {sigE:.3e} J "
      f"(the energy of a {sigE / H_PLANCK / 1e9:.0f} GHz photon)")

f = np.array([0.0, 1e7, 1e8])
nep = budget.nep_spectrum(T, tauA, f)
for fi, a, p in zip(f, nep["andreev"], nep["phonon"]):
    print(f"NEP at {fi:.0e} Hz: Andreev {a:.2e}, phonon {p:.2e} W/sqrt(Hz)")
```

```
T = 0.225 K, thermal time tau_th = 0.78 ns
temperature resolution in 1 s: Andreev 40.48 uK, phonon 3.58 uK
tauA/C_A = 1.172e+15 s K/J, tau_th/C_e = 9.145e+12 s K/J
energy resolution: 1.810e-22 J (the energy of a 273 GHz photon)
NEP at 0e+00 Hz: Andreev 6.26e-18, phonon 5.53e-19 W/sqrt(Hz)
NEP at 1e+07 Hz: Andreev 6.27e-18, phonon 5.53e-19 W/sqrt(Hz)
NEP at 1e+08 Hz: Andreev 6.97e-18, phonon 5.53e-19 W/sqrt(Hz)
```

At this operating point the occupation noise, not the phonon noise,
sets the temperature resolution. That matches the criterion stated in
the code: occupation noise dominates when `tauA/C_A` is larger than
`tau_th/C_e`. In the NEP, the phonon part is flat at every frequency
(`sqrt(4 kB T^2 G)`), and the occupation part rises as
`sqrt(1 + (2 pi f tau_th)^2)`.

### 2. The temperature bound, finite length, and pair processes

```python
import numpy as np
from absnoise import RECIPES, ShortJunction, FiniteLJunction, sigma_spectrum

r = RECIPES[3]                           # Ti/Al(thick): L = 0.3 um, xi = 3.6 um
T = 0.3 * r.Tc

sj = ShortJunction(r)                    # every channel has the same level
sj.calibrate()
achieved, bound, _ = sj.temperature_bound(T, tauA=1e-6, t_int=1.0)
print(f"short-junction model: achieved / bound = {achieved / bound:.6f}")

fl = FiniteLJunction(r)                  # exact levels at the real length
print(f"finite-length model at phi = 1: achieved / bound = "
      f"{fl.saturation_ratio(1.0, T):.4f}")

# pair processes can only shorten the correlation time
for Gp in (0.0, 1.0, 10.0):
    _, S0, var, tau_eff = sigma_spectrum(0.25, 1.0, Gp, np.array([0.0]))
    print(f"Gp = {Gp:4.1f} Gs: tau_eff = {tau_eff:.4f} / Gs")
```

```
short-junction model: achieved / bound = 1.000000
finite-length model at phi = 1: achieved / bound = 1.0392
Gp =  0.0 Gs: tau_eff = 1.0000 / Gs
Gp =  1.0 Gs: tau_eff = 0.7500 / Gs
Gp = 10.0 Gs: tau_eff = 0.3750 / Gs
```

A short junction whose channels all have the same transparency sits
exactly on the bound. At the real length of this junction the levels
move differently with phase, and the resolution is a little worse than
the bound (never better). `sigma_spectrum` shows that pair processes
(two quasiparticles entering or leaving together, at rate `Gp`) can
only shorten the effective correlation time, so the single-quasiparticle
case is the worst one. Here `Gs` is the single-quasiparticle rate.

### 3. Characterize your junction from a critical-current curve

```python
import numpy as np
from absnoise import ic_model, fit_ic_curve, validate_ic_data, SensorBudget

# Synthetic "measurement": 10 temperatures, 1 % noise, fixed seed.
# The true values (illustrative) are Ic0 = 2 uA, Tc = 0.75 K, tau = 0.78.
T = np.linspace(0.05, 0.65, 10)
rng = np.random.default_rng(1)
sigma = 0.02e-6
Ic = ic_model(T, 2.0e-6, 0.75, 0.78) + rng.normal(0.0, sigma, T.size)

print(validate_ic_data(T, Ic))                   # nothing suspicious
print(validate_ic_data(T * 1e3, Ic)["flags"][:2])  # millikelvin typed as kelvin?
fit = fit_ic_curve(T, Ic, sigma_Ic=sigma)
print(f"Ic0 = {fit.Ic0 * 1e6:.3f} +- {fit.sigma['Ic0'] * 1e6:.3f} uA")
print(f"Tc  = {fit.Tc:.4f} +- {fit.sigma['Tc']:.4f} K")
print(f"tau = {fit.tau:.3f} +- {fit.sigma['tau']:.3f}")
print(f"chi2 = {fit.chi2:.2f} for {fit.chi2_dof} degrees of freedom")

# Package the fit (plus your own measured geometry) for the budgets.
rec = fit.to_recipe(name="my junction", label="mine", xi=5e-6,
                    L=0.2e-6, W=2e-6, Vbg=30.0, Rn=50.0)
b = SensorBudget(rec)
print(f"Josephson inductance at 0.2 K: {b.LJ(0.2) * 1e9:.3f} nH")
```

```
{'n_points': 10, 'flags': []}
[(0, 'T_large', 50.0), (1, 'T_large', 116.66666666666667)]
Ic0 = 2.007 +- 0.012 uA
Tc  = 0.7530 +- 0.0057 K
tau = 0.794 +- 0.032
chi2 = 3.79 for 7 degrees of freedom
Josephson inductance at 0.2 K: 0.226 nH
```

`fit_ic_curve` fits the package's own short-junction model for three
numbers: `Ic0` (the critical current at zero temperature), `Tc` and
`tau`. The error bars are the standard linearized estimate at the
solution (from the slopes of the model at the best fit; accurate when
the errors are small). `chi2` is given when you pass measurement
errors; a value near the number of degrees of freedom (the number of
points minus the three fitted numbers) means model and errors are
consistent. `validate_ic_data` never rescales anything: it returns flags
for values that look like a unit mix-up (above 30 K, or above 1 mA, by
default). `to_recipe` needs your measured geometry, because an Ic(T)
curve does not determine it.

### 4. Plan the measurements before the cooldown

```python
import numpy as np
from absnoise import (plan_ic_measurement, design_ic_temperatures,
                      psd_band_for_tau, plan_psd_measurement,
                      averages_for_tau)

# Expected junction (illustrative): Ic0 = 2 uA, Tc = 0.75 K, tau = 0.78.
Ic0, Tc, tau = 2.0e-6, 0.75, 0.78
sig = 0.02e-6                                  # expected error per point (A)

for top in (0.2, 0.65):                        # highest temperature reached (K)
    T = np.linspace(0.05, top, 8)
    p = plan_ic_measurement(T, sig, Ic0, Tc, tau)
    print(f"run up to {top} K: sigma(Tc) = {p['sigma']['Tc'] * 1e3:.1f} mK, "
          f"covers_tc_knee = {p['covers_tc_knee']}")

cand = np.linspace(0.03, 0.70, 15)             # temperatures the fridge can hold
pick = design_ic_temperatures(cand, 5, sig, Ic0, Tc, tau)
print("measure at:", np.round(np.sort(cand[pick["indices"]]), 3), "K")

f_lo, f_hi, f_knee = psd_band_for_tau(3e-4)    # expect tau near 0.3 ms
print(f"PSD band {f_lo:.1f} to {f_hi:.0f} Hz (knee {f_knee:.0f} Hz)")
f = np.geomspace(f_lo, f_hi, 48)
plan = plan_psd_measurement(f, 64, S0=1e-3, tau_s=3e-4, floor=1e-5)
print(f"64 averages: sigma(tau) = {plan['sigma']['tau_s'] * 1e6:.2f} us")
n, _ = averages_for_tau(3e-6, f, S0=1e-3, tau_s=3e-4, floor=1e-5)
print(f"averages needed for sigma(tau) <= 3 us: {n}")
```

```
run up to 0.2 K: sigma(Tc) = 1031.1 mK, covers_tc_knee = False
run up to 0.65 K: sigma(Tc) = 6.1 mK, covers_tc_knee = True
measure at: [0.03  0.078 0.365 0.413 0.7  ] K
PSD band 132.6 to 2122 Hz (knee 531 Hz)
64 averages: sigma(tau) = 15.50 us
averages needed for sigma(tau) <= 3 us: 1708
```

The planned error bars are the same matrix the fit reports, computed
from the package's own model before any data exist. A run that stays
far below `Tc` barely feels `Tc`; the plan flags this as
`covers_tc_knee = False` (the fit warns about the same case).
`design_ic_temperatures` picks the most informative temperatures one at
a time (a greedy method, not a proof of the best set).
`psd_band_for_tau` returns a band from `f_knee/4` to `4 f_knee` that
satisfies the rules `fit_telegraph_psd` enforces. Error bars fall as
`1/sqrt(n_avg)`, and `averages_for_tau` inverts that.

### 5. Occupation noise from a spectrum, and the Allan variance

```python
import numpy as np
from absnoise import (telegraph_traces, psd_single_sided, fit_telegraph_psd,
                      allan_variance, avar_exponential)

# Simulated occupation noise: 64 channels, f = 0.3, tau = 0.2 ms, 5 us steps.
rng = np.random.default_rng(0)
f_occ, tau, dt = 0.3, 2e-4, 5e-6
tr = telegraph_traces(f_occ, tau, dt, 40000, 64, rng)

# Channel-averaged periodogram, then averaged into 40 log-spaced bins.
S = np.mean([psd_single_sided(tr[:, c].astype(float), dt)[1]
             for c in range(64)], axis=0)[1:]
fr = np.fft.rfftfreq(40000, dt)[1:]
edges = np.geomspace(fr[0], fr[-1], 41)
fb, Sb = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    m = (fr >= lo) & (fr < hi)
    if m.sum() >= 3:
        fb.append(fr[m].mean())
        Sb.append(S[m].mean())

fit = fit_telegraph_psd(np.array(fb), np.array(Sb))
print(f"fitted tau = {fit.tau_s * 1e6:.0f} +- {fit.sigma['tau_s'] * 1e6:.0f} us "
      f"(true {tau * 1e6:.0f} us)")
print(f"variance under the fitted Lorentzian {fit.variance:.4f}, "
      f"trace variance {tr.astype(float).var():.4f}")

# Allan variance of the 64-channel sum against the closed form.
x = tr.astype(float).sum(axis=1)
taus, avar = allan_variance(x, dt, m_list=[20, 80, 320])
closed = avar_exponential(64 * f_occ * (1 - f_occ), tau, taus)
for T, a, c in zip(taus, avar, closed):
    print(f"T = {T * 1e3:.1f} ms: measured {a:.2e}, closed form {c:.2e}")
```

```
fitted tau = 198 +- 2 us (true 200 us)
variance under the fitted Lorentzian 0.2112, trace variance 0.2112
T = 0.1 ms: measured 3.14e+00, closed form 3.13e+00
T = 0.4 ms: measured 5.26e+00, closed form 5.12e+00
T = 1.6 ms: measured 2.34e+00, closed form 2.73e+00
```

`fit_telegraph_psd` fits `S0 / (1 + (2 pi f tau)^2) + floor` in
logarithmic units. The variance under the fitted Lorentzian,
`S0/(4 tau)`, is compared here with the variance of the simulated
traces. The Allan variance of the simulated signal follows the closed
form `avar_exponential`; the longest window has the fewest independent
samples and scatters most (the test suite checks agreement within 12 %
on a longer record).

### 6. Decode a noisy readout trace

```python
import os, tempfile
import numpy as np
from absnoise import telegraph_traces, save_trace_csv, load_trace_csv, fit_hmm

# A simulated readout: occupation f = 0.3, tau = 1 ms, sampled every 50 us,
# with Gaussian readout noise 0.3 (levels 0 and 1). Illustrative values.
rng = np.random.default_rng(0)
dt = 5e-5
n = telegraph_traces(0.3, 1e-3, dt, 20000, 1, rng)[:, 0]
y = n + rng.normal(0.0, 0.3, n.size)

path = os.path.join(tempfile.mkdtemp(), "readout.csv")
save_trace_csv(path, dt, y)              # header t_s,y
dt2, y2 = load_trace_csv(path)           # refuses a non-uniform time grid

model, logliks = fit_hmm(y2)
f_hat, tau_hat = model.rates(dt2)
print(f"f = {f_hat:.3f}, tau = {tau_hat * 1e3:.3f} ms, "
      f"levels {model.means[0]:.3f} and {model.means[1]:.3f}, noise {model.sigma:.3f}")
print(f"EM iterations: {len(logliks)}")
path_hat = model.viterbi(y2)
print(f"samples decoded correctly: {(path_hat == n).mean() * 100:.2f} %")
```

```
f = 0.297, tau = 0.977 ms, levels -0.001 and 1.003, noise 0.302
EM iterations: 8
samples decoded correctly: 99.69 %
```

`fit_hmm` finds the occupation fraction, the correlation time, the two
readout levels and the noise from the trace alone, by
expectation-maximization (a standard method that improves the
estimates step by step until they stop changing). `viterbi` gives the most probable filled/empty
sequence and `posterior` the probability of "filled" at every sample.
The trace goes through the documented file format (`t_s,y`) on the way.

### 7. A single photon: matched design, click, and a refusal

```python
from absnoise import (matched_Tc, matched_recipe, SensorBudget, click_template,
                      click_monte_carlo, two_temperature_click, H_PLANCK)

T0 = 0.05                                 # bath temperature (K)
print(f"matched junction Tc for T0 = 50 mK: {matched_Tc(T0):.4f} K")
b = SensorBudget(matched_recipe(T0, W=5.3e-6, L=0.5e-6))   # illustrative geometry
E = H_PLANCK * 26e9                       # one 26 GHz photon

tpl = click_template(b, E, tauA=1e-7, T0=T0)
print(f"peak electron temperature {tpl['Te_peak'] * 1e3:.1f} mK, "
      f"occupation dip {(1 - tpl['ms'].min() / tpl['m0']) * 100:.1f} %")

mc = click_monte_carlo(b, E, tauA=1e-7, T0=T0, n_trials=200, seed=3)
print(f"Monte Carlo: SNR {mc['snr']:.1f}, efficiency {mc['efficiency']:.2f}, "
      f"dark fraction {mc['dark_fraction']:.2f}")

try:
    two_temperature_click(b, E, 1e-7, T0)        # no phonon parameters given
except ValueError as err:
    print("refused:", err)
```

```
matched junction Tc for T0 = 50 mK: 0.0776 K
peak electron temperature 121.8 mK, occupation dip 1.5 %
Monte Carlo: SNR 1.9, efficiency 0.82, dark fraction 0.21
refused: c_ph, kappa_pb and delta_pb carry no vetted defaults: the phonon heat capacity and boundary escape law of a real stack are measurements of that stack. Supply all three from your device's thermal characterization (or a cited equivalent stack).
```

`matched_Tc` chooses the junction `Tc` so that the gap at the operating
temperature is `2.3994 kB T0` (the `MATCHED_RATIO` in the code).
`click_template` follows the electron temperature and the occupations
after the photon is absorbed, without noise; `click_monte_carlo` adds
occupation, phonon and readout noise and detects clicks with a matched
filter. Its numbers are statistics-limited; the tests only check
ordering relations. `two_temperature_click` also lets the local phonons
heat up, but it has no default phonon parameters and refuses to run
without them.

## What is in the package

**Junctions and levels**

- `Recipe`, `RECIPES` -- a junction design; the six measured recipes of
  Jung et al. (index 0 to 5: Ta/Ti/Au, Ti/Al/Au, Ti/Al(thin),
  Ti/Al(thick), Ti/Nb/NbN, MoRe).
- `abs_energies(phi, tau, c, Delta)` -- the exact Andreev levels of one
  channel of finite length, found by solving a closed-form equation.
- `gap_bcs(T, Tc, Delta0)` -- the gap `Delta(T)`, from the gap equation
  of BCS theory (the standard Bardeen-Cooper-Schrieffer theory of
  superconductivity) solved numerically, not the common interpolation
  formula.
- `ShortJunction` -- the closed-form short-junction model: levels,
  current-phase relation, `Ic`, `dIdphi0`, occupation sums and
  `temperature_bound`.
- `FiniteLJunction` -- the same sums from the exact finite-length
  levels, and `saturation_ratio` (1 means on the bound).
- `JunctionModel`, `junction_properties` -- the full mode-summed model
  of a recipe, including states above the gap (the continuum),
  calibrated to the measured `Ic` at 20 mK.
- `continuum_share`, `occupation_heat_capacities` -- how much of the
  current, and of the occupation heat capacity, comes from the
  continuum.

**Graphene thermal model** (`materials`)

- `carrier_density`, `fermi_energy`, `dos_ef`, `n_modes` -- electron
  density from the gate voltage, and quantities derived from it.
- `heat_capacity`, `ep_power`, `gth` -- electron heat capacity,
  electron-phonon cooling power and thermal conductance.
- `steady_temperature` -- the electron temperature at which cooling
  balances a steady absorbed power (readout self-heating).

**Occupation noise**

- `channel_generator`, `sigma_spectrum` -- the four-state master
  equation of one channel (the equations for how the probabilities of
  its four occupation states change in time), with single and pair
  processes, and its exact spectrum.
- `tau_activated`, `noneq_penalty` -- an energy-dependent (thermally
  activated) correlation time, and the resolution penalty of an excess,
  non-equilibrium occupation.
- `telegraph_traces`, `psd_single_sided` -- Monte Carlo telegraph
  signals and their periodogram.
- `allan_variance`, `avar_exponential`, `avar_white` -- the Allan
  variance of data, and its closed forms for exponentially correlated
  and white noise.

**Device budget and single photons**

- `SensorBudget` -- one recipe as a detector: `dT_andreev`,
  `dT_phonon`, `freq_noise_spectrum`, `energy_resolution`,
  `energy_resolution_analytic_A`, `nep_spectrum`, plus `Ce`, `Gep`,
  `tau_th`, `LJ` and `participation` (the share of the resonator's
  inductance that the junction supplies).
- `matched_Tc`, `matched_recipe` -- the matched-level design.
- `click_template`, `click_monte_carlo` -- the response to one photon,
  noiseless and with noise.
- `two_temperature_click`, `total_energy` -- the same with a heated
  local phonon bath, and its conserved energy.

**From measured data**

- `fit_ic_curve`, `IcFit`, `ic_model` -- the Ic(T) fit, its result
  (with `to_recipe`), and the model curve.
- `fit_telegraph_psd`, `TelegraphPSDFit`, `telegraph_psd_model` -- the
  noise-spectrum fit, its result, and the model spectrum.
- `fit_hmm`, `TelegraphHMM` -- the readout-trace decoder
  (`posterior`, `viterbi`, `forward_backward`, `rates`, `from_rates`).
- `load_ic_csv`, `save_ic_csv` (header `T_K,Ic_A` or
  `T_K,Ic_A,sigma_Ic_A`), `load_trace_csv`, `save_trace_csv` (header
  `t_s,y`), `validate_ic_data` -- the file formats and unit checks.

**Planning**

- `plan_ic_measurement`, `design_ic_temperatures` -- planned Ic(T)
  error bars, and which temperatures to measure.
- `psd_band_for_tau`, `plan_psd_measurement`, `averages_for_tau` -- the
  frequency band, the planned spectrum-fit error bars, and the number
  of averages for a target error on `tau`.

**Constants** -- `HBAR`, `KB`, `E_CHARGE`, `H_PLANCK`, `PHI0`, `EPS0`
(CODATA 2018) and `V_F` (graphene Fermi velocity, 1.0e6 m/s).

Each function's docstring (`help(absnoise.fit_ic_curve)`, for example)
gives its inputs, units and conventions.

## When it refuses, and why

`absnoise` raises an error instead of guessing when:

- `two_temperature_click` is called without all three phonon parameters
  (`c_ph`, `kappa_pb`, `delta_pb`): they are measurements of a real
  device, and none are shipped. It also refuses a non-positive `c_ph`
  or `delta_pb`, a negative `kappa_pb` or a negative deposited energy.
- `fit_telegraph_psd` gets a spectrum that changes by less than a
  factor of 4 across the band, or whose fitted knee lies outside
  `[2 f_min, f_max/2]`: such a band cannot determine `tau`. It also
  needs at least 8 points, positive frequencies (drop the zero-frequency
  point), a positive PSD and, if given, `n_avg >= 1`.
- `plan_psd_measurement` and `averages_for_tau` get a planned band the
  fit would refuse (the same rules, applied in advance), `n_avg < 1`,
  or a non-positive target; `psd_band_for_tau` gets a margin below 4.
- `fit_ic_curve` gets fewer than 4 points, fewer than 5 points without
  measurement errors (the noise level must then be estimated from the
  residuals), non-finite values, non-positive temperatures or currents,
  or non-positive errors. It raises `RuntimeError` if the fit does not
  converge or the data cannot separate the three parameters. If the
  data stay below 0.3 `Tc`, it warns (it does not refuse) that `Tc`
  and `tau` are weakly determined.
- `plan_ic_measurement` and `design_ic_temperatures` get `tau` outside
  (0, 1], a non-positive `Ic0`, `Tc`, temperature or error, a number of
  picks outside 3 to the number of candidates, or candidates that
  cannot determine the parameters even all together.
- a data file has the wrong header or number of fields, fewer than 2
  rows, temperatures that do not strictly increase, or (for a trace) a
  time grid that is not uniform; the file is refused, not repaired.
  `validate_ic_data` refuses non-finite values, non-positive
  temperatures and negative currents.
- `allan_variance` gets fewer than 4 samples, non-finite samples, a
  non-positive `dt` or a window outside `1 <= m <= len(x)//2`; the
  closed forms get a negative variance or PSD, or non-positive times.
- `TelegraphHMM` gets flip probabilities outside (0, 1), a non-positive
  noise or not exactly two levels.
- `steady_temperature` gets a negative power; `nep_spectrum` gets a
  negative or non-finite frequency; `click_template` and
  `two_temperature_click` get a scenario other than `"C"` or `"T"`.
- `telegraph_traces` uses a time step too coarse for the rates (a flip
  probability per step of 0.12 or more). This check is a Python
  `assert`, so it is skipped under `python -O`.

## How the results are checked

88 automated tests run on every push and pull request, on Python 3.9,
3.10, 3.11, 3.12, 3.13 and 3.14, and once more on Python 3.9 with the
oldest NumPy (1.24.0) and SciPy (1.10.0) the package allows. The
numerical checks compare the package with an exact formula, a second
calculation done a different way, or a simulation. A few checks
compare with stored numbers on purpose, for example the recipe table
(locked to its source) and one tabulated value of the BCS gap. Others
check that refusals fire. The main checks:

**Levels and gap**

- Short-junction limit: the exact solver gives
  `Delta sqrt(1 - tau sin^2(phi/2))` to 1e-12 (relative to `Delta`),
  for 6 transparencies and 4 phases.
- Fully transparent, finite length: the levels satisfy the known
  exact level condition for that case (the Kulik condition) to
  1e-10 rad.
- One clean channel at 1 mK: `Ic = 2 e Delta / hbar` within 0.5 %.
- BCS gap: equals `Delta0` at 0.01 `Tc` to 1e-10, is below 0.05
  `Delta0` at 0.99999 `Tc`, equals the tabulated 0.956887 at `Tc/2`
  within 2e-4, and falls monotonically.
- Short junction: the continuum carries no phase dependence (below
  1e-14).

**Occupation noise and the bound**

- A uniform short junction reaches the Cauchy-Schwarz bound to 1e-9;
  a finite-length junction (Ti/Al(thick), two phases) never goes below
  it (to 1e-9).
- The occupation responsivity (how much the signal changes per
  kelvin) matches numerical differentiation to 1e-5, and `dIdphi0` matches its closed form to 1e-12.
- Master equation: without pair processes the variance is `2f(1-f)`
  and the correlation time `1/Gs`, both to 1e-12; the variance does not
  depend on the pair rate (1e-12); the correlation time never rises
  with the pair rate; probability is conserved (1e-14); the spectrum
  integrates to the variance within 5e-4.
- Telegraph Monte Carlo: the plateau within 10 %, half the plateau at
  the knee within 0.12, and the variance of 50-second averages equals
  `S(0)/(2t)` within 25 %.
- `occupation_heat_capacities`: the bound part equals the level sum
  `C_A` of `FiniteLJunction` within 1e-5 (two code paths); the
  continuum part is exactly 0 at zero length; for Ti/Al/Au at 0.3 `Tc`
  it is positive and below 10 % of the total.

**Allan variance**

- Linear drift: the estimator gives `(c T)^2 / 2` to 1e-12.
- The closed form matches numerical integration of the defining
  integrals to 1e-6, and both limits to 1e-3.
- Telegraph Monte Carlo matches the closed form within 12 %, white
  noise matches `S0/(2T)` within 5 %.

**Device budget**

- The frequency-noise spectrum is exactly half the plateau at the knee
  and 1/101 of it at ten times the knee (1e-12).
- The phonon resolution equals its closed form (1e-18 K); the
  occupation resolution respects the bound; the full energy resolution
  is never better than 0.99 times the occupation-only closed form.
- NEP: the phonon part equals `sqrt(4 kB T^2 G)` at every frequency to
  1e-14; the occupation part has exactly the shape
  `sqrt(1 + (2 pi f tau_th)^2)` (1e-12); the zero-frequency ratio of
  the two equals the documented criterion (1e-12) and doubles when
  `tauA` doubles (1e-12); the readout part rises faster.
- The energy resolution built from the NEP equals the one from
  `energy_resolution` to 1e-10, with and without a readout floor, for
  both readouts `"L"` and `"I"` (two code paths).

**Recipes and thermal model**

- The label, `Tc`, `tau`, `Ic20` and `Rn` of all six recipes equal
  Table I of Jung et al. exactly.
- Gate voltage 20 V gives the quoted density 1.7e16 m^-2 within 5 %;
  the heat capacity at 1 um^2, 1.7e16 m^-2 and 0.1 K lies between 4 and
  8 `kB`.
- `steady_temperature` round trip: `ep_power(steady_temperature(P))`
  returns `P` within `max(1e-12 P, 1e-30 W)`.

**Single photons**

- Matched design: `Delta(T0) / (kB T0) = 2.3994` to 1e-9.
- The starting electron temperature satisfies
  `gamma (T_pk^2 - T0^2) / 2 = E_gamma` to 1e-12 (this checks the
  starting point, not the time integration).
- The noiseless template dips and returns within 0.5 % of its start;
  the Monte Carlo has signal-to-noise above 1 and efficiency above the
  dark fraction.
- Two-temperature model: with nothing deposited, both temperatures
  stay exactly at the start value (not even rounding drift); with no
  escape, energy is conserved within 1e-5 and both temperatures end at
  the independently computed common temperature within 1e-5; with a
  very large phonon heat capacity it reproduces `click_template`
  within 1e-3; with escape on, both temperatures shed more than 99 %
  of their excursion.

**Fits and decoder**

- Ic(T) model: the closed-form zero-temperature maximum equals a dense
  grid maximum within 1e-8 (transparencies 0.05, 0.3, 0.78, 0.99 and
  1); `ic_model`
  equals `ShortJunction.Ic` within 1e-12 for all six recipes.
- Ic(T) fit: noise-free data return `Ic0` and `Tc` within 1e-6 and
  `tau` within 1e-5; over 60 noisy fits the scatter is between 0.5
  and 1.5 times the reported error bars; a fitted recipe fed back into
  `ShortJunction` reproduces the curve within 1e-4.
- Spectrum fit: noise-free data return `S0` and `tau` within 1e-6 and
  the floor within 1e-4; on telegraph Monte Carlo, `tau` within 15 %,
  the variance within 15 % of the trace variance, `S0` within 20 %.
- Decoder: above 99.9 % of samples correct at noise 0.15; at noise 0.5
  the stated confidence matches the real accuracy within 0.01; the fit
  finds `f` and `tau` within 4 statistical errors and the levels and
  noise within 0.02; the log-likelihood never decreases (within 1e-6
  relative).
- Files: Ic files round-trip exactly; trace files round-trip the
  samples exactly and `dt` within 1e-18 s.

**Planning**

- Planned Ic(T) error bars equal the fit's within 5 %; 250 simulated
  cooldowns scatter in `Tc` within 20 % of the plan; a run below 0.3
  `Tc` is flagged and its `Tc` error is more than 10 times larger; the
  greedy design is never worse than 20 random choices.
- Planned spectrum-fit error bars equal the fit's within 0.1 %; the
  `1/sqrt(n_avg)` scaling holds to 1e-9; `averages_for_tau` returns the
  smallest sufficient number; 300 simulated spectra scatter in `tau`
  within 20 % of the plan; a band from `psd_band_for_tau` is accepted
  by the fit and returns `tau` within 1e-4.

## Corrections in earlier versions

**0.10.1 (this release) fixes four problems in 0.10.0.**

- **NumPy 1.x.** The package allows NumPy 1.24 and newer, but four
  integrals called `numpy.trapezoid`, which exists only from NumPy 2.0.
  On NumPy 1.x, `energy_resolution`, the current and free-energy
  methods of `JunctionModel` (`Ic`, `calibrate`, ...),
  `junction_properties`, `continuum_share` and
  `occupation_heat_capacities` stopped with an `AttributeError`. They
  now work on both, and NumPy 1.24.0 with SciPy 1.10.0 is tested.
- **Current readout in `energy_resolution`.** With `which="I"` it used
  the phase 1e-4 rad instead of the phase of maximum current that
  `freq_noise_spectrum`, `nep_spectrum` and
  `energy_resolution_analytic_A` use. Its result disagreed with the
  NEP-based value (it was 66 % larger without a readout floor in the
  example setup of the tests). The default `which="L"` was not affected.
- **Full transparency.** `ic_model` returned NaN at `tau = 1`, which
  the fit's bounds and the planner allow; `plan_ic_measurement` then
  reported such a junction as not identifiable. It now uses the
  correct limit, `Delta0/2`.
- **`n_avg` below 1.** `fit_telegraph_psd` accepted `n_avg = 0` or a
  negative value and returned infinite or NaN error bars. It now
  refuses, as the planner already did.

Also: `TelegraphHMM` and `fit_hmm` were importable from `absnoise` but
missing from `__all__`; they are now listed. Docstrings that described
tests that do not exist (a Monte Carlo check of the pair-process
model, a convergence-order check of the two-temperature integrator, a
continuity check of the continuum sign, a comparison of the telegraph
simulation with the occupation sums) now say what is and is not
tested, and the spectrum-fit notes now give the floor's real test
tolerance (1e-4, not 1e-6).

**What the 0.10.0 README overstated.** It said the checks run on
Python 3.9-3.14 (Python 3.10 was not in the CI matrix), gave "every
spectrum" a Monte-Carlo check (the master-equation spectrum has none),
called the Ic(T) covariance "exact" (it is the linearized estimate),
said the tests lock "every recipe number" (they lock `Tc`, `tau`,
`Ic20`, `Rn` and the label, not the geometry and gate voltage), said
no check uses a stored number (the recipe table and one gap value do),
described a machine-precision "energy conservation of the click
integrator" (the check covers the starting temperature only), and
said the file loaders flag unit mix-ups (that is `validate_ic_data`).

No earlier release records a correction. The full history is in
[CHANGELOG.md](CHANGELOG.md).

## Limits

- The correlation time `tauA` is an input, not a prediction.
- The device budget (`SensorBudget`) uses the short-junction model and
  converts the signal to resonator frequency through the Josephson
  inductance at zero phase. `which="I"` is accepted and evaluated at
  the phase of maximum current, but the frequency conversion is still
  that inductance.
- The finite-length occupation sums (`FiniteLJunction`) count bound
  levels only; `occupation_heat_capacities` gives the size of the
  continuum part at one phase (the tests check its sign and size only
  for Ti/Al/Au).
- The continuum occupation *noise* is not modelled: it would need a
  correlation-time model with material parameters this package does
  not invent (its thermodynamic weight is computed).
- The phonons are one lumped temperature; a non-thermal phonon model has
  no cited parameters at these scales. The two-temperature model needs
  your measured phonon parameters.
- `click_monte_carlo` results are statistics-limited; the tests check
  only ordering relations.
- `fit_hmm` models Gaussian readout noise. Its default start splits the
  trace at the median; a trace with more than half of its samples
  exactly at the maximum value (for example a noise-free trace that is
  filled most of the time) makes that start fail with an error about
  flip probabilities. Pass `init=` in that case.
- The greedy temperature design is a good heuristic, not a proof of
  the best subset.

## Where it comes from

Methodological basis:

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Andreev occupation
> noise sets the sensitivity limit of proximity Josephson thermal
> detectors"; code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/andreev-occupation-noise

This package is the general-purpose engine; the paper repository
reproduces the specific study.

Six junction recipes ship with full provenance (Table I of W. Jung et
al., Phys. Rev. Applied 26, 014078 (2026), arXiv:2503.06850);
graphene electron-phonon cooling follows G.-H. Lee et al., Nature
586, 42 (2020); physical constants are CODATA 2018. The phonon NEP is
compared with J. C. Mather, Appl. Opt. 21, 1125 (1982). The Allan
variance follows D. W. Allan, Proc. IEEE 54, 221 (1966) and W. J.
Riley, Handbook of Frequency Stability Analysis, NIST Special
Publication 1065 (2008). The temperature design follows F. Pukelsheim,
Optimal Design of Experiments, SIAM (2006). A replacement recipe number
must arrive with a new source. No phonon-stack parameters are shipped
at all -- they are measurements of a real device, and calls without
them raise.

## Citing, support and license

If `absnoise` helps your work, please cite it with the concept DOI
[10.5281/zenodo.22048608](https://doi.org/10.5281/zenodo.22048608),
which always resolves to the latest version; every release is archived
on Zenodo. [CITATION.cff](CITATION.cff) has the details.

Written and maintained by Tanvir Mahmud Mahim (Department of
Electrical and Electronic Engineering, BRAC University), who reviews
every change and takes the final decision on scope and releases.
Design questions are discussed in the open in issues and pull
requests, and the standing rule of
[CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly as it
binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the
[issue tracker](https://github.com/TaN-MM-Org/absnoise/issues). Usage
questions are welcome alongside bug reports; a docstring that left a
unit or a sign convention unclear is treated as a documentation bug,
not user error. While the version is below 1.0 the API may still move
between minor versions; such changes are called out in the release
notes.

Licensed under Apache-2.0.
