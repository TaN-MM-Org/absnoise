# absnoise

[![PyPI](https://img.shields.io/pypi/v/absnoise)](https://pypi.org/project/absnoise/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22048608-blue)](https://doi.org/10.5281/zenodo.22048608) [![tests](https://github.com/TaN-MM-Org/absnoise/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/absnoise/actions)

How well can a superconducting junction measure temperature? In a
proximity Josephson junction, the discrete Andreev states that carry
the supercurrent flicker between occupied and empty even in perfect
equilibrium, and that flicker -- not the readout electronics -- is
what ultimately limits the sensor. `absnoise` computes the whole
chain: the level structure, the size and speed of the occupation
flicker, and the resulting temperature, frequency-noise and
single-photon energy resolutions of a real device. It also works
backward from measured data: critical-current curves, readout time
traces, and noise spectra.

## Install

```
pip install absnoise
```

For development: clone the repository and `pip install -e .[test]`.

## Predict a device budget

```python
import numpy as np
from absnoise import RECIPES, SensorBudget

budget = SensorBudget(RECIPES[1])          # Ti/Al/Au recipe, Jung et al.
T = 0.3 * budget.recipe.Tc                 # operating temperature (K)
tauA = 1e-6                                # occupation correlation time (s)

achieved, sums = budget.dT_andreev(T, tauA, t=1.0)
Sy, Snu = budget.freq_noise_spectrum(T, tauA, np.array([0.0, 1e3, 1e6]))
sigE = budget.energy_resolution(T, tauA)
```

Units are SI throughout; PSDs are single-sided with the convention
var(t-average) = S(0)/(2t), validated by Monte Carlo in the tests.

## Analyze your own measurements

The three standard data products of these experiments each have a
dedicated inverse tool, and the files enter through documented
contracts (`load_ic_csv`, `load_trace_csv`, new in v0.7) that refuse
malformed input and flag likely unit mix-ups (millikelvin entered as
kelvin, microamps entered as amps) instead of silently rescaling:

- **Critical current vs temperature** -> `fit_ic_curve`: fits the
  package's own junction model to a measured Ic(T) curve. The shape
  of the suppression identifies the transparency; with measurement
  sigmas you get an exact covariance and a chi-square check, and a
  curve that never leaves its low-temperature plateau triggers a
  warning that the parameters are weakly determined.
  `IcFit.to_recipe` packages the result so every budget in the
  package applies to *your* junction.
- **Readout time trace** -> `fit_hmm`: a hidden-Markov decoder
  returns the posterior occupation at every sample, the most probable
  state path, and maximum-likelihood rates -- from the raw trace
  alone.
- **Noise spectrum** -> `fit_telegraph_psd` (new in v0.7): when the
  dynamics are too fast to resolve in time, the spectrum still
  carries them. Fits the Lorentzian occupation-noise model
  S0/(1 + (2 pi f tau)^2) + floor and returns the plateau, the
  correlation time and the background with uncertainties -- and
  REFUSES a frequency band that never sees the knee, because such a
  band cannot determine tau, instead of returning one of a continuum
  of answers.

`allan_variance` and `avar_exponential` answer the practical
stability question -- how long is it worth averaging -- with the
closed form this package's own noise obeys, checked against its
defining integrals and Monte Carlo.

## Plan the measurements those fits need

`fit_ic_curve` and `fit_telegraph_psd` work on data that already
exist; the `lab` tools answer the planning questions that come first:

```python
from absnoise import (plan_ic_measurement, design_ic_temperatures,
                      psd_band_for_tau)

# Would this cooldown determine (Ic0, Tc, tau), and how well?
plan = plan_ic_measurement(T_K, sigma_Ic_A=20e-9, Ic0=1.1e-6,
                           Tc=1.5, tau=0.8)
print(plan["sigma"], plan["covers_tc_knee"])

# Which reachable temperatures are worth the fridge time?
pick = design_ic_temperatures(candidates_K, 6, 20e-9, 1.1e-6, 1.5, 0.8)

# Which frequency band can determine the correlation time at all?
f_lo, f_hi, f_knee = psd_band_for_tau(tau_expected_s=3e-4)
```

The planned error bars are the same (J^T W J)^-1 matrix the fit
reports, computed through the package's own `ic_model` before any
data exist; the fit's low-temperature leverage warning (a run that
never approaches 0.3 Tc barely feels Tc) reappears in the plan as
`covers_tc_knee`; and the PSD band planner applies, in advance,
exactly the knee-visibility rules `fit_telegraph_psd` enforces after
the fact -- a band it emits is one the fit will accept.

The PSD planner also answers the averaging question in closed form:
`plan_psd_measurement` predicts the (S0, tau, floor) error bars of a
planned averaging depth -- the same log-space matrix
`fit_telegraph_psd` reports, before any spectrum exists -- and
`averages_for_tau` inverts the exact 1/sqrt(n_avg) scaling into the
number of periodogram averages a target tau error bar costs. Both
refuse, in advance and with the same explanations, exactly the
bands the fit refuses after the fact.

## What is inside the physics engine

- The exact finite-length Andreev spectrum from its closed-form
  secular equation, and the closed-form short-junction ensemble:
  current-phase relation, critical current, Josephson inductance,
  responsivities, noise sums, heat capacity.
- The BCS gap Delta(T) solved from the gap equation itself, because
  the widely used interpolation formula misrepresents its
  low-temperature slope.
- The Cauchy-Schwarz temperature-resolution bound
  var(T) >= 2 kB T^2 tauA / (C_A t), with the level-resolved sums
  that show when it saturates.
- A four-state master equation for the pair processes, telegraph
  Monte Carlo, and the continuum (above-gap) contribution to the
  occupation heat capacity, cross-validated through an independent
  code path.
- Detector budgets: phonon thermal-fluctuation noise, resonator
  frequency-noise spectra, matched-filter energy resolution, and
  (new in v0.8) the frequency-resolved noise-equivalent power
  NEP(f) per channel in closed form -- the phonon channel exactly
  flat at Mather's 4 kB T^2 G (Appl. Opt. 1982), the occupation
  channel's Lorentzian cancelling exactly against its own lag; the
  matched-level design condition, the nonlinear single-photon click
  response, and a two-temperature (electron + local phonon) click
  model whose phonon parameters must be measured, not invented.

## Cited constants

Six junction recipes ship with full provenance (Table I of W. Jung et
al., Phys. Rev. Applied 26, 014078 (2026), arXiv:2503.06850);
graphene electron-phonon cooling follows G.-H. Lee et al., Nature
586, 42 (2020); physical constants are CODATA 2018. The test suite
locks every recipe number to its source; a replacement must arrive
with a new source. No phonon-stack parameters are shipped at all --
they are measurements of a real device, and calls without them raise.

## How it is checked

83 tests (Python 3.9-3.14, run in CI on every push), every physics
claim anchored to a closed form, an exact identity, or two
independent code paths -- never a stored number. Highlights: the
short-junction limit to 1e-12 and Kulik levels to 1e-10; the BCS
midpoint and both asymptotes; exact saturation (and non-violation) of
the resolution bound; the master equation's exact singles limit and
sum rules; Monte-Carlo validation of every spectrum and of the Allan
closed form; energy conservation of the click integrator to machine
precision; EM log-likelihood monotonicity in the decoder; exact
noise-free recovery in both fitters, with Monte-Carlo scatter
matching the reported sigmas; the PSD fit's variance tied to the
generating traces by Parseval; and exact file-contract round trips.

Deliberate scope, designed out with reasons: the continuum occupation
*noise* would need a correlation-time model with material parameters
this package refuses to invent (its thermodynamic weight IS
computed); and the phonon subsystem is one lumped temperature,
because a nonthermal phonon model has no cited parameters at these
scales.

## Methodological basis

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Andreev occupation
> noise sets the sensitivity limit of proximity Josephson thermal
> detectors"; code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/andreev-occupation-noise

This package is the general-purpose engine; the paper repository
reproduces the specific study.

## Support and governance

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

## License

Apache-2.0. Every release is archived on Zenodo under the concept DOI
[10.5281/zenodo.22048608](https://doi.org/10.5281/zenodo.22048608),
which always resolves to the latest version.
