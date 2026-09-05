# absnoise

[![PyPI](https://img.shields.io/pypi/v/absnoise)](https://pypi.org/project/absnoise/) [![DOI](https://img.shields.io/badge/DOI-10.5281%2Fzenodo.22048608-blue)](https://doi.org/10.5281/zenodo.22048608) [![tests](https://github.com/TaN-MM-Org/absnoise/actions/workflows/ci.yml/badge.svg)](https://github.com/TaN-MM-Org/absnoise/actions)

**Andreev-bound-state occupation noise** in proximity Josephson
junctions, computed from the level structure to the **detector budget**:
the exact finite-length Andreev spectrum, the intrinsic
occupation-noise limit of Andreev thermometry, and the resulting
temperature, frequency-noise and calorimetric energy resolutions of an
inductively read junction. The package exists because the occupation of
Andreev levels fluctuates even in equilibrium, and that fluctuation, not
the readout, is what ultimately limits a proximity Josephson thermal
detector.

## Decoding occupation from readout traces (new in v0.4)

The `decode` module solves the inverse problem the rest of the package
predicts: given a noisy sampled readout of the two-state occupation, a
hidden-Markov decoder returns the exact posterior occupation probability
at every sample (forward-backward), the most probable state path
(Viterbi), and maximum-likelihood estimates of f, tauA and the readout
levels from the trace alone (Baum-Welch EM, whose log-likelihood is
provably non-decreasing, a property the tests check). Validated against
the package's own telegraph Monte Carlo: state recovery above 99.9
percent at high SNR, posterior confidence matching realized accuracy to
better than a percent, and rate estimates landing within the
transition-count statistical error of the ground truth.

```python
from absnoise.decode import TelegraphHMM, fit_hmm
model, logliks = fit_hmm(y_trace)          # EM from the raw trace
f_hat, tau_hat = model.rates(dt)
posterior = model.posterior(y_trace)       # P(occupied) per sample
```

## Status

v0.5.0 (alpha). Implemented and tested (56 tests, Python 3.9-3.13):

- exact finite-length ABS solver from the closed-form secular equation
  cos(2 arccos(E/Delta) - eta(E)) = 1 - tau + tau cos(phi), with the
  continuum (E > Delta) free energy from the scattering phase
- weak-coupling BCS gap Delta(T) solved from the gap equation itself
  (tabulated once per process; no data file), because the widely used
  tanh interpolation misrepresents dDelta/dT at low temperature
- closed-form short-junction ensemble: current-phase relation, critical
  current, Josephson inductance, occupation-channel responsivity and
  noise sums, Andreev heat capacity, and the Cauchy-Schwarz
  temperature-resolution bound var(T) >= 2 kB T^2 tauA / (C_A t)
- level-resolved finite-length sums and the bound-saturation deficit
- four-state (pair-process) master equation with exact spectra,
  activated exchange times, and the nonequilibrium occupation penalty
- telegraph-noise Monte Carlo and single-sided PSD estimation
- device budgets: phonon thermal-fluctuation noise, resonator
  fractional-frequency noise spectra, matched-filter calorimetric
  energy resolution
- six cited graphene junction contact recipes (Jung et al., Table I)
  and graphene electronic thermodynamics (heat capacity,
  electron-phonon cooling, Lee et al.)
- **matched-level design and nonlinear click dynamics (new in v0.2)**:
  the matched condition Delta*(T0) = 2.3994 kB T0 solved from the gap
  equation (`matched_Tc`, `matched_recipe`), the full nonlinear
  post-deposit response with a stiffness-safe exponential integrator
  (`click_template`), a whitened matched-filter single-photon click
  Monte Carlo with all three noise channels (`click_monte_carlo`), and
  the exact self-heating steady state (`steady_temperature`)

Verified against closed forms in the test suite rather than asserted:
the short-junction limit E = Delta sqrt(1 - tau sin^2(phi/2)) to 1e-12;
Kulik levels at tau = 1 to 1e-10; the ballistic anchor
Ic = 2 e Delta / hbar per orbital mode to 5e-3 (phase-grid limited);
exact vanishing of the continuum phase dependence at L = 0; the BCS
midpoint u(0.5) = 0.956887 and both asymptotes; exact saturation of the
Cauchy-Schwarz bound by uniform-transparency short junctions to 1e-9,
and its non-violation by dispersing finite-length levels; the analytic
occupation responsivity to 1e-5; the pair-process master equation's
exact singles limit, equilibrium-variance invariance, monotone
shortening of the correlation time, probability conservation, and
spectrum-to-variance sum rule; the telegraph Monte Carlo Lorentzian
plateau, knee, and the variance convention var = S(0)/(2t)
(statistics-limited tolerances); and the exact Lorentzian knee of the
predicted resonator frequency-noise spectrum.

For v0.2 the test suite additionally asserts: the matched-level
condition to 1e-9 from the solved gap equation; energy conservation of
the click integrator to machine precision through the exact
peak-temperature identity; dip-and-recovery of the occupation template
in both exchange scenarios; photon-versus-dark ordering of the click
Monte Carlo; and the exact closed-form round trip of the self-heating
steady state.

Both previously stated limitations are now closed (v0.5):

- **Continuum occupation channel quantified**
  (`occupation_heat_capacities`): the continuum (E > Delta)
  contribution to the occupation-channel heat capacity, computed as
  -T d2F/dT2 at frozen gap from the validated scattering-phase free
  energy -- the Krein spectral-shift piece the bound-only sums
  neglect. Cross-validated, not asserted: the bound part of the same
  derivative reproduces the level-sum C_A of `andreev_sums` through a
  completely independent code path (< 1e-5 relative), and the
  continuum part vanishes identically at L = 0. For the recipe set at
  0.3 Tc the continuum share is at the percent level, which is now a
  number in a test rather than a hope in a docstring.
- **Dynamic phonon bath** (`two_temperature_click`): the coupled
  electron/local-phonon temperature dynamics with a finite phonon
  heat capacity and a boundary escape law, integrated with
  stiffness-aware RK4. No phonon parameters are shipped -- c_ph,
  kappa_pb and delta_pb are measurements of a real stack, and calls
  without them raise. Anchors: T0 is an exact fixed point (preserved
  bitwise), the isolated flow conserves gamma Te^2/2 + c_ph Tp^4/4
  and equilibrates to the independently computed quartic-root common
  temperature (< 1e-5), and the infinite-bath limit reproduces
  `click_template` exactly where it should.

Deliberate scope, designed out with reasons: the continuum occupation
*noise* (as opposed to its thermodynamic weight) would need the
kinetics of continuum quasiparticles -- a correlation-time model with
material parameters this package refuses to invent; and the phonon
subsystem is one lumped temperature, not a spectral phonon
distribution, because a nonthermal phonon model has no cited
parameters at these device scales either.

## Install and use

```
pip install absnoise
```

For development, clone the repository and `pip install -e .[test]`.

```python
import numpy as np
from absnoise import RECIPES, SensorBudget

budget = SensorBudget(RECIPES[1])          # Ti/Al/Au recipe, Jung et al.
T = 0.3 * budget.recipe.Tc                 # operating temperature (K)
tauA = 1e-6                                # occupation correlation time (s)

achieved, sums = budget.dT_andreev(T, tauA, t=1.0)
print(f"temperature resolution {achieved*1e6:.2f} uK in 1 s")

Sy, Snu = budget.freq_noise_spectrum(T, tauA, np.array([0.0, 1e3, 1e6]))
sigE = budget.energy_resolution(T, tauA)
print(f"matched-filter energy resolution {sigE:.3e} J")
```

Units are SI throughout; PSDs are single-sided with the variance
convention var(t-average) = S(0)/(2t), validated by Monte Carlo in the
test suite.

## Allan variance (new in v0.3)

Occupation noise is exponentially correlated, so the practical question
for a thermometer readout is stability versus integration time:
averaging helps until the correlation time is passed, then improves
only as the white floor S(0)/(2T). `allan_variance` computes the
overlapping two-sample variance of a sampled series (Allan, Proc. IEEE
54, 221 (1966); Riley, NIST SP 1065, 2008), and `avar_exponential`
gives the closed form this package's own noise obeys,

    AVAR(T) = var (tau/T^2) (2T - 3 tau + 4 tau e^(-T/tau)
              - tau e^(-2T/tau)),

with `avar_white` the S0/(2T) floor. The closed form is not taken on
faith: the test suite integrates the defining double integrals
numerically and checks both limits, a machine-precision drift identity,
and seeded telegraph Monte Carlo against it.

## Cited constants

The six junction recipes ship with full provenance: Table I of W. Jung,
E. G. Arnault, B. Huang, J. Park, S. Jang, K. Watanabe, T. Taniguchi,
D. Englund, K. C. Fong and G.-H. Lee, "Engineering Andreev Bound States
for Thermal Sensing in Proximity Josephson Junctions", Phys. Rev.
Applied 26, 014078 (2026) (arXiv:2503.06850). Graphene electron-phonon
cooling follows the measured coupling of G.-H. Lee et al., Nature 586,
42 (2020) (resonant-supercollision regime, delta = 3,
Sigma ~ 2 W m^-2 K^-3). Physical constants are CODATA 2018. The
test suite locks every recipe number to the source; a change to any of
them must arrive with a new source.

## Methodological basis

> T. M. Mahim, A. S. M. Mohsin and M. M. Rahman, "Andreev occupation
> noise sets the sensitivity limit of proximity Josephson thermal
> detectors"; code for the paper:
> https://github.com/Tanvir-Mahmud-Mahim/andreev-occupation-noise

This package is the general-purpose engine (v0.2 includes the
matched-level design and the nonlinear click Monte Carlo); the paper
repository reproduces the specific study: the device grids, the
figures, the approximation-resolution analysis, and the archived trial
data.

## Citing the tool

The repository carries a `CITATION.cff` file with citation metadata. If
this software contributes to a publication, please cite the versioned
DOI you used.

## Support and governance

The package is written and maintained by Tanvir Mahmud Mahim
(Department of Electrical and Electronic Engineering, BRAC University),
who reviews every change and takes the final decision on scope and
releases. There is no separate governance body; design questions are
discussed in the open in issues and pull requests, and the standing
rule of [CONTRIBUTING.md](CONTRIBUTING.md) binds the maintainer exactly
as it binds contributors: a change that touches physics arrives with a
test, and a constant arrives with its source.

Support runs through the issue tracker at
https://github.com/TaN-MM-Org/absnoise/issues. Usage questions are
welcome there alongside bug reports; a docstring that left a unit or a
sign convention unclear is treated as a documentation bug, not as user
error. The maintainer aims to respond within a week.

While the version is below 1.0 the API may still move between minor
versions; such changes are called out in the release notes. Recipe
constants are governed by the same rule as code: the test suite locks
every number to its cited source, and a replacement number must arrive
with a new source.

## License

Apache-2.0
