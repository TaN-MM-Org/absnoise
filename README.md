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

## Status

v0.2.0 (alpha). Implemented and tested:

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

Not yet implemented, stated plainly because they matter physically:
continuum contributions to the occupation channel are neglected (bound
levels dominate for L < xi; the largest L/xi in the recipe set is
0.43; `continuum_share` quantifies the supercurrent-channel analog),
and phonon-bath heating by the substrate is treated only through the
steady-state electron temperature, not dynamically.

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

## License

Apache-2.0
