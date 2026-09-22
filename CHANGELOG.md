# Changelog

## 0.10.1 (2026-09-22)

Bug fixes, a CI update and a rewritten README.

### Fixed

- NumPy 1.x support: four integrals called `numpy.trapezoid`, which
  exists only from NumPy 2.0, although `pyproject.toml` allows
  NumPy >= 1.24. On NumPy 1.x, `SensorBudget.energy_resolution`,
  the current and free-energy methods of `JunctionModel` (`Ic`,
  `calibrate`, ...), `junction_properties`, `continuum_share` and
  `occupation_heat_capacities` raised `AttributeError` (9 of 83 tests
  failed with NumPy 1.24.0 / SciPy 1.10.0). A small helper
  (`absnoise._compat.trapezoid`) now uses `numpy.trapezoid` or, on
  NumPy 1.x, `numpy.trapz`.
- `SensorBudget.energy_resolution(which="I")` evaluated the current
  readout at phase 1e-4 rad instead of the phase of maximum current
  used by `freq_noise_spectrum`, `nep_spectrum` and
  `energy_resolution_analytic_A`, so it disagreed with the NEP-based
  value. The default `which="L"` is unchanged. Its docstring also
  wrote the matched-filter integral with a factor 2; the code uses 4,
  which is correct for single-sided spectra.
- `ic_model` (and so `fit_ic_curve` and `plan_ic_measurement`)
  returned NaN at transparency `tau = 1`, which their documented range
  (0, 1] allows; the zero-temperature normalization now uses its limit
  `Delta0 / 2` there.
- `fit_telegraph_psd` accepted `n_avg < 1` and returned infinite or NaN
  error bars; it now raises `ValueError`, as `plan_psd_measurement`
  already did.
- `TelegraphHMM` and `fit_hmm` were exported but missing from
  `__all__`.
- Docstrings that described tests that do not exist were corrected:
  a Monte Carlo check of the pair-process master equation
  (`master`), an RK4 convergence-order check (`twotemp`), a continuity
  check of the continuum sign (`levels`), and a comparison of telegraph
  Monte Carlo with `andreev_sums` (`telegraph`). The `psdfit` module
  docstring said noise-free recovery holds to 1e-6 for all three
  parameters; the test asserts 1e-4 for the floor.

### Tests

- New `tests/test_v0101.py` (5 tests, each failing before its fix):
  integrals without `numpy.trapezoid`, the NEP matched-filter identity
  for `which="I"`, `ic_model` at `tau = 1`, the `n_avg` refusal, and
  `__all__` completeness. 88 tests in total (83 before).
- Tests that called `numpy.trapezoid` use the same helper.
- CI: Python 3.10 added to the matrix (now 3.9 to 3.14), and a new
  `oldest-dependencies` job runs the suite on Python 3.9 with
  NumPy 1.24.0 and SciPy 1.10.0 (also passed locally on Python 3.10).

### Changed

- README rewritten for readers outside the field: a guide to the
  words used, units and conventions, seven examples with their exact
  output, every public name, the refusals, and the main test checks
  with their real tolerances.
- CONTRIBUTING.md: the dependencies are NumPy and SciPy (it said
  NumPy only).

### Corrections to earlier notes

- 0.10.0 has no entry in this file; its README describes
  `plan_psd_measurement` and `averages_for_tau`, which the 0.9.0 entry
  does not list.
- 0.5.0 lists the CI matrix as Python 3.9, 3.11, 3.12, 3.13; Python
  3.10 was not tested until this release, although `requires-python`
  is `>=3.9`.
- 0.7.0 says noise-free PSD recovery "to 1e-6": the test asserts 1e-6
  for `S0` and `tau` and 1e-4 for the floor.
- 0.6.0 calls the known-noise covariance "exact" and the noise-free
  recovery "exact": the covariance is the linearized estimate, and
  the test asserts recovery within 1e-6 (`Ic0`, `Tc`) and 1e-5 (`tau`).
- 0.5.0 says the continuum share is "at the percent level for the
  recipe set": the test checks one recipe (Ti/Al/Au), requiring a
  positive share below 10 %. Its "exact peak-temperature deposit
  identity" checks the starting temperature formula, not the time
  integration.

## 0.9.0 (2026-09-17)

Lab adaptability: the measurements planned before the fridge time is
spent.

- `lab.plan_ic_measurement`: predicted (Ic0, Tc, tau) error bars for
  a planned Ic(T) run -- the same (J^T W J)^-1 matrix `fit_ic_curve`
  reports, through the package's own `ic_model`, with a
  scale-invariant identifiability verdict and the fit's low-
  temperature leverage warning surfaced in advance as
  `covers_tc_knee`.
- `lab.design_ic_temperatures`: greedy D-optimal choice of which
  reachable temperatures to measure (Pukelsheim, Optimal Design of
  Experiments, SIAM (2006)).
- `lab.psd_band_for_tau`: a frequency band satisfying, in advance,
  exactly the knee-visibility rules `fit_telegraph_psd` enforces
  after the fact.
- Anchors: planned sigmas match the fit's on the same design and 250
  seeded Monte-Carlo cooldowns match the planned Tc scatter; the
  below-0.3-Tc design flagged and its Tc error bar an order of
  magnitude worse; the greedy design never loses to a random subset;
  the band planner cross-validated against the fit in both
  directions (emitted bands accepted with the generating tau
  recovered, knee-blind bands refused).

## 0.8.0 (2026-09-13)

Physics upgrade: the noise-equivalent power spectrum -- the
bolometric figure of merit every detector paper quotes -- with each
channel referred to the input power through the full signal chain in
closed form.

### Added

- `SensorBudget.nep_spectrum`: frequency-resolved NEP of the Andreev
  occupation channel, the phonon thermal-fluctuation channel and the
  readout imprecision, in W/sqrt(Hz). Two exact cancellations do the
  work: the phonon channel is EXACTLY flat at Mather's classic
  4 kB T^2 G_ep (Appl. Opt. 21, 1125 (1982)) because the temperature-
  fluctuation rolloff cancels against the responsivity rolloff, and
  the occupation channel's Lorentzian cancels against the occupation
  lag, leaving only the thermal rolloff to undo.

### Anchors (asserted in `tests/test_nep.py`, not stated)

- The phonon NEP equals Mather's closed form to 1e-14 at every
  frequency.
- The matched-filter identity sigma_E = [4 int df / NEP^2]^(-1/2)
  built from the NEP closed forms reproduces the package's
  independent `energy_resolution` signal-chain integral to 1e-10,
  with and without a readout floor -- two code paths through the
  same physics.
- The DC crossover NEP_A(0) vs NEP_ph(0) is exactly the
  tauA/C_A vs tau_th/C_e criterion the budget docstring has always
  stated, and scales exactly linearly in tauA.
- The occupation channel's shape is exactly
  sqrt(1 + (2 pi f tau_th)^2), and the readout channel rises faster
  (it carries both rolloffs).

## 0.7.0 (2026-09-12)

From-the-instrument release: the third standard data product of these
experiments -- a measured noise spectrum -- gets its inverse tool,
and measurement files enter through documented contracts.

### Added

- `fit_telegraph_psd` / `TelegraphPSDFit` / `telegraph_psd_model`:
  extraction of the occupation-noise parameters (plateau S0,
  correlation time tau, white floor) from a measured single-sided
  PSD, fitted in log space (the variance-stabilizing choice for
  averaged periodograms, stated rather than hidden), with
  uncertainties and an optional chi-square check when the number of
  averages is given. Identifiability enforced, not hoped for: a band
  that never sees the knee -- flat spectra, or a fitted knee outside
  the measured band -- is refused with an explanation, because such
  a band cannot determine tau. Anchors: noise-free recovery to 1e-6;
  the fitted Lorentzian's exact integral S0/(4 tau) matching the
  generating telegraph Monte-Carlo trace variance (Parseval, two
  independent code paths); the fitted tau and plateau matching the
  generating values within statistics; both out-of-band refusals.
- `load_ic_csv` / `save_ic_csv` (header `T_K,Ic_A[,sigma_Ic_A]`) and
  `load_trace_csv` / `save_trace_csv` (header `t_s,y`, uniform grid
  enforced): documented plain-text contracts for the inputs of
  `fit_ic_curve`, `fit_hmm`, `psd_single_sided` and
  `allan_variance`, with exact round trips and refusals.
- `validate_ic_data`: structural problems raise; unit-plausibility
  findings (temperatures that look like millikelvin entered as
  kelvin, currents that look like microamps entered as amps) are
  returned as flags for the user to judge -- never silently
  rescaled. Thresholds are explicit keyword parameters.

### Changed

- README rewritten: organized by what the package does (predict a
  budget / analyze your measurements / the physics engine) rather
  than by release history, in plainer language, same facts.

## 0.6.0 (2026-09-10)

### Added

- `fit_ic_curve` / `IcFit` / `ic_model`: junction characterization
  from a measured Ic(T) curve -- bounded nonlinear least squares of
  the package's short-junction ensemble for (Ic0, Tc, tau), exact
  known-noise covariance and chi-square check when sigma_Ic is given
  (residual-variance covariance, requiring n >= 5, otherwise), a
  weak-constraint warning when the data never leave the
  low-temperature plateau, and `IcFit.to_recipe` to package the fit
  (plus the user's measured geometry) as a `Recipe` for the rest of
  the pipeline.
- Anchors: closed-form T = 0 maximizing phase
  sin^2(phi*/2) = (1 - sqrt(1 - tau))/tau against a dense grid
  maximum; the fit's forward model against the independent
  `ShortJunction.Ic` implementation to machine precision on every
  shipped recipe; exact noise-free recovery; Monte-Carlo scatter
  compatible with reported sigmas; refusals on malformed and
  unconstraining data; fitted-recipe round trip through the
  calibrated device pipeline.

## 0.5.0 (2026-09-05)

Closes both limitations stated in v0.4's "not yet implemented"
paragraph, and documents what remains out as deliberate scope with
reasons.

### Added

- `occupation_heat_capacities`: bound and continuum contributions to
  the occupation-channel heat capacity at fixed phase, computed as
  -T d2F/dT2 with the gap frozen. The bound part is cross-validated
  against the level-sum C_A of `andreev_sums` through an independent
  code path (< 1e-5 relative); the continuum part is the Krein
  spectral-shift contribution from the validated scattering-phase
  free energy, exactly zero at L = 0 (asserted) and at the percent
  level for the recipe set at 0.3 Tc -- the previously neglected
  channel, now quantified.
- `two_temperature_click` / `total_energy`: dynamic phonon bath. The
  coupled electron/local-phonon system with a finite phonon heat
  capacity C_p = c_ph T^3 and boundary escape law
  kappa_pb A (Tp^delta_pb - T0^delta_pb), integrated by RK4 with
  substeps bounded by both the local timescale (accuracy) and the
  linearized stiff rates (stability). No phonon parameter values are
  shipped: calls without c_ph, kappa_pb, delta_pb raise. Anchors
  asserted in the tests: exact bitwise fixed point at T0; isolated
  (kappa_pb = 0) conservation of gamma Te^2/2 + c_ph Tp^4/4 and
  equilibration to the independently computed quartic-root common
  temperature; infinite-bath reduction to `click_template`; the exact
  peak-temperature deposit identity; monotone relaxation back to the
  bath.

### Changed

- README: the "not yet implemented" paragraph replaced by the two
  closures above plus the deliberate-scope statement (no invented
  continuum kinetics, no invented nonthermal phonon model).
- CI matrix: Python 3.9, 3.11, 3.12, 3.13.

## 0.4.0 (2026-08-29)

- `decode`: telegraph hidden-Markov decoder (forward-backward
  posterior, Viterbi, Baum-Welch EM with non-decreasing likelihood).

## 0.3.0

- Allan variance: overlapping estimator, closed form for exponential
  noise.

## 0.2.0

- Matched-level design (`matched_Tc`, `matched_recipe`), nonlinear
  click dynamics (`click_template`, `click_monte_carlo`),
  `steady_temperature`.

## 0.1.0

- Initial release: exact finite-length ABS solver, BCS gap from the
  gap equation, short-junction ensemble, Cauchy-Schwarz temperature
  bound, master equation, telegraph Monte Carlo, sensor budgets;
  analytic testbench (28 tests).
