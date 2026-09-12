# Changelog

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
