# Changelog

## 0.15.0 (2026-10-18)

The exact critical current is now the default calibration.

### Changed

- `JunctionModel.calibrate` and `junction_properties` use
  `method="matsubara"` by default: the scale is set so that the exact
  maximum of the Matsubara-sum current (`Ic_matsubara`) at 20 mK equals
  the measured `Ic20`. Up to 0.14.0 the default was `method="levels"`,
  the largest value of the level-based `current_phase` on its phase
  grid, which lies 1.7 % above the exact maximum for MoRe. A model
  calibrated by default therefore had an exact critical current at
  20 mK of 0.98344 `Ic20` for MoRe; now it is `Ic20` (1e-9). Scales of
  the shipped recipes, 0.14.0 -> 0.15.0: Ta(10)/Ti(60)/Au(5) 0.383553
  -> 0.382872 (-0.18 %), Ti(6)/Al(60)/Au(5) 0.517814 -> 0.517578
  (-0.05 %), Ti(6)/Al(70) 0.554064 -> 0.553802 (-0.05 %), Ti(6)/Al(200)
  0.658375 -> 0.658100 (-0.04 %), Ti(6)/Nb(5)/NbN(50) 0.133312 ->
  0.133331 (+0.01 %), MoRe(50) 0.492841 -> 0.501142 (+1.68 %). Every
  current-like quantity of a calibrated `JunctionModel` scales with it;
  README Example 11 changes in the last digit of `L_J` (0.2195 ->
  0.2196 nH, 0.2219 -> 0.2220 nH), no other example or tutorial
  changes. `method="levels"` reproduces 0.14.0. The MoRe calibration
  takes about 27 s instead of 4 s on the development machine.
  `ShortJunction.calibrate` was always exact and is unchanged. Tests:
  `tests/test_v015.py` (fails on 0.14.0); the 0.14.0 tests that
  relied on the old default now pass `method="levels"`.

## 0.14.0 (2026-10-18)

The matched-filter energy resolution for a readout of finite
bandwidth, and the bandwidth a target resolution needs; the exact
critical current of the full finite-length model and an exact
calibration option; the covariance of the noise-spectrum fit; one
correction (found by a test that fails on 0.13.0); a faster Markov
dwell test; and a third tutorial. No function was renamed or removed,
and README Examples 1 to 12 print the same output as with 0.13.0.

### Added

- `energy_resolution(..., f_max=None)` and `matched_filter_sigma(...,
  f_max=None)`: the matched filter restricted to the readout band
  0 <= f <= f_max, sigma_E^-2 = 4 int_0^f_max df / NEP^2, in closed
  form (partial fractions of 1 / (a + b w^2 + c w^4), with a series
  limit where the two roots nearly coincide). The unlimited integral
  is the optimal-filter result of S. H. Moseley, J. C. Mather and
  D. McCammon, J. Appl. Phys. 56, 1257 (1984) (Eq. (5) of their NASA
  TM 80092 preprint). `SensorBudget.bandwidth_for_resolution(T, tauA,
  factor=1.1, S_ro_y=0, which="L")`: the bandwidth at which the
  resolution is `factor` times the unlimited value. README Example 13.
- `JunctionModel.Ic_matsubara(T, ...)`: the critical current of the
  full model as the maximum of the Matsubara-sum current (grid of 31
  phases, refined by a bounded Brent search), without the energy and
  phase grids of `Ic`. At 20 mK (scale 1) the level-based `Ic` is
  4.2e-4 to 1.8e-3 below it for Ta/Ti/Au, Ti/Al/Au, Ti/Al(thin) and
  Ti/Al(thick), 1.5e-4 above it for Ti/Nb/NbN, and 1.7 % above it
  for MoRe, whose level-based current scatters by up to 3 % between
  neighboring grid phases near its maximum (6.04982 to 6.35094 uA
  over three grid points, against a smooth 6.23 to 6.21 uA from the
  Matsubara sum). `JunctionModel.calibrate(T0, method="matsubara")`
  and `junction_properties(..., method="matsubara")` use it; the
  default `method="levels"` is unchanged, so every earlier calibration
  stays as it was. README Example 14.
- `TelegraphPSDFit.cov`: the covariance of (S0, tau_s[, floor]) in
  their own units, with the correlations that `sigma` leaves out.
- `examples/tutorial_readout_bandwidth.py`.

### Fixed

- `FiniteLJunction.level_data` (and so `andreev_sums` and
  `saturation_ratio`) for |phi| below its difference step dphi =
  1e-3 rad. The levels at phi - dphi, phi and phi + dphi were matched
  by their order in energy, but the two levels of an Andreev doublet
  cross at phi = 0, so the stencil switched branch: the slope of a
  doublet level came out as -alpha phi / dphi instead of -alpha and
  its curvature as about -2 alpha / dphi. Ti/Al(thick) at 0.3 Tc,
  `saturation_ratio(phi, T, "L")`: 853.79 -> 1.00747 at phi = 1e-4,
  965.59 -> 1.00747 at phi = 0, 94.87 -> 1.00747 at phi = 9e-4
  (1.0074676 at phi = 1e-3, unchanged). Each level is now followed
  along its branch (signed angle 2 asin(sqrt(tau) sin(phi/2))) when
  the stencil crosses phi = 0 (mod 2 pi); elsewhere the code and the
  results are those of 0.13.0. `C_A` was not affected.
- Documentation: the module docstring of `absnoise.levels` said the
  junction current counts bound states only; the code has included
  the continuum since 0.1.0.

### Changed

- `FiniteLJunction.andreev_sums` and `saturation_ratio` refuse a
  `which` other than "I" or "L" (it was taken as "L") and a
  temperature that is not positive or is at or above Tc (a NaN ratio
  before). `SensorBudget` refuses a `sigma`, `delta`, `L_r` or `nu_r`
  that is not finite and positive (a ZeroDivisionError or NaN later,
  before).
- `absnoise.levels.matsubara_free_energy_terms` moved its summation
  into a helper shared with `Ic_matsubara`; the operations are the
  same, in the same order.
- Type hints on the public functions touched.
- Packaging: a new `MANIFEST.in` puts `examples/`, `CHANGELOG.md`,
  `CITATION.cff` and `CONTRIBUTING.md` into the source distribution.
  Without the examples the tutorial tests were skipped when the suite
  ran from the 0.13.0 sdist; from the 0.14.0 sdist all 267 tests run
  and pass.

### Performance

- `markov_dwell_test`: the simulated traces are drawn from the same
  random stream in the same order and decoded in batches
  (`_viterbi_batch`: the Viterbi recursion over a vector of traces,
  the same additions and comparisons as `viterbi`). The paths are
  integers and are compared exactly with `viterbi` in the tests, and
  the result equals that of the 0.13.0 loop exactly. README Example 10
  call (20 000 samples, 500 traces), median of 5 runs, two interleaved
  rounds on a shared machine: 11.25 s -> 3.69 s and 16.17 s -> 5.87 s
  (3.0 and 2.8 times).

### Tutorials

- `examples/tutorial_readout_bandwidth.py`: a noise record of the
  resonator frequency -> averaged spectrum -> tauA, readout floor and
  plateau with their covariance -> energy resolution for 1, 3 and
  10 MHz of readout bandwidth and the bandwidth for 1.1 times the
  unlimited value, with error bars.

### Tests

- New `tests/test_v014.py` (67 tests, about 20 s). The fix against
  level slopes and curvatures from implicit differentiation of the
  level condition (coded in the test; fails on 0.13.0); `Ic_matsubara`
  against the closed-form short-junction maximum and the level path;
  the band integral against adaptive quadrature for every branch, the
  band-limited resolution against quadrature of 4 / NEP^2 from
  `nep_spectrum`, its low-bandwidth limit and the 0.10.x grid integral
  over its own band, the bandwidth against its closed form without a
  floor; `TelegraphPSDFit.cov` against 400 simulated spectra; the
  batched decoder and the dwell test against the 0.13.0 code (kept in
  the tests); the tutorial output.
- 267 tests in total (200 in 0.13.0).

## 0.13.0 (2026-10-17)

Exact inductance and Kerr term for the full finite-length model, a fit
of the electron-phonon cooling law to a measured heating curve, two
end-to-end tutorials, six corrections (each found by a test that fails
on 0.12.1), and faster telegraph simulation and level search. No
function was renamed or removed; README Examples 1 to 10 print the
same output as with 0.12.0.

### Added

- `JunctionModel.cpr_series(T)`: the exact Taylor coefficients
  (a1, a3) of I(phi) = a1 phi + a3 phi^3 + ... of the finite-length
  model, bound states and continuum together, from the Matsubara-sum
  free energy F = -2 kB T sum_n ln D(i w_n) (P. W. Brouwer and
  C. W. J. Beenakker, Chaos, Solitons & Fractals 8, 1249 (1997),
  Eqs. (6) and (8)) with D(i w) = 1 + 2 R x + x^2 and
  x = exp(-2 arcsinh(w/Delta) - c w). `JunctionModel.cpr_matsubara(T, phis)`: the whole
  current-phase relation the same way (no level search, no finite
  differences; for Ti/Al/Au with 121 phases 0.21 s against 2.15 s for
  the 0.13.0 `current_phase`, median of 5).
  `absnoise.levels.matsubara_free_energy_terms` is the per-channel
  building block. The sum runs to 2000 Delta and adds the rest of its
  large-frequency form in closed form (Legendre chi function), so the
  truncation is about 1e-10. README Example 11.
- `fit_ep_coupling(P_W, Te_K, area, Tp_K, sigma_Te=None, delta=None,
  fit_Tp=False)` and `EPCouplingFit` (with `G(T)` and `propagate`):
  Sigma, delta and optionally the bath temperature of
  P = Sigma A (Te^delta - Tp^delta) from a measured electron
  temperature versus heater power, fitted in Te with the closed form
  of `steady_temperature`, as G.-H. Lee et al., Nature 586, 42 (2020)
  did (their Fig. 4c and Table I). Linearized covariance, chi-square
  with Te errors; refuses an exponent at the end of its search range
  (1 to 8) and inseparable parameters; warns when the data stay in the
  linear regime. `ep_temperature`: the model curve. README Example 12.
- `examples/tutorial_detector_budget.py` and
  `examples/tutorial_readout_trace.py`, and a "Tutorials" section in
  the README with their output.

### Fixed

- `JunctionModel.dIdphi0` fitted a cubic to the first 8 points of
  `current_phase`, whose first point is a one-sided difference (about
  14 times too large) and whose next points scatter by up to 17 %
  where bound levels leave for the continuum. Zero-length junction
  (closed form known): 44 % low -> exact (1e-9). Ti/Al/Au at 0.3 Tc
  (scale 1): 1.229e-6 -> 2.865e-6 A/rad. It now returns the exact
  slope; `k` is ignored with a DeprecationWarning.
- `junction_properties` took dIdphi0 and a3 from `cpr_coeffs`, a
  two-term fit over 0 < phi < 0.76 rad. Ti/Al/Au at 0.3 Tc:
  a3 -1.0077e-7 -> -9.495e-8 A/rad^3 (6.1 %), dIdphi0
  1.484052e-6 -> 1.483368e-6 A/rad (4.6e-4); zero length: a3 2.3 %
  off -> exact (1e-8). Now from `cpr_series`. `cpr_coeffs` is
  unchanged (its docstring now states the bias).
- `abs_energies` merged the two levels of a degenerate pair (phi = 0,
  and tau = 1 at phi = pi), and arccos of the rounded
  R = 1 - tau + tau cos(phi) gave a = 0 for every |phi| below about
  2e-8/sqrt(tau). Levels at phi = 0 (tau = 0.7, c Delta = 3): 1 -> 2;
  first-order doublet splitting at phi = 1e-6: off by 7.6e-5 -> 1.6e-10;
  `FiniteLJunction` C_A for Ti/Al(thick) at 0.3 Tc, phi = 0:
  1.741741e-21 -> 3.483482e-21 J/K (the value at phi = 1e-7); tau = 1
  at phi = pi - 1e-9: 0 -> 1 level (E = 2.0e-10 Delta). Where
  1 - |R| >= 1e-3 the levels are bit-identical to 0.12.1.
- `ShortJunction.dlnIc_dT` / `dlnI1_dT` used the step 2e-3 Tc
  always. MoRe at 10 mK: inf / NaN -> 0.0 / 0.0; Ti/Al/Au at
  0.999 Tc: -inf -> -1320.07 1/K (fine derivative of `ic_model`:
  -1315.8); at 0.997 Tc: -532.63 -> -443.60 (reference -442.12). The
  step is now min(rel Tc, T/2, (Tc - T)/10), unchanged between
  0.004 Tc and 0.98 Tc; T <= 0 and T >= Tc are refused.
- `sigma_spectrum`: f = 1e-200 and f = 0 or 1: AssertionError ->
  closed form (var 2f(1-f), tau_eff 1/Gs); Gs = Gp = 1e-15:
  S0 Gs = 0, tau_eff Gs = 0 -> 1.125, 0.75 (the values for
  Gs = Gp = 1). The symmetrized generator is built from its closed-form
  entries and only the stationary mode is dropped; other results agree
  with 0.12.1 to 1e-12.
- `continuum_share`: ZeroDivisionError below about 0.03 Tc (Ti/Al/Au
  at 20 mK) -> share_dIdT NaN (dI/dT is below rounding there), and
  below 0.002 Tc the step no longer reaches negative temperatures.
- `SensorBudget.nep_spectrum` accepted a negative readout floor
  (readout NaN, total 6.28418e-18 below the Andreev plus phonon
  6.28420e-18 W/sqrt(Hz) in the Example 1 setup); now refused, as in
  `energy_resolution`.

### Changed

- `sigma_spectrum` refuses f outside [0, 1], Gs <= 0 and Gp < 0;
  `continuum_share` and the log slopes refuse T outside (0, Tc).
- `junction_properties` converts `T_grid` to a float array.
- Type hints on the public functions touched.

### Performance

Median of 7 runs, same machine, 0.12.1 -> 0.13.0, two interleaved
rounds (the machine was shared with other jobs):

- `telegraph_traces`, vectorized in time with the same random stream:
  identical traces (tested against the 0.12.1 loop). README Example 5
  call (40 000 steps, 64 channels): 0.364 s -> 0.084 s and
  0.362 s -> 0.089 s in the two rounds (4.1 to 4.3 times); one
  channel, 20 000 steps: 0.161 s -> 0.46 ms (about 350 times).
- `abs_energies`: scalar path of the root function without
  `numpy.clip` (bit-identical); `JunctionModel.compute_levels` for
  Ti/Al/Au with 41 phases: 0.51 s -> 0.17 s (2.7 to 3.0 times).
- `free_energy_multi` and `free_energy_components`: continuum part for
  all phases of a mode at once (equal to the 0.12.1 loop to 1e-13 in
  the tests, bit-identical on the development machine). Ti/Al/Au, 41
  phases: 1.25 s -> 0.49 to 0.54 s (2.3 to 2.6 times);
  `JunctionModel.calibrate` end to end: 1.73 to 1.82 s -> 0.70 to
  0.74 s (2.4 to 2.5 times).
- The full test suite: 160 tests in 87 s (0.12.1) -> 200 tests in
  53 s.

### Tutorials

- `examples/tutorial_detector_budget.py`: Ic(T) curve and heating
  curve -> fitted junction and cooling law -> thermal time, phonon NEP
  and energy resolution with the error bars of both fits.
- `examples/tutorial_readout_trace.py`: readout trace -> f and tauA
  with error bars, Markov check, a second tauA from the spectrum, and
  the implied temperature resolution.

### Tests

- New `tests/test_v013.py` (40 tests, about 15 s, 10 s of it the
  readout tutorial). Fixes: against the exactly computed branch angle
  and the first-order splitting, the closed-form short-junction slope
  and cubic term (derived in the test), `ic_model` derivatives, the
  closed form of independent spins and scale covariance; each fails on
  0.12.1. Features: the Matsubara current against the closed form
  (1e-9) and against `current_phase` at finite length (2e-3), the chi
  function against its series; the cooling-law fit against noise-free
  data, a Nelder-Mead search on an independently coded chi-square, the
  closed-form one-parameter error bar, 150-fit Monte Carlo and
  `SensorBudget.Gep`; propagation exact for linear and quadratic
  functions. Speed-ups against the 0.12.1 code kept in the tests. Both
  tutorials are run and their printed numbers checked.
- 200 tests in total (160 in 0.12.1).

## 0.12.1 (2026-10-10)

Citation corrections and packaging.

### Fixed (documentation)

- `ep_power` docstring: Lee et al., Nature 586, 42 (2020) fit Sigma =
  2.14, 2.04, 2.74 and 3.30 W m^-2 K^-3, so the range is 2.04-3.30, not
  2.1-3.3. The default of 2.0 is unchanged.
- `constants.py`: the graphene mass density (7.6e-7 kg/m^2) and LA
  sound velocity (2.0e4 m/s) now cite E. H. Hwang and S. Das Sarma,
  Phys. Rev. B 77, 115449 (2008), whose values they match. The comment
  on the deformation potential called 18 eV "mid-range of 10-30 eV";
  it now says that the paper uses 19 eV and quotes 10-30 eV, and that
  18 eV lies in that range. None of the three constants is used by the
  package, and their values are unchanged.
- Packaging: the license is now given as the SPDX expression
  `license = "Apache-2.0"` with `license-files = ["LICENSE"]`, and the
  old license table and license classifier are gone. Building no longer
  prints setuptools' deprecation warnings (their support ends on
  2027-02-18). Building from source now needs setuptools 77 or newer;
  installing the package is unchanged.

## 0.12.0 (2026-10-10)

Two tools for working with your own measurements (error bars on any
budget number from an Ic(T) fit, and a check that a readout trace
follows the single-rate telegraph model), and five corrections, each
found by a test that fails on 0.11.0. No function was renamed or
removed; every README example of 0.11.0 prints the same output.

### Added

- `IcFit.propagate(func, xi, L, W, Vbg, Rn, ...)`: one-standard-
  deviation error (and covariance) of any number computed from the
  fitted junction, `func(recipe)`, from the fit covariance of
  (Ic0, Tc, tau), with the measured geometry held fixed. Sigma points
  at plus and minus one standard deviation along the principal axes of
  the covariance; exact for a function linear in the parameters.
  Refuses when the second-order change over one standard deviation
  exceeds `max_nonlinearity` (default 0.1) times the error bar, when a
  parameter would leave its range within one standard deviation, or
  when `func` returns a non-finite value. README Example 9.
- `markov_dwell_test(y, model, n_boot=200, seed=0, min_dwells=30)`:
  tests whether the stays of a decoded trace follow the geometric law
  of a single-rate Markov process, with the dispersion index
  var(k) / (mean(k) (mean(k) - 1)) (1 for a geometric law) and a
  parametric bootstrap that simulates and decodes the fitted model, so
  decoding errors enter the reference too. Refuses fewer than
  `min_dwells` complete stays per state and a mean stay below 2
  samples. README Example 10.
- `dwell_times(path, dt=1.0)`: the complete stays in each state of a
  0/1 path (first and last stay left out, since the record cuts them).
- README Examples 9 and 10 and a "What changed in 0.12.0" section.

### Fixed

- `gap_bcs` near Tc. The spline in t = T/Tc ended at t = 0.9995 and
  returned 0 above it, and a cubic spline cannot follow the
  square-root closing of the gap. Before -> after (u = Delta/Delta0):
  t = 0.9997: 0 -> 0.030723; t = 0.9995: 0 -> 0.039328;
  t = 0.999: 0.054884 -> 0.055252; t = 0.998: 0.077956 -> 0.077854;
  t = 0.995: 0.1227057 -> 0.1227072. The new values are direct
  solutions of the same gap equation (agreement better than 1e-9),
  taken for t >= 0.99 from a table of u / sqrt(1 - t/t1) built from the
  Matsubara-sum form of the equation. Below t = 0.99 nothing changes.
  `SensorBudget` no longer refuses 0.9995 Tc <= T < Tc as "at or above
  Tc".
- `avar_exponential` lost its accuracy for T << tau, where the four
  terms of the closed form cancel to order (T/tau)^3. With
  var = tau = 1: T = 1e-4: 6.66578e-5 -> 6.66617e-5; T = 1e-5:
  7.77e-6 -> 6.67e-6; T = 1e-6: -1.1e-4 (negative) -> 6.67e-7. Below
  T = tau/2 the bracket is now summed as its Taylor series; for
  T >= tau/2 the computation is unchanged.
- `ic_model` and `ShortJunction.Ic` (`current_phase`) returned NaN at
  and above Tc; they return 0. `plan_ic_measurement` with one planned
  temperature above Tc (Example 4's 8 temperatures plus 0.8 K):
  identifiable False, no error bars -> identifiable True with the same
  error bars as without that point (sigma(Tc) = 6.06 mK).
- `absnoise.levels.continuum_free_energy` had the opposite sign of the
  continuum free energy used by `JunctionModel`, `continuum_share` and
  `occupation_heat_capacities`. In the test case (c Delta = 1.5,
  tau = 0.7, T = 0.3 Tc) F_bound + F_cont jumped by 2.013 Delta across
  the 0.06 rad where a bound level leaves for the continuum; it now
  changes by 0.007 Delta (the bound part alone jumps by 1.010 Delta).
  At phi = 1: -0.587970 Delta -> +0.587970 Delta. `JunctionModel` is
  unchanged.
- `plan_psd_measurement` truncated `n_avg` with int(), so a
  non-integer `n_avg`, which `fit_telegraph_psd` accepts, was planned
  as the integer below. Example 4's band with n_avg = 1.5:
  sigma(tau) 159.0 us -> 119.9 us (the fit reports 119.9 us).

### Changed

- `click_template` refuses a negative or non-finite deposited energy
  and a non-positive tauA or T0 (a negative deposit started below T0
  and never relaxed); a zero deposit returns the flat template with
  `energy_residual` 0 instead of raising ZeroDivisionError.
- `avar_exponential` refuses non-finite var, tau_c and averaging
  times.
- `plan_psd_measurement` refuses a non-finite `n_avg`.
- Docstrings of `ic_model`, `ShortJunction.current_phase`, `gap_bcs`
  and `continuum_free_energy` state the behavior at and above Tc, the
  near-Tc table and the sign convention.

### Tests

- New `tests/test_v012.py` (29 tests). Fixes: the gap near Tc against
  direct solutions of the gap equation in integral form (1e-9) and in
  Matsubara-sum form (1e-9), the two forms against each other (1e-9),
  the Ginzburg-Landau limit derived from the sum (2e-5), continuity at
  0.99 Tc and unchanged values below; Ic exactly 0 at and above Tc and
  a plan with a point above Tc equal to the plan without it (1e-12);
  `avar_exponential` against the defining integral rewritten without
  cancellation (1e-9, T/tau from 1e-8 to 3) and its Taylor series
  (1e-12); continuity of F_bound + F_cont at a bound-level exit and
  agreement of `continuum_free_energy` with the `JunctionModel` path
  (5e-4); planned against fitted error bars for n_avg = 1.5, 2.5, 7.3
  (1e-3); `click_template` input checks. All of these fail on 0.11.0.
  Features: `propagate` exact for a linear function (1e-10), the
  quadratic closed form (1e-9), and Monte Carlo over 400 parameter
  sets (12 %); dwell times counted by hand, the geometric identity of
  the dispersion index, the closed-form statistics of the simulated
  reference paths, p-value calibration on 12 Markov traces and
  detection of a two-time-scale trace.
- 160 tests in total (131 in 0.11.0).

## 0.11.0 (2026-09-30)

Exact where 0.10.x approximated: the matched-filter energy resolution
in closed form, exact sampling of the telegraph process in the
simulator and the decoder, the exact statistics of averaged
periodograms in the spectrum fit and its planner; and error bars for
the readout-trace decoder.

### Added

- `matched_filter_sigma(P, Q, W, tau_th, tauA)`: closed-form
  matched-filter energy resolution for the budget's three noise
  channels (phonon, occupation, readout floor), from
  `int_0^inf dx / (a + b x^2 + c x^4) = pi / (2 sqrt(a) sqrt(b + 2 sqrt(a c)))`.
  `SensorBudget.energy_resolution` uses it by default;
  `method="grid"` keeps the 0.10.x numerical integral.
- `TelegraphHMM.uncertainties(y, dt=None)`: covariance and
  one-standard-deviation errors of (p_up, p_dn, m0, m1, sigma), and of
  f and tauA when dt is given, from the observed information (central
  differences of the exact gradient). `TelegraphHMM.score` (exact
  gradient of the log-likelihood by Fisher's identity) and
  `TelegraphHMM.loglik`.
- `flip_probabilities(f, tauA, dt)`: exact per-sample flip
  probabilities of a sampled two-state process.
- README Example 8 (a noise spectrum with few averages) and error bars
  in Example 6; a "What changed in 0.11.0" section.

### Changed

- `fit_hmm` starts from the exact two-group split of the samples (the
  threshold with the smallest within-group sum of squares), with the
  upper group's fraction as the starting occupation, instead of the
  median split. This lifts a limit the 0.10.1 README listed: a trace
  with more than half of its samples exactly at the maximum could not
  start. `fit_hmm` now refuses fewer than 4 samples, non-finite samples
  and constant traces.
- `TelegraphHMM.forward_backward` and `viterbi` run on Python floats:
  on 20 000 samples about 10 times (forward-backward) and 17 times
  (Viterbi) faster than 0.10.1, with results equal to the 0.10.1
  matrix code to 1e-12 and identical Viterbi paths (tested). The
  Gaussian densities are formed in log space and divided by their
  larger value at each sample (the log-likelihood adds the divisor
  back), so a far outlier no longer underflows: with one 200-sigma
  sample, 0.10.1 returned a NaN log-likelihood and `fit_hmm` stopped
  with "flip probabilities must lie in (0, 1)"; now both are finite.
  The start split uses samples clipped to their 0.1 % and 99.9 %
  quantiles, so a lone outlier cannot become a "state". Single-sample
  traces still work in `forward_backward`, `posterior` and `viterbi`
  (same values as 0.10.1); an empty trace is refused.
- `SensorBudget` methods refuse T <= 0, T at or above Tc (gap closed;
  `Ce`, `Gep`, `tau_th` and `dT_phonon` still accept it), non-positive
  or non-finite tauA and averaging time, `which` other than "L"/"I"
  and a negative readout floor. Before: ZeroDivisionError at T = 0 or
  T >= Tc, NaN for negative tauA, and any other `which` was silently
  treated as an inductance coupling at the maximum-current phase.
- `telegraph_traces` validates f, tauA, dt, n_steps and n_channels
  with ValueError; the `assert` on the time step is gone (no longer
  needed).
- `matched_Tc` refuses a non-positive T0.

### Fixed

- `SensorBudget.energy_resolution` stopped its integral at
  10 / (2 pi min(tauA, tau_th)), where the integrand has not decayed
  (it falls as 1/f^2 where occupation noise dominates and is flat
  where phonon noise dominates), and so overestimated sigma_E.
- `TelegraphHMM.rates` / `from_rates` and `telegraph_traces` used the
  first-order flip probabilities dt f / tauA and dt (1-f) / tauA. For
  a continuous-time process read every dt the exact values are
  f (1 - e^(-dt/tauA)) and (1 - f)(1 - e^(-dt/tauA)); `rates` now
  returns tauA = -dt / ln(1 - p_up - p_dn) and raises when
  p_up + p_dn >= 1.
- `fit_telegraph_psd(n_avg=n)` treated the log of an average of n
  periodograms as unbiased with variance 1/n. It has mean
  ln S + psi(n) - ln n and variance psi'(n) (Gamma(n, 1/n) scatter,
  exact for Gaussian noise); the fit now removes the offset and uses
  psi'(n). `plan_psd_measurement` and `averages_for_tau` use psi'(n)
  too; `averages_for_tau` searches for the smallest sufficient
  integer instead of the 1/sqrt(n) closed form.
- `matched_Tc` used Delta0 = 1.7639 kB Tc, while `Recipe.Delta` (and
  so the budget of `matched_recipe`) uses 1.764 kB Tc.
- `psd_single_sided` doubled the Nyquist bin for an even number of
  samples; `sum(S) df` now equals the variance exactly.
- The decode module docstring promised likelihood-based error bars
  that the module did not compute.

### Behaviour changes (before -> after)

- Energy resolution, Ti/Al/Au at 0.3 Tc (README Example 1,
  tauA = 1 us): 1.810e-22 J -> 1.752e-22 J (-3.2 %); with
  tauA = 1 ns: -9.7 % ("L") and -15 % ("I") (grid overestimates of
  10.7 % and 18 %). With a large readout floor (S_ro_y = 1e-18) and
  tauA = 1 us the change is below 0.1 % (both readouts), but with
  tauA = 1 ns it is still -2.6 % ("L") and -10 % ("I").
- HMM tauA from the same fitted flip probabilities: multiplied by
  s / (-ln(1 - s)), s = p_up + p_dn, i.e. 2.5 % smaller at
  dt = 0.05 tauA and 21 % smaller at dt = 0.5 tauA (the old value was
  27 % too long there). README Example 6:
  f = 0.297, tau = 0.977 ms -> f = 0.290 +- 0.020,
  tau = 0.993 +- 0.056 ms (new trace, new start, new conversion; true
  values 0.3 and 1 ms).
- `telegraph_traces` output changes wherever a random number falls
  between the old and new flip thresholds (the random stream is the
  same). README Example 5: fitted tau 198 -> 201 us, Lorentzian
  variance 0.2112 -> 0.2116, trace variance 0.2112 -> 0.2107, Allan
  values 3.14/5.26/2.34 -> 3.12/5.17/2.33.
- `fit_telegraph_psd` with n_avg: S0 and floor larger by
  exp(ln n - psi(n)) (1.78 for n = 1, 1.31 for n = 2, 1.14 for n = 4,
  1.0079 for n = 64); tau unchanged (to the optimizer tolerance,
  about 3e-9); relative error bars larger by sqrt(n psi'(n)) (1.28
  for n = 1, 1.0039 for n = 64), so sigma(tau) grows by that factor
  and sigma(S0), sigma(floor) by it times the S0 factor; chi2 smaller
  by the factor 1 / (n psi'(n)).
  Without n_avg nothing changes.
- PSD planner (README Example 4): sigma(tau) at 64 averages
  15.50 us -> 15.56 us; averages needed for 3 us 1708 -> 1709.
- `matched_Tc(0.05)`: 0.0776088 K -> 0.0776059 K (-3.6e-5 relative).
- `psd_single_sided`: only the last bin of an even-length record
  changes (halved).

### Tests

- New `tests/test_v011.py` (43 tests): closed form against adaptive
  quadrature over all frequencies (1e-9 bare, 1e-8 through the budget
  and `nep_spectrum`, both readouts) and its two exact limits (1e-14);
  the grid overestimate pinned to the documented 3.3 % and 10.7 %;
  budget refusals; flip probabilities against `scipy.linalg.expm`
  (1e-9) and the first-order limit (1e-5); Monte Carlo
  autocorrelation e^-k at dt = tauA (0.01); exact rate inversion
  (1e-9); coarse-trace tau within 4 error bars; score against finite
  differences (1e-5); binomial limit of the error bars (0.5 %); Monte
  Carlo calibration of the error bars (z-score standard deviation in
  [0.65, 1.4] over 30 traces); the two-group start on a noise-free
  trace filled 90 % of the time and against brute force (1e-12); the
  fast recursions against the 0.10.1 matrix code (1e-12); the PSD
  offset as an exact rescaling (1e-7) and a 300-spectrum Monte Carlo
  of an average of two periodograms (unbiased log S0, chi2/dof within
  0.06 of 1, tau scatter within 20 % of the plan, and no refused fit
  in the 300 draws, asserted); Parseval (1e-12); a 200-sigma outlier
  (log-likelihood equal to a log-space logsumexp forward recursion to
  1e-10, finite and monotone EM); a single-sample trace against its
  closed form (1e-14).
- Updated: the NEP matched-filter identities in `test_nep.py` and
  `test_v0101.py` and the NumPy 1.x integral check now name
  `method="grid"` (the path they were written for); the planner tests
  use psi'(n) scaling and a noise-free spectrum at the log-mean; the
  matched-design test checks the ratio inside `SensorBudget`.
- 131 tests in total (88 before).

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
