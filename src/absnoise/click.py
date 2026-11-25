"""Nonlinear single-photon click dynamics and matched-level design.

Removes the linear-response and Gaussian-statistics assumptions of the
budget-level analysis:

* the matched-level design condition: a junction critical temperature
  chosen so that the proximity gap sits at Delta*(T0) = 2.3994 kB T0,
  the occupation-readout optimum (solved from the BCS gap equation, not
  the tanh interpolation);
* the full nonlinear heat balance C_e(T) dT/dt = -Sigma A (T^d - T0^d)
  after a photon deposit E_gamma, integrated with a stiffness-safe
  exponential integrator whose energy bookkeeping is anchored by the
  exact peak-temperature identity (1/2) gamma (T_pk^2 - T0^2) = E_gamma
  for C_e = gamma T (verified to machine precision in the test suite);
* Andreev occupations m(t) relaxing toward tanh(Delta*/2 kB Te(t)) with
  an exchange time that is either the cold value tau_A ("C") or
  thermally activated, tau(Te) = tau_A exp[-(Delta*/kB)(1/T0 - 1/Te)]
  floored at 1 ns ("T"); the electrodes stay cold, so Delta* is fixed;
* a whitened matched-filter click Monte Carlo with the three noise
  channels of the budget analysis: occupation telegraph noise, phonon
  TFN entering through the occupation lag, and a white readout floor.

The specific published study (device grids, figure scripts, archived
trial data) remains in the paper repository; this module is the
general-purpose engine.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from .constants import BCS_RATIO, E_CHARGE, HBAR, KB
from .levels import gap_bcs
from .materials import Recipe

MATCHED_RATIO = 2.3994   # Delta*(T0) / kB T0 at the matched-level optimum


def matched_Tc(T0, ratio=MATCHED_RATIO):
    """Junction critical temperature whose BCS-suppressed proximity gap
    satisfies Delta*(T0) = ratio * kB * T0 at operating temperature T0.

    Uses the solved gap equation (gap_bcs); the widely used tanh
    interpolation would return a visibly different Tc. The
    zero-temperature gap is Delta0 = 1.764 kB Tc (BCS_RATIO), the
    convention of `Recipe.Delta` and so of every budget built on
    `matched_recipe` (fixed in 0.11.0: before, this solve used
    1.7639 kB Tc, and the budget of the returned recipe then had
    Delta*(T0) / kB T0 = 2.39954 instead of 2.3994). The test suite
    verifies the defining equation, through `SensorBudget`, to 1e-9."""
    T0 = float(T0)
    if not (np.isfinite(T0) and T0 > 0.0):
        raise ValueError("T0 must be finite and positive (K)")
    ratio = float(ratio)

    def g(Tc):
        return gap_bcs(T0, Tc, BCS_RATIO * KB * Tc) - ratio * KB * T0
    # (changed in 0.17.0) a ratio outside the searched range of Tc,
    # 1.01 T0 to 6 T0, is refused with that range; up to 0.16.0 brentq
    # stopped with "f(a) and f(b) must have different signs"
    r_lo = gap_bcs(T0, 1.01 * T0, BCS_RATIO * KB * 1.01 * T0) / (KB * T0)
    r_hi = gap_bcs(T0, 6.0 * T0, BCS_RATIO * KB * 6.0 * T0) / (KB * T0)
    if not (r_lo < ratio < r_hi):
        raise ValueError(
            f"ratio = Delta*(T0) / kB T0 must lie between {r_lo:.4g} "
            f"(Tc = 1.01 T0) and {r_hi:.4g} (Tc = 6 T0), the range "
            f"searched for Tc; got {ratio!r}")
    return float(brentq(g, T0 * 1.01, T0 * 6.0))


def matched_recipe(T0, W, L, tau=0.3, Vbg=30.0, Ic20=1.08e-6, Rn=43.3,
                   ratio=MATCHED_RATIO):
    """A Recipe at the matched-level design condition.

    Geometry (W, L in meters) and contact transparency are free design
    choices; Tc is fixed by matched_Tc. Default Ic20 and Rn follow the
    Ta/Ti/Au recipe of Jung et al. and act only as the calibration
    anchor of the ensemble scale."""
    Tc = matched_Tc(T0, ratio)
    return Recipe("matched", "matched", Tc, 1e-6, L, W, Vbg, tau,
                  Ic20, Rn)


def click_template(budget, E_gamma: float, tauA: float, T0: float,
                   scenario: str = "C", dt: float = 1e-10,
                   t_end: float = 2e-7, dt2: float | None = None,
                   t_end2: float | None = None) -> dict:
    """Noiseless occupation response m(t) to a photon deposit (J).

    budget : a SensorBudget built on the operating recipe.
    Returns dict(ts, ms, Te_peak, m0, energy_residual) where
    energy_residual is |(1/2) gamma (T_pk^2 - T0^2) - E_gamma|/E_gamma,
    the exact peak-temperature identity for C_e = gamma T (zero up to
    floating point; asserted in the test suite)."""
    if scenario not in ("C", "T"):
        raise ValueError("scenario must be 'C' (cold exchange time) or "
                         "'T' (thermally activated)")
    # input checks (new in 0.12.0): a negative deposit gave a start
    # below T0 that never relaxed, and E_gamma = 0 a ZeroDivisionError
    E_gamma, tauA, T0 = float(E_gamma), float(tauA), float(T0)
    if not (np.isfinite(E_gamma) and E_gamma >= 0.0):
        raise ValueError("deposited energy E_gamma must be finite and "
                         ">= 0 (J)")
    if not (np.isfinite(tauA) and tauA > 0.0):
        raise ValueError("tauA must be finite and positive (s)")
    if not (np.isfinite(T0) and T0 > 0.0):
        raise ValueError("T0 must be finite and positive (K)")
    if dt2 is None:
        dt2 = min(2e-8, tauA / 6.0)
    if t_end2 is None:
        t_end2 = 60.0 * tauA
    # (fixed in 0.17.0) the time grid is checked: up to 0.16.0 a
    # negative dt silently skipped the first stage and dt = 0 raised
    # ZeroDivisionError
    for name, val in (("dt", dt), ("dt2", dt2)):
        if not (np.isfinite(float(val)) and float(val) > 0.0):
            raise ValueError(f"{name} must be finite and positive (s)")
    for name, val in (("t_end", t_end), ("t_end2", t_end2)):
        if not (np.isfinite(float(val)) and float(val) >= 0.0):
            raise ValueError(f"{name} must be finite and >= 0 (s)")
    dt, dt2 = float(dt), float(dt2)
    t_end, t_end2 = float(t_end), float(t_end2)
    sj = budget.sj
    A = budget.recipe.area
    gam = budget.Ce(T0) / T0                # C_e = gamma T
    Sig, dlt = budget.sigma, budget.delta
    D = sj.Delta(T0)
    m0 = float(np.tanh(D / (2 * KB * T0)))
    Tpk = float(np.sqrt(T0 ** 2 + 2 * E_gamma / gam))
    residual = (abs(0.5 * gam * (Tpk ** 2 - T0 ** 2) - E_gamma) / E_gamma
                if E_gamma > 0.0 else 0.0)

    def tau_of_T(Te):
        if scenario == "C":
            return tauA
        expo = -(D / KB) * (1.0 / T0 - 1.0 / max(Te, 1e-4))
        return max(1e-9, tauA * float(np.exp(np.clip(expo, -500, 0))))

    def meq_of_T(Te):
        return float(np.tanh(D / (2 * KB * max(Te, 1e-4))))

    Te, m, t = Tpk, m0, 0.0
    ts, ms = [0.0], [m0]
    for step, stop in ((dt, t_end), (dt2, t_end2)):
        n = int((stop - t) / step)
        for _ in range(n):
            if Te - T0 > 1e-9:
                P = Sig * A * (Te ** dlt - T0 ** dlt)
                k = P / (gam * Te) / (Te - T0)      # local decay rate
                Te = T0 + (Te - T0) * np.exp(-k * step)
            tl = tau_of_T(Te)
            m = m + (meq_of_T(Te) - m) * (1 - np.exp(-step / tl))
            t += step
            ts.append(t)
            ms.append(m)
    return dict(ts=np.array(ts), ms=np.array(ms), Te_peak=Tpk, m0=m0,
                energy_residual=float(residual))


def click_monte_carlo(budget, E_gamma, tauA, T0, scenario="C",
                      n_trials=400, S_ro_nu=None, threshold=0.5,
                      dt=5e-10, seed=0):
    """Whitened matched-filter click Monte Carlo.

    Simulates n_trials photon and n_trials dark records of the readout
    resonator frequency around the operating point, with occupation
    telegraph noise (variance 2 f (1-f) / Nch on the mean occupation,
    correlation time tauA), phonon thermal-fluctuation noise entering
    through the occupation lag, and a white frequency-noise floor
    S_ro_nu (Hz^2/Hz; default: a quantum-limited-scale floor as in the
    accompanying study). Detection is a whitened matched filter built
    from the analytic noise PSD and the noiseless template.

    Returns dict(efficiency, dark_fraction, snr, m_dip, ...). The
    numbers are statistics-limited; the test suite asserts ordering
    relations (efficiency above dark fraction, positive SNR), not tight
    values.

    Since 0.17.0 the result also holds the standard errors of the three
    statistics: `efficiency_err` and `dark_fraction_err` are the
    binomial errors sqrt(p (1 - p) / n_trials) of a fraction of
    n_trials independent records, and `snr_err` is
    sqrt((2 + snr^2 / 4) / (n - 1)) with n = n_trials: the first-order
    error of a difference of two means of n Gaussian scores divided by
    their pooled standard deviation (the scores are linear in the
    Gaussian noise of the records; each sample variance, taken with
    divisor n, has variance 2 (n - 1) s^4 / n^2 for scores of standard
    deviation s). The tests compare all three with the scatter over
    independent seeds. n_trials must be at least 2, and dt, the
    threshold and S_ro_nu are checked (up to 0.16.0 n_trials = 1
    returned snr = inf)."""
    from scipy.signal import lfilter

    n_trials = float(n_trials)
    if not (np.isfinite(n_trials) and n_trials == np.floor(n_trials)
            and n_trials >= 2):
        raise ValueError("n_trials must be a whole number >= 2")
    n_trials = int(n_trials)
    dt = float(dt)
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive (s)")
    if not np.isfinite(float(threshold)):
        raise ValueError("threshold must be finite")
    if S_ro_nu is not None and not (np.isfinite(float(S_ro_nu))
                                    and float(S_ro_nu) >= 0.0):
        raise ValueError("S_ro_nu must be finite and >= 0 (Hz^2/Hz)")

    rng = np.random.default_rng(seed)
    sj = budget.sj
    D = sj.Delta(T0)
    Nch = sj.Nch
    m0 = float(np.tanh(D / (2 * KB * T0)))
    f0 = 1.0 / (np.exp(D / (KB * T0)) + 1.0)
    I1_0 = sj.dIdphi0(T0)
    LJ0 = (HBAR / (2 * E_CHARGE)) / I1_0
    Lr, nu0 = budget.L_r, budget.nu_r
    Cres = 1.0 / ((2 * np.pi * nu0) ** 2 * (Lr + LJ0))

    def nu_of_m(m):
        I1 = I1_0 * np.maximum(m, 1e-6) / m0
        LJ = (HBAR / (2 * E_CHARGE)) / I1
        return 1.0 / (2 * np.pi * np.sqrt((Lr + LJ) * Cres))

    tpl = click_template(budget, E_gamma, tauA, T0, scenario)
    nwin = int(max(6.0 * tauA, 4e-7) / dt)
    tgrid = np.arange(nwin) * dt
    i0 = nwin // 4
    dnu_tpl = np.interp(tgrid, tpl["ts"], nu_of_m(tpl["ms"]) - nu0)

    var_m = 2 * f0 * (1 - f0) / Nch
    sigT_eq = float(np.sqrt(KB * T0 ** 2 / budget.Ce(T0)))
    dmeq_dT = float(-(D / (2 * KB * T0 ** 2)) /
                    np.cosh(D / (2 * KB * T0)) ** 2)
    if S_ro_nu is None:
        kap = 2 * np.pi * 1e6
        S_ro_nu = (kap / 4) ** 2 * 2 / (30.0 * kap) / (2 * np.pi) ** 2
    conv = (nu_of_m(m0 * (1 + 1e-6)) - nu0) / (m0 * 1e-6)
    tau_th = budget.tau_th(T0)
    aA = np.exp(-dt / tauA)
    ath = np.exp(-dt / tau_th)
    sig_white = np.sqrt(S_ro_nu / (2 * dt))

    def batch(photon):
        xiT = rng.standard_normal((n_trials, nwin))
        dT = lfilter([sigT_eq * np.sqrt(1 - ath ** 2)], [1, -ath],
                     xiT, axis=1)
        drv = (np.sqrt(var_m * (1 - aA ** 2)) *
               rng.standard_normal((n_trials, nwin))
               + dmeq_dT * dT * (dt / tauA))
        x = lfilter([1.0], [1, -aA], drv, axis=1)
        y = conv * x + sig_white * rng.standard_normal((n_trials, nwin))
        if photon:
            y[:, i0:] += dnu_tpl[None, :nwin - i0]
        return y

    tpl_full = np.zeros(nwin)
    tpl_full[i0:] = dnu_tpl[:nwin - i0]
    freqs = np.fft.rfftfreq(nwin, dt)
    w = 2 * np.pi * freqs
    HA2 = 1.0 / (1.0 + (w * tauA) ** 2)
    S_A = conv ** 2 * 4 * var_m * tauA * HA2
    S_TFN = conv ** 2 * dmeq_dT ** 2 * \
        (4 * sigT_eq ** 2 * tau_th / (1 + (w * tau_th) ** 2)) * HA2
    S_n = S_A + S_TFN + S_ro_nu
    s_hat = np.fft.rfft(tpl_full)
    wgt = np.conj(s_hat) / S_n
    norm = float(np.real(np.sum(wgt * s_hat)))

    def score(Y):
        return np.real(np.fft.rfft(Y, axis=1) @ wgt) / norm

    out0 = score(batch(False))
    out1 = score(batch(True))
    snr = float((out1.mean() - out0.mean()) /
                np.sqrt(0.5 * (out0.var() + out1.var())))
    eff = float(np.mean(out1 > threshold))
    dark = float(np.mean(out0 > threshold))
    n = n_trials
    return dict(
        efficiency=eff,
        dark_fraction=dark,
        snr=snr,
        efficiency_err=float(np.sqrt(eff * (1.0 - eff) / n)),
        dark_fraction_err=float(np.sqrt(dark * (1.0 - dark) / n)),
        snr_err=float(np.sqrt((2.0 + 0.25 * snr ** 2) / (n - 1))),
        m_dip=float((m0 - tpl["ms"].min()) / m0),
        energy_residual=tpl["energy_residual"],
        Te_peak=tpl["Te_peak"],
    )
