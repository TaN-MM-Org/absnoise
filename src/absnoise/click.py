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

from .constants import E_CHARGE, HBAR, KB
from .levels import gap_bcs
from .materials import Recipe

MATCHED_RATIO = 2.3994   # Delta*(T0) / kB T0 at the matched-level optimum


def matched_Tc(T0, ratio=MATCHED_RATIO):
    """Junction critical temperature whose BCS-suppressed proximity gap
    satisfies Delta*(T0) = ratio * kB * T0 at operating temperature T0.

    Uses the solved gap equation (gap_bcs); the widely used tanh
    interpolation would return a visibly different Tc. The test suite
    verifies the defining equation to 1e-9."""
    def g(Tc):
        return gap_bcs(T0, Tc, 1.7639 * KB * Tc) - ratio * KB * T0
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


def click_template(budget, E_gamma, tauA, T0, scenario="C",
                   dt=1e-10, t_end=2e-7, dt2=None, t_end2=None):
    """Noiseless occupation response m(t) to a photon deposit (J).

    budget : a SensorBudget built on the operating recipe.
    Returns dict(ts, ms, Te_peak, m0, energy_residual) where
    energy_residual is |(1/2) gamma (T_pk^2 - T0^2) - E_gamma|/E_gamma,
    the exact peak-temperature identity for C_e = gamma T (zero up to
    floating point; asserted in the test suite)."""
    if scenario not in ("C", "T"):
        raise ValueError("scenario must be 'C' (cold exchange time) or "
                         "'T' (thermally activated)")
    if dt2 is None:
        dt2 = min(2e-8, tauA / 6.0)
    if t_end2 is None:
        t_end2 = 60.0 * tauA
    sj = budget.sj
    A = budget.recipe.area
    gam = budget.Ce(T0) / T0                # C_e = gamma T
    Sig, dlt = budget.sigma, budget.delta
    D = sj.Delta(T0)
    m0 = float(np.tanh(D / (2 * KB * T0)))
    Tpk = float(np.sqrt(T0 ** 2 + 2 * E_gamma / gam))
    residual = abs(0.5 * gam * (Tpk ** 2 - T0 ** 2) - E_gamma) / E_gamma

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
    values."""
    from scipy.signal import lfilter

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
    return dict(
        efficiency=float(np.mean(out1 > threshold)),
        dark_fraction=float(np.mean(out0 > threshold)),
        snr=snr,
        m_dip=float((m0 - tpl["ms"].min()) / m0),
        energy_residual=tpl["energy_residual"],
        Te_peak=tpl["Te_peak"],
    )
