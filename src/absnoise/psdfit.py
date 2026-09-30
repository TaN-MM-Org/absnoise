"""Extract occupation-noise parameters from a measured noise spectrum
(new in v0.7).

The HMM decoder of `absnoise.decode` works on resolved time traces;
when the occupation dynamics are faster than the sampling, or only a
spectrum analyzer trace exists, the data product is a single-sided PSD
instead. For the exponentially correlated occupation noise this
package predicts, that PSD is the Lorentzian of the module docstring
of `absnoise.telegraph`,

    S(f) = S0 / (1 + (2 pi f tau)^2) + floor,

with S0 the zero-frequency plateau, tau the occupation correlation
time (the knee sits at f_c = 1/(2 pi tau)), and `floor` the white
readout background. `fit_telegraph_psd` fits that model to a measured
averaged PSD in log space (the variance-stabilizing choice for the
multiplicative scatter of averaged periodograms, stated rather than
hidden) and returns the three parameters with uncertainties.

Identifiability is enforced, not hoped for: a band that never sees the
knee cannot determine tau -- a spectrum flat across the whole band is
equally well fit by any sufficiently small or large tau, with S0 and
the floor trading against each other. The fit therefore REFUSES,
with an explanation, when the fitted knee lands outside the measured
band (below 2 f_min or above f_max / 2), instead of returning one of
a continuum of minimizers with a well-formatted covariance.

Anchors asserted in the tests rather than stated: noise-free synthetic
spectra return the generating S0 and tau to better than 1e-6 and the
floor to better than 1e-4;
the total occupation variance recovered from the fitted Lorentzian's
exact integral, S0 / (4 tau), matches the variance of the generating
telegraph Monte-Carlo traces (Parseval, two independent code paths);
the fitted tau matches the Monte-Carlo generating value within
statistical tolerance; and both out-of-band knee cases are refused.

Statistics of an averaged periodogram (changed in 0.11.0). For Gaussian
noise, one periodogram bin (not the zero or Nyquist bin) scatters as S
times an exponential variable, and the average of n independent
periodograms as S times a Gamma(n, 1/n) variable. Its logarithm then
has mean ln S + psi(n) - ln n and variance psi'(n) (psi is the digamma
function, psi' the trigamma function; standard properties of the Gamma
distribution). When `n_avg` is given, the fit subtracts that offset and
uses psi'(n_avg) as the per-bin variance. Up to 0.10.x it used no offset
and the variance 1/n_avg: S0 and the floor came out low by the factor
exp(psi(n) - ln n) (0.56 for n = 1, 0.76 for n = 2, 0.992 for n = 64),
and chi2 came out high by n psi'(n) (1.64 for n = 1, 1.008 for n = 64).
tau does not depend on the offset (it scales S0 and the floor
together). Without `n_avg` nothing is corrected, because the offset is
unknown; for n_avg = n the bias of S0 and floor is then the factor
above. The Gamma statistics are exact for Gaussian noise and hold
approximately for telegraph noise over records much longer than tau.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy.optimize import least_squares
from scipy.special import digamma, polygamma

__all__ = ["TelegraphPSDFit", "fit_telegraph_psd", "telegraph_psd_model"]


def telegraph_psd_model(f_hz, S0, tau_s, floor=0.0):
    """The single-sided model PSD S0 / (1 + (2 pi f tau)^2) + floor --
    the closed form of `absnoise.telegraph`, exposed for plotting and
    cross-checks."""
    f = np.asarray(f_hz, dtype=float)
    return S0 / (1.0 + (2.0 * np.pi * f * tau_s) ** 2) + floor


@dataclasses.dataclass
class TelegraphPSDFit:
    """Fitted Lorentzian occupation-noise spectrum.

    S0 : zero-frequency plateau (units of the input PSD).
    tau_s : occupation correlation time (s); knee at 1/(2 pi tau).
    floor : fitted white background (0 if not fitted).
    sigma : dict of 1-sigma uncertainties for the fitted parameters.
    variance : the total variance the fitted Lorentzian carries,
        S0 / (4 tau) -- its exact single-sided integral, useful as a
        cross-check against the trace variance.
    chi2, chi2_dof : residual chi-square of the log-space fit when
        `n_avg` was given (each averaged PSD bin has log-variance
        psi'(n_avg), the trigamma function; about 1/n_avg for large
        n_avg).
    model : the fitted model evaluated at the data frequencies.
    """

    S0: float
    tau_s: float
    floor: float
    sigma: dict
    variance: float
    chi2: float | None
    chi2_dof: int | None
    model: np.ndarray


def fit_telegraph_psd(f_hz, S_meas, fit_floor=True, n_avg=None) -> TelegraphPSDFit:
    """Fit the Lorentzian occupation-noise model to a measured PSD.

    f_hz : (n,) positive frequencies (drop the DC bin).
    S_meas : (n,) measured single-sided PSD, positive (an averaged
        periodogram, e.g. Welch or the channel-averaged output of
        `psd_single_sided`).
    fit_floor : also fit a white background floor (default True; set
        False for background-subtracted spectra).
    n_avg : number of independent periodograms averaged in each bin
        (it need not be an integer: an effective number is accepted).
        When given, the fit removes the known offset psi(n) - ln n of
        the logarithm of an averaged periodogram, so S0 and the floor
        are not biased low, and uses the exact log-variance psi'(n) per
        bin for the error bars and a chi-square consistency check (see
        the module docstring; changed in 0.11.0).

    Raises ValueError on malformed input and on the out-of-band knee
    (see module docstring: a band that never sees the knee cannot
    determine tau).
    """
    f = np.asarray(f_hz, dtype=float).ravel()
    S = np.asarray(S_meas, dtype=float).ravel()
    if f.size != S.size or f.size < 8:
        raise ValueError("need >= 8 matching (frequency, PSD) points")
    if np.any(f <= 0.0) or not np.all(np.isfinite(f)):
        raise ValueError("frequencies must be positive and finite "
                         "(drop the DC bin)")
    if np.any(S <= 0.0) or not np.all(np.isfinite(S)):
        raise ValueError("the PSD must be positive and finite; average "
                         "more periodograms or mask bad bins")
    if float(S.max() / S.min()) < 4.0:
        raise ValueError(
            "the spectrum varies by less than a factor of 4 across "
            "the band: no knee rolloff is visible, so tau (and the "
            "S0/floor split) cannot be determined from this band. "
            "Extend the band past the knee, or determine tau from a "
            "resolved time trace with absnoise.decode")
    if n_avg is not None and not (np.isfinite(float(n_avg))
                                  and float(n_avg) >= 1.0):
        raise ValueError("n_avg must be >= 1 (the number of averaged "
                         "periodograms behind each bin)")
    if n_avg is None:
        log_offset = 0.0
    else:
        log_offset = float(digamma(float(n_avg)) - np.log(float(n_avg)))
    logS = np.log(S) - log_offset

    # starting guesses from the data themselves
    # (divided by exp(log_offset), so that with n_avg the problem is
    # exactly the uncorrected one shifted by a constant in log space)
    S0_0 = float(np.median(S[f <= np.percentile(f, 20)])) \
        * np.exp(-log_offset)
    fl_0 = float(np.median(S[f >= np.percentile(f, 80)])) \
        * np.exp(-log_offset)
    tau_0 = 1.0 / (2.0 * np.pi * float(np.median(f)))

    if fit_floor:
        p0 = [np.log(S0_0), np.log(tau_0),
              np.log(max(fl_0, 1e-12 * S0_0))]
    else:
        p0 = [np.log(S0_0), np.log(tau_0)]

    def unpack(p):
        S0 = np.exp(p[0])
        tau = np.exp(p[1])
        fl = np.exp(p[2]) if fit_floor else 0.0
        return S0, tau, fl

    def resid(p):
        S0, tau, fl = unpack(p)
        return np.log(telegraph_psd_model(f, S0, tau, fl)) - logS

    res = least_squares(resid, p0, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    if not res.success:
        raise RuntimeError(f"PSD fit failed: {res.message}")
    S0, tau, fl = unpack(res.x)

    f_knee = 1.0 / (2.0 * np.pi * tau)
    if f_knee < 2.0 * f.min() or f_knee > 0.5 * f.max():
        raise ValueError(
            f"the fitted knee frequency {f_knee:.3g} Hz lies outside "
            f"the measured band [{f.min():.3g}, {f.max():.3g}] Hz "
            "(needs a factor-2 margin on both sides). A band that "
            "never sees the knee cannot determine tau: S0 and the "
            "floor trade against it. Extend the band, or determine "
            "tau from a resolved time trace with absnoise.decode")

    J = res.jac
    dof = f.size - len(res.x)
    try:
        cov_log = np.linalg.inv(J.T @ J)
    except np.linalg.LinAlgError:
        raise RuntimeError(
            "singular information matrix: the spectrum does not "
            "determine the parameters")
    rss = float(2.0 * res.cost)
    if n_avg is None:
        cov_log = cov_log * (rss / max(dof, 1))
        chi2 = chi2_dof = None
    else:
        var_bin = float(polygamma(1, float(n_avg)))
        cov_log = cov_log * var_bin
        chi2, chi2_dof = rss / var_bin, dof
    s_log = np.sqrt(np.maximum(np.diag(cov_log), 0.0))
    names = ["S0", "tau_s"] + (["floor"] if fit_floor else [])
    vals = [S0, tau, fl][:len(names)]
    sigma = {nm: float(sv * v) for nm, sv, v in zip(names, s_log, vals)}
    return TelegraphPSDFit(
        S0=float(S0), tau_s=float(tau), floor=float(fl), sigma=sigma,
        variance=float(S0 / (4.0 * tau)), chi2=chi2, chi2_dof=chi2_dof,
        model=telegraph_psd_model(f, S0, tau, fl))
