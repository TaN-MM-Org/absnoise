"""v0.7 from-the-instrument anchors: the PSD parameter extraction held
to exact recovery, a Parseval cross-check against the generating
Monte Carlo, and its out-of-band refusals; and the file contracts'
exact round trips, refusals, and unit-plausibility flags."""
import numpy as np
import pytest

import absnoise as ab
from absnoise.psdfit import telegraph_psd_model


def test_psd_fit_exact_recovery_and_model_identity():
    S0, tau, fl = 3.2e-3, 1.4e-5, 1.1e-6
    f = np.logspace(2.5, 6.5, 60)                 # knee ~1.1e4 in band
    S = telegraph_psd_model(f, S0, tau, fl)
    fit = ab.fit_telegraph_psd(f, S)
    assert abs(fit.S0 - S0) / S0 < 1e-6
    assert abs(fit.tau_s - tau) / tau < 1e-6
    assert abs(fit.floor - fl) / fl < 1e-4
    # the exact single-sided integral of the Lorentzian part:
    # definitional identity on the fitted parameters, and close to truth
    assert fit.variance == fit.S0 / (4.0 * fit.tau_s)
    assert abs(fit.variance - S0 / (4.0 * tau)) / (S0 / (4 * tau)) < 1e-5
    # background-subtracted path
    fit0 = ab.fit_telegraph_psd(f, telegraph_psd_model(f, S0, tau),
                                fit_floor=False)
    assert abs(fit0.tau_s - tau) / tau < 1e-8 and fit0.floor == 0.0


def test_psd_fit_matches_the_telegraph_monte_carlo():
    """Two independent code paths: the Markov trace generator plus the
    periodogram on one side, the Lorentzian fit on the other. The
    fitted tau must match the generating tau, and the variance carried
    by the fitted Lorentzian must match the trace variance
    (Parseval)."""
    rng = np.random.default_rng(0)
    fvals, tauA, dt = 0.3, 2e-4, 5e-6
    n_steps, n_ch = 40000, 64
    tr = ab.telegraph_traces(fvals, tauA, dt, n_steps, n_ch, rng)
    # channel-averaged periodogram of the summed observable per channel
    Ss = []
    for c in range(n_ch):
        fr, S = ab.psd_single_sided(tr[:, c].astype(float), dt)
        Ss.append(S)
    Sbar = np.mean(Ss, axis=0)[1:]
    fr = fr[1:]
    # coarse log-binning tames periodogram scatter further
    nb = 40
    edges = np.logspace(np.log10(fr[0]), np.log10(fr[-1]), nb + 1)
    fb, Sb = [], []
    for i in range(nb):
        m = (fr >= edges[i]) & (fr < edges[i + 1])
        if m.sum() >= 3:
            fb.append(fr[m].mean())
            Sb.append(Sbar[m].mean())
    fit = ab.fit_telegraph_psd(np.asarray(fb), np.asarray(Sb))
    assert abs(fit.tau_s - tauA) / tauA < 0.15
    var_trace = float(tr.astype(float).var())
    assert abs(fit.variance - var_trace) / var_trace < 0.15
    # and the theoretical plateau 4 f (1-f) tau
    S0_th = 4.0 * fvals * (1 - fvals) * tauA
    assert abs(fit.S0 - S0_th) / S0_th < 0.2


def test_psd_fit_refuses_a_band_that_never_sees_the_knee():
    S0, tau, fl = 1e-3, 1e-5, 1e-7                # knee at 1.6e4 Hz
    f_low = np.logspace(0.0, 2.0, 30)             # band far below knee
    with pytest.raises(ValueError, match="knee|rolloff"):
        ab.fit_telegraph_psd(f_low, telegraph_psd_model(f_low, S0, tau, fl))
    f_high = np.logspace(6.0, 8.0, 30)            # band far above knee
    with pytest.raises(ValueError, match="knee|rolloff"):
        ab.fit_telegraph_psd(f_high,
                             telegraph_psd_model(f_high, S0, tau, fl))
    with pytest.raises(ValueError, match="positive"):
        ab.fit_telegraph_psd(np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0,
                                       6.0, 7.0]),
                             np.ones(8))


def test_ic_csv_round_trip_refusals_and_flags(tmp_path):
    T = np.linspace(0.05, 1.2, 12)
    Ic = np.linspace(5e-6, 1e-6, 12)
    sig = np.full(12, 5e-8)
    p = tmp_path / "ic.csv"
    ab.save_ic_csv(p, T, Ic, sig)
    T2, Ic2, s2 = ab.load_ic_csv(p)
    assert np.array_equal(T, T2) and np.array_equal(Ic, Ic2)
    assert np.array_equal(sig, s2)
    ab.save_ic_csv(p, T, Ic)                      # two-column variant
    _, _, s3 = ab.load_ic_csv(p)
    assert s3 is None
    q = tmp_path / "bad.csv"
    q.write_text("T,Ic\n0.1,1e-6\n0.2,1e-6\n")
    with pytest.raises(ValueError, match="header"):
        ab.load_ic_csv(q)
    q.write_text("T_K,Ic_A\n0.2,1e-6\n0.1,1e-6\n")
    with pytest.raises(ValueError, match="strictly increasing"):
        ab.load_ic_csv(q)
    # plausibility flags: mK-as-K and uA-as-A, flagged not rescaled
    rep = ab.validate_ic_data(np.array([0.05, 40.0]),
                              np.array([5e-6, 5e-3]))
    kinds = {fl[1] for fl in rep["flags"]}
    assert kinds == {"T_large", "Ic_large"}
    rep_ok = ab.validate_ic_data(T, Ic)
    assert rep_ok["flags"] == [] and rep_ok["n_points"] == 12
    with pytest.raises(ValueError):
        ab.validate_ic_data(np.array([0.1, -0.2]), np.array([1e-6, 1e-6]))


def test_trace_csv_round_trip_and_uniformity_refusal(tmp_path):
    rng = np.random.default_rng(1)
    y = rng.normal(size=50)
    dt = 2.5e-6
    p = tmp_path / "trace.csv"
    ab.save_trace_csv(p, dt, y)
    dt2, y2 = ab.load_trace_csv(p)
    assert abs(dt2 - dt) < 1e-18 and np.array_equal(y, y2)
    lines = p.read_text().splitlines()
    lines[10] = "1e-3," + lines[10].split(",")[1]   # break uniformity
    q = tmp_path / "warped.csv"
    q.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="uniform"):
        ab.load_trace_csv(q)


def test_file_to_decoder_pipeline(tmp_path):
    """The complete path: telegraph trace -> file -> loader -> HMM
    decoder, recovering the generating rates as the direct-array path
    does (same arrays, asserted equal)."""
    rng = np.random.default_rng(2)
    f0, tauA, dt = 0.4, 1e-3, 5e-5
    tr = ab.telegraph_traces(f0, tauA, dt, 6000, 1, rng)[:, 0]
    y = tr + rng.normal(0.0, 0.15, tr.size)
    p = tmp_path / "readout.csv"
    ab.save_trace_csv(p, dt, y)
    dt2, y2 = ab.load_trace_csv(p)
    assert np.array_equal(y, y2)
    model, _ = ab.fit_hmm(y2)
    f_hat, tau_hat = model.rates(dt2)
    assert abs(f_hat - f0) < 0.1
    assert abs(tau_hat - tauA) / tauA < 0.35
