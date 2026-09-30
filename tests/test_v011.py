"""0.11.0 changes, each checked against an independent reference:

* the matched-filter energy resolution in closed form, against adaptive
  quadrature over all frequencies and two exact limits;
* exact sampling of the telegraph process, against the matrix
  exponential of the rate matrix and the Monte Carlo autocorrelation;
* the exact rate inversion of the HMM decoder, its error bars (exact
  gradient against finite differences, the binomial limit, Monte Carlo
  calibration), its default starting point, and its fast recursions
  against the 0.10.x matrix formulation;
* the log-periodogram statistics of the PSD fit (exact offset symmetry,
  Monte Carlo unbiasedness and chi-square calibration);
* Parseval for `psd_single_sided`, and the budget's input refusals.
"""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import expm
from scipy.special import digamma, polygamma

from absnoise import (RECIPES, SensorBudget, TelegraphHMM, fit_hmm,
                      fit_telegraph_psd, flip_probabilities,
                      matched_filter_sigma, plan_psd_measurement,
                      psd_single_sided, telegraph_psd_model,
                      telegraph_traces)
from absnoise.decode import _two_cluster_split

B = SensorBudget(RECIPES[1])
T = 0.3 * B.recipe.Tc


def _quad_sigma(nep2, t_scale):
    """sigma_E = [4 int_0^inf df / NEP^2(f)]^(-1/2) by adaptive
    quadrature, with f = tan(theta) / (2 pi t_scale) mapping [0, inf)
    onto [0, pi/2) so the slowly decaying tail is integrated too."""
    def g(th):
        f = np.tan(th) / (2 * np.pi * t_scale)
        return 1.0 / nep2(f) / (2 * np.pi * t_scale * np.cos(th) ** 2)
    val, _ = quad(g, 0.0, np.pi / 2, limit=500, epsabs=0.0,
                  epsrel=1e-13)
    return 1.0 / np.sqrt(4.0 * val)


# ---- energy resolution --------------------------------------------------

@pytest.mark.parametrize("P, Q, W, tth, tA", [
    (1.0, 1.0, 0.0, 1.0, 1.0),
    (3.0, 0.2, 0.0, 2e-9, 1e-6),
    (1.0, 0.5, 2.0, 1e-9, 3e-8),
    (0.0, 1.0, 1e-3, 1e-6, 1e-9),
    (50.0, 1e-3, 1e-4, 1e-9, 1e-9),
])
def test_closed_form_matches_quadrature(P, Q, W, tth, tA):
    def nep2(f):
        u = 1.0 + (2 * np.pi * f * tth) ** 2
        v = 1.0 + (2 * np.pi * f * tA) ** 2
        return P + Q * u + W * u * v
    ref = _quad_sigma(nep2, tth)
    assert abs(matched_filter_sigma(P, Q, W, tth, tA) / ref - 1) < 1e-9


def test_closed_form_exact_limits():
    # occupation noise only: sigma^2 = Q tau_th (the closed form of
    # energy_resolution_analytic_A)
    assert abs(matched_filter_sigma(0.0, 2.5, 0.0, 3e-9, 1e-6) ** 2
               / (2.5 * 3e-9) - 1) < 1e-14
    # readout floor only: int_0^inf dw / ((1 + w^2 a^2)(1 + w^2 b^2))
    # = pi / (2 (a + b)), so sigma^2 = W (tau_th + tauA)
    assert abs(matched_filter_sigma(0.0, 0.0, 4.0, 3e-9, 1e-6) ** 2
               / (4.0 * (3e-9 + 1e-6)) - 1) < 1e-14
    with pytest.raises(ValueError):
        matched_filter_sigma(1.0, 0.0, 0.0, 1e-9, 1e-6)   # unbounded
    with pytest.raises(ValueError):
        matched_filter_sigma(1.0, -1.0, 0.0, 1e-9, 1e-6)


@pytest.mark.parametrize("tauA", [1e-6, 1e-9])
@pytest.mark.parametrize("S_ro", [0.0, 1e-22])
@pytest.mark.parametrize("which", ["L", "I"])
def test_energy_resolution_matches_quadrature_of_nep(tauA, S_ro, which):
    """Two code paths: the closed form inside energy_resolution, and
    adaptive quadrature over all frequencies of the NEP spectrum."""
    exact = B.energy_resolution(T, tauA, S_ro_y=S_ro, which=which)

    def nep2(f):
        return B.nep_spectrum(T, tauA, np.atleast_1d(f), S_ro_y=S_ro,
                              which=which)["total"][0] ** 2
    ref = _quad_sigma(nep2, B.tau_th(T))
    assert abs(exact / ref - 1) < 1e-8
    # adding phonon and readout noise never helps
    assert exact >= B.energy_resolution_analytic_A(T, tauA, which) \
        * (1 - 1e-12)


def test_grid_method_overestimates_as_documented():
    """The 0.10.x band-limited integral (method="grid") lies above the
    exact value by the amounts the docstring and CHANGELOG quote."""
    for tauA, lo, hi in ((1e-6, 0.033, 0.034), (1e-9, 0.106, 0.108)):
        ex = B.energy_resolution(T, tauA)
        gr = B.energy_resolution(T, tauA, method="grid")
        assert lo < gr / ex - 1 < hi


def test_budget_refuses_what_it_cannot_evaluate():
    Tc = B.recipe.Tc
    for bad_T in (0.0, -0.1, Tc, 1.2 * Tc, np.nan):
        with pytest.raises(ValueError):
            B.dT_andreev(bad_T, 1e-6, 1.0)
        with pytest.raises(ValueError):
            B.energy_resolution(bad_T, 1e-6)
    for bad in (0.0, -1e-6, np.inf):
        with pytest.raises(ValueError):
            B.energy_resolution(T, bad)
        with pytest.raises(ValueError):
            B.dT_andreev(T, 1e-6, bad)
        with pytest.raises(ValueError):
            B.nep_spectrum(T, bad, np.array([1.0]))
    with pytest.raises(ValueError):
        B.dT_phonon(0.0, 1.0)
    with pytest.raises(ValueError):
        B.energy_resolution(T, 1e-6, which="X")
    with pytest.raises(ValueError):
        B.energy_resolution(T, 1e-6, method="simpson")
    with pytest.raises(ValueError):
        B.energy_resolution(T, 1e-6, S_ro_y=-1.0)
    # the phonon part does not need the gap
    assert np.isfinite(B.dT_phonon(1.2 * Tc, 1.0))


# ---- exact telegraph sampling -------------------------------------------

@pytest.mark.parametrize("f", [0.02, 0.3, 0.9])
@pytest.mark.parametrize("x", [1e-6, 0.1, 1.0, 7.0])
def test_flip_probabilities_are_the_matrix_exponential(f, x):
    tau, dt = 2.0, 2.0 * x
    Qm = np.array([[-f, 1.0 - f], [f, -(1.0 - f)]]) / tau  # Q[i,j]: j->i
    P = expm(Qm * dt)
    p_up, p_dn = flip_probabilities(f, tau, dt)
    assert abs(p_up / P[1, 0] - 1) < 1e-9
    assert abs(p_dn / P[0, 1] - 1) < 1e-9
    # small-step limit: the 0.10.x first-order values dt f / tau
    if x < 1e-3:
        assert abs(p_up / (dt * f / tau) - 1) < 1e-5


def test_coarse_sampling_autocorrelation_is_exact():
    """dt = tauA was refused before 0.11.0 (the first-order flip
    probabilities would exceed 0.12, and give zero correlation between
    samples); the exact sampling gives autocorrelation e^-k at lag k."""
    rng = np.random.default_rng(3)
    f = 0.4
    tr = telegraph_traces(f, 1.0, 1.0, 20000, 64, rng).astype(float)
    assert abs(tr.mean() - f) < 0.01
    x = tr - f
    var = f * (1 - f)
    for k in (1, 2, 3):
        rho = float(np.mean(x[k:] * x[:-k])) / var
        assert abs(rho - np.exp(-k)) < 0.01
    with pytest.raises(ValueError):
        telegraph_traces(1.5, 1.0, 0.1, 10, 1, rng)
    with pytest.raises(ValueError):
        telegraph_traces(0.3, 1.0, 0.0, 10, 1, rng)


def test_hmm_rates_are_the_exact_inverse():
    for f, tau, dt in ((0.3, 1e-3, 5e-5), (0.3, 1.0, 2.0),
                       (0.8, 1.0, 1e-7)):
        hmm = TelegraphHMM.from_rates(f, tau, dt, (0.0, 1.0), 0.2)
        assert (hmm.p_up, hmm.p_dn) == flip_probabilities(f, tau, dt)
        f2, tau2 = hmm.rates(dt)
        assert abs(f2 - f) < 1e-12 and abs(tau2 / tau - 1) < 1e-9
    with pytest.raises(ValueError, match="anti-correlated"):
        TelegraphHMM(0.6, 0.6, (0.0, 1.0), 0.1).rates(1.0)


def test_coarse_trace_fit_recovers_tau_within_its_error_bar():
    """dt = tauA / 2: the exact inversion recovers tauA within 4 of its
    own error bars; the 0.10.x inversion dt / (p_up + p_dn) lands about
    27 % high on the same fit."""
    rng = np.random.default_rng(5)
    dt = 0.5
    n = telegraph_traces(0.3, 1.0, dt, 8000, 1, rng)[:, 0]
    y = n + rng.normal(0.0, 0.3, n.size)
    model, _ = fit_hmm(y)
    f_hat, tau_hat = model.rates(dt)
    sig = model.uncertainties(y, dt=dt)["sigma"]
    assert abs(tau_hat - 1.0) < 4 * sig["tauA"]
    assert abs(f_hat - 0.3) < 4 * sig["f"]
    old = dt / (model.p_up + model.p_dn)
    assert old / tau_hat - 1 > 0.2


# ---- HMM error bars, start and speed ------------------------------------

@pytest.fixture(scope="module")
def hmm_trace():
    rng = np.random.default_rng(0)
    n = telegraph_traces(0.3, 1e-3, 1e-4, 6000, 1, rng)[:, 0]
    return n, n + rng.normal(0.0, 0.4, n.size)


def test_score_is_the_gradient_of_the_loglikelihood(hmm_trace):
    """Fisher's identity (one forward-backward pass) against central
    differences of the log-likelihood itself."""
    _, y = hmm_trace
    model = TelegraphHMM(0.03, 0.06, (0.05, 0.9), 0.45)
    p = model._params()
    num = np.empty(5)
    for k in range(5):
        h = 1e-6 * abs(p[k])
        pp, pm = p.copy(), p.copy()
        pp[k] += h
        pm[k] -= h
        num[k] = (TelegraphHMM._from_params(pp).loglik(y)
                  - TelegraphHMM._from_params(pm).loglik(y)) / (2 * h)
    assert np.allclose(model.score(y), num, rtol=1e-5, atol=0.0)


def test_error_bars_reach_the_binomial_limit():
    """At high signal-to-noise the states are known, and the flip
    probabilities are binomial estimates: sigma(p_up) =
    sqrt(p (1 - p) / N0), N0 the number of samples in state 0 that
    have a successor (N1 likewise for p_dn)."""
    rng = np.random.default_rng(11)
    n = telegraph_traces(0.3, 1e-3, 1e-4, 20000, 1, rng)[:, 0]
    y = n + rng.normal(0.0, 0.05, n.size)
    model, _ = fit_hmm(y)
    sig = model.uncertainties(y)["sigma"]
    N0 = np.sum(n[:-1] == 0)
    N1 = np.sum(n[:-1] == 1)
    ref_up = np.sqrt(model.p_up * (1 - model.p_up) / N0)
    ref_dn = np.sqrt(model.p_dn * (1 - model.p_dn) / N1)
    assert abs(sig["p_up"] / ref_up - 1) < 5e-3
    assert abs(sig["p_dn"] / ref_dn - 1) < 5e-3


def test_error_bars_are_calibrated_by_monte_carlo():
    """30 seeded traces: the errors of the fitted tauA and f, each in
    units of that trace's own reported error bar, have a standard
    deviation near 1 (the standard error of that check is about 0.13)."""
    rng = np.random.default_rng(3)
    dt, z_tau, z_f = 0.2, [], []
    for _ in range(30):
        n = telegraph_traces(0.3, 1.0, dt, 3000, 1, rng)[:, 0]
        y = n + rng.normal(0.0, 0.3, n.size)
        model, _ = fit_hmm(y)
        f_hat, tau_hat = model.rates(dt)
        s = model.uncertainties(y, dt=dt)["sigma"]
        z_tau.append((tau_hat - 1.0) / s["tauA"])
        z_f.append((f_hat - 0.3) / s["f"])
    for z in (np.array(z_tau), np.array(z_f)):
        assert 0.65 < z.std(ddof=1) < 1.4
        assert abs(z.mean()) < 0.6


def test_default_start_handles_a_mostly_filled_noise_free_trace():
    """The 0.10.x median split failed here (more than half of the
    samples exactly at the maximum); the two-cluster split does not."""
    rng = np.random.default_rng(8)
    n = telegraph_traces(0.9, 1e-3, 1e-4, 4000, 1, rng)[:, 0]
    y = 2.0 + 3.0 * n.astype(float)
    model, _ = fit_hmm(y)
    assert abs(model.means[0] - 2.0) < 1e-6
    assert abs(model.means[1] - 5.0) < 1e-6
    assert np.array_equal(model.viterbi(y), n)
    with pytest.raises(ValueError, match="constant"):
        fit_hmm(np.ones(100))
    with pytest.raises(ValueError, match="non-finite"):
        fit_hmm(np.array([0.0, 1.0, np.nan, 1.0, 0.0]))


def _log_forward(hmm, y):
    """Independent reference: the forward recursion in log space with
    logsumexp, straight from the Gaussian log-densities."""
    from scipy.special import logsumexp
    z = (np.asarray(y)[:, None] - hmm.means[None, :]) / hmm.sigma
    lb = -0.5 * z ** 2 - np.log(np.sqrt(2 * np.pi) * hmm.sigma)
    la = np.log(hmm._transition())                 # la[i, j]: j -> i
    a = np.log(hmm._stationary()) + lb[0]
    for t in range(1, lb.shape[0]):
        a = lb[t] + logsumexp(la + a[None, :], axis=1)
    return float(logsumexp(a))


def test_far_outlier_does_not_underflow():
    """A 200-sigma sample makes both Gaussian densities underflow to 0
    (0.10.1 returned NaN, the first 0.11.0 draft ZeroDivisionError).
    The scaled emissions keep everything finite, and the log-likelihood
    equals a log-space forward recursion to 1e-10."""
    rng = np.random.default_rng(0)
    n = telegraph_traces(0.3, 1e-3, 1e-4, 4000, 1, rng)[:, 0]
    y = n + rng.normal(0.0, 0.2, n.size)
    y[1000] = 1.0 + 200 * 0.2
    hmm = TelegraphHMM.from_rates(0.3, 1e-3, 1e-4, (0.0, 1.0), 0.2)
    assert hmm._emission(y)[1000].max() == 0.0     # the raw underflow
    gamma, xi, ll = hmm.forward_backward(y)
    assert np.all(np.isfinite(gamma)) and np.all(np.isfinite(xi))
    assert abs(ll / _log_forward(hmm, y) - 1) < 1e-10
    assert gamma[1000, 1] > 0.999                  # nearer the upper level
    assert (hmm.viterbi(y) == n).mean() > 0.99
    # the full fit runs, its likelihood never decreases, and the rates
    # are defined; the Gaussian model widens its noise to absorb the
    # outlier (0.2 -> about 0.7; see the README limits)
    model, lls = fit_hmm(y)
    assert np.all(np.isfinite(lls))
    assert np.all(np.diff(lls) > -1e-6 * np.abs(lls[1:]))
    f_hat, tau_hat = model.rates(1e-4)
    assert 0.0 < f_hat < 1.0 and tau_hat > 0.0
    assert model.means[0] < 0.2 and model.means[1] > 0.8


def test_single_sample_trace():
    """One sample (accepted by 0.10.1, briefly refused in the 0.11.0
    draft): the posterior is the stationary law times the emission,
    normalized, and the log-likelihood is log sum_s pi_s N(y; m_s, s)."""
    hmm = TelegraphHMM(0.1, 0.3, (0.0, 1.0), 0.4)
    y = np.array([0.35])
    gamma, xi, ll = hmm.forward_backward(y)
    pi = hmm._stationary()
    dens = np.exp(-0.5 * ((y[0] - hmm.means) / 0.4) ** 2) \
        / (np.sqrt(2 * np.pi) * 0.4)
    assert np.allclose(gamma[0], pi * dens / np.sum(pi * dens),
                       rtol=1e-14)
    assert xi.shape == (0, 2, 2)
    assert abs(ll - np.log(np.sum(pi * dens))) < 1e-14
    assert hmm.viterbi(y)[0] == int(np.argmax(np.log(pi) + np.log(dens)))
    with pytest.raises(ValueError, match="empty"):
        hmm.forward_backward(np.array([]))


def test_two_cluster_split_is_the_brute_force_optimum():
    rng = np.random.default_rng(2)
    y = np.concatenate([rng.normal(0, 0.3, 70), rng.normal(1, 0.3, 30)])
    m_lo, m_hi, sd, frac = _two_cluster_split(y)
    ys = np.sort(y)
    best = min((np.sum((ys[:k] - ys[:k].mean()) ** 2)
                + np.sum((ys[k:] - ys[k:].mean()) ** 2), k)
               for k in range(1, ys.size))
    k = best[1]
    assert abs(m_lo - ys[:k].mean()) < 1e-12
    assert abs(m_hi - ys[k:].mean()) < 1e-12
    assert abs(sd - np.sqrt(best[0] / ys.size)) < 1e-12
    assert frac == (ys.size - k) / ys.size


def _reference_forward_backward(hmm, y):
    """The 0.10.1 matrix formulation, kept here as an independent
    reference for the scalar recursions."""
    B_ = hmm._emission(y)
    A = hmm._transition()
    n = B_.shape[0]
    alpha, c = np.empty((n, 2)), np.empty(n)
    a = hmm._stationary() * B_[0]
    c[0] = a.sum()
    alpha[0] = a / c[0]
    for t in range(1, n):
        a = B_[t] * (A @ alpha[t - 1])
        c[t] = a.sum()
        alpha[t] = a / c[t]
    beta = np.empty((n, 2))
    beta[-1] = 1.0
    for t in range(n - 2, -1, -1):
        beta[t] = (A.T @ (B_[t + 1] * beta[t + 1])) / c[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi = np.empty((n - 1, 2, 2))
    for t in range(n - 1):
        m = (B_[t + 1] * beta[t + 1])[:, None] * A * alpha[t][None, :]
        xi[t] = (m / c[t + 1]).T
    logB = np.log(np.maximum(B_, 1e-300))
    logA = np.log(A)
    delta = np.log(hmm._stationary()) + logB[0]
    back = np.empty((n, 2), dtype=np.int8)
    for t in range(1, n):
        cand = logA + delta[None, :]
        back[t] = np.argmax(cand, axis=1)
        delta = logB[t] + np.max(cand, axis=1)
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(delta))
    for t in range(n - 2, -1, -1):
        path[t] = back[t + 1][path[t + 1]]
    return gamma, xi, float(np.sum(np.log(c))), path


def test_fast_recursions_match_the_matrix_formulation(hmm_trace):
    _, y = hmm_trace
    hmm = TelegraphHMM(0.02, 0.05, (0.0, 1.0), 0.4)
    g, x, ll = hmm.forward_backward(y)
    g0, x0, ll0, path0 = _reference_forward_backward(hmm, y)
    assert np.abs(g - g0).max() < 1e-12
    assert np.abs(x - x0).max() < 1e-12
    assert abs(ll - ll0) < 1e-12 * abs(ll0)
    assert np.array_equal(hmm.viterbi(y), path0)


# ---- PSD fit statistics --------------------------------------------------

F = np.geomspace(50.0, 20000.0, 48)
S0, TAU, FLOOR = 1e-3, 3e-4, 1e-5


def test_log_offset_is_an_exact_rescaling():
    """With n_avg the fit removes the constant psi(n) - ln n from
    log S: tau is unchanged and S0 and the floor scale together by
    exp(ln n - psi(n)). (Tolerance 1e-7: the optimizer's stopping
    point, about 3e-9 here, not the offset.)"""
    rng = np.random.default_rng(4)
    S = telegraph_psd_model(F, S0, TAU, FLOOR) * rng.gamma(4, 0.25, F.size)
    a = fit_telegraph_psd(F, S)
    for n in (1, 4, 64):
        b = fit_telegraph_psd(F, S, n_avg=n)
        k = np.exp(np.log(n) - digamma(n))
        assert abs(b.tau_s / a.tau_s - 1) < 1e-7
        assert abs(b.S0 / (k * a.S0) - 1) < 1e-7
        assert abs(b.floor / (k * a.floor) - 1) < 1e-7
        rss = float(np.sum((np.log(b.model) - np.log(S)
                            + digamma(n) - np.log(n)) ** 2))
        assert abs(b.chi2 / (rss / polygamma(1, n)) - 1) < 1e-9


def test_averaged_periodogram_monte_carlo():
    """300 spectra drawn with the exact scatter of an average of two
    periodograms (S times Gamma(2, 1/2)). In 0.10.x the mean log S0
    error was psi(2) - ln 2 = -0.27 and chi2/dof was 2 psi'(2) = 1.29."""
    n_avg = 2
    rng = np.random.default_rng(9)
    S_true = telegraph_psd_model(F, S0, TAU, FLOOR)
    plan = plan_psd_measurement(F, n_avg, S0, TAU, FLOOR)
    lS0, lfl, chi, taus = [], [], [], []
    failed = 0
    for _ in range(300):
        S = S_true * rng.gamma(n_avg, 1.0 / n_avg, F.size)
        # a fit can be refused when the scatter pushes the fitted knee
        # outside the band's factor-2 margins (the fit's own
        # identifiability rule) or the optimizer fails; with this seed
        # and band none is refused, and the assertion keeps it that way
        # rather than silently dropping draws
        try:
            fit = fit_telegraph_psd(F, S, n_avg=n_avg)
        except (ValueError, RuntimeError):
            failed += 1
            continue
        lS0.append(np.log(fit.S0 / S0))
        lfl.append(np.log(fit.floor / FLOOR))
        chi.append(fit.chi2 / fit.chi2_dof)
        taus.append(fit.tau_s)
    assert failed == 0
    lS0 = np.array(lS0)
    se = lS0.std() / np.sqrt(lS0.size)
    assert abs(lS0.mean()) < 3 * se            # se is about 0.012
    # the floor keeps a small bias from the nonlinearity of the fit
    # (about -5 % here), far below the 0.10.x offset of -27 %
    assert abs(np.mean(lfl)) < 0.1
    assert abs(np.mean(chi) - 1.0) < 0.06
    assert abs(np.std(taus, ddof=1) / plan["sigma"]["tau_s"] - 1) < 0.2


# ---- periodogram normalization -------------------------------------------

@pytest.mark.parametrize("n", [1000, 1001])
def test_psd_single_sided_obeys_parseval(n):
    rng = np.random.default_rng(n)
    x = rng.normal(size=n) + 0.3
    dt = 1e-3
    fr, S = psd_single_sided(x, dt)
    df = 1.0 / (n * dt)
    assert abs(np.sum(S) * df / np.var(x) - 1) < 1e-12
    assert fr[1] - fr[0] == pytest.approx(df, rel=1e-12)
