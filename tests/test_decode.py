"""HMM decoding validated against the package's own telegraph Monte Carlo:
state recovery, posterior calibration, EM parameter recovery within the
transition-count statistical error, and the EM monotonicity theorem."""
import numpy as np
import pytest

from absnoise.decode import TelegraphHMM, fit_hmm
from absnoise.telegraph import telegraph_traces

F, TAU_A, DT, N_STEPS = 0.3, 1.0e-3, 5e-5, 40000


@pytest.fixture(scope="module")
def trace():
    rng = np.random.default_rng(0)
    n = telegraph_traces(F, TAU_A, DT, N_STEPS, 1, rng)[:, 0]
    return n, rng


def test_rates_roundtrip():
    hmm = TelegraphHMM.from_rates(F, TAU_A, DT, (0.0, 1.0), 0.2)
    f_back, tau_back = hmm.rates(DT)
    assert np.isclose(f_back, F, rtol=1e-12)
    assert np.isclose(tau_back, TAU_A, rtol=1e-12)


def test_high_snr_state_recovery(trace):
    n, rng = trace
    y = n + rng.normal(0, 0.15, N_STEPS)
    hmm = TelegraphHMM.from_rates(F, TAU_A, DT, (0.0, 1.0), 0.15)
    assert (hmm.viterbi(y) == n).mean() > 0.999
    assert ((hmm.posterior(y) > 0.5) == n).mean() > 0.999


def test_posterior_is_calibrated_at_moderate_snr(trace):
    """The mean maximum posterior equals the realized accuracy to within
    a percent: the decoder knows how often it is right."""
    n, rng = trace
    y = n + rng.normal(0, 0.5, N_STEPS)
    hmm = TelegraphHMM.from_rates(F, TAU_A, DT, (0.0, 1.0), 0.5)
    post = hmm.posterior(y)
    acc = ((post > 0.5) == n).mean()
    confidence = np.maximum(post, 1.0 - post).mean()
    assert acc > 0.95
    assert abs(confidence - acc) < 0.01


def test_em_recovers_ground_truth_within_statistical_error(trace):
    """f and tauA fitted from the raw noisy trace alone land within a few
    times the 1/sqrt(N_transitions) statistical error of the truth."""
    n, rng = trace
    y = n + rng.normal(0, 0.3, N_STEPS)
    model, lls = fit_hmm(y)
    f_hat, tau_hat = model.rates(DT)
    n_trans = int(np.sum(np.abs(np.diff(n.astype(int)))))
    rel = 1.0 / np.sqrt(n_trans)
    assert abs(f_hat - F) < 4 * rel * F
    assert abs(tau_hat - TAU_A) < 4 * rel * TAU_A
    assert abs(model.means[0] - 0.0) < 0.02
    assert abs(model.means[1] - 1.0) < 0.02
    assert abs(model.sigma - 0.3) < 0.02


def test_em_loglikelihood_is_monotone(trace):
    n, rng = trace
    y = n + rng.normal(0, 0.4, N_STEPS)
    _, lls = fit_hmm(y, n_iter=40)
    diffs = np.diff(lls)
    assert np.all(diffs > -1e-6 * np.abs(lls[1:]))


def test_forward_backward_posteriors_normalize(trace):
    n, rng = trace
    y = (n + rng.normal(0, 0.4, N_STEPS))[:2000]
    hmm = TelegraphHMM.from_rates(F, TAU_A, DT, (0.0, 1.0), 0.4)
    gamma, xi, ll = hmm.forward_backward(y)
    assert np.allclose(gamma.sum(axis=1), 1.0)
    assert np.allclose(xi.sum(axis=(1, 2)), 1.0)
    assert np.isfinite(ll)
    # xi marginals agree with gamma
    assert np.allclose(xi.sum(axis=2), gamma[:-1], atol=1e-9)


def test_input_validation():
    with pytest.raises(ValueError):
        TelegraphHMM(0.0, 0.1, (0.0, 1.0), 0.1)
    with pytest.raises(ValueError):
        TelegraphHMM(0.1, 0.1, (0.0, 1.0), 0.0)
    with pytest.raises(ValueError):
        TelegraphHMM(0.1, 0.1, (0.0, 1.0, 2.0), 0.1)
