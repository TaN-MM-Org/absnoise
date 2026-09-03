"""Decoding Andreev occupation from noisy readout traces (hidden Markov).

The telegraph module simulates what the junction does; a real experiment
sees it only through a noisy detector.  This module solves the inverse
problem: given a sampled readout trace y(t) that is a noisy image of a
two-state occupation n(t) with detailed-balance rates
    p_up = dt f / tauA  (0 -> 1),      p_dn = dt (1 - f) / tauA  (1 -> 0)
(the exact convention of ``telegraph.telegraph_traces``) and Gaussian
readout noise, it computes

* the exact posterior occupation probability at every sample
  (forward-backward algorithm, scaled for numerical stability),
* the most probable state path (Viterbi), and
* maximum-likelihood estimates of the physical parameters (f, tauA, the
  two readout levels and the noise) from the trace alone, by
  expectation-maximization (Baum-Welch), whose log-likelihood is
  guaranteed non-decreasing at every iteration - a property the test
  suite checks, along with recovery of ground-truth rates from the
  package's own telegraph Monte Carlo.

This turns the package's noise model into a measurement tool: occupation
lifetimes and equilibrium occupations become quantities extracted from a
readout record with likelihood-based error bars, not just predicted.
"""
from __future__ import annotations

import numpy as np


class TelegraphHMM:
    """Two-state hidden Markov model of a telegraph occupation readout.

    p_up, p_dn: per-sample flip probabilities (0->1 and 1->0);
    means: readout levels (m0, m1) of the two occupation states;
    sigma: Gaussian readout noise per sample.
    """

    def __init__(self, p_up: float, p_dn: float, means, sigma: float):
        if not (0.0 < p_up < 1.0 and 0.0 < p_dn < 1.0):
            raise ValueError("flip probabilities must lie in (0, 1)")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.p_up = float(p_up)
        self.p_dn = float(p_dn)
        self.means = np.asarray(means, dtype=float)
        if self.means.shape != (2,):
            raise ValueError("means must be (m0, m1)")
        self.sigma = float(sigma)

    @classmethod
    def from_rates(cls, f: float, tauA: float, dt: float, means, sigma: float):
        """Build from the physical parameters, in the exact convention of
        telegraph.telegraph_traces: p_up = dt f / tauA, p_dn = dt (1-f)/tauA."""
        return cls(dt * f / tauA, dt * (1.0 - f) / tauA, means, sigma)

    def rates(self, dt: float):
        """(f, tauA) implied by the flip probabilities at sample time dt."""
        s = self.p_up + self.p_dn
        return self.p_up / s, dt / s

    # ------------------------------------------------------------------
    def _emission(self, y):
        y = np.asarray(y, dtype=float)
        z = (y[:, None] - self.means[None, :]) / self.sigma
        return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * self.sigma)

    def _transition(self):
        return np.array([[1.0 - self.p_up, self.p_dn],
                         [self.p_up, 1.0 - self.p_dn]])

    def _stationary(self):
        s = self.p_up + self.p_dn
        return np.array([self.p_dn / s, self.p_up / s])

    def forward_backward(self, y):
        """Scaled forward-backward pass.

        Returns (gamma, xi, loglik): gamma[t, s] is the exact posterior
        probability of state s at sample t; xi[t, s, s'] the posterior of
        the transition s -> s' between samples t and t+1; loglik the data
        log-likelihood.
        """
        B = self._emission(y)                     # (T, 2)
        A = self._transition()                    # A[i, j] = P(j -> i)
        T = B.shape[0]
        alpha = np.empty((T, 2))
        c = np.empty(T)
        a = self._stationary() * B[0]
        c[0] = a.sum()
        alpha[0] = a / c[0]
        for t in range(1, T):
            a = B[t] * (A @ alpha[t - 1])
            c[t] = a.sum()
            alpha[t] = a / c[t]
        beta = np.empty((T, 2))
        beta[-1] = 1.0
        for t in range(T - 2, -1, -1):
            beta[t] = (A.T @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        xi = np.empty((T - 1, 2, 2))
        for t in range(T - 1):
            m = (B[t + 1] * beta[t + 1])[:, None] * A * alpha[t][None, :]
            xi[t] = (m / c[t + 1]).T              # xi[t, from, to]
        return gamma, xi, float(np.sum(np.log(c)))

    def posterior(self, y):
        """Posterior occupation probability P(n = 1 | data) per sample."""
        gamma, _, _ = self.forward_backward(y)
        return gamma[:, 1]

    def viterbi(self, y):
        """Most probable state path (int8 array of 0/1)."""
        logB = np.log(np.maximum(self._emission(y), 1e-300))
        logA = np.log(self._transition())
        T = logB.shape[0]
        delta = np.log(self._stationary()) + logB[0]
        back = np.empty((T, 2), dtype=np.int8)
        for t in range(1, T):
            cand = logA + delta[None, :]          # cand[i, j]: from j to i
            back[t] = np.argmax(cand, axis=1)
            delta = logB[t] + np.max(cand, axis=1)
        path = np.empty(T, dtype=np.int8)
        path[-1] = int(np.argmax(delta))
        for t in range(T - 2, -1, -1):
            path[t] = back[t + 1][path[t + 1]]
        return path


def fit_hmm(y, n_iter: int = 60, init: TelegraphHMM | None = None,
            tol: float = 1e-8):
    """Baum-Welch maximum-likelihood fit of the telegraph HMM to a trace.

    Returns (model, logliks): the fitted :class:`TelegraphHMM` and the
    per-iteration log-likelihoods, which are non-decreasing (up to
    floating-point rounding) by the EM theorem.  The default
    initialization splits the trace at its median.
    """
    y = np.asarray(y, dtype=float)
    if init is None:
        med = np.median(y)
        lo = y[y <= med]
        hi = y[y > med]
        spread = max(float(y.std()), 1e-12)
        init = TelegraphHMM(0.05, 0.05,
                            (float(lo.mean()), float(hi.mean())),
                            max(0.25 * spread, 1e-6))
    model = init
    logliks = []
    for _ in range(int(n_iter)):
        gamma, xi, ll = model.forward_backward(y)
        logliks.append(ll)
        # M step
        occ = gamma.sum(axis=0)                   # expected time per state
        trans = xi.sum(axis=0)                    # trans[from, to]
        p_up = trans[0, 1] / max(trans[0, 0] + trans[0, 1], 1e-300)
        p_dn = trans[1, 0] / max(trans[1, 0] + trans[1, 1], 1e-300)
        m0 = float(np.dot(gamma[:, 0], y) / max(occ[0], 1e-300))
        m1 = float(np.dot(gamma[:, 1], y) / max(occ[1], 1e-300))
        var = float((np.dot(gamma[:, 0], (y - m0) ** 2)
                     + np.dot(gamma[:, 1], (y - m1) ** 2)) / max(occ.sum(), 1e-300))
        model = TelegraphHMM(np.clip(p_up, 1e-8, 1 - 1e-8),
                             np.clip(p_dn, 1e-8, 1 - 1e-8),
                             (m0, m1), max(np.sqrt(var), 1e-12))
        if len(logliks) > 1 and abs(logliks[-1] - logliks[-2]) < tol * abs(logliks[-1]):
            break
    return model, np.asarray(logliks)
