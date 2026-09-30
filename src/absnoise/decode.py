"""Decoding Andreev occupation from noisy readout traces (hidden Markov).

The telegraph module simulates what the junction does; a real experiment
sees it only through a noisy detector.  This module solves the inverse
problem: given a sampled readout trace y(t) that is a noisy image of a
two-state occupation n(t) with detailed-balance rates
    p_up = f (1 - e^(-dt/tauA))  (0 -> 1),
    p_dn = (1 - f)(1 - e^(-dt/tauA))  (1 -> 0)
(the convention of ``telegraph.telegraph_traces``) and Gaussian
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
readout record, with likelihood-based error bars from
`TelegraphHMM.uncertainties` (new in 0.11.0; before, this docstring
promised error bars that the module did not compute).

Changed in 0.11.0:

* The per-sample flip probabilities are those of the continuous-time
  process sampled every dt, p_up = f (1 - e^(-dt/tauA)) and
  p_dn = (1 - f)(1 - e^(-dt/tauA)) (`telegraph.flip_probabilities`),
  so `rates` returns tauA = -dt / ln(1 - p_up - p_dn). The 0.10.x
  first-order inversion tauA = dt / (p_up + p_dn) overestimated tauA
  by the factor s / (-ln(1 - s)), s = 1 - e^(-dt/tauA): 2.5 % at
  dt = 0.05 tauA, 27 % at dt = 0.5 tauA. f is unchanged.
* The default starting point of `fit_hmm` is the exact two-cluster
  split of the samples (the threshold that minimizes the
  within-cluster sum of squares), which works for any occupation;
  the median split failed when more than half of the samples sat
  exactly at the maximum.
* The forward-backward and Viterbi recursions run on Python floats
  instead of per-sample NumPy calls: on 20 000 samples about 10 times
  faster (forward-backward) and 17 times faster (Viterbi) than
  0.10.x, with the same results to rounding (checked in the tests
  against the 0.10.x matrix formulation).
"""
from __future__ import annotations

import numpy as np

from .telegraph import flip_probabilities


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
        """Build from the physical parameters, in the convention of
        telegraph.telegraph_traces (exact sampling of the continuous
        process): p_up = f (1 - e^(-dt/tauA)), p_dn = (1-f)(1 - e^(-dt/tauA))."""
        p_up, p_dn = flip_probabilities(f, tauA, dt)
        return cls(p_up, p_dn, means, sigma)

    def rates(self, dt: float):
        """(f, tauA) implied by the flip probabilities at sample time dt:
        f = p_up / s and tauA = -dt / ln(1 - s), s = p_up + p_dn (the
        exact inverse of `from_rates`). Raises ValueError if s >= 1: an
        anti-correlated chain is not a sampled telegraph process."""
        s = self.p_up + self.p_dn
        if not s < 1.0:
            raise ValueError(
                f"p_up + p_dn = {s:.6g} >= 1: successive samples are "
                "anti-correlated, which no continuous-time two-state "
                "process produces; tauA is undefined")
        return self.p_up / s, -float(dt) / float(np.log1p(-s))

    # ------------------------------------------------------------------
    def _emission(self, y):
        y = np.asarray(y, dtype=float)
        z = (y[:, None] - self.means[None, :]) / self.sigma
        return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * self.sigma)

    def _log_emission(self, y):
        """Log Gaussian densities, shape (T, 2); never underflows."""
        y = np.asarray(y, dtype=float).ravel()
        z = (y[:, None] - self.means[None, :]) / self.sigma
        return -0.5 * z ** 2 - np.log(np.sqrt(2 * np.pi) * self.sigma)

    def _scaled_emission(self, y):
        """Emissions divided by their larger value at each sample, and
        the log of that divisor. Posteriors are unchanged by a
        per-sample factor; the log-likelihood gets the sum of the
        log-divisors back. This keeps a far outlier (where both raw
        densities underflow to 0) finite: 0.10.1 returned NaN there and
        the first 0.11.0 draft raised ZeroDivisionError."""
        logB = self._log_emission(y)
        if logB.shape[0] < 1:
            raise ValueError("the trace is empty")
        if not np.all(np.isfinite(logB)):
            raise ValueError("the trace contains non-finite samples")
        top = logB.max(axis=1)
        return np.exp(logB - top[:, None]), float(np.sum(top))

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
        B, log_scale = self._scaled_emission(y)   # (T, 2), max 1 per row
        T = B.shape[0]
        u, d = self.p_up, self.p_dn
        a00, a01, a10, a11 = 1.0 - u, d, u, 1.0 - d   # A[i, j] = P(j -> i)
        b0 = B[:, 0].tolist()
        b1 = B[:, 1].tolist()
        al0 = [0.0] * T
        al1 = [0.0] * T
        c = [0.0] * T
        pi0, pi1 = self._stationary()
        x0 = float(pi0) * b0[0]
        x1 = float(pi1) * b1[0]
        ct = x0 + x1
        c[0], al0[0], al1[0] = ct, x0 / ct, x1 / ct
        p0, p1 = al0[0], al1[0]
        for t in range(1, T):
            x0 = b0[t] * (a00 * p0 + a01 * p1)
            x1 = b1[t] * (a10 * p0 + a11 * p1)
            ct = x0 + x1
            p0 = x0 / ct
            p1 = x1 / ct
            c[t], al0[t], al1[t] = ct, p0, p1
        be0 = [1.0] * T
        be1 = [1.0] * T
        q0 = q1 = 1.0
        for t in range(T - 2, -1, -1):
            e0 = b0[t + 1] * q0
            e1 = b1[t + 1] * q1
            ct = c[t + 1]
            q0 = (a00 * e0 + a10 * e1) / ct
            q1 = (a01 * e0 + a11 * e1) / ct
            be0[t], be1[t] = q0, q1
        alpha = np.column_stack([al0, al1])
        beta = np.column_stack([be0, be1])
        cc = np.asarray(c)
        gamma = alpha * beta
        gamma /= gamma.sum(axis=1, keepdims=True)
        A = self._transition()
        m = ((B[1:] * beta[1:])[:, :, None] * A[None, :, :]
             * alpha[:-1][:, None, :]) / cc[1:, None, None]
        xi = np.transpose(m, (0, 2, 1))           # xi[t, from, to]
        return gamma, xi, float(np.sum(np.log(cc))) + log_scale

    def loglik(self, y):
        """Data log-likelihood (forward pass only)."""
        return self.forward_backward(y)[2]

    def posterior(self, y):
        """Posterior occupation probability P(n = 1 | data) per sample."""
        gamma, _, _ = self.forward_backward(y)
        return gamma[:, 1]

    def viterbi(self, y):
        """Most probable state path (int8 array of 0/1)."""
        logB = self._log_emission(y)
        if logB.shape[0] < 1:
            raise ValueError("the trace is empty")
        la = np.log(self._transition())
        l00, l01, l10, l11 = (float(la[0, 0]), float(la[0, 1]),
                              float(la[1, 0]), float(la[1, 1]))
        lb0 = logB[:, 0].tolist()
        lb1 = logB[:, 1].tolist()
        T = len(lb0)
        ls = np.log(self._stationary())
        d0 = float(ls[0]) + lb0[0]
        d1 = float(ls[1]) + lb1[0]
        back0 = bytearray(T)
        back1 = bytearray(T)
        for t in range(1, T):
            # into state 0 from 0 or 1; ties go to the lower index,
            # as numpy.argmax does
            c00, c01 = l00 + d0, l01 + d1
            c10, c11 = l10 + d0, l11 + d1
            if c01 > c00:
                back0[t], n0 = 1, c01
            else:
                n0 = c00
            if c11 > c10:
                back1[t], n1 = 1, c11
            else:
                n1 = c10
            d0 = lb0[t] + n0
            d1 = lb1[t] + n1
        path = np.empty(T, dtype=np.int8)
        state = 1 if d1 > d0 else 0
        path[-1] = state
        for t in range(T - 1, 0, -1):
            state = back1[t] if state else back0[t]
            path[t - 1] = state
        return path

    # ------------------------------------------------------------------
    def _params(self):
        return np.array([self.p_up, self.p_dn, self.means[0],
                         self.means[1], self.sigma])

    @classmethod
    def _from_params(cls, p):
        return cls(p[0], p[1], (p[2], p[3]), p[4])

    def score(self, y):
        """Gradient of the log-likelihood with respect to
        (p_up, p_dn, m0, m1, sigma), exact, from one forward-backward
        pass (Fisher's identity: the gradient of the data
        log-likelihood equals the posterior expectation of the gradient
        of the complete-data log-likelihood). The tests compare it with
        finite differences of `loglik`."""
        y = np.asarray(y, dtype=float)
        gamma, xi, _ = self.forward_backward(y)
        u, d = self.p_up, self.p_dn
        s = u + d
        n = xi.sum(axis=0)                        # n[from, to]
        g0 = gamma[0]
        # initial state drawn from the stationary law (p_dn, p_up) / s
        du = g0[1] / u - 1.0 / s + n[0, 1] / u - n[0, 0] / (1.0 - u)
        dd = g0[0] / d - 1.0 / s + n[1, 0] / d - n[1, 1] / (1.0 - d)
        sg = self.sigma
        r0 = y - self.means[0]
        r1 = y - self.means[1]
        dm0 = float(np.dot(gamma[:, 0], r0)) / sg ** 2
        dm1 = float(np.dot(gamma[:, 1], r1)) / sg ** 2
        ss = float(np.dot(gamma[:, 0], r0 * r0)
                   + np.dot(gamma[:, 1], r1 * r1))
        dsg = ss / sg ** 3 - y.size / sg
        return np.array([du, dd, dm0, dm1, dsg])

    def uncertainties(self, y, dt=None, rel_step=1e-4):
        """Error bars of the fitted parameters from the observed
        information (new in 0.11.0).

        Call it on the model returned by `fit_hmm` with the same trace.
        The observed information is minus the Hessian of the
        log-likelihood at the fit, obtained by central differences of
        the exact gradient `score` (10 forward-backward passes); its
        inverse is the covariance of (p_up, p_dn, m0, m1, sigma) --
        the standard large-sample (asymptotic) estimate, accurate when
        the trace contains many transitions. With dt (s) it also gives
        f and tauA through the exact relations of `rates` (first-order
        error propagation).

        Returns dict(cov, sigma): cov is the 5 x 5 covariance in the
        order above; sigma maps 'p_up', 'p_dn', 'm0', 'm1', 'sigma'
        (and 'f', 'tauA' when dt is given) to one-standard-deviation
        errors. Raises ValueError if the information matrix is not
        positive definite (the trace does not determine the
        parameters, or the model is not at a maximum).
        """
        y = np.asarray(y, dtype=float)
        p = self._params()
        h = rel_step * np.array([min(p[0], 1.0 - p[0]),
                                 min(p[1], 1.0 - p[1]),
                                 p[4], p[4], p[4]])
        H = np.empty((5, 5))
        for k in range(5):
            pp, pm = p.copy(), p.copy()
            pp[k] += h[k]
            pm[k] -= h[k]
            H[:, k] = (self._from_params(pp).score(y)
                       - self._from_params(pm).score(y)) / (2.0 * h[k])
        info = -0.5 * (H + H.T)
        try:
            np.linalg.cholesky(info)
        except np.linalg.LinAlgError:
            raise ValueError(
                "the observed information is not positive definite: "
                "the trace does not determine all five parameters, or "
                "this model is not the maximum-likelihood fit "
                "(run fit_hmm first)") from None
        cov = np.linalg.inv(info)
        names = ("p_up", "p_dn", "m0", "m1", "sigma")
        sig = {nm: float(np.sqrt(cov[i, i])) for i, nm in enumerate(names)}
        if dt is not None:
            dt = float(dt)
            u, d = self.p_up, self.p_dn
            s = u + d
            if not s < 1.0:
                raise ValueError("p_up + p_dn >= 1: tauA is undefined")
            L = float(np.log1p(-s))
            jf = np.array([d / s ** 2, -u / s ** 2])
            dtau_ds = -dt / ((1.0 - s) * L ** 2)
            jt = np.array([dtau_ds, dtau_ds])
            c2 = cov[:2, :2]
            sig["f"] = float(np.sqrt(jf @ c2 @ jf))
            sig["tauA"] = float(np.sqrt(jt @ c2 @ jt))
        return dict(cov=cov, sigma=sig)


def _two_cluster_split(y):
    """Exact 1-D two-means: the threshold between sorted samples that
    minimizes the within-cluster sum of squares (equivalently,
    maximizes the between-cluster variance). Returns (m_lo, m_hi,
    pooled standard deviation, fraction in the upper cluster)."""
    ys = np.sort(y)
    n = ys.size
    cs = np.cumsum(ys)
    k = np.arange(1, n)                            # lower-cluster sizes
    distinct = ys[1:] > ys[:-1]                    # split between values
    if not np.any(distinct):
        raise ValueError("the trace is constant: there is no second "
                         "readout level to fit")
    mean_lo = cs[:-1] / k
    mean_hi = (cs[-1] - cs[:-1]) / (n - k)
    between = k * (n - k) * (mean_hi - mean_lo) ** 2
    between = np.where(distinct, between, -np.inf)
    j = int(np.argmax(between))
    m_lo, m_hi = float(mean_lo[j]), float(mean_hi[j])
    kk = j + 1
    within = (float(np.sum((ys[:kk] - m_lo) ** 2))
              + float(np.sum((ys[kk:] - m_hi) ** 2)))
    return m_lo, m_hi, float(np.sqrt(within / n)), (n - kk) / n


def fit_hmm(y, n_iter: int = 60, init: TelegraphHMM | None = None,
            tol: float = 1e-8):
    """Baum-Welch maximum-likelihood fit of the telegraph HMM to a trace.

    Returns (model, logliks): the fitted :class:`TelegraphHMM` and the
    per-iteration log-likelihoods, which are non-decreasing (up to
    floating-point rounding) by the EM theorem.  The default
    initialization (changed in 0.11.0) splits the samples (clipped to
    their 0.1 % and 99.9 % quantiles, so a few far outliers cannot form
    a cluster of their own) into the two
    clusters with the smallest within-cluster sum of squares, taking
    the cluster means as the levels (state 1 is the upper level) and
    the upper-cluster fraction as the starting occupation. Error bars:
    ``model.uncertainties(y, dt)``.
    """
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4:
        raise ValueError("need at least 4 samples")
    if not np.all(np.isfinite(y)):
        raise ValueError("the trace contains non-finite samples")
    if init is None:
        # the split is made on samples clipped to the 0.1 % and 99.9 %
        # quantiles, so that a few far outliers cannot form a group of
        # their own (the fit itself still sees every sample)
        lo_q, hi_q = np.quantile(y, [0.001, 0.999])
        yc = np.clip(y, lo_q, hi_q)
        if not np.any(yc != yc[0]):
            yc = y
        m_lo, m_hi, sd, frac = _two_cluster_split(yc)
        spread = max(m_hi - m_lo, 1e-300)
        frac = min(max(frac, 0.01), 0.99)
        init = TelegraphHMM(0.05 * frac, 0.05 * (1.0 - frac),
                            (m_lo, m_hi), max(sd, 1e-3 * spread))
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
