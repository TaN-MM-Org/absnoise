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

New in 0.12.0: `dwell_times` (the complete stays in each state of a
decoded path) and `markov_dwell_test`, which checks whether those
stays follow the geometric law of the single-rate Markov model this
module assumes, against a parametric bootstrap that simulates and
decodes the fitted model.

New in 0.17.0: a readout noise width for each state
(`TelegraphHMM(..., sigma=(s0, s1))`, `fit_hmm(..., equal_noise=False)`),
with the exact gradient and error bars for the six parameters; and in
`markov_dwell_test` a second check, the correlation of successive
stays.
"""
from __future__ import annotations

import numpy as np

from .telegraph import flip_probabilities


class TelegraphHMM:
    """Two-state hidden Markov model of a telegraph occupation readout.

    p_up, p_dn: per-sample flip probabilities (0->1 and 1->0);
    means: readout levels (m0, m1) of the two occupation states;
    sigma: Gaussian readout noise per sample, one number for both
    states, or (new in 0.17.0) a pair (s0, s1), one width per state.

    `sigma` keeps what was given (a float, or an array of two widths);
    `sigmas` is always the array (s0, s1) and `equal_noise` tells which
    model it is. With one width the model has five parameters (p_up,
    p_dn, m0, m1, sigma), with two widths six (sigma0, sigma1 in place
    of sigma). A pair of two equal widths is the six-parameter model
    with equal values: it gives the same likelihood as one width.
    """

    def __init__(self, p_up: float, p_dn: float, means, sigma):
        if not (0.0 < p_up < 1.0 and 0.0 < p_dn < 1.0):
            raise ValueError("flip probabilities must lie in (0, 1)")
        self.equal_noise = np.ndim(sigma) == 0
        sg = np.asarray(sigma, dtype=float).ravel()
        if sg.size not in (1, 2) or (not self.equal_noise
                                     and sg.size != 2):
            raise ValueError("sigma must be one number or a pair (s0, s1)")
        # (fixed in 0.17.0) a NaN noise or level was accepted, and every
        # later call failed with "the trace contains non-finite samples"
        if not (np.all(np.isfinite(sg)) and np.all(sg > 0.0)):
            raise ValueError("sigma must be finite and positive")
        self.p_up = float(p_up)
        self.p_dn = float(p_dn)
        self.means = np.asarray(means, dtype=float)
        if self.means.shape != (2,):
            raise ValueError("means must be (m0, m1)")
        if not np.all(np.isfinite(self.means)):
            raise ValueError("the readout levels must be finite")
        if self.equal_noise:
            self.sigma = float(sigma)
            self.sigmas = np.array([self.sigma, self.sigma])
        else:
            self.sigmas = sg.copy()
            self.sigma = sg.copy()

    @classmethod
    def from_rates(cls, f: float, tauA: float, dt: float, means, sigma):
        """Build from the physical parameters, in the convention of
        telegraph.telegraph_traces (exact sampling of the continuous
        process): p_up = f (1 - e^(-dt/tauA)), p_dn = (1-f)(1 - e^(-dt/tauA))."""
        p_up, p_dn = flip_probabilities(f, tauA, dt)
        return cls(p_up, p_dn, means, sigma)

    def rates(self, dt: float):
        """(f, tauA) implied by the flip probabilities at sample time dt:
        f = p_up / s and tauA = -dt / ln(1 - s), s = p_up + p_dn (the
        exact inverse of `from_rates`). Raises ValueError if s >= 1: an
        anti-correlated chain is not a sampled telegraph process, and
        (since 0.16.0) if dt is not finite and positive (a negative dt
        gave a negative tauA)."""
        dt = float(dt)
        if not (np.isfinite(dt) and dt > 0.0):
            raise ValueError("dt must be finite and positive (s)")
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
        sg = self.sigma if self.equal_noise else self.sigmas[None, :]
        z = (y[:, None] - self.means[None, :]) / sg
        return np.exp(-0.5 * z ** 2) / (np.sqrt(2 * np.pi) * sg)

    def _log_emission(self, y):
        """Log Gaussian densities, shape (T, 2); never underflows."""
        y = np.asarray(y, dtype=float).ravel()
        if self.equal_noise:
            z = (y[:, None] - self.means[None, :]) / self.sigma
            return -0.5 * z ** 2 - np.log(np.sqrt(2 * np.pi) * self.sigma)
        z = (y[:, None] - self.means[None, :]) / self.sigmas[None, :]
        return -0.5 * z ** 2 - np.log(np.sqrt(2 * np.pi)
                                      * self.sigmas)[None, :]

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
        if self.equal_noise:
            return np.array([self.p_up, self.p_dn, self.means[0],
                             self.means[1], self.sigma])
        return np.array([self.p_up, self.p_dn, self.means[0],
                         self.means[1], self.sigmas[0], self.sigmas[1]])

    @classmethod
    def _from_params(cls, p):
        if len(p) == 5:
            return cls(p[0], p[1], (p[2], p[3]), p[4])
        return cls(p[0], p[1], (p[2], p[3]), (p[4], p[5]))

    def _names(self):
        if self.equal_noise:
            return ("p_up", "p_dn", "m0", "m1", "sigma")
        return ("p_up", "p_dn", "m0", "m1", "sigma0", "sigma1")

    def score(self, y):
        """Gradient of the log-likelihood with respect to
        (p_up, p_dn, m0, m1, sigma), exact, from one forward-backward
        pass (Fisher's identity: the gradient of the data
        log-likelihood equals the posterior expectation of the gradient
        of the complete-data log-likelihood). For a model with one
        noise width per state (new in 0.17.0) the last entry is
        replaced by the two derivatives with respect to sigma0 and
        sigma1. The tests compare it with finite differences of
        `loglik`."""
        y = np.asarray(y, dtype=float)
        gamma, xi, _ = self.forward_backward(y)
        u, d = self.p_up, self.p_dn
        s = u + d
        n = xi.sum(axis=0)                        # n[from, to]
        g0 = gamma[0]
        # initial state drawn from the stationary law (p_dn, p_up) / s
        du = g0[1] / u - 1.0 / s + n[0, 1] / u - n[0, 0] / (1.0 - u)
        dd = g0[0] / d - 1.0 / s + n[1, 0] / d - n[1, 1] / (1.0 - d)
        r0 = y - self.means[0]
        r1 = y - self.means[1]
        if not self.equal_noise:
            s0, s1 = float(self.sigmas[0]), float(self.sigmas[1])
            dm0 = float(np.dot(gamma[:, 0], r0)) / s0 ** 2
            dm1 = float(np.dot(gamma[:, 1], r1)) / s1 ** 2
            n0, n1 = float(gamma[:, 0].sum()), float(gamma[:, 1].sum())
            ds0 = float(np.dot(gamma[:, 0], r0 * r0)) / s0 ** 3 - n0 / s0
            ds1 = float(np.dot(gamma[:, 1], r1 * r1)) / s1 ** 3 - n1 / s1
            return np.array([du, dd, dm0, dm1, ds0, ds1])
        sg = self.sigma
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
        errors. For a model with one noise width per state (new in
        0.17.0) cov is 6 x 6 and 'sigma0', 'sigma1' replace 'sigma'.
        Raises ValueError if the information matrix is not
        positive definite (the trace does not determine the
        parameters, or the model is not at a maximum).
        """
        y = np.asarray(y, dtype=float)
        p = self._params()
        k_par = p.size
        s_min = float(np.min(self.sigmas))
        h = rel_step * np.array([min(p[0], 1.0 - p[0]),
                                 min(p[1], 1.0 - p[1])]
                                + [s_min] * (k_par - 2))
        H = np.empty((k_par, k_par))
        for k in range(k_par):
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
                f"the trace does not determine all {k_par} parameters, "
                "or this model is not the maximum-likelihood fit "
                "(run fit_hmm first)") from None
        cov = np.linalg.inv(info)
        names = self._names()
        sig = {nm: float(np.sqrt(cov[i, i])) for i, nm in enumerate(names)}
        if dt is not None:
            dt = float(dt)
            if not (np.isfinite(dt) and dt > 0.0):
                raise ValueError("dt must be finite and positive (s)")
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


def dwell_times(path, dt: float = 1.0) -> dict:
    """Complete dwell times of a two-state path (new in 0.12.0).

    path : sequence of 0/1 states, for example `TelegraphHMM.viterbi`.
    dt : sample interval (s); the default 1 gives dwells in samples.

    Returns {0: array, 1: array}: the durations of every stay in state
    0 and in state 1, in time order, as (number of samples) * dt. The
    first and the last stay are left out, because the record cuts them
    (their true length is unknown).
    """
    s = np.asarray(path).ravel()
    if s.size < 1:
        raise ValueError("the path is empty")
    if not np.all((s == 0) | (s == 1)):
        raise ValueError("the path must contain only the states 0 and 1")
    dt = float(dt)
    if not (np.isfinite(dt) and dt > 0.0):
        raise ValueError("dt must be finite and positive")
    s = s.astype(np.int8)
    change = np.flatnonzero(s[1:] != s[:-1]) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [s.size]]))
    states = s[starts]
    lengths, states = lengths[1:-1], states[1:-1]
    return {0: lengths[states == 0] * dt, 1: lengths[states == 1] * dt}


def _dispersion(k):
    """var(k) / (mean(k) (mean(k) - 1)) of dwell lengths k in samples.
    A geometric law P(k) = (1 - p)^(k-1) p has mean 1/p and variance
    (1 - p)/p^2 = mean (mean - 1), so its population value is 1."""
    m = float(np.mean(k))
    return float(np.var(k, ddof=1)) / (m * (m - 1.0))


def _successive_correlation(path):
    """Correlation of each complete stay with the next one (new in
    0.17.0). The complete stays (first and last left out, as in
    `dwell_times`) are taken in time order; each is replaced by its
    rank among the stays of its own state, scaled to (0, 1) (ties get
    their average rank), so that the two states, whose stays have
    different mean lengths, are on the same footing. Returns the
    Pearson correlation of successive scaled ranks (a Spearman-type
    rank correlation), or NaN with fewer than 4 stays. A Markov chain
    has independent stays, so the population value is 0."""
    from scipy.stats import rankdata
    s = np.asarray(path).ravel().astype(np.int8)
    change = np.flatnonzero(s[1:] != s[:-1]) + 1
    starts = np.concatenate([[0], change])
    lengths = np.diff(np.concatenate([starts, [s.size]]))[1:-1]
    states = s[starts][1:-1]
    if lengths.size < 4:
        return float("nan")
    u = np.empty(lengths.size)
    for st in (0, 1):
        sel = states == st
        if np.sum(sel) < 2:
            return float("nan")
        u[sel] = (rankdata(lengths[sel]) - 0.5) / np.sum(sel)
    a, b = u[:-1], u[1:]
    a = a - a.mean()
    b = b - b.mean()
    den = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / den) if den > 0.0 else float("nan")


def _simulate_path(p_up, p_dn, n, rng):
    """A path of n samples of the two-state Markov chain, started from
    its stationary law, built from geometric stays (vectorized)."""
    s = p_up + p_dn
    state = int(rng.random() < p_up / s)
    mean_pair = 1.0 / p_up + 1.0 / p_dn
    out = np.empty(0, dtype=np.int8)
    while out.size < n:
        m = int(2.0 * n / mean_pair) + 8
        st0 = rng.geometric(p_up, m)          # stays in state 0
        st1 = rng.geometric(p_dn, m)          # stays in state 1
        if state == 0:
            lens = np.column_stack([st0, st1]).ravel()
            vals = np.tile(np.array([0, 1], dtype=np.int8), m)
        else:
            lens = np.column_stack([st1, st0]).ravel()
            vals = np.tile(np.array([1, 0], dtype=np.int8), m)
        out = np.concatenate([out, np.repeat(vals, lens)])
        state = int(out[-1]) ^ 1
    return out[:n]


def _viterbi_batch(model, Y):
    """`TelegraphHMM.viterbi` for several traces of equal length at once
    (rows of Y), returning an int8 array of paths with the shape of Y
    (new in 0.14.0, used by `markov_dwell_test`).

    The recursion is the one of `viterbi`, step for step: the same
    additions of the same float64 numbers, the same strict comparisons
    (ties go to state 0) and the same back-tracking, applied to a
    vector of traces instead of one Python float per trace. Additions
    and comparisons of IEEE doubles give the same result in NumPy and
    in Python, so the paths are identical (asserted in the tests)."""
    Y = np.asarray(Y, dtype=float)
    B, T = Y.shape
    if model.equal_noise:
        sg0 = sg1 = model.sigma
    else:
        sg0, sg1 = float(model.sigmas[0]), float(model.sigmas[1])
    const0 = np.log(np.sqrt(2 * np.pi) * sg0)
    const1 = np.log(np.sqrt(2 * np.pi) * sg1)
    z0 = (Y - model.means[0]) / sg0
    z1 = (Y - model.means[1]) / sg1
    lb0 = np.ascontiguousarray((-0.5 * z0 ** 2 - const0).T)    # (T, B)
    lb1 = np.ascontiguousarray((-0.5 * z1 ** 2 - const1).T)
    del z0, z1
    la = np.log(model._transition())
    l00, l01, l10, l11 = (float(la[0, 0]), float(la[0, 1]),
                          float(la[1, 0]), float(la[1, 1]))
    ls = np.log(model._stationary())
    d0 = float(ls[0]) + lb0[0]
    d1 = float(ls[1]) + lb1[0]
    back0 = np.zeros((T, B), dtype=bool)
    back1 = np.zeros((T, B), dtype=bool)
    for t in range(1, T):
        c00 = l00 + d0
        c01 = l01 + d1
        c10 = l10 + d0
        c11 = l11 + d1
        b0 = c01 > c00
        b1 = c11 > c10
        back0[t] = b0
        back1[t] = b1
        d0 = lb0[t] + np.where(b0, c01, c00)
        d1 = lb1[t] + np.where(b1, c11, c10)
    path = np.empty((T, B), dtype=np.int8)
    state = d1 > d0
    path[-1] = state
    for t in range(T - 1, 0, -1):
        state = np.where(state, back1[t], back0[t])
        path[t - 1] = state
    return np.ascontiguousarray(path.T)


def markov_dwell_test(y, model: TelegraphHMM, n_boot: int = 200,
                      seed: int = 0, min_dwells: int = 30) -> dict:
    """Do the dwells in a decoded trace follow the single-rate
    (Markov) telegraph model? (new in 0.12.0)

    The decoder assumes a two-state Markov process: every stay in a
    state ends with the same probability per sample, so the stays are
    geometric (exponential in continuous time), with one correlation
    time. Several trap states, slow drifts of the rates or a dead time
    break this, and then f and tauA from `fit_hmm` describe no single
    process. This test asks whether the data show such a departure.

    y : the readout trace; model : the `TelegraphHMM` fitted to it (by
    `fit_hmm`). The trace is decoded with `model.viterbi`, and for the
    complete stays in each state (see `dwell_times`) the dispersion
    index

        D = var(k) / (mean(k) (mean(k) - 1))

    is computed from the stay lengths k (in samples). A geometric law
    has variance mean (mean - 1), so D is near 1 for a Markov process;
    D > 1 means too many very short and very long stays (several time
    scales), D < 1 too regular stays (for example a dead time).

    The reference distribution of D comes from a parametric bootstrap
    of the whole chain: n_boot traces of the same length are simulated
    from `model` (Markov path plus Gaussian noise) and decoded with
    the same Viterbi step. Decoding errors (missed short stays at low
    signal-to-noise) therefore enter the data and the reference in the
    same way. The p-value of each state is two-sided,
    2 min(P(D* >= D), P(D* <= D)), with the usual +1 correction of a
    finite bootstrap; `p_value` combines the two states (twice the
    smaller one, Bonferroni). A small p-value (say below 0.01) means
    the data are not consistent with the single-rate model; a large one
    means no departure was detected, not that the model is proven.

    A second property is checked since 0.17.0: in a Markov chain each
    stay is independent of the one before. `correlation` is the rank
    correlation of each complete stay with the next one (each stay
    ranked among the stays of its own state; see
    `_successive_correlation`), and `p_correlation` its two-sided
    p-value from the same simulated traces. It detects, for example,
    slow changes of the rates that make long stays follow long stays,
    or stays of the two states that are linked, even when each state
    alone has exactly geometric stays (which the dispersion index
    cannot see). `p_value` is unchanged: it is the dispersion test
    alone, as in earlier releases; combine the two yourself if you use
    both (for example twice the smaller p-value).

    Returns dict(p_value, p_state, dispersion, n_dwells, mean_dwell,
    correlation, p_correlation), p_state to mean_dwell keyed by state
    (0, 1); mean_dwell is in samples.
    Refuses traces with fewer than `min_dwells` complete stays in a
    state (too few to judge a distribution) or a mean stay below 2
    samples (sampling too coarse to resolve the stays).
    """
    if not isinstance(model, TelegraphHMM):
        raise TypeError("model must be a TelegraphHMM (from fit_hmm)")
    y = np.asarray(y, dtype=float).ravel()
    if y.size < 4 or not np.all(np.isfinite(y)):
        raise ValueError("y must hold at least 4 finite samples")
    n_boot, min_dwells = int(n_boot), int(min_dwells)
    if n_boot < 19:
        raise ValueError("n_boot must be at least 19 (a p-value "
                         "resolution of 0.1 for the two-sided test)")
    if min_dwells < 10:
        raise ValueError("min_dwells must be at least 10")

    def stats(path):
        d = dwell_times(path)
        return d, {s: (_dispersion(d[s]) if d[s].size >= 2
                       and d[s].mean() > 1.0 else np.nan)
                   for s in (0, 1)}

    path0 = model.viterbi(y)
    dw, disp = stats(path0)
    corr = _successive_correlation(path0)
    for s in (0, 1):
        if dw[s].size < min_dwells:
            raise ValueError(
                f"only {dw[s].size} complete stays in state {s} (need "
                f"{min_dwells}): too few to judge their distribution; "
                "record a longer trace")
        if dw[s].mean() < 2.0:
            raise ValueError(
                f"the mean stay in state {s} is {dw[s].mean():.2f} "
                "samples: the sampling is too coarse to resolve the "
                "stays; sample faster than the occupation flips")
    rng = np.random.default_rng(seed)
    boot = {0: [], 1: []}
    boot_c = []
    # (changed in 0.14.0) the simulated traces are drawn in the same
    # order from the same stream as before and decoded in batches by
    # `_viterbi_batch`, which returns the same paths as `viterbi`
    chunk = max(1, min(n_boot, (1 << 22) // y.size))
    done = 0
    while done < n_boot:
        k = min(chunk, n_boot - done)
        Yb = np.empty((k, y.size))
        for i in range(k):
            path = _simulate_path(model.p_up, model.p_dn, y.size, rng)
            if model.equal_noise:
                Yb[i] = model.means[path] \
                    + model.sigma * rng.standard_normal(y.size)
            else:
                Yb[i] = model.means[path] \
                    + model.sigmas[path] * rng.standard_normal(y.size)
        for row in _viterbi_batch(model, Yb):
            _, db = stats(row)
            for s in (0, 1):
                boot[s].append(db[s])
            boot_c.append(_successive_correlation(row))
        done += k
    p_state = {}
    for s in (0, 1):
        b = np.asarray(boot[s])
        b = b[np.isfinite(b)]
        if b.size < n_boot // 2:
            raise ValueError("the simulated traces of this model rarely "
                             "contain enough stays; record a longer "
                             "trace")
        hi = (1.0 + np.sum(b >= disp[s])) / (b.size + 1.0)
        lo = (1.0 + np.sum(b <= disp[s])) / (b.size + 1.0)
        p_state[s] = float(min(1.0, 2.0 * min(hi, lo)))
    bc = np.asarray(boot_c)
    bc = bc[np.isfinite(bc)]
    p_corr = float("nan")
    if np.isfinite(corr) and bc.size >= n_boot // 2:
        hi = (1.0 + np.sum(bc >= corr)) / (bc.size + 1.0)
        lo = (1.0 + np.sum(bc <= corr)) / (bc.size + 1.0)
        p_corr = float(min(1.0, 2.0 * min(hi, lo)))
    return dict(p_value=float(min(1.0, 2.0 * min(p_state.values()))),
                p_state=p_state, dispersion=disp,
                n_dwells={s: int(dw[s].size) for s in (0, 1)},
                mean_dwell={s: float(dw[s].mean()) for s in (0, 1)},
                correlation=corr, p_correlation=p_corr)


def fit_hmm(y, n_iter: int = 60, init: TelegraphHMM | None = None,
            tol: float = 1e-8, equal_noise: bool = True):
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

    equal_noise (new in 0.17.0): True (the default, the model of earlier
    releases) fits one Gaussian noise width for both states; False fits
    one width per state, for a readout whose noise differs between the
    two levels (the M step then takes each width from the samples
    weighted by the posterior of its own state). The default start uses
    the pooled width for both. A given `init` decides the model by
    itself: with a pair of widths the fit keeps two, with one width it
    keeps one, whatever `equal_noise` says.
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
        s_init = max(sd, 1e-3 * spread)
        init = TelegraphHMM(0.05 * frac, 0.05 * (1.0 - frac),
                            (m_lo, m_hi),
                            s_init if equal_noise else (s_init, s_init))
    model = init
    per_state = not model.equal_noise
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
        if per_state:
            v0 = float(np.dot(gamma[:, 0], (y - m0) ** 2)
                       / max(occ[0], 1e-300))
            v1 = float(np.dot(gamma[:, 1], (y - m1) ** 2)
                       / max(occ[1], 1e-300))
            sig = (max(np.sqrt(v0), 1e-12), max(np.sqrt(v1), 1e-12))
        else:
            var = float((np.dot(gamma[:, 0], (y - m0) ** 2)
                         + np.dot(gamma[:, 1], (y - m1) ** 2))
                        / max(occ.sum(), 1e-300))
            sig = max(np.sqrt(var), 1e-12)
        model = TelegraphHMM(np.clip(p_up, 1e-8, 1 - 1e-8),
                             np.clip(p_dn, 1e-8, 1 - 1e-8),
                             (m0, m1), sig)
        if len(logliks) > 1 and abs(logliks[-1] - logliks[-2]) < tol * abs(logliks[-1]):
            break
    return model, np.asarray(logliks)
