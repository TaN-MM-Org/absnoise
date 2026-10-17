"""Generalizations that remove the stated limitations of the baseline
occupation-noise model.

1. Four-state channel (both spins) with single-quasiparticle AND pair
   (two-quasiparticle) exchange processes, exact spectrum from the
   master-equation generator. Detailed balance fixes all rates:
       spin flip 0->1 at Gs f,  1->0 at Gs (1-f)          (singles)
       (00)->(11) at Gp f^2,  (11)->(00) at Gp (1-f)^2     (pairs)
   The equilibrium variance of sigma = 1 - n_up - n_down is 2f(1-f)
   independent of the dynamics; pair processes only add relaxation
   channels, so the effective correlation time
       tau_eff := S_sigma(0) / (4 var sigma)
   can only decrease. The Cauchy-Schwarz bound therefore holds with
   tau_A -> tau_eff per level, and the single-quasiparticle value is the
   worst case (checked numerically in the tests: tau_eff does not rise
   as the pair rate Gp grows; there is no Monte Carlo of the four-state
   channel).

2. Energy-dependent activated exchange, tau(E) = tau0 exp[(Delta-E)/kBT]
   (removal of a trapped quasiparticle requires bridging Delta - E;
   detailed balance then fixes the in-rate), used for the scenario-B
   analysis of the phase-bias route.

3. Nonequilibrium occupation floor: a steady-state trapped probability
   q >= f adds occupation variance 2q(1-q) without adding thermal
   responsivity, degrading the resolution by sqrt[q(1-q)/f(1-f)].
"""

import numpy as np

from .constants import KB


def channel_generator(f, Gs, Gp):
    """Rate matrix W (W[i,j] = rate j->i) for states [00, 01, 10, 11]."""
    W = np.zeros((4, 4))
    up, dn = Gs * f, Gs * (1.0 - f)
    # single-spin flips: 00<->01, 00<->10, 01<->11, 10<->11
    W[1, 0] += up; W[2, 0] += up
    W[0, 1] += dn; W[3, 1] += up
    W[0, 2] += dn; W[3, 2] += up
    W[1, 3] += dn; W[2, 3] += dn
    # pair processes: 00<->11
    W[3, 0] += Gp * f * f
    W[0, 3] += Gp * (1.0 - f) ** 2
    np.fill_diagonal(W, 0.0)
    np.fill_diagonal(W, -W.sum(axis=0))
    return W


def sigma_spectrum(f: float, Gs: float, Gp: float, omegas):
    """Single-sided PSD S_sigma(omega) of sigma = 1 - n_up - n_dn, and
    (S0, var, tau_eff). Uses detailed-balance symmetrization for a
    stable eigendecomposition.

    f in [0, 1]; Gs > 0 and Gp >= 0 finite rates (any unit; omega in
    the same unit). Changed in 0.13.0: the symmetrized generator
    D^(-1/2) W D^(1/2) (D = diag of the stationary law) is built from
    its closed-form entries, e.g. Gs sqrt(f (1 - f)) and Gp f (1 - f),
    instead of dividing by sqrt(pi); and the stationary mode is the
    eigenvector along sqrt(pi) instead of every eigenvalue above
    -1e-14. Up to 0.12.1 an occupation f below about 1e-162 (f^2
    underflows) or f = 0 or 1 stopped with an AssertionError, and
    rates below about 1e-14 in the chosen unit dropped every mode
    (S0 = 0, tau_eff = 0 instead of 1/Gs). Other results agree with
    0.12.1 to rounding (tested to 1e-12).
    """
    f, Gs, Gp = float(f), float(Gs), float(Gp)
    if not (0.0 <= f <= 1.0):
        raise ValueError("occupation f must lie in [0, 1]")
    if not (np.isfinite(Gs) and Gs > 0.0 and np.isfinite(Gp)
            and Gp >= 0.0):
        raise ValueError("rates must be finite, Gs > 0 and Gp >= 0")
    g = 1.0 - f
    rf = np.sqrt(f * g)
    d = np.array([g, rf, rf, f])            # sqrt of the stationary law
    # sigma = 1 - n_up - n_dn takes the values (1, 0, 0, -1)
    # detailed-balance symmetrization A = D^{-1/2} W D^{1/2}: the
    # off-diagonal entries are sqrt(W[i, j] W[j, i])
    s1, sp = Gs * rf, Gp * f * g
    Ws = np.array([
        [-(2.0 * Gs * f + Gp * f * f), s1, s1, sp],
        [s1, -Gs, 0.0, s1],
        [s1, 0.0, -Gs, s1],
        [sp, s1, s1, -(2.0 * Gs * g + Gp * g * g)]])
    lam, U = np.linalg.eigh(Ws)             # lam <= 0
    # weighted fluctuation vector d * (sig - <sig>), <sig> = 1 - 2f,
    # with each entry written without cancellation
    x = d * np.array([2.0 * f, f - g, f - g, -2.0 * g])
    w = (U.T @ x) ** 2                      # mode weights, sum = var
    var = float(np.sum(w))
    k0 = int(np.argmax(np.abs(U.T @ d)))    # the stationary mode
    om = np.asarray(omegas, dtype=float)
    S = np.zeros_like(om)
    S0 = 0.0
    for k, (wk, lk) in enumerate(zip(w, lam)):
        if k != k0 and lk < 0.0:
            S += 4.0 * wk * (-lk) / (lk ** 2 + om ** 2)
            S0 += 4.0 * wk / (-lk)
    tau_eff = S0 / (4.0 * var) if var > 0 else 0.0
    return S, float(S0), var, float(tau_eff)


def tau_activated(E, Delta, T, tau0):
    """Scenario-B exchange time tau(E) = tau0 exp[(Delta - E)/kB T]."""
    return tau0 * np.exp(np.clip((Delta - E) / (KB * T), 0.0, 500.0))


def noneq_penalty(f, q):
    """Resolution degradation factor sqrt[q(1-q) / f(1-f)] from a
    nonequilibrium steady-state occupation q >= f."""
    f = np.asarray(f, dtype=float)
    q = np.asarray(q, dtype=float)
    return np.sqrt(np.maximum(q * (1 - q), 0) /
                   np.maximum(f * (1 - f), 1e-300))
