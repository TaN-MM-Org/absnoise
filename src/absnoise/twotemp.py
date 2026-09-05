"""Dynamic phonon bath: the two-temperature extension of the click
dynamics -- the second of the two stated v0.4 limitations, lifted.

Everywhere else in the package the substrate phonons are an infinite
bath at fixed T0; `click_template` cools the electrons against that
bath but never heats it. Physically the local phonons have a finite
heat capacity and a finite escape conductance to the substrate, so a
deposit heats them too, and the electron temperature relaxes against a
*moving* target. This module integrates that coupled system exactly as
stated:

    C_e(T_e) dT_e/dt = -Sigma A (T_e^delta - T_p^delta)
    C_p(T_p) dT_p/dt = +Sigma A (T_e^delta - T_p^delta)
                        - kappa_pb A (T_p^delta_pb - T0^delta_pb)
    dm/dt = (m_eq(T_e) - m) / tau(T_e)

with C_e = gamma T_e (degenerate 2D electrons, as in `SensorBudget`)
and C_p = c_ph T_p^3 (the user-supplied cubic coefficient). NO values
of c_ph, kappa_pb or delta_pb are shipped: the phonon heat capacity
and boundary (Kapitza-type) escape law of a real stack are material
measurements, and this package refuses to invent them -- calls without
all three raise, with a pointer to where such values must come from
(the device's own thermal characterization).

Integrator: classical RK4 with fixed step. The anchors are structural
identities of the stated equations, asserted in the tests rather than
trusted:

* T_e = T_p = T0 is an exact fixed point (the integrator preserves it
  bitwise: every RHS evaluates to exactly zero).
* With kappa_pb = 0 (no escape) the total energy
  U = gamma T_e^2 / 2 + c_ph T_p^4 / 4 is a conserved quantity of the
  flow; the RK4 drift is small and falls as dt^4 (order verified by
  halving dt).
* In the infinite-bath limit (large phonon heat capacity with a
  finite escape) the phonons pin to T0 and the electron and occupation
  trajectories reproduce the single-temperature `click_template`
  dynamics.
* Isolated (kappa_pb = 0), both temperatures equilibrate to the common
  final temperature solving gamma T^2/2 + c_ph T^4/4 = U0 -- a quartic
  root computed independently in the tests.
"""
from __future__ import annotations

import numpy as np

from .constants import KB
from .materials import ep_power


def two_temperature_click(budget, E_gamma, tauA, T0, c_ph=None,
                          kappa_pb=None, delta_pb=None, scenario="C",
                          dt=1e-10, t_end=2e-7):
    """Occupation response m(t) to a photon deposit with a dynamic
    phonon bath.

    budget : SensorBudget on the operating recipe (supplies C_e, the
        electron-phonon law Sigma A (Te^delta - Tp^delta), the gap and
        the exchange-time model, exactly as in `click_template`).
    E_gamma : deposited energy (J), absorbed by the electrons at t = 0:
        T_e(0) = sqrt(T0^2 + 2 E_gamma / gamma), T_p(0) = T0 -- the
        same exact C_e = gamma T identity `click_template` asserts.
    c_ph : phonon heat capacity coefficient, C_p = c_ph T^3 (J K^-4).
    kappa_pb : phonon-substrate escape coefficient (W m^-2 K^-delta_pb),
        P_esc = kappa_pb A (T_p^delta_pb - T0^delta_pb).
    delta_pb : escape-law exponent.
    All three must be supplied with a cited or measured basis; None
    raises.

    Returns dict(ts, Te, Tp, ms, m0, Te_peak).
    """
    if c_ph is None or kappa_pb is None or delta_pb is None:
        raise ValueError(
            "c_ph, kappa_pb and delta_pb carry no vetted defaults: the "
            "phonon heat capacity and boundary escape law of a real "
            "stack are measurements of that stack. Supply all three "
            "from your device's thermal characterization (or a cited "
            "equivalent stack).")
    c_ph = float(c_ph)
    kappa_pb = float(kappa_pb)
    delta_pb = float(delta_pb)
    if c_ph <= 0.0 or kappa_pb < 0.0 or delta_pb <= 0.0:
        raise ValueError("c_ph and delta_pb must be positive, "
                         "kappa_pb non-negative")
    if E_gamma < 0.0:
        raise ValueError("deposited energy must be non-negative")
    if scenario not in ("C", "T"):
        raise ValueError("scenario must be 'C' or 'T'")

    A = budget.recipe.area
    Sig, dlt = budget.sigma, budget.delta
    gam = budget.Ce(T0) / T0                    # C_e = gamma T
    D = budget.sj.Delta(T0)

    def tau_of_T(Te):
        if scenario == "C":
            return tauA
        expo = -(D / KB) * (1.0 / T0 - 1.0 / max(Te, 1e-4))
        return max(1e-9, tauA * float(np.exp(np.clip(expo, -500, 0))))

    def meq_of_T(Te):
        return float(np.tanh(D / (2 * KB * max(Te, 1e-4))))

    def rhs(y):
        Te, Tp, m = y
        Pep = Sig * A * (Te ** dlt - Tp ** dlt)
        Pesc = kappa_pb * A * (Tp ** delta_pb - T0 ** delta_pb)
        dTe = -Pep / (gam * Te)
        dTp = (Pep - Pesc) / (c_ph * Tp ** 3)
        dm = (meq_of_T(Te) - m) / tau_of_T(Te)
        return np.array([dTe, dTp, dm])

    def rk4(y, h):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    m0 = meq_of_T(T0)
    Te0 = float(np.sqrt(T0 ** 2 + 2.0 * E_gamma / gam))
    y = np.array([Te0, T0, m0])
    n = int(t_end / dt)
    ts = np.empty(n + 1)
    out = np.empty((n + 1, 3))
    ts[0], out[0] = 0.0, y
    def stiff_rate(y):
        """Largest local linearized relaxation rate (1/s): keeps every
        RK4 substep inside the stability region even at the quasi-
        equilibrium points where the raw derivatives vanish but the
        stiff eigenvalues do not."""
        Te, Tp, _ = y
        lam_e = dlt * Sig * A * Te ** (dlt - 1) / (gam * Te)
        lam_p = (dlt * Sig * A * Tp ** (dlt - 1)
                 + delta_pb * kappa_pb * A * Tp ** (delta_pb - 1)) \
            / (c_ph * Tp ** 3)
        return max(lam_e, lam_p, 1.0 / tau_of_T(Te))

    for i in range(1, n + 1):
        # substep against both the local timescale of the trajectory
        # (accuracy) and the stiff linearized rates (stability): a
        # fixed step would trade energy conservation for speed
        # silently, or ring at the stiff quasi-equilibrium
        remaining = dt
        while remaining > 1e-30:
            f = rhs(y)
            with np.errstate(divide="ignore", invalid="ignore"):
                scales = np.abs(y[:2]) / np.maximum(np.abs(f[:2]), 1e-300)
            h = min(remaining, 0.02 * float(scales.min()),
                    0.5 / stiff_rate(y))
            y = rk4(y, h)
            remaining -= h
        ts[i], out[i] = i * dt, y
    return dict(ts=ts, Te=out[:, 0], Tp=out[:, 1], ms=out[:, 2],
                m0=m0, Te_peak=Te0)


def total_energy(budget, T0, Te, Tp, c_ph):
    """Conserved energy of the isolated (kappa_pb = 0) two-temperature
    flow: U = gamma Te^2/2 + c_ph Tp^4/4 (J), the exact integrals of
    C_e = gamma T and C_p = c_ph T^3. Used by the conservation anchor."""
    gam = budget.Ce(T0) / T0
    return gam * np.asarray(Te) ** 2 / 2.0 + \
        float(c_ph) * np.asarray(Tp) ** 4 / 4.0
