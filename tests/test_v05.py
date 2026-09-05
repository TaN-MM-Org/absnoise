"""v0.5 anchors: the continuum occupation-channel quantification
against the level-sum identity and the short-junction limit, and the
two-temperature phonon-bath dynamics against its structural
invariants (exact fixed point, energy conservation with RK4-order
convergence, the independently computed isolated common temperature,
and the pinned-bath reduction to `click_template`)."""
import dataclasses

import numpy as np
import pytest

from absnoise import (RECIPES, FiniteLJunction, SensorBudget,
                      click_template, occupation_heat_capacities,
                      total_energy, two_temperature_click)

RECIPE = RECIPES[1]


# ------------------- continuum occupation channel -------------------


def test_bound_heat_capacity_two_code_paths_agree():
    """-T d2F_bound/dT2 at frozen gap must equal the level-resolved
    occupation sum C_A of `FiniteLJunction.andreev_sums` -- the same
    physical quantity through the free energy and through the level
    sums, agreeing to finite-difference accuracy."""
    T = 0.3 * RECIPE.Tc
    out = occupation_heat_capacities(RECIPE, T, phi0=np.pi / 2)
    s = FiniteLJunction(RECIPE).andreev_sums(np.pi / 2, T)
    assert abs(out["C_bound"] - s["C_A"]) / s["C_A"] < 1e-5


def test_continuum_share_vanishes_in_the_short_junction_limit():
    """At L = 0 the continuum scattering determinant is real positive
    (arg D = 0 identically), so the continuum occupation heat capacity
    is exactly zero."""
    T = 0.3 * RECIPE.Tc
    r0 = dataclasses.replace(RECIPE, L=0.0)
    out = occupation_heat_capacities(r0, T)
    assert out["C_cont"] == 0.0
    assert out["share_CA"] == 0.0


def test_continuum_share_is_small_but_nonzero_at_finite_length():
    """The stated justification for the bound-only sums -- bound levels
    dominate for L < xi -- is now a number: positive, and small for
    the recipe set's L/xi."""
    T = 0.3 * RECIPE.Tc
    out = occupation_heat_capacities(RECIPE, T)
    assert out["C_cont"] > 0.0
    assert 0.0 < out["share_CA"] < 0.1


# ---------------------- two-temperature bath -----------------------


def _setup():
    b = SensorBudget(RECIPE)
    T0 = 0.3 * RECIPE.Tc
    gam = b.Ce(T0) / T0
    return b, T0, gam


def test_missing_phonon_parameters_are_refused():
    b, T0, gam = _setup()
    with pytest.raises(ValueError):
        two_temperature_click(b, 1e-22, 1e-6, T0)
    with pytest.raises(ValueError):
        two_temperature_click(b, 1e-22, 1e-6, T0, c_ph=1e-9)
    with pytest.raises(ValueError):
        two_temperature_click(b, 1e-22, 1e-6, T0, c_ph=1e-9,
                              kappa_pb=1.0)
    with pytest.raises(ValueError):
        two_temperature_click(b, 1e-22, 1e-6, T0, c_ph=-1.0,
                              kappa_pb=1.0, delta_pb=4.0)
    with pytest.raises(ValueError):
        two_temperature_click(b, -1e-22, 1e-6, T0, c_ph=1e-9,
                              kappa_pb=1.0, delta_pb=4.0)


def test_equilibrium_is_an_exact_fixed_point():
    """With no deposit every right-hand side is exactly zero, and the
    integrator preserves T0 bitwise."""
    b, T0, gam = _setup()
    r = two_temperature_click(b, 0.0, 1e-6, T0, c_ph=gam / T0 ** 2,
                              kappa_pb=100.0, delta_pb=4.0,
                              dt=1e-9, t_end=2e-8)
    assert np.abs(r["Te"] - T0).max() == 0.0
    assert np.abs(r["Tp"] - T0).max() == 0.0
    assert np.abs(r["ms"] - r["m0"]).max() == 0.0


def test_isolated_energy_conservation():
    """kappa_pb = 0: U = gamma Te^2/2 + c_ph Tp^4/4 is conserved by
    the stated equations; the integrator drift is tiny."""
    b, T0, gam = _setup()
    cph = gam / T0 ** 2
    r = two_temperature_click(b, 2e-22, 1e-6, T0, c_ph=cph,
                              kappa_pb=0.0, delta_pb=4.0,
                              dt=2e-10, t_end=5e-8)
    U = total_energy(b, T0, r["Te"], r["Tp"], cph)
    assert np.abs(U - U[0]).max() / U[0] < 1e-5


def test_isolated_common_temperature_matches_the_quartic_root():
    """Independent computation: both temperatures must equilibrate to
    the T solving gamma T^2/2 + c_ph T^4/4 = U0 (unique positive
    root), never having been told it."""
    b, T0, gam = _setup()
    cph = gam / T0 ** 2
    r = two_temperature_click(b, 2e-22, 1e-6, T0, c_ph=cph,
                              kappa_pb=0.0, delta_pb=4.0,
                              dt=2e-10, t_end=5e-7)
    U0 = float(total_energy(b, T0, r["Te"][0], T0, cph))
    roots = np.roots([cph / 4.0, 0.0, gam / 2.0, 0.0, -U0])
    Tf = float(roots[np.isreal(roots) & (roots.real > 0)].real.max())
    assert abs(r["Te"][-1] - r["Tp"][-1]) < 1e-12
    assert abs(r["Te"][-1] - Tf) / Tf < 1e-5


def test_pinned_bath_reproduces_click_template():
    """Infinite-bath limit (large phonon heat capacity, finite
    escape): the phonons stay at T0 and the occupation trajectory is
    the single-temperature `click_template` one."""
    b, T0, gam = _setup()
    ct = click_template(b, 2e-22, 1e-6, T0, scenario="C", dt=1e-10,
                        t_end=2e-7, dt2=1e-10, t_end2=2e-7)
    r = two_temperature_click(b, 2e-22, 1e-6, T0, c_ph=1e-9,
                              kappa_pb=100.0, delta_pb=4.0,
                              dt=1e-10, t_end=2e-7)
    assert r["Tp"].max() - T0 < 1e-9
    mi = np.interp(r["ts"], ct["ts"], ct["ms"])
    assert np.abs(r["ms"] - mi).max() < 1e-3


def test_peak_temperature_identity_and_relaxation_to_bath():
    """Te(0) satisfies the exact deposit identity, and with escape on,
    both temperatures return to the bath."""
    b, T0, gam = _setup()
    Eg = 2e-22
    cph = gam / T0 ** 2
    r = two_temperature_click(b, Eg, 1e-6, T0, c_ph=cph,
                              kappa_pb=100.0, delta_pb=4.0,
                              dt=1e-9, t_end=2e-6)
    assert abs(0.5 * gam * (r["Te_peak"] ** 2 - T0 ** 2) - Eg) / Eg \
        < 1e-12
    # both temperatures have shed > 99 percent of their peak excursion
    assert abs(r["Te"][-1] - T0) < 0.01 * (r["Te_peak"] - T0)
    assert abs(r["Tp"][-1] - T0) < 0.01 * (r["Tp"].max() - T0)
    assert r["Tp"].max() > T0               # the bath actually heated
    # and the decay is monotone over the tail
    tail = r["Te"][r["ts"] > 5e-7]
    assert np.all(np.diff(tail) <= 1e-15)
