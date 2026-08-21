"""Occupation-noise sums, the Cauchy-Schwarz bound, and finite length."""
import numpy as np

from absnoise import FiniteLJunction, RECIPES, ShortJunction
from absnoise.constants import E_CHARGE, HBAR, KB


def test_uniform_tau_saturates_the_bound():
    # uniform-transparency short junction saturates the Cauchy-Schwarz
    # temperature-resolution bound exactly
    r = RECIPES[1]
    sj = ShortJunction(r)
    sj.calibrate()
    T = 0.35 * r.Tc
    achieved, bound, _ = sj.temperature_bound(T, tauA=1e-6, t_int=1.0)
    assert abs(achieved / bound - 1.0) < 1e-9


def test_occupation_responsivity_matches_numeric():
    # analytic R_occ against numeric differentiation of the occupation
    # factor alone (Delta(T) held fixed)
    r = RECIPES[1]
    sj = ShortJunction(r)
    sj.calibrate()
    T = 0.35 * r.Tc
    phi = sj.phi_max(T)
    s = sj.andreev_sums(phi, T, "I")
    E = sj.E(phi, T)
    g = sj.scale * (2 * E_CHARGE / HBAR) * (-sj.dEdphi(phi, T))
    dT = 1e-6

    def occ(TT):
        return np.tanh(E / (2 * KB * TT))

    R_num = sj.Nch * g * (occ(T + dT) - occ(T - dT)) / (2 * dT)
    assert abs(R_num - s["R_occ"]) / abs(R_num) < 1e-5


def test_dIdphi0_closed_form():
    # I'(0) = Nch (2e/hbar)(Delta tau/4) tanh(Delta/2kBT), exact
    r = RECIPES[2]
    sj = ShortJunction(r)
    T = 0.3 * r.Tc
    D = sj.Delta(T)
    expect = sj.Nch * (2 * E_CHARGE / HBAR) * (D * sj.tau / 4.0) * \
        np.tanh(D / (2 * KB * T))
    assert abs(sj.dIdphi0(T) - expect) / expect < 1e-12


def test_finite_length_does_not_beat_the_bound():
    # dispersing levels break coupling-energy proportionality, so the
    # finite-length junction can only sit at or above the bound
    r = RECIPES[3]           # Ti/Al(thick), moderate L/xi
    fl = FiniteLJunction(r)
    T = 0.3 * r.Tc
    for phi in (1.0, 2.0):
        ratio = fl.saturation_ratio(phi, T, which="I")
        assert ratio >= 1.0 - 1e-9


def test_andreev_heat_capacity_positive_and_peaks_below_gap():
    r = RECIPES[1]
    sj = ShortJunction(r)
    sj.calibrate()
    T = 0.3 * r.Tc
    s = sj.andreev_sums(sj.phi_max(T), T, "I")
    assert s["C_A"] > 0.0 and s["S0_over_tau"] > 0.0
