"""Cited recipe set locked to its source, and graphene thermodynamics."""
import numpy as np

from absnoise import (RECIPES, carrier_density, gth, heat_capacity,
                      n_modes)
from absnoise.constants import KB


def test_recipes_locked_to_jung_table():
    # Table I of Jung et al., Phys. Rev. Applied 26, 014078 (2026)
    # (arXiv:2503.06850); a change to any number must arrive with a
    # new source
    expect = [
        ("Ta/Ti/Au", 0.57, 0.30, 1.08e-6, 43.3),
        ("Ti/Al/Au", 0.75, 0.78, 2.11e-6, 42.0),
        ("Ti/Al(thin)", 0.99, 0.53, 2.11e-6, 64.9),
        ("Ti/Al(thick)", 1.17, 0.42, 3.17e-6, 48.2),
        ("Ti/Nb/NbN", 2.5, 0.58, 0.985e-6, 72.9),
        ("MoRe", 7.4, 0.27, 3.13e-6, 133.0),
    ]
    assert len(RECIPES) == 6
    for r, (label, Tc, tau, Ic20, Rn) in zip(RECIPES, expect):
        assert r.label == label
        assert (r.Tc, r.tau, r.Ic20, r.Rn) == (Tc, tau, Ic20, Rn)


def test_carrier_density_matches_quoted_value():
    # Vbg = 20 V on 280 nm SiO2 gives n ~ 1.7e16 m^-2 as quoted by
    # Jung et al.
    n = carrier_density(20.0)
    assert abs(n - 1.7e16) / 1.7e16 < 0.05


def test_heat_capacity_reproduces_quoted_anchor():
    # ~6 kB for A = 1 um^2, n = 1.7e16 m^-2, T = 0.1 K
    C = heat_capacity(0.1, 1.7e16, 1e-12)
    assert 4.0 < C / KB < 8.0


def test_thermal_conductance_positive_and_monotone():
    T = np.linspace(0.05, 1.0, 20)
    G = gth(T, 1e-12)
    assert np.all(G > 0) and np.all(np.diff(G) > 0)


def test_mode_counts_are_physical():
    for r in RECIPES:
        N = n_modes(r)
        assert 1 <= N <= 2000
