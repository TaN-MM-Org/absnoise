"""Analytic anchors of the ABS level solver and the BCS gap."""
import numpy as np

from absnoise import (JunctionModel, RECIPES, Recipe, abs_energies,
                      gap_bcs)
from absnoise.constants import E_CHARGE, HBAR
from absnoise.levels import continuum_delta, continuum_phase_grid

DELTA = 1.5e-23  # J


def test_short_junction_formula():
    # L -> 0: E = Delta sqrt(1 - tau sin^2(phi/2)), machine precision
    worst = 0.0
    for tau in (0.1, 0.3, 0.5, 0.78, 0.99, 1.0):
        for phi in (0.3, 1.0, 2.0, 3.0):
            E = abs_energies(phi, tau, 0.0, DELTA)
            Ean = DELTA * np.sqrt(1 - tau * np.sin(phi / 2) ** 2)
            assert len(E) == 1
            worst = max(worst, abs(E[0] - Ean) / DELTA)
    assert worst < 1e-12


def test_kulik_levels():
    # tau = 1, finite L: 2 arccos(E/Delta) - cE = +-phi mod 2 pi
    c = 3.0 / DELTA
    worst = 0.0
    for phi in (0.5, 1.5, 2.5):
        for E in abs_energies(phi, 1.0, c, DELTA):
            th = 2 * np.arccos(E / DELTA) - c * E
            r = min(abs(((th - phi) + np.pi) % (2 * np.pi) - np.pi),
                    abs(((th + phi) + np.pi) % (2 * np.pi) - np.pi))
            worst = max(worst, r)
    assert worst < 1e-10


def test_ballistic_IcRn():
    # single ballistic channel, short junction, T -> 0:
    # Ic = 2 e Delta / hbar (two spin-degenerate channels; valley
    # factor handled by keeping one orbital mode)
    r = Recipe("test", "test", 1.0, 1e-5, 1e-9, 12e-9, 30.0, 1.0,
               1e-6, 1.0)
    m = JunctionModel(r, n_phi=601)
    assert m.N >= 1
    m.cos_th = m.cos_th[:1]
    m.cs = m.cs[:1]
    m.N = 1
    m._levels = None
    Ic = m.Ic(0.001)
    Ican = 2.0 * E_CHARGE * r.Delta / HBAR
    assert abs(Ic - Ican) / Ican < 5e-3


def test_short_junction_continuum_vanishes():
    # c = 0: the secular function above the gap is real and positive,
    # so the continuum carries no phase dependence at all
    Efac = continuum_phase_grid(400)
    for phi in (0.4, 1.3, 2.8):
        d = continuum_delta(phi, 0.6, 0.0, DELTA, Efac)
        assert np.max(np.abs(d)) < 1e-14


def test_bcs_gap_asymptotes_and_midpoint():
    Tc, D0 = 1.0, 1.0
    # low-T asymptote: gap essentially closed to Delta0
    assert abs(gap_bcs(0.01, Tc, D0) - 1.0) < 1e-10
    # closes at Tc
    assert gap_bcs(0.99999, Tc, D0) < 0.05
    # universal midpoint u(0.5) = 0.956887 (solved gap equation)
    assert abs(gap_bcs(0.5, Tc, D0) - 0.956887) < 2e-4
    # monotone decreasing
    ts = np.linspace(0.05, 0.95, 19)
    us = gap_bcs(ts, Tc, D0)
    assert np.all(np.diff(us) < 0.0)


def test_recipe_calibration_is_order_one():
    for r in RECIPES[:2]:
        m = JunctionModel(r, n_phi=41)
        s = m.calibrate()
        assert 0.1 < s < 10.0
