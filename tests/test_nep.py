"""v0.8 NEP anchors: Mather's exactly flat phonon-TFN NEP, the
matched-filter identity against the package's independent
energy-resolution integral (two code paths), the exact DC crossover
equivalence, and the exact shape of the occupation channel.
Reference: J. C. Mather, Appl. Opt. 21, 1125 (1982)."""
import numpy as np
import pytest

from absnoise import KB, RECIPES, SensorBudget

B = SensorBudget(RECIPES[1])
T = 0.3 * B.recipe.Tc
TAUA = 1e-6


def test_phonon_nep_is_mathers_flat_closed_form():
    f = np.logspace(0, 8, 50)
    nep = B.nep_spectrum(T, TAUA, f)
    ref = np.sqrt(4.0 * KB * T ** 2 * B.Gep(T))
    assert np.abs(nep["phonon"] / ref - 1.0).max() < 1e-14


def test_matched_filter_identity_reproduces_energy_resolution():
    """sigma_E = [4 int df / NEP^2]^(-1/2) computed from the NEP
    closed forms must equal the package's own signal-chain integral
    -- two independent code paths through the same physics."""
    for S_ro in (0.0, 1e-22):
        tth = B.tau_th(T)
        fmax = 10.0 / (2 * np.pi * min(TAUA, tth))
        f = np.logspace(np.log10(1.0 / (200 * 2 * np.pi *
                                        max(TAUA, tth))),
                        np.log10(fmax), 6000)
        nep = B.nep_spectrum(T, TAUA, f, S_ro_y=S_ro)
        sig_from_nep = 1.0 / np.sqrt(
            4.0 * np.trapezoid(1.0 / nep["total"] ** 2, f))
        sig_direct = B.energy_resolution(T, TAUA, S_ro_y=S_ro)
        assert abs(sig_from_nep - sig_direct) / sig_direct < 1e-10


def test_dc_crossover_matches_documented_criterion():
    """NEP_A(0) > NEP_ph(0) iff tauA S_T_A(0)-style comparison; when
    the occupation bound is saturated S_T_A(0) = 4 kB T^2 tauA / C_A,
    so the crossover is exactly tauA / C_A vs tau_th / C_e -- the
    criterion the module docstring has always stated."""
    f0 = np.array([0.0])
    nep = B.nep_spectrum(T, TAUA, f0)
    s = B.sj.andreev_sums(1e-4, T, "L")
    ST_A0 = s["S0_over_tau"] * TAUA / s["R_occ"] ** 2
    lhs = nep["andreev"][0] ** 2 / nep["phonon"][0] ** 2
    rhs = ST_A0 * B.Gep(T) / (4.0 * KB * T ** 2)
    assert abs(lhs - rhs) / rhs < 1e-12
    # and the ratio scales exactly linearly in tauA
    nep2 = B.nep_spectrum(T, 2 * TAUA, f0)
    r1 = nep["andreev"][0] ** 2 / nep["phonon"][0] ** 2
    r2 = nep2["andreev"][0] ** 2 / nep2["phonon"][0] ** 2
    assert abs(r2 - 2 * r1) / r1 < 1e-12


def test_occupation_channel_shape_is_exactly_thermal_rolloff():
    """The occupation Lorentzian cancels against the occupation lag,
    leaving NEP_A(f) = NEP_A(0) sqrt(1 + (2 pi f tau_th)^2)."""
    f = np.logspace(1, 7, 40)
    nep = B.nep_spectrum(T, TAUA, np.concatenate([[0.0], f]))
    tth = B.tau_th(T)
    ref = nep["andreev"][0] * np.sqrt(1 + (2 * np.pi * f * tth) ** 2)
    assert np.abs(nep["andreev"][1:] / ref - 1.0).max() < 1e-12
    # readout channel carries BOTH rolloffs (it is not filtered by
    # the physics), so it must rise faster than the Andreev channel
    nep_ro = B.nep_spectrum(T, TAUA, f, S_ro_y=1e-22)["readout"]
    ratio = nep_ro / nep_ro[0]
    ratio_A = nep["andreev"][1:] / nep["andreev"][1]
    assert ratio[-1] > ratio_A[-1]
    with pytest.raises(ValueError):
        B.nep_spectrum(T, TAUA, np.array([-1.0]))
