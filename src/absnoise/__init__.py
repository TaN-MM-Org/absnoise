"""absnoise: Andreev-bound-state occupation noise in proximity Josephson junctions.

Computes the Andreev level structure of ballistic proximity Josephson
junctions (exact finite-length solver and closed-form short-junction
ensemble), the intrinsic occupation-noise limit of Andreev thermometry
(Cauchy-Schwarz bound over channels), master-equation and Monte Carlo
noise spectra, and device-level detector budgets: temperature resolution,
resonator frequency-noise spectra, and matched-filter calorimetric energy
resolution.

Methodological basis: T. M. Mahim, A. S. M. Mohsin and M. M. Rahman,
"Andreev occupation noise sets the sensitivity limit of proximity
Josephson thermal detectors" (manuscript; code and data:
github.com/Tanvir-Mahmud-Mahim/andreev-occupation-noise); measured
junction recipes from W. Jung et al., Phys. Rev. Applied 26, 014078
(2026) (arXiv:2503.06850).
"""
from .constants import HBAR, KB, E_CHARGE, H_PLANCK, PHI0, EPS0, V_F
from .materials import (RECIPES, Recipe, carrier_density, dos_ef,
                        ep_power, fermi_energy, gth, heat_capacity,
                        n_modes, steady_temperature)
from .levels import (JunctionModel, abs_energies, continuum_share,
                     gap_bcs, junction_properties)
from .shortjunction import ShortJunction
from .finitelength import FiniteLJunction
from .master import (channel_generator, noneq_penalty, sigma_spectrum,
                     tau_activated)
from .telegraph import psd_single_sided, telegraph_traces
from .budgets import SensorBudget
from .click import (click_monte_carlo, click_template, matched_Tc,
                    matched_recipe)

__version__ = "0.2.0"
__all__ = [
    "Recipe", "RECIPES", "carrier_density", "fermi_energy", "dos_ef",
    "heat_capacity", "ep_power", "gth", "n_modes", "steady_temperature",
    "abs_energies", "gap_bcs", "JunctionModel", "junction_properties",
    "continuum_share", "ShortJunction", "FiniteLJunction",
    "channel_generator", "sigma_spectrum", "tau_activated",
    "noneq_penalty", "telegraph_traces", "psd_single_sided",
    "SensorBudget",
    "matched_Tc", "matched_recipe", "click_template",
    "click_monte_carlo",
    "HBAR", "KB", "E_CHARGE", "H_PLANCK", "PHI0", "EPS0", "V_F",
]
