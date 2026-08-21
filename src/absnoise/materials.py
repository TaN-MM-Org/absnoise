"""Junction recipes and graphene thermodynamic functions.

Recipe parameters are taken from Table I of Jung et al.,
"Engineering Andreev Bound States for Thermal Sensing in Proximity
Josephson Junctions", arXiv:2503.06850 (six measured graphene Josephson
junction contact recipes). Thermal parameters follow the measured
electron-phonon coupling of Lee et al., Nature 586, 42 (2020)
(delta = 3 resonant-supercollision regime, Sigma_3 ~ 2 W m^-2 K^-3).
"""

from dataclasses import dataclass
import numpy as np

from .constants import HBAR, KB, E_CHARGE, EPS0, V_F, BCS_RATIO


@dataclass(frozen=True)
class Recipe:
    name: str          # contact stack
    label: str         # short label for plots
    Tc: float          # junction critical temperature T_c* (K)
    xi: float          # coherence length (m), as quoted
    L: float           # channel length (m)
    W: float           # channel width (m)
    Vbg: float         # back-gate voltage (V)
    tau: float         # contact transparency
    Ic20: float        # measured critical current at 20 mK (A)
    Rn: float          # normal-state resistance (Ohm)

    @property
    def Delta(self) -> float:
        """Proximity-induced gap Delta* = 1.764 kB Tc* (J)."""
        return BCS_RATIO * KB * self.Tc

    @property
    def area(self) -> float:
        return self.L * self.W


# Table I of Jung et al., arXiv:2503.06850 (film thicknesses in nm in name)
RECIPES = [
    Recipe("Ta(10)/Ti(60)/Au(5)", "Ta/Ti/Au", 0.57, 7.2e-6, 0.2e-6, 5.3e-6,
           30.0, 0.30, 1.08e-6, 43.3),
    Recipe("Ti(6)/Al(60)/Au(5)", "Ti/Al/Au", 0.75, 5.1e-6, 0.2e-6, 1.8e-6,
           30.0, 0.78, 2.11e-6, 42.0),
    Recipe("Ti(6)/Al(70)", "Ti/Al(thin)", 0.99, 4.4e-6, 0.3e-6, 2.9e-6,
           20.0, 0.53, 2.11e-6, 64.9),
    Recipe("Ti(6)/Al(200)", "Ti/Al(thick)", 1.17, 3.6e-6, 0.3e-6, 2.9e-6,
           45.0, 0.42, 3.17e-6, 48.2),
    Recipe("Ti(6)/Nb(5)/NbN(50)", "Ti/Nb/NbN", 2.5, 0.38e-6, 0.1e-6, 1.6e-6,
           30.0, 0.58, 0.985e-6, 72.9),
    Recipe("MoRe(50)", "MoRe", 7.4, 0.46e-6, 0.2e-6, 1.8e-6,
           30.0, 0.27, 3.13e-6, 133.0),
]

# Gate capacitance for 280 nm SiO2 (Jung et al. wafers) and CNP offset
T_SIO2 = 280e-9
C_GATE = 3.9 * EPS0 / T_SIO2      # F/m^2
V_CNP = -2.0                       # V (chosen so that Vbg = 20 V gives
                                   # n ~ 1.7e16 m^-2 as quoted by Jung et al.)


def carrier_density(Vbg: float) -> float:
    """Electron density (1/m^2) from parallel-plate gate model."""
    return C_GATE * (Vbg - V_CNP) / E_CHARGE


def fermi_energy(n: float) -> float:
    """Graphene Fermi energy (J) at density n (1/m^2)."""
    return HBAR * V_F * np.sqrt(np.pi * n)


def dos_ef(n: float) -> float:
    """Graphene DOS per area per energy at E_F (1/(J m^2)), 4-fold degen."""
    return 2.0 * fermi_energy(n) / (np.pi * HBAR**2 * V_F**2)


def heat_capacity(Te, n: float, area: float):
    """Electronic heat capacity (J/K). Degenerate 2D Fermi gas,
    C = (pi^2/3) kB^2 T nu(E_F) A. Validated: reproduces ~6 kB for
    A = 1 um^2, n = 1.7e16 m^-2, T = 0.1 K (Jung et al.) and the 0.6 ns
    thermal time of Lee et al. Nature 586, 42 (2020)."""
    return (np.pi**2 / 3.0) * KB**2 * np.asarray(Te) * dos_ef(n) * area


def ep_power(Te, Tp: float, area: float, sigma: float = 2.0,
             delta: int = 3):
    """Electron-phonon cooling power (W), P = Sigma A (Te^d - Tp^d).
    Default: resonant-supercollision regime delta = 3 with
    Sigma = 2.0 W m^-2 K^-3 (measured for hBN-encapsulated graphene JJ
    bolometers, Lee et al. Nature 586, 42 (2020): 2.1-3.3)."""
    Te = np.asarray(Te, dtype=float)
    return sigma * area * (Te**delta - Tp**delta)


def gth(Te, area: float, sigma: float = 2.0, delta: int = 3):
    """Differential thermal conductance G = dP/dTe (W/K)."""
    Te = np.asarray(Te, dtype=float)
    return delta * sigma * area * Te**(delta - 1)


def n_modes(recipe: Recipe) -> int:
    """Number of orbital transverse modes (spin+valley counted separately),
    N = kF W / pi, as in Jung et al. (R_Q = (h/4e^2)/N)."""
    n = carrier_density(recipe.Vbg)
    kf = np.sqrt(np.pi * n)
    return max(1, int(np.floor(kf * recipe.W / np.pi)))
