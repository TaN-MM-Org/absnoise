"""Physical constants (SI units, CODATA 2018)."""

HBAR = 1.054571817e-34      # J s
KB = 1.380649e-23           # J / K
E_CHARGE = 1.602176634e-19  # C
H_PLANCK = 6.62607015e-34   # J s
PHI0 = H_PLANCK / (2.0 * E_CHARGE)   # magnetic flux quantum, Wb
EPS0 = 8.8541878128e-12     # F / m

# Graphene band parameters
V_F = 1.0e6                 # graphene Fermi velocity, m/s
RHO_M = 7.6e-7              # graphene mass density, kg/m^2
S_LA = 2.0e4                # LA phonon sound velocity, m/s
D_DEF = 18.0 * E_CHARGE     # deformation potential, J (18 eV, mid-range of 10-30 eV)

# BCS gap ratio
BCS_RATIO = 1.764           # Delta0 = 1.764 kB Tc
