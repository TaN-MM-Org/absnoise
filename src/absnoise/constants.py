"""Physical constants (SI units, CODATA 2018)."""

HBAR = 1.054571817e-34      # J s
KB = 1.380649e-23           # J / K
E_CHARGE = 1.602176634e-19  # C
H_PLANCK = 6.62607015e-34   # J s
PHI0 = H_PLANCK / (2.0 * E_CHARGE)   # magnetic flux quantum, Wb
EPS0 = 8.8541878128e-12     # F / m

# Graphene band parameters
V_F = 1.0e6                 # graphene Fermi velocity, m/s
# RHO_M and S_LA are the values used by E. H. Hwang and S. Das Sarma,
# Phys. Rev. B 77, 115449 (2008) (7.6e-8 g/cm^2 and 2e6 cm/s). They
# use D = 19 eV and note that quoted values range over D ~ 10-30 eV;
# D_DEF = 18 eV lies in that range. None of the three is used by the
# package's calculations; they are kept for user code.
RHO_M = 7.6e-7              # graphene mass density, kg/m^2
S_LA = 2.0e4                # LA phonon sound velocity, m/s
D_DEF = 18.0 * E_CHARGE     # deformation potential, J (18 eV)

# BCS gap ratio
BCS_RATIO = 1.764           # Delta0 = 1.764 kB Tc
