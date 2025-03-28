
try:
    import scipy.constants as c
    hbar = c.hbar
    Boltzmann = c.Boltzmann
    eV = c.eV
    atomic_mass = c.atomic_mass
except ImportError:
    hbar = 1.0545718e-34
    Boltzmann = 1.38064852e-23
    eV = 1.60217662e-19
    atomic_mass = 1.660539040e-27