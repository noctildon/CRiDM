"""
Cosmic rays flux
"""
import numpy as np
from scipy.interpolate import interp1d
import pkg_resources
from .const import *


# Rigidity [GV], Modulated Spectrum [(m^2 s sr GV)^{-1}], LIS
proton_data = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/TABLE_Protons_R.txt'))
he4_data = np.genfromtxt(pkg_resources.resource_filename(__name__, 'data/TABLE_Helium_R.txt'))

rays = {
    "mi" : {'proton': mp, 'he4':4.002602*amu},
    "Zei": {'proton': 1, 'he4':2},
    "Ai" : {'proton': 1, 'he4':4},
    "ji" : {'proton': 0.5, 'he4': 0}
}

def dIdR(r, spec):
    """
    Interpolate the data
    specs: 'proton' or 'helium'
    r: rigidity
    """
    if spec == 'proton':
        raw = proton_data
    elif spec == 'he4':
        raw = he4_data
    else:
        raise ValueError('spec must be proton or helium')

    f = interp1d(raw[:, 0], raw[:, 1], fill_value='extrapolate')
    return f(r)


def T2R(T, spec):
    """
    Convert rigidity to kinetic energy
    T: kinetic energy
    spec: proton or helium
    """
    return np.sqrt(T**2 + 2*T*rays['mi'][spec]) / rays['Zei'][spec]

def R2T(R, spec):
    """
    Convert kinetic energy to rigidity
    R: rigidity
    spec: proton or helium
    """
    return np.sqrt(rays['mi'][spec]**2 + (R * rays['Zei'][spec])**2) - rays['mi'][spec]

TdatMin = (R2T(proton_data[0,0], 'proton'), R2T(he4_data[0,0], 'he4'))
TdatMax = (R2T(proton_data[-1,0], 'proton'), R2T(he4_data[-1,0], 'he4'))

def dRdT(t, spec):
    """
    flux of cosmic rays
    t: kinetic energy
    spec: proton or helium
    """
    Zei = rays["Zei"][spec]
    mi = rays["mi"][spec]
    return (2*mi + 2*t) / np.sqrt(2*mi*t + t**2) / (2*Zei)


def dPhidT(t, spec):
    """
    flux of cosmic rays
    t: kinetic energy
    spec: proton or helium
    """
    res = 4*np.pi/100**2 / (Centimeter**2 * Second * GeV)
    res *= dIdR(T2R(t, spec), spec) * dRdT(t, spec)
    return res