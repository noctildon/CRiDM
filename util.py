import numpy as np
from scipy.integrate import quad
from .const import *


def log_int(func, a, b, mode='log', **kwargs):
    """
    Integrate a function in log space, designed for extreme small/large functions
    func: the function to integrate
    a, b: integration range (original, not in log space)
    return: integral
    """
    match mode:
        case 'log':
            def func_log(u):
                return np.exp(u)*func(np.exp(u))
            return quad(func_log, np.log(a), np.log(b), **kwargs)[0]

        case 'linear':
            return quad(func, a, b, **kwargs)[0]

    raise ValueError('Invalid mode')


def Fhelm(q, A):
    """
    Helm form factor
    q: momentum transfer [GeV]
    A: mass number of the nucleus
    """
    s = 0.9
    r = np.sqrt((1.23*A**(1/3)-.6)**2 + 7/3*np.pi**2*0.52**2 - 5*0.9**2)
    qr = q*r/hbarc
    qs = q*s/hbarc

    if 0 < q < 4:
        return 3 * (np.sin(qr)-qr*np.cos(qr)) / qr**3 * np.exp(-qs**2/2)
    elif q > 4:
        return 0
    else:
        return 1


def nuclear_res(q, A, Ji):
    """
    Nuclear response function FM
    q: momentum transfer [GeV]
    A: mass number of the nucleus
    Ji: spin of the nucleus
    """
    if A == 1:
        res = 0.0397887 * gi(q**2, 0.77)**2
    elif A == 4:
        res = 0.31831 * gi(q**2, 0.41)**2
    else:
        res = A**2 * Fhelm(q, A)**2 /(16*np.pi)
    return 4*np.pi * 4 * res /(2*Ji+1)


def gi(q2, l):
    """
    Dipole form factor
    q2: momentum transfer squared [GeV^2]
    l: Lambda, charge radius of proton and neutron: 0.770 & 0.410
    """
    return (1+q2/l**2)**(-2)


def reduced_mass(m1, m2):
    """
    reduced mass
    m1, m2: masses
    """
    return m1*m2 / (m1+m2)


def xsec_XP(g, mchi, mphi, kk=0):
    """
    NR cross section
    g: coupling
    mchi: DM mass
    mphi: mediator mass
    """
    return 4 * g**4 * reduced_mass(mchi, mp)**2 / (np.pi * mphi**4 * (1+4*kk**2/mphi**2)) / Centimeter**2 / GeV**2


