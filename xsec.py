import numpy as np
from .const import *

def xsec_Tchi(g_chi, g_N, m_chi, m_phi, m_i, T_chi, T_i, A, ff, current):
    """
    Differential cross sections of DM-cosmic ray nucleus. Eq2 and Eq3 in 1907.03782
    g_chi: vector/axial DM coupling
    g_N: vector/axial nucleon coupling
    m_chi: DM mass
    m_phi: mediator mass
    m_i: incoming cosmic ray mass
    T_chi: DM kinetic energy
    T_i: incoming CR kinetic energy
    A: mass number
    ff: squared form factor
    current: 'vector', 'axial'
    """
    if current == 'vector':
        res = A**2*ff * (m_chi*(mn+T_i)**2 - T_chi*((mn+m_chi)**2 + 2*m_chi*T_i) + m_chi*T_chi**2 )
        res /= 4*np.pi*(2*m_chi*T_chi + m_phi**2)**2 * (T_i**2 + 2*m_i*T_i)
    elif current == 'axial':
        res = 4*m_chi*mn**2 + 2*T_chi*(m_chi**2+mn**2) + m_chi*T_chi**2
        res /= 8*np.pi*(2*m_chi*T_chi + m_phi**2)**2 * (T_i**2 + 2*m_i*T_i)
    res *= g_chi**2 * g_N**2
    return res


def xsec_er(er, g_chi, g_N, m_chi, m_phi, m_T, T_chi, A, ff, current):
    """
    Differential cross section of DM-detector nucleus. Eq8 and Eq9 in 1907.03782
    er: recoil energy
    g_chi: vector/axial DM coupling
    g_N: vector/axial nucleon coupling
    m_chi: DM mass
    m_phi: mediator mass
    m_T: target mass
    T_chi: DM kinetic energy
    A: mass number
    ff: form factor
    current: 'vector', 'axial'
    """
    if current == 'vector':
        res = ff*A**2 * (2*(m_chi+T_chi)**2 - er/mn**2 * (m_T*m_chi**2 + mn**2*(m_T+2*(m_chi+T_chi))) + er**2 )
        res /= np.pi*(2*m_T*er + m_phi**2) * (T_chi**2 + 2*m_chi*T_chi)
    elif current == 'axial':
        res = m_T*(2*(m_chi+T_chi)**2 - er*(m_T*(1+ (m_chi/mn)**2) + 2*(m_chi+T_chi) - er))
        res /= np.pi*(2*m_T*er + m_phi**2)**2 * (T_chi**2 + 2*m_chi*T_chi)
    res *= g_chi**2 * g_N**2 * m_T
    return res