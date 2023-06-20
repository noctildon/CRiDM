"""
All unit in GeV, unless stated otherwise
This code is regarding 1907.03782.
Mass splitting DM (2108.00583) is in CRDM_split.py
"""

import numpy as np
import matplotlib.pyplot as plt
from .xsec import xsec_Tchi
from .util import *
from .kinematics import *
from .rays import *


def dsigmadTx(Tx, Ti, mx, eps, mA, iS):
    """
    Differential cross sections of DM-cosmic ray nucleus
    Tx: DM kinetic energy
    Ti: CR kinetic energy
    mx: DM mass
    eps: effective g_Nv=e*eps*Q, g_{chi v} = g_D
    mA: mediator mass
    iS: species of CR 'proton' or 'he4'
    return: [GeV^-3]
    """
    mi = rays['mi'][iS]
    fm = nuclear_res(np.sqrt(2*mx*Tx**2), rays['Ai'][iS], rays['ji'][iS])

    gD = np.sqrt(4*np.pi*alpha_D) # = g_{chi v}
    g_N = e_charge * eps * rays['Zei'][iS] # = g_Nv
    return xsec_Tchi(gD, g_N, mx, mA, mi, Tx, Ti, rays['Ai']['he4'], fm, 'vector')


def dphidTx(Tx, mx, eps, mA, temp_scale=1e80, limit=50):
    """
    Differential upscattered chi dark matter flux, contributions from protons and helium included
    Tx: outgoing DM kinetic energy
    mx: DM mass
    eps: coupling
    mA: mediator mass
    limit: quad integration limit
    temp_scale: to fix the precision problem
    """
    s = 0
    for idx, iS in zip([0, 1], ['proton', 'he4']):
        def ff(Tii):
            phi = dPhidT(Tii, iS)
            return temp_scale*phi*dsigmadTx(Tx, Tii, mx, eps, mA, iS)

        ff_min = max(TdatMin[idx], TiMin(Tx, rays['mi'][iS], mx, 0))
        ff_max = max(TdatMax[idx], TiMin(Tx, rays['mi'][iS], mx, 0))
        # s += quad(ff, ff_min, ff_max)[0] # integrate in linear space
        s += log_int(ff, ff_min, ff_max, limit=limit) # integrate in log space

    return s * Deff * rho_X / mx / temp_scale


def CRDM_phi(mx, eps, mA):
    """
    Upscattered chi dark matter flux, contributions from protons and helium included
    mx: DM mass
    eps: coupling
    mA: mediator mass
    """
    Tx_min = 0
    Tx_max = 100
    def ff(Tx):
        return dphidTx(Tx, mx, eps, mA)
    return log_int(ff, Tx_min, Tx_max)


def _plot_phi():
    mx = 1e-3 # DM mass [GeV]
    mA = 1e-3 # mediator mass [GeV]
    eps = 1e-3 # coupling

    tt = np.logspace(-5, 1, 100) # DM kinematic energy T_x [GeV]
    yy = [unitsCM2S * t * dphidTx(t, mx, eps, mA) for t in tt] # the differential flux [cm^-2 s^-1]

    plt.figure(figsize=(8, 6))
    plt.plot(tt, yy)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$T_x$ [GeV]', fontsize=20)
    plt.ylabel('$T_x\,\, d\Phi/dT_x$ [cm$^{-2}$ s$^{-1}$]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()
