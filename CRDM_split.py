import matplotlib.pyplot as plt
from scipy.integrate import quad
from .util import *
from .kinematics import *
from .rays import *


def dsigmadTxSplit(Tx, Ti, mx, de, gxi, mA, iS):
    """
    Mass splitting DM: differential cross sections of DM-cosmic ray nucleus (vector case)
    Tx: incoming DM kinetic energy
    Ti: incoming CR kinetic energy
    mx: DM mass
    de: mass splitting
    gxi: x/nucleon to mediator coupling (assume they are the same)
    mA: mediator mass
    iS: species of CR 'proton' or 'he4'
    return: [GeV^-3]
    """
    mi = rays['mi'][iS]
    n1 = gxi**4 * (4*mx*(mi+Ti)**2 -2*((mi+mx)**2+2*mx*Ti)*Tx + 2*mx*Tx**2 - 4*mx*(mi+Ti)*de + (mx-Tx)*de**2)
    d1 = 2*np.pi*Ti * (2*mi+Ti) * (mA**2+2*mx*Tx-de**2)**2 / GeV**3
    fm = nuclear_res(np.sqrt(2*mx*Tx-de**2), rays['Ai'][iS], rays['ji'][iS])
    return n1/d1 * fm


def dphidTx2Split(Tx2, mx, de, gxi, mA, temp_scale=1e80):
    """
    Mass splitting DM: Upscattered chi2 dark matter flux, contributions from protons and helium included
    Tx2: outgoing DM kinetic energy
    mx: DM mass
    de: mass splitting
    gxi: x/nucleon to mediator coupling (assume they are the same)
    mA: mediator mass
    temp_scale: to fix the precision problem
    """
    s = 0
    phis = []
    for idx, iS in zip([0, 1], ['proton', 'he4']):
        if Tx2 > TxMin(TdatMax[idx], rays['mi'][iS], mx, de):
            def ff(Tii):
                phi = dPhidT(Tii, iS)
                phis.append(phi)
                return temp_scale*phi*dsigmadTxSplit(Tx2, Tii, mx, de, gxi, mA, iS)

            ff_min = max(TdatMin[idx], TiMin(Tx2, rays['mi'][iS], mx, de))
            ff_max = max(TdatMax[idx], TiMin(Tx2, rays['mi'][iS], mx, de))
            s += quad(ff, ff_min, ff_max)[0]
    return s * Deff * rho_X / mx / temp_scale


def plot_phi_split():
    tt = np.logspace(-5, 1, 100)
    gxLight = np.sqrt(.001)
    mx = 1e-3
    mA = 1e-3
    yy = [unitsCM2S * t * dphidTx2Split(t, mx, .0001, gxLight, mA) for t in tt]
    plt.figure(figsize=(8, 6))
    plt.plot(tt, yy)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('$T_x$ [GeV]', fontsize=20)
    plt.ylabel('$T_x\,\, d\Phi/dT_x$ [cm$^{-2}$ s$^{-1}$]', fontsize=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.show()


if __name__ == '__main__':
    plot_phi_split()
