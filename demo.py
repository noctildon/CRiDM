"""
This file demonstrates the use of the CRiDM package
"""
from CRDM import *

def plot_phi():
    mA = 1e-3 # mediator mass [GeV]
    mx = 1e-3 # DM mass [GeV]
    eps = 1e-6 # coupling
    tt = np.logspace(-5, 1, 100) # DM kinematic energy T_x [GeV]

    plt.figure(figsize=(8, 6))
    yy = [1e-3*unitsCM2S*dphidTx(t, mx, eps, mA) for t in tt] # the differential flux [cm^-2 s^-1 MeV^-1]
    plt.plot(tt*1e3, yy)

    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1e-2, 1e4)
    plt.ylim(1e-32, 1e-6)
    plt.legend(fontsize=16, loc='lower left')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('$T_x$ [MeV]', fontsize=24)
    plt.ylabel('$d\Phi/dT_x$ [cm$^{-2}$ s$^{-1}$ MeV${^-1}$]', fontsize=24)
    plt.show()


if __name__ == "__main__":
    plot_phi()