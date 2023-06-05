import numpy as np


def deMax(Tx, mx):
    # delta Max
    return np.sqrt(2*mx*Tx)

def TxMax(Ti, mi, mx, de):

    if de == 0:
        return 0
    d1 = 2*(mi+mx)**2 + 4*mx*Ti
    n1 = 2*mx*Ti*(2*mi+Ti) - 2*(mi*(mi+mx)+mx*Ti)*de + (mi+mx+Ti)*de**2
    n2 = -Ti*(2*mi+Ti) * (-2*mx*Ti+2*(mi+mx)*de+de**2) * (2*mx*(Ti-de)-de**2 + 2*mi*(2*mx+de))
    return (n1 + np.sqrt(n2)) / d1


def TiMin(Tx, mii, mx, de):
    if de > deMax(Tx, mx):
        return np.inf
    sqroot = Tx * (2*mx+Tx+2*de) * (2*mx*Tx-de**2) * (4*mii**2 + 2*mx*Tx - de**2)
    return (-2*mii + Tx + de + np.sqrt(sqroot) / (2*mx*Tx-de**2)) / 2


def TxMin(Ti, mi, mx, de):
    """
    Minimum kinetic energy of the DM
    Ti: kinetic energy of the cosmic ray
    mi: mass of the cosmic ray
    mx: mass of the DM
    de: mass spliting
    """
    if de == 0:
        return 0
    d1 = 2*(mi+mx)**2 + 4*mx*Ti
    n1 = 2*mx*Ti*(2*mi+Ti) - 2*(mi*(mi+mx)+mx*Ti)*de + (mi+mx+Ti)*de**2
    n2 = -Ti*(2*mi+Ti) * (-2*mx*Ti+2*(mi+mx)*de+de**2) * (2*mx*(Ti-de)-de**2 + 2*mi*(2*mx+de))
    return (n1 - np.sqrt(n2)) / d1


def TxMinGlobal(mx, de):
    return de**2 / (2*mx)