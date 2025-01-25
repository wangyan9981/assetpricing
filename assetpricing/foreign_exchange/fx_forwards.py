# assetpricing/foreign_exchange/fx_forwards.py

import numpy as np 

def fx_forward_rate(spot_rate, domestic_rate, foreign_rate, T, continuous=True):
    """
    Calculate FX forward rate using interest rate parity.
    """
    if continuous:
        return spot_rate * np.exp((domestic_rate - foreign_rate) * T)
    else:
        return spot_rate * (1 + domestic_rate*T) / (1 + foreign_rate*T)