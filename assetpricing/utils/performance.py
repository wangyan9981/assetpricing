# assetpricing/utils/performance.py
from numba import njit
import numpy as np

@njit
def numba_black_scholes(S, K, T, r, sigma):
    """Numba-accelerated Black-Scholes implementation"""
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    call = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    return call

@njit(parallel=True)
def monte_carlo_numba(S, K, T, r, sigma, n_sims=1000000):
    """Parallelized Monte Carlo using Numba"""
    payoff = np.empty(n_sims)
    for i in range(n_sims):
        z = np.random.normal()
        ST = S * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*z)
        payoff[i] = max(ST - K, 0)
    return np.exp(-r*T) * payoff.mean()