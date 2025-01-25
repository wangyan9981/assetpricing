# assetpricing/interest_rate/advanced_models.py
import numpy as np
from scipy.linalg import cholesky

class HJMModel:
    """
    Heath-Jarrow-Morton framework for forward rate modeling
    """
    def __init__(self, num_factors, num_periods, delta_t):
        self.num_factors = num_factors
        self.num_periods = num_periods
        self.delta_t = delta_t
        self.forward_rates = None
        self.sigma = None

    def calibrate_volatility(self, market_vols):
        """Calibrate volatility structure to market data"""
        self.sigma = np.array(market_vols)
        
    def simulate_forward_rates(self, initial_curve, num_simulations):
        """Simulate forward rate paths"""
        n = self.num_periods
        self.forward_rates = np.zeros((num_simulations, n+1, n+1))
        self.forward_rates[:,0,:] = initial_curve
        
        for sim in range(num_simulations):
            for t in range(1, n+1):
                dw = np.random.normal(size=self.num_factors)
                for T in range(t, n+1):
                    drift = self._calculate_drift(t, T)
                    volatility = self._calculate_vol(t, T)
                    self.forward_rates[sim,t,T] = self.forward_rates[sim,t-1,T] + \
                        drift*self.delta_t + np.dot(volatility, dw)*np.sqrt(self.delta_t)

        return self.forward_rates

class LMModel:
    """
    Libor Market Model (BGM Model) implementation
    """
    def __init__(self, num_periods, delta_t, initial_rates, corr_matrix):
        self.num_periods = num_periods
        self.delta_t = delta_t
        self.initial_rates = initial_rates
        self.corr_matrix = corr_matrix
        self.vols = None
        self.chol = cholesky(corr_matrix, lower=True)

    def simulate_forward_rates(self, num_simulations, vols):
        """Simulate forward LIBOR rates"""
        self.vols = np.array(vols)
        rates = np.zeros((num_simulations, self.num_periods+1, self.num_periods+1))
        rates[:,0,:] = self.initial_rates
        
        for sim in range(num_simulations):
            for t in range(1, self.num_periods+1):
                dw = self.chol @ np.random.normal(size=self.num_periods)
                for T in range(t, self.num_periods+1):
                    mu = self._calculate_drift(rates[sim,t-1,:], t, T)
                    rates[sim,t,T] = rates[sim,t-1,T] * np.exp(
                        (mu - 0.5*self.vols[T-1]**2)*self.delta_t + 
                        self.vols[T-1]*np.sqrt(self.delta_t)*dw[T-1]
                    )
        return rates