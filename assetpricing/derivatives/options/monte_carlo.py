# assetpricing/derivatives/options/monte_carlo.py
import numpy as np

class MonteCarloPricer:
    """
    Path-dependent option pricing using Monte Carlo simulation.
    """
    def __init__(self, S, K, T, r, sigma, n_simulations=100000, n_steps=100):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.n_simulations = n_simulations
        self.n_steps = n_steps
        self.dt = T/n_steps
        
    def asian_option(self, option_type='call'):
        """
        Arithmetic Asian option pricing.
        """
        paths = self._generate_paths()
        averages = paths.mean(axis=1)
        
        if option_type == 'call':
            payoffs = np.maximum(averages - self.K, 0)
        else:
            payoffs = np.maximum(self.K - averages, 0)
            
        return np.exp(-self.r*self.T) * payoffs.mean()
        
    def _generate_paths(self):
        paths = np.zeros((self.n_simulations, self.n_steps+1))
        paths[:,0] = self.S
        
        for t in range(1, self.n_steps+1):
            z = np.random.normal(size=self.n_simulations)
            paths[:,t] = paths[:,t-1] * np.exp(
                (self.r - 0.5*self.sigma**2)*self.dt + 
                self.sigma*np.sqrt(self.dt)*z
            )
            
        return paths