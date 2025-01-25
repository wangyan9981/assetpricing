# assetpricing/derivatives/options/exotics.py
import numpy as np
from scipy.stats import norm

class BarrierOption:
    """
    Pricing barrier options using closed-form solution
    """
    def __init__(self, S, K, H, T, r, sigma, option_type='call', barrier_type='down-and-out'):
        self.S = S
        self.K = K
        self.H = H
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type
        self.barrier_type = barrier_type

    def price(self):
        phi = 1 if self.option_type == 'call' else -1
        eta = 1 if 'in' in self.barrier_type else -1
        mu = (self.r - 0.5*self.sigma**2) / self.sigma**2
        lambd = np.sqrt(mu**2 + 2*self.r/(self.sigma**2))
        
        z1 = (np.log(self.S/self.H) / (self.sigma*np.sqrt(self.T))) + (1 + mu)*self.sigma*np.sqrt(self.T)
        z2 = (np.log(self.H/self.S) / (self.sigma*np.sqrt(self.T))) + (1 + mu)*self.sigma*np.sqrt(self.T)
        x1 = np.log(self.S/self.K) / (self.sigma*np.sqrt(self.T)) + (1 + mu)*self.sigma*np.sqrt(self.T)
        x2 = np.log(self.H**2/(self.S*self.K)) / (self.sigma*np.sqrt(self.T)) + (1 + mu)*self.sigma*np.sqrt(self.T)
        
        term1 = self.S * norm.cdf(phi*x1) - self.K*np.exp(-self.r*self.T)*norm.cdf(phi*x1 - phi*self.sigma*np.sqrt(self.T))
        term2 = self.S * norm.cdf(phi*z1) - self.K*np.exp(-self.r*self.T)*norm.cdf(phi*z1 - phi*self.sigma*np.sqrt(self.T))
        term3 = self.S*(self.H/self.S)**(2*(mu + 1)) * norm.cdf(eta*x2) - \
                self.K*np.exp(-self.r*self.T)*(self.H/self.S)**(2*mu) * norm.cdf(eta*x2 - eta*self.sigma*np.sqrt(self.T))
        
        if 'out' in self.barrier_type:
            return term1 - term2 + term3
        else:
            return term2 + term3 - term1

class LookbackOption:
    """
    Floating strike lookback option pricing
    """
    def price(self, S, T, r, sigma, option_type='call'):
        a1 = (np.log(S/S) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        a2 = a1 - sigma*np.sqrt(T)
        if option_type == 'call':
            return S*norm.cdf(a1) - S*np.exp(-r*T)*norm.cdf(a2) - \
                   S*sigma**2/(2*r)*(norm.cdf(a1) - np.exp(-r*T)*(S/S)**(-2*r/sigma**2)*norm.cdf(a2))
        else:
            return S*np.exp(-r*T)*norm.cdf(-a2) - S*norm.cdf(-a1) + \
                   S*sigma**2/(2*r)*(norm.cdf(a1) - np.exp(-r*T)*(S/S)**(-2*r/sigma**2)*norm.cdf(a2))