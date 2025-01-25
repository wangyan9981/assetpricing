# assetpricing/risk/sensitivity_analysis.py
import numpy as np
from scipy.stats import multivariate_normal

class GreekCalculator:
    """
    Compute option Greeks using finite difference methods
    """
    @staticmethod
    def delta(pricer, bump=0.01):
        """First order delta calculation"""
        original = pricer(S=pr.S)
        bumped = pricer(S=pr.S * (1 + bump))
        return (bumped - original) / (pr.S * bump)
    
    @staticmethod
    def gamma(pricer, bump=0.01):
        """Second order gamma calculation"""
        up = pricer(S=pr.S * (1 + bump))
        down = pricer(S=pr.S * (1 - bump))
        return (up - 2*original + down) / (pr.S**2 * bump**2)
    
    @staticmethod
    def vega(pricer, bump=0.001):
        """Vega calculation using volatility bump"""
        original = pricer(sigma=pr.sigma)
        bumped = pricer(sigma=pr.sigma + bump)
        return (bumped - original) / bump

class ValueAtRisk:
    """
    Value-at-Risk calculation using variance-covariance method
    """
    def __init__(self, portfolio, confidence_level=0.95):
        self.portfolio = portfolio
        self.confidence = confidence_level
        
    def calculate(self, returns, window=252):
        """Parametric VaR calculation"""
        mu = returns.mean()
        sigma = returns.std()
        z_score = multivariate_normal.ppf(self.confidence)
        return self.portfolio * (mu - z_score * sigma) * np.sqrt(window)