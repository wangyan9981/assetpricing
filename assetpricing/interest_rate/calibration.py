# assetpricing/interest_rate/calibration.py
import numpy as np
from scipy.optimize import minimize

class HullWhiteCalibrator:
    """
    Calibrate Hull-White model parameters to market instruments.
    """
    def __init__(self, mean_reversion_guess, volatility_guess):
        self.a = mean_reversion_guess
        self.sigma = volatility_guess
        
    def calibrate(self, market_prices, maturities, strikes, model):
        """
        market_prices: List of market prices for calibration instruments
        model: Function that calculates model prices given (a, sigma)
        """
        def loss(params):
            a, sigma = params
            model_prices = [model(a, sigma, T, K) for T, K in zip(maturities, strikes)]
            return np.sum((np.array(model_prices) - np.array(market_prices))**2)
            
        result = minimize(loss, [self.a, self.sigma], 
                         bounds=[(0.01, 1), (0.001, 0.1)])
        self.a, self.sigma = result.x
        return result