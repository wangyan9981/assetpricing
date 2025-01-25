# assetpricing/interest_rate/short_rate_models.py (CIR model addition)
import numpy as np

class CIRModel:
    """
    Cox-Ingersoll-Ross model: dr_t = a(b - r_t)dt + σ√r_t dW_t
    """
    def __init__(self, a, b, sigma, r0):
        self.a = a  # Mean reversion speed
        self.b = b  # Long-term mean
        self.sigma = sigma  # Volatility
        self.r0 = r0  # Initial rate

    def simulate(self, T, steps, n_paths):
        dt = T/steps
        rates = np.zeros((steps+1, n_paths))
        rates[0] = self.r0
        
        for t in range(1, steps+1):
            dW = np.random.normal(scale=np.sqrt(dt), size=n_paths)
            drift = self.a*(self.b - rates[t-1])*dt
            diffusion = self.sigma * np.sqrt(np.abs(rates[t-1])) * dW
            rates[t] = np.maximum(rates[t-1] + drift + diffusion, 0)
            
        return rates