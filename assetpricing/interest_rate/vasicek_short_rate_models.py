# assetpricing/interest_rate/short_rate_models.py
import numpy as np

class VasicekModel:
    """
    Vasicek short rate model: dr_t = a(b - r_t)dt + σdW_t
    Enhanced with visualization capabilities
    
    Parameters:
    a (float): Mean reversion speed
    b (float): Long-term mean rate
    sigma (float): Volatility
    r0 (float): Initial interest rate
    """
    def __init__(self, a, b, sigma, r0):
        self.a = a
        self.b = b
        self.sigma = sigma
        self.r0 = r0

    def simulate(self, T, steps, n_paths):
        """
        Simulate interest rate paths
        
        Args:
            T (float): Time horizon in years
            steps (int): Number of time steps
            n_paths (int): Number of paths to simulate
            
        Returns:
            np.ndarray: Array of simulated rates (steps+1 x n_paths)
        """
        dt = T/steps
        rates = np.zeros((steps+1, n_paths))
        rates[0] = self.r0
        
        for t in range(1, steps+1):
            dW = np.random.normal(scale=np.sqrt(dt), size=n_paths)
            rates[t] = rates[t-1] + self.a*(self.b - rates[t-1])*dt + self.sigma*dW
            
        return rates

    def plot_simulation(self, rates, T, figsize=(10, 6), title=None, show_plot=True):
        """
        Visualize simulated rate paths
        
        Args:
            rates (np.ndarray): Simulated rates from simulate() method
            T (float): Time horizon used in simulation
            figsize (tuple): Figure dimensions
            title (str): Custom plot title
            show_plot (bool): Whether to immediately display the plot
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib is required for plotting. Install with 'pip install matplotlib'")
        
        plt.figure(figsize=figsize)
        time = np.linspace(0, T, rates.shape[0])
        
        # Plot individual paths
        for i in range(rates.shape[1]):
            plt.plot(time, rates[:, i], lw=1, alpha=0.6)
            
        # Plot theoretical mean
        plt.plot(time, self._theoretical_mean(time), 'k--', lw=2, label='Theoretical Mean')
        
        # Formatting
        plt.title(title or f"Vasicek Model Simulation\n(a={self.a}, b={self.b}, σ={self.sigma}, r0={self.r0})")
        plt.xlabel('Time (years)')
        plt.ylabel('Interest Rate')
        plt.grid(True)
        plt.legend()
        
        if show_plot:
            plt.show()

    def _theoretical_mean(self, t):
        """Calculate theoretical mean rate at time t"""
        return self.b + (self.r0 - self.b)*np.exp(-self.a*t)

# Example usage:
if __name__ == "__main__":
    model = VasicekModel(a=0.1, b=0.05, sigma=0.02, r0=0.03)
    simulated_rates = model.simulate(T=5, steps=250, n_paths=10)
    model.plot_simulation(simulated_rates, T=5)