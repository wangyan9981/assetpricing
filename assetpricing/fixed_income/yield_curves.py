# assetpricing/fixed_income/yield_curves.py
import numpy as np
from ..utils.newton_numerical_methods import newton_raphson

class YieldCurveBootstrapper:
    """
    Bootstrap zero-coupon yield curve from par rates with visualization
    """
    def __init__(self, instruments):
        """
        instruments: List of tuples (maturity, par_rate)
        """
        self.instruments = sorted(instruments, key=lambda x: x[0])
        self.zero_rates = []
        self.zero_curve = {}
        
    def bootstrap(self):
        """Calculate zero rates through bootstrapping"""
        self.zero_rates = []
        self.zero_curve = {}
        
        for T, par_rate in self.instruments:
            if T == 0:
                self.zero_curve[T] = par_rate
                continue
                
            def func(z):
                discount = np.exp(-z*T)
                coupon = par_rate * sum(np.exp(-z*t) for t, _ in self.instruments if t <= T)
                return coupon + discount - 1
                
            z = newton_raphson(func, x0=par_rate)
            self.zero_curve[T] = z
            
        return self.zero_curve

    def plot_curve(self, plot_forward=True, figsize=(10, 6)):
        """
        Visualize the yield curve
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Matplotlib required for plotting. Install with 'pip install matplotlib'")
        
        plt.figure(figsize=figsize)
        times = sorted(self.zero_curve.keys())
        zeros = [self.zero_curve[t] for t in times]
        
        # Plot zero curve
        plt.plot(times, zeros, 'bo-', label='Zero Rates')
        
        # Calculate and plot forward rates
        if plot_forward and len(times) > 1:
            forwards = []
            forward_times = []
            for i in range(1, len(times)):
                T1, T2 = times[i-1], times[i]
                fr = self.forward_rate(T1, T2)
                forwards.append(fr)
                forward_times.append((T1 + T2)/2)
                
            plt.plot(forward_times, forwards, 'rs--', label='1Y Forward Rates')
        
        plt.title("Yield Curve Bootstrapping Results")
        plt.xlabel("Maturity (Years)")
        plt.ylabel("Rate")
        plt.legend()
        plt.grid(True)
        plt.show()

    def forward_rate(self, T1, T2):
        """
        Calculate forward rate between T1 and T2
        """
        z1 = self.zero_curve[T1]
        z2 = self.zero_curve[T2]
        return (z2*T2 - z1*T1)/(T2 - T1)