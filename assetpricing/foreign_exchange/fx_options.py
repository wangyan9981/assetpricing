# assetpricing/foreign_exchange/fx_options.py
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

def garman_kohlhagen(S, K, T, rd, rf, sigma, option_type='call'):
    """
    FX option pricing using Garman-Kohlhagen model.
    """
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    if option_type == 'call':
        price = S*np.exp(-rf*T)*norm.cdf(d1) - K*np.exp(-rd*T)*norm.cdf(d2)
    else:
        price = K*np.exp(-rd*T)*norm.cdf(-d2) - S*np.exp(-rf*T)*norm.cdf(-d1)
        
    return price

def fx_delta(S, K, T, rd, rf, sigma, option_type='call'):
    """Delta sensitivity for FX options"""
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == 'call':
        return np.exp(-rf*T) * norm.cdf(d1)
    else:
        return np.exp(-rf*T) * (norm.cdf(d1) - 1)

def fx_gamma(S, K, T, rd, rf, sigma):
    """Gamma sensitivity for FX options"""
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return np.exp(-rf*T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))

def fx_vega(S, K, T, rd, rf, sigma):
    """Vega sensitivity for FX options"""
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * np.exp(-rf*T) * norm.pdf(d1) * np.sqrt(T)

def fx_theta(S, K, T, rd, rf, sigma, option_type='call'):
    """Theta sensitivity for FX options"""
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    term1 = -S * np.exp(-rf*T) * norm.pdf(d1) * sigma / (2*np.sqrt(T))
    term2 = rf * S * np.exp(-rf*T) * norm.cdf(d1)
    term3 = rd * K * np.exp(-rd*T) * norm.cdf(d2)
    
    if option_type == 'call':
        return term1 - term2 + term3
    else:
        return term1 + term2 - term3

def fx_rho(S, K, T, rd, rf, sigma, option_type='call'):
    """Rho sensitivity for domestic interest rate"""
    d2 = (np.log(S/K) + (rd - rf - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    if option_type == 'call':
        return K * T * np.exp(-rd*T) * norm.cdf(d2)
    else:
        return -K * T * np.exp(-rd*T) * norm.cdf(-d2)

def fx_phi(S, K, T, rd, rf, sigma, option_type='call'):
    """Phi sensitivity for foreign interest rate"""
    d1 = (np.log(S/K) + (rd - rf + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    
    if option_type == 'call':
        return -S * T * np.exp(-rf*T) * norm.cdf(d1)
    else:
        return S * T * np.exp(-rf*T) * norm.cdf(-d1)

def implied_volatility(price, S, K, T, rd, rf, option_type='call', tol=1e-6):
    """Calculate implied volatility using Newton-Raphson"""
    def vol_func(sigma):
        return garman_kohlhagen(S, K, T, rd, rf, sigma, option_type) - price
    
    try:
        return newton_raphson(vol_func, x0=0.2, tol=tol)
    except:
        # Fallback to binary search
        low, high = 0.001, 2.0
        for _ in range(100):
            mid = (low + high)/2
            diff = vol_func(mid)
            if abs(diff) < tol:
                return mid
            if diff > 0:
                high = mid
            else:
                low = mid
        raise ValueError("Implied vol failed to converge")

# In price_sensitivity function, adjust the parameter key for spot
def price_sensitivity(S, K, T, rd, rf, sigma, param_range, parameter):
    prices = []
    for val in param_range:
        args = {
            'S': S,
            'K': K,
            'T': T,
            'rd': rd,
            'rf': rf,
            'sigma': sigma
        }
        # Map 'spot' parameter to 'S' expected by the pricing model
        param_key = 'S' if parameter == 'spot' else parameter
        args[param_key] = val
        prices.append(garman_kohlhagen(**args))
    return np.array(prices)

def volatility_smile(S, T, rd, rf, strikes, prices, option_type='call'):
    """
    Calculate volatility smile from market prices
    
    Returns:
    dict: {strike: implied_vol}
    """
    smile = {}
    for K, price in zip(strikes, prices):
        try:
            iv = implied_volatility(price, S, K, T, rd, rf, option_type)
            smile[K] = iv
        except:
            smile[K] = np.nan
    return smile



def plot_sensitivity(param_range, parameter, S, K, T, rd, rf, sigma, show=True, save_path=None):
    """
    Visualize option price sensitivity to different parameters
    
    Parameters:
    param_range (array-like): Range of parameter values to test
    parameter (str): Name of parameter to analyze ('spot', 'sigma', 'T', 'rd', 'rf')
    S (float): Spot price
    K (float): Strike price
    T (float): Time to maturity (in years)
    rd (float): Domestic risk-free rate
    rf (float): Foreign risk-free rate
    sigma (float): Volatility
    show (bool): Whether to display the plot immediately
    save_path (str): Optional path to save the figure
    
    Returns:
    fig, ax: Matplotlib figure and axis objects
    """
    # Parameter name mapping to model parameters
    param_mapping = {
        'spot': 'S',
        'sigma': 'sigma',
        'T': 'T',
        'rd': 'rd',
        'rf': 'rf'
    }
    
    # Calculate prices
    prices = []
    for val in param_range:
        kwargs = {
            'S': S,
            'K': K,
            'T': T,
            'rd': rd,
            'rf': rf,
            'sigma': sigma
        }
        kwargs[param_mapping[parameter]] = val
        prices.append(garman_kohlhagen(**kwargs))
    
    # Create labels
    labels = {
        'spot': ('EUR/USD Spot Rate', 'Spot Sensitivity'),
        'sigma': ('Volatility (σ)', 'Volatility Sensitivity'),
        'T': ('Time to Maturity (years)', 'Time Sensitivity'),
        'rd': ('Domestic Interest Rate', 'Rate Sensitivity (Domestic)'),
        'rf': ('Foreign Interest Rate', 'Rate Sensitivity (Foreign)')
    }
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(param_range, prices, lw=2.5, color='royalblue')
    ax.set_title(f"FX Call Option {labels[parameter][1]}\n(K={K}, T={T:.2f} yrs)")
    ax.set_xlabel(labels[parameter][0])
    ax.set_ylabel('Option Premium')
    ax.grid(True, alpha=0.3)
    ax.spines[['top', 'right']].set_visible(False)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
    
    if show:
        plt.show()
    
    return fig, ax

def plot_volatility_surface(S, K, T_values, rd, rf, sigma_range):
    """
    Create 3D volatility surface plot
    
    Parameters:
    S (float): Spot price
    K (float): Strike price
    T_values (array-like): Range of time to maturity values
    rd (float): Domestic risk-free rate
    rf (float): Foreign risk-free rate
    sigma_range (array-like): Range of volatility values
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    T_grid, sigma_grid = np.meshgrid(T_values, sigma_range)
    prices = np.zeros_like(T_grid)
    
    for i in range(T_grid.shape[0]):
        for j in range(T_grid.shape[1]):
            prices[i,j] = garman_kohlhagen(S, K, T_grid[i,j], rd, rf, sigma_grid[i,j])
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(T_grid, sigma_grid, prices, cmap='viridis',
                          linewidth=0, antialiased=True)
    
    ax.set_xlabel('Time to Maturity (years)')
    ax.set_ylabel('Volatility (σ)')
    ax.set_zlabel('Option Price')
    ax.set_title('FX Option Volatility Surface')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()