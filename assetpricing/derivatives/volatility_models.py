# assetpricing/derivatives/volatility_models.py
import numpy as np
from scipy.stats import norm
from ..derivatives.options.black_scholes import black_scholes_price


def implied_volatility(price, S, K, T, r, option_type='call', tol=1e-6, max_iter=100):
    """
    Robust implied volatility calculation with fallback methods
    
    Parameters:
    price (float): Market price of the option
    S (float): Spot price
    K (float): Strike price
    T (float): Time to expiration (years)
    r (float): Risk-free rate
    option_type (str): 'call' or 'put'
    tol (float): Tolerance for convergence
    max_iter (int): Maximum iterations
    
    Returns:
    float: Implied volatility
    """
    # Newton-Raphson with proper vega calculation
    sigma = 0.5  # More conservative initial guess
    for _ in range(max_iter):
        # Calculate Black-Scholes price and derivatives
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        if option_type == 'call':
            bs_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
        else:
            bs_price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
            
        diff = bs_price - price
        vega = S * norm.pdf(d1) * np.sqrt(T)
        
        if abs(diff) < tol:
            return sigma
            
        # Avoid division by zero and numerical instability
        if vega < 1e-12:
            break
            
        sigma -= diff/vega

    # Fallback to bisection if Newton-Raphson fails
    low, high = 0.001, 5.0
    for _ in range(max_iter):
        mid = (low + high) / 2
        d1_mid = (np.log(S/K) + (r + 0.5*mid**2)*T) / (mid*np.sqrt(T))
        d2_mid = d1_mid - mid*np.sqrt(T)
        
        if option_type == 'call':
            price_mid = S * norm.cdf(d1_mid) - K * np.exp(-r*T) * norm.cdf(d2_mid)
        else:
            price_mid = K * np.exp(-r*T) * norm.cdf(-d2_mid) - S * norm.cdf(-d1_mid)
            
        if abs(price_mid - price) < tol:
            return mid
            
        if price_mid < price:
            low = mid
        else:
            high = mid

    raise ValueError(f"Implied vol failed to converge (Price: {price:.2f}, S/K: {S}/{K})")


def plot_volatility_surface(S, r, strikes, maturities, prices, option_type='call', figsize=(12, 6)):
    """
    Visualize 3D volatility surface from market prices
    
    Parameters:
    S (float): Spot price
    r (float): Risk-free rate
    strikes (np.array): Array of strike prices
    maturities (np.array): Array of maturities in years
    prices (2D np.array): Matrix of market prices [strike × maturity]
    """
    try:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
    except ImportError:
        raise ImportError("Matplotlib required. Install with 'pip install matplotlib'")
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    X, Y = np.meshgrid(strikes, maturities)
    iv_matrix = np.zeros_like(X)
    
    for i, T in enumerate(maturities):
        for j, K in enumerate(strikes):
            try:
                iv = implied_volatility(prices[i,j], S, K, T, r, option_type)
                iv_matrix[i,j] = iv
            except:
                iv_matrix[i,j] = np.nan
    
    surf = ax.plot_surface(X, Y, iv_matrix, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Strike Price')
    ax.set_ylabel('Time to Maturity')
    ax.set_zlabel('Implied Volatility')
    ax.set_title(f"Volatility Surface (S={S}, r={r})")
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

def plot_volatility_smile(S, T, r, strikes, prices, option_type='call', figsize=(10,6)):
    """
    Plot volatility smile from market prices
    
    Parameters:
    S (float): Spot price
    T (float): Time to maturity
    r (float): Risk-free rate
    strikes (np.array): Strike prices
    prices (np.array): Market prices corresponding to strikes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required. Install with 'pip install matplotlib'")
    
    ivs = []
    for K, price in zip(strikes, prices):
        try:
            iv = implied_volatility(price, S, K, T, r, option_type)
            ivs.append(iv)
        except:
            ivs.append(np.nan)
    
    plt.figure(figsize=figsize)
    plt.plot(strikes, ivs, 'bo-')
    plt.title(f"Volatility Smile (S={S}, T={T}, r={r})")
    plt.xlabel("Strike Price")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.show()

def plot_volatility_term_structure(S, r, K, maturities, prices, option_type='call', figsize=(10,6)):
    """
    Plot volatility term structure
    
    Parameters:
    S (float): Spot price
    r (float): Risk-free rate
    K (float): Strike price
    maturities (np.array): Array of maturities in years
    prices (np.array): Market prices corresponding to maturities
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Matplotlib required. Install with 'pip install matplotlib'")
    
    ivs = []
    for T, price in zip(maturities, prices):
        try:
            iv = implied_volatility(price, S, K, T, r, option_type)
            ivs.append(iv)
        except:
            ivs.append(np.nan)
    
    plt.figure(figsize=figsize)
    plt.plot(maturities, ivs, 'ro-')
    plt.title(f"Term Structure (S={S}, K={K}, r={r})")
    plt.xlabel("Time to Maturity")
    plt.ylabel("Implied Volatility")
    plt.grid(True)
    plt.show()