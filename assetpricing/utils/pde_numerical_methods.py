# assetpricing/utils/numerical_methods.py (PDE solver addition)
import numpy as np

def finite_difference_pde(S_max, S_min, K, T, M, N, r, sigma, option_type='call'):
    """
    Finite difference method for PDE option pricing (Explicit scheme).
    """
    dS = (S_max - S_min)/M
    dt = T/N
    S = np.linspace(S_min, S_max, M+1)
    V = np.zeros((M+1, N+1))
    
    # Terminal condition
    if option_type == 'call':
        V[:, -1] = np.maximum(S - K, 0)
    else:
        V[:, -1] = np.maximum(K - S, 0)
        
    # Boundary conditions
    V[0, :] = 0 if option_type == 'call' else K*np.exp(-r*dt*(N - np.arange(N+1)))
    V[-1, :] = (S_max - K*np.exp(-r*dt*(N - np.arange(N+1)))) if option_type == 'call' else 0
    
    # Explicit scheme
    for j in reversed(range(N)):
        for i in range(1, M):
            alpha = 0.5*sigma**2*S[i]**2*dt/dS**2
            beta = 0.5*r*S[i]*dt/dS
            V[i,j] = (alpha*(V[i+1,j+1] - 2*V[i,j+1] + V[i-1,j+1]) +
                     beta*(V[i+1,j+1] - V[i-1,j+1]) +
                     (1 - r*dt)*V[i,j+1])
            
    return V