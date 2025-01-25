# assetpricing/utils/numerical_methods.py
def newton_raphson(f, x0, tol=1e-6, max_iter=100):
    """
    Newton-Raphson root-finding method
    
    Parameters:
    f: function - Target function to find root for
    x0: float - Initial guess
    tol: float - Tolerance for convergence
    max_iter: int - Maximum iterations
    """
    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        fpx = (f(x + 1e-6) - fx) / 1e-6  # Numerical derivative
        x = x - fx/fpx
    raise ValueError(f"Newton-Raphson failed to converge after {max_iter} iterations")