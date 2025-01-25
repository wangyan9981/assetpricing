# assetpricing/utils/error_handling.py
import functools
import numpy as np

def validate_inputs(func):
    """Decorator for input validation"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Check numerical inputs
        for arg in args:
            if isinstance(arg, (int, float)) and not np.isfinite(arg):
                raise ValueError(f"Invalid input value: {arg}")
                
        # Check array inputs
        arr_args = [a for a in args if isinstance(a, np.ndarray)]
        for arr in arr_args:
            if not np.all(np.isfinite(arr)):
                raise ValueError("Array contains invalid values")
                
        return func(*args, **kwargs)
    return wrapper

def check_positive(func):
    """Ensure positive values for specified parameters"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        positive_params = ['S', 'K', 'T', 'sigma']
        for name, value in kwargs.items():
            if name in positive_params and value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        return func(*args, **kwargs)
    return wrapper