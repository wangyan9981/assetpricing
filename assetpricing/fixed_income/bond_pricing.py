# assetpricing/fixed_income/bond_pricing.py
import numpy as np
from ..utils.newton_numerical_methods import newton_raphson
from ..utils.date_handling import day_count

def zero_coupon_pv(face_value, time_to_maturity, ytm, compounding_freq=1):
    """
    Calculate present value of a zero-coupon bond.
    """
    if compounding_freq == 0:  # Continuous compounding
        return face_value * np.exp(-ytm * time_to_maturity)
    return face_value / (1 + ytm/compounding_freq)**(compounding_freq*time_to_maturity)

def fixed_coupon_bond_pv(coupon_rate, face_value, times, ytm, compounding_freq=1):
    """
    Calculate present value of a fixed coupon bond.
    """
    cash_flows = [coupon_rate * face_value / compounding_freq] * len(times)
    cash_flows[-1] += face_value
    discount_factors = [1/(1 + ytm/compounding_freq)**(compounding_freq*t) for t in times]
    return np.dot(cash_flows, discount_factors)

def calculate_ytm(price, face_value, coupon_rate, times, compounding_freq=1, guess=0.05):
    """
    Calculate yield-to-maturity using Newton-Raphson method.
    """
    def ytm_func(y):
        return fixed_coupon_bond_pv(coupon_rate, face_value, times, y, compounding_freq) - price
    
    return newton_raphson(ytm_func, guess)