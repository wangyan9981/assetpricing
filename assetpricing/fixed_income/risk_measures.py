# assetpricing/fixed_income/risk_measures.py
import numpy as np
from ..utils.date_handling import day_count  # Add this import if needed

def macaulay_duration(cash_flows, times, ytm, compounding_freq=1):
    """
    Calculate Macaulay duration for a series of cash flows.
    """
    discounted_cf = [cf / (1 + ytm/compounding_freq)**(compounding_freq*t) 
                    for cf, t in zip(cash_flows, times)]
    weights = np.array(discounted_cf) / sum(discounted_cf)
    return np.dot(weights, times)

def modified_duration(macaulay_dur, ytm, compounding_freq=1):
    """
    Convert Macaulay duration to modified duration.
    """
    return macaulay_dur / (1 + ytm/compounding_freq)

def convexity(cash_flows, times, ytm, compounding_freq=1):
    """
    Calculate bond convexity with proper array handling
    """
    times = np.array(times)  # Convert to numpy array
    discounted_cf = np.array([cf / (1 + ytm/compounding_freq)**(compounding_freq*t) 
                            for cf, t in zip(cash_flows, times)])
    weights = discounted_cf / discounted_cf.sum()
    
    # Element-wise operations using numpy arrays
    t_term = times * (times + 1/compounding_freq)
    return np.sum(weights * t_term) / (1 + ytm/compounding_freq)**2