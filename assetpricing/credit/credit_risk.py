# assetpricing/credit/credit_risk.py
import numpy as np

class CreditDefaultSwap:
    """
    Fixed CreditDefaultSwap implementation with proper term structure handling
    """
    def __init__(self, notional, spread, recovery_rate, term_structure):
        """
        term_structure: dict {year: discount_factor}
        """
        self.notional = notional
        self.spread = spread
        self.recovery = recovery_rate
        self.terms = term_structure  # Store as dictionary

    def survival_probability(self, hazard_rate, t):
        return np.exp(-hazard_rate * t)
    
    def pv_premium_leg(self, hazard_rate):
        pv = 0
        for t, df in self.terms.items():  # Fixed iteration
            pv += self.spread * df * self.survival_probability(hazard_rate, t)
        return self.notional * pv
    
    def pv_default_leg(self, hazard_rate):
        pv = 0
        prev_sp = 1
        for t, df in self.terms.items():  # Fixed iteration
            sp = self.survival_probability(hazard_rate, t)
            default_prob = prev_sp - sp
            pv += default_prob * df * (1 - self.recovery)
            prev_sp = sp
        return self.notional * pv
    
    def fair_spread(self, hazard_rate):
        return self.pv_default_leg(hazard_rate) / self.pv_premium_leg(1)