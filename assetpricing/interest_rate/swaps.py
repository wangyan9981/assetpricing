# assetpricing/interest_rate/swaps.py
import numpy as np

class InterestRateSwap:
    """
    Vanilla interest rate swap pricing.
    """
    def __init__(self, notional, fixed_rate, tenor, day_count='30/360'):
        self.notional = notional
        self.fixed_rate = fixed_rate
        self.tenor = tenor  # in years
        self.day_count = day_count
        
    def pv_fixed_leg(self, discount_factors):
        """
        Calculate present value of fixed leg.
        """
        cf = self.notional * self.fixed_rate * self.tenor
        return cf * discount_factors[self.tenor]
    
    def pv_floating_leg(self, forward_rates, discount_factors):
        """
        Calculate present value of floating leg.
        """
        cf = self.notional * forward_rates * self.tenor
        return cf * discount_factors[self.tenor]
    
    def swap_rate(self, discount_factors, floating_leg_pv):
        """
        Calculate fair swap rate.
        """
        annuity = sum([df * t for t, df in discount_factors.items()])
        return floating_leg_pv / (self.notional * annuity)