# assetpricing/derivatives/futures.py
import numpy as np

class CostOfCarry:
    """
    Updated implementation with storage cost handling
    """
    @staticmethod
    def fair_value(spot_price, risk_free_rate, time_to_maturity,
                  storage_cost=0, convenience_yield=0, dividend_yield=0):
        cost_of_carry = risk_free_rate + storage_cost - convenience_yield - dividend_yield
        return spot_price * np.exp(cost_of_carry * time_to_maturity)

    @staticmethod
    def implied_convenience_yield(futures_price, spot_price, risk_free_rate,
                                 time_to_maturity, storage_cost=0, dividend_yield=0):
        """
        Corrected formula including storage costs:
        F = S * e^{(r + c - y - q)T}
        => y = r + c - q - (ln(F/S)/T)
        """
        return (np.log(spot_price/futures_price)/time_to_maturity) + risk_free_rate + storage_cost - dividend_yield