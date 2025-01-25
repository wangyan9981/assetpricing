# assetpricing/foreign_exchange/fx_spot.py
import numpy as np

class FXSpotArbitrage:
    """
    Detect triangular arbitrage opportunities in FX markets.
    """
    def __init__(self, base_currency, rates):
        """
        rates: Dictionary of exchange rates {currency_pair: rate}
        """
        self.base = base_currency
        self.rates = rates
        
    def check_triangular_arbitrage(self, triangle):
        """
        Check arbitrage for currency triangle (e.g., ['USD', 'EUR', 'GBP'])
        """
        try:
            rate1 = self.rates[f"{triangle[0]}{triangle[1]}"]
            rate2 = self.rates[f"{triangle[1]}{triangle[2]}"]
            rate3 = self.rates[f"{triangle[2]}{triangle[0]}"]
            
            implied_rate = rate1 * rate2 * rate3
            return implied_rate < 0.9999 or implied_rate > 1.0001
        except KeyError:
            return False

    def calculate_arbitrage_profit(self, triangle, notional):
        rate1 = self.rates[f"{triangle[0]}{triangle[1]}"]
        rate2 = self.rates[f"{triangle[1]}{triangle[2]}"]
        rate3 = self.rates[f"{triangle[2]}{triangle[0]}"]
        
        cycle_return = notional * rate1 * rate2 * rate3
        return cycle_return - notional