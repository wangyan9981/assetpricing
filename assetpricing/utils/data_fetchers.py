# assetpricing/utils/data_fetchers.py
import pandas as pd
import yfinance as yf
from datetime import datetime

class MarketDataFetcher:
    @staticmethod
    def get_risk_free_rate(maturity=10):
        """Fetch 10-year Treasury yield as risk-free rate proxy"""
        treasury = yf.Ticker("^TNX")
        hist = treasury.history(period="1d")
        return hist['Close'].iloc[-1]/100  # Convert from percentage
    
    @staticmethod
    def get_fx_rates(base_currency, target_currency):
        """Get current FX spot rate"""
        pair = f"{base_currency}{target_currency}=X"
        data = yf.Ticker(pair)
        return data.history(period="1d")['Close'].iloc[-1]
    
    @staticmethod
    def get_equity_data(ticker, start_date, end_date=datetime.today()):
        """Get historical stock data"""
        return yf.download(ticker, start=start_date, end=end_date)