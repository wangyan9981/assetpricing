# assetpricing/utils/date_handling.py
from datetime import datetime

def day_count(start_date, end_date, convention='ACT/360'):
    """
    Calculate day count fraction between two dates
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        
    delta = end_date - start_date
    days = delta.days
    
    if convention == 'ACT/360':
        return days / 360
    elif convention == 'ACT/365':
        return days / 365
    elif convention == '30/360':
        d1 = min(30, start_date.day)
        d2 = min(30, end_date.day) if d1 == 30 else end_date.day
        return ((end_date.year - start_date.year) * 360 +
                (end_date.month - start_date.month) * 30 +
                d2 - d1) / 360
    else:
        raise ValueError(f"Unknown convention: {convention}")