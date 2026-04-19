import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime


# ================================
# 1. GET CURRENT STOCK PRICE
# ================================

def get_stock_price(ticker):
    """
    Fetch the latest stock price safely
    """

    import yfinance as yf

    try:
        stock = yf.Ticker(ticker)

        # Try fast_info
        price = stock.fast_info.get("lastPrice", None)

        # Fallback 1
        if price is None:
            data = stock.history(period="1d")
            price = data["Close"].iloc[-1]

        return float(price)

    except Exception as e:
        return None


# ================================
# 2. GET HISTORICAL DATA
# ================================

def get_historical_data(ticker, period="1y"):
    """
    Fetch historical stock data
    Default: 1 year (used for volatility)
    """

    stock = yf.Ticker(ticker)
    data = stock.history(period=period)

    if data.empty:
        return None

    return data["Close"]


# ================================
# 3. COMPUTE VOLATILITY
# ================================

def compute_volatility(price_series):
    """
    Compute annualized historical volatility
    """

    if price_series is None or len(price_series) < 2:
        return None

    log_returns = np.log(price_series / price_series.shift(1)).dropna()

    daily_vol = np.std(log_returns)

    annual_vol = daily_vol * np.sqrt(252)

    return float(annual_vol)


# ================================
# 4. GET OPTION CHAIN
# ================================

def get_option_chain(ticker):
    """
    Fetch all available option expiries and chains
    """

    stock = yf.Ticker(ticker)
    expiries = stock.options

    return stock, expiries


# ================================
# 5. FIND CLOSEST MARKET OPTION
# ================================

def get_closest_option_price(ticker, K, T):
    """
    Safe fetch of the closest European option price
    Returns None if it fails 
    """

    import yfinance as yf
    import pandas as pd
    from datetime import datetime

    try:
        stock = yf.Ticker(ticker)

        expiries = stock.options

        if not expiries:
            return None

        today = datetime.today()
        target_days = int(T * 365)
        target_date = today + pd.Timedelta(days=target_days)

        expiry_dates = [pd.to_datetime(e) for e in expiries]
        closest_expiry = min(expiry_dates, key=lambda d: abs(d - target_date))

        chain = stock.option_chain(closest_expiry.strftime("%Y-%m-%d"))
        calls = chain.calls.copy()

        calls["diff"] = abs(calls["strike"] - K)
        closest_row = calls.loc[calls["diff"].idxmin()]

        return {
            "price": float(closest_row["lastPrice"]),
            "strike": float(closest_row["strike"]),
            "expiry": closest_expiry.strftime("%Y-%m-%d")
        }

    except Exception:
        return None
