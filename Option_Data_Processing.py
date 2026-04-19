import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime


# ================================
# 1. GET CURRENT STOCK PRICE
# ================================

def get_stock_price(ticker):
    """
    Fetch latest stock price (S0)
    """

    stock = yf.Ticker(ticker)

    price = stock.fast_info.get("lastPrice", None)

    # Fallback if fast_info fails
    if price is None:
        data = stock.history(period="1d")
        if data.empty:
            return None
        price = data["Close"].iloc[-1]

    return float(price)


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
    Find closest European option from market data

    Returns:
    dict with:
    - price
    - strike used
    - expiry used
    """

    stock = yf.Ticker(ticker)

    expiries = stock.options

    if len(expiries) == 0:
        return None

    # Target expiry date
    today = datetime.today()
    target_days = int(T * 365)
    target_date = today + pd.Timedelta(days=target_days)

    # Find closest expiry
    expiry_dates = [pd.to_datetime(e) for e in expiries]
    closest_expiry = min(expiry_dates, key=lambda d: abs(d - target_date))

    # Fetch option chain
    try:
        chain = stock.option_chain(closest_expiry.strftime("%Y-%m-%d"))
    except:
        return None

    calls = chain.calls

    if calls is None or calls.empty:
        return None

    # Clean data (important for robustness)
    calls = calls.dropna()

    if calls.empty:
        return None

    # Find closest strike
    calls = calls.copy()  
    calls["diff"] = abs(calls["strike"] - K)
    closest_row = calls.loc[calls["diff"].idxmin()]

    # ================================
    # BETTER PRICE: MID PRICE
    # ================================

    bid = closest_row.get("bid", 0)
    ask = closest_row.get("ask", 0)

    if bid > 0 and ask > 0:
        market_price = (bid + ask) / 2
    else:
        market_price = closest_row.get("lastPrice", None)

    if market_price is None:
        return None

    return {
        "price": float(market_price),
        "strike": float(closest_row["strike"]),
        "expiry": closest_expiry.strftime("%Y-%m-%d")
    }