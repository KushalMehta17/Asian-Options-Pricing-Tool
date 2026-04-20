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

    try:
        stock = yf.Ticker(ticker)

        price = stock.fast_info.get("lastPrice", None)

        if price is None:
            data = stock.history(period="1d")
            if not data.empty:
                price = data["Close"].iloc[-1]

        return float(price) if price else None

    except Exception:
        return None


# ================================
# 2. GET HISTORICAL DATA
# ================================

def get_historical_data(ticker, period="1y"):
    """
    Fetch historical stock data
    """

    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period)

        if data.empty:
            return None

        return data["Close"]

    except Exception:
        return None


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

    return float(daily_vol * np.sqrt(252))


# ================================
# 4. GET OPTION CHAIN
# ================================

def get_option_chain(ticker):
    """
    Fetch all available option expiries
    """

    try:
        stock = yf.Ticker(ticker)
        expiries = stock.options
        return stock, expiries
    except Exception:
        return None, []


# ================================
# 5. FIND CLOSEST MARKET OPTION
# ================================

def get_closest_option_price(ticker, K, T, option_type="call"):
    """
    Fetch closest European option (CALL or PUT)

    option_type: "call" or "put"
    """

    try:
        stock = yf.Ticker(ticker)

        expiries = stock.options
        if not expiries:
            return None

        # ------------------------
        # Find closest expiry
        # ------------------------
        today = datetime.today()
        target_days = int(T * 365)
        target_date = today + pd.Timedelta(days=target_days)

        expiry_dates = [pd.to_datetime(e) for e in expiries]
        closest_expiry = min(expiry_dates, key=lambda d: abs(d - target_date))

        # ------------------------
        # Fetch option chain
        # ------------------------
        chain = stock.option_chain(closest_expiry.strftime("%Y-%m-%d"))

        if option_type == "call":
            options_df = chain.calls.copy()
        else:
            options_df = chain.puts.copy()

        if options_df.empty:
            return None

        # ------------------------
        # Find closest strike
        # ------------------------
        options_df["diff"] = abs(options_df["strike"] - K)
        closest_row = options_df.loc[options_df["diff"].idxmin()]

        # ------------------------
        # Extract price safely
        # ------------------------
        price = closest_row.get("lastPrice", None)

        if price is None or np.isnan(price):
            price = closest_row.get("bid", 0)

        if price is None or np.isnan(price):
            price = closest_row.get("ask", 0)

        if price is None:
            return None

        return {
            "price": float(price),
            "strike": float(closest_row["strike"]),
            "expiry": closest_expiry.strftime("%Y-%m-%d"),
            "type": option_type
        }

    except Exception:
        return None