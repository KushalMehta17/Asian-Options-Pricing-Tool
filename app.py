import streamlit as st
import numpy as np
import pandas as pd

from Monte_Carlo import simulate_gbm
from Variance_Reduction_Methods import antithetic_mc, control_variate_mc, black_scholes_price
from Option_Data_Processing import get_stock_price, get_historical_data, compute_volatility, get_closest_option_price
from Plotting import (
    plot_convergence,
    plot_payoff_distribution,
    plot_price_comparison,
    plot_variance_comparison
)

# ================================
# PAGE CONFIG
# ================================

st.set_page_config(page_title="Asian Option Pricer", layout="wide")

# ================================
# SIDEBAR INPUTS
# ================================

st.sidebar.header("Input Parameters")

ticker = st.sidebar.selectbox(
    "Select Stock",
    [
        # US Stocks
        "AAPL", "MSFT", "GOOG", "AMZN", "TSLA", "META", "NVDA",

        # Indian Stocks
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "ICICIBANK.NS", "SBIN.NS", "HINDUNILVR.NS"
    ]
)

@st.cache_data(ttl=300)
def cached_price(ticker):
    return get_stock_price(ticker)

S0 = cached_price(ticker)

if S0 is None:
    st.warning("Live data unavailable. Using fallback price.")
    S0 = 100

st.sidebar.write(f"**Current Price (S₀):** {S0:.2f}")

option_type = st.sidebar.radio("Option Type", ["call", "put"])

K = st.sidebar.slider(
    "Strike Price (K)",
    min_value=float(0.5 * S0),
    max_value=float(1.5 * S0),
    value=float(S0)
)

T = st.sidebar.slider("Time to Maturity (Years)", 0.1, 2.0, 1.0)
steps = int(252 * T)
st.sidebar.write(f"Time Steps (T x 252): {steps}")

vol_mode = st.sidebar.radio("Volatility Mode", ["Auto", "Manual"])

@st.cache_data(ttl=300)
def cached_volatility(ticker):
    hist_data = get_historical_data(ticker)
    return compute_volatility(hist_data)
    
if vol_mode == "Auto":
    sigma = cached_volatility(ticker)
    if sigma is None:
        st.warning("Volatility unavailable. Using default value.")
        sigma = 0.2
    st.sidebar.write(f"σ (Historical): {sigma:.4f}")
else:
    sigma = st.sidebar.slider("Volatility (σ)", 0.1, 0.6, 0.2)

r = 0.05
st.sidebar.write(f"Risk-Free Rate (r): {r}")

simulations = st.sidebar.slider("Simulations", 20000, 60000, 50000, step=5000)

methods = st.sidebar.multiselect(
    "Variance Reduction Methods",
    ["Standard MC", "Antithetic", "Control Variate"],
    default=["Standard MC", "Antithetic", "Control Variate"]
)

# Submit button
run = st.sidebar.button("Submit")

# ================================
# LANDING PAGE
# ================================

if not run:
    st.title("Asian Option Pricing Tool")

    st.markdown("""
    ### Welcome!

    This app allows you to:
    - Price Asian Options using Monte Carlo
    - Compare with European Option Prices (Black-Scholes & Real Market)
    - Analyze Variance Reduction Techniques

    ### How to Use:
    1. Select stock
    2. Choose Call/Put
    3. Adjust strike (ATM default)
    4. Click **Submit**
    """)
    st.stop()

# ================================
# MAIN COMPUTATION
# ================================

np.random.seed(42)  

paths = simulate_gbm(S0, r, sigma, T, steps, simulations)

avg_price = np.mean(paths, axis=1)
ST = paths[:, -1]

# Payoffs
if option_type == "call":
    asian_payoffs = np.maximum(avg_price - K, 0)
    european_payoffs = np.maximum(ST - K, 0)
else:
    asian_payoffs = np.maximum(K - avg_price, 0)
    european_payoffs = np.maximum(K - ST, 0)

asian_price = np.exp(-r * T) * np.mean(asian_payoffs)
euro_mc_price = np.exp(-r * T) * np.mean(european_payoffs)

# ================================
# VARIANCE METHODS
# ================================

price_dict = {}
std_dev_dict = {}
std_error_dict = {}
payoffs_dict = {}

if "Standard MC" in methods:
    discounted = np.exp(-r * T) * asian_payoffs
    price_dict["Standard MC"] = np.mean(discounted)
    std_dev_dict["Standard MC"] = np.std(discounted)
    std = np.std(discounted)
    std_error_dict["Standard MC"] = std / np.sqrt(simulations)
    payoffs_dict["Standard MC"] = discounted

if "Antithetic" in methods:
    p, s, se, pay = antithetic_mc(S0, K, r, sigma, T, steps, simulations, option_type)
    price_dict["Antithetic"] = p
    std_dev_dict["Antithetic"] = s
    std_error_dict["Antithetic"] = se
    payoffs_dict["Antithetic"] = pay

if "Control Variate" in methods:
    p, s, se, pay = control_variate_mc(S0, K, r, sigma, T, steps, simulations, option_type)
    price_dict["Control Variate"] = p
    std_dev_dict["Control Variate"] = s
    std_error_dict["Control Variate"] = se
    payoffs_dict["Control Variate"] = pay

# ================================
# OTHER PRICES
# ================================

bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)

@st.cache_data(ttl=600)
def cached_option_price(ticker, K, T):
    try:
        return get_closest_option_price(ticker, K, T)
    except Exception:
        return None

market_data = cached_option_price(ticker, K, T)

market_price = market_data["price"] if market_data and "price" in market_data else None

# ================================
# DISPLAY
# ================================

st.header("Results")

cols = st.columns(4)
cols[0].metric("Asian (MC)", f"{asian_price:.2f}")
cols[1].metric("European (MC)", f"{euro_mc_price:.2f}")
cols[2].metric("Black-Scholes", f"{bs_price:.2f}")

if market_price:
    cols[3].metric("Real Market Price", f"{market_price:.2f}")

# ================================
# VARIANCE TABLE
# ================================

st.subheader("Variance Comparison")

ordered_methods = ["Standard MC", "Control Variate", "Antithetic"]

data = []
for m in ordered_methods:
    if m in price_dict:
        data.append({
            "Method": m,
            "Price": round(price_dict[m], 2),
            "Std Error": round(std_error_dict[m], 2)
        })

st.dataframe(pd.DataFrame(data), use_container_width=True)

# ================================
# PLOTS 
# ================================

import matplotlib.pyplot as plt
import seaborn as sns

# Build convergence data
sim_points = np.linspace(1000, simulations, 5, dtype=int)
conv = {}

for method in methods:
    prices = []
    for n in sim_points:
        np.random.seed(42)

        if method == "Standard MC":
            paths_tmp = simulate_gbm(S0, r, sigma, T, steps, n)
            avg = np.mean(paths_tmp, axis=1)

            if option_type == "call":
                pay = np.maximum(avg - K, 0)
            else:
                pay = np.maximum(K - avg, 0)

            prices.append(np.exp(-r*T)*np.mean(pay))

        elif method == "Antithetic":
            p, _, _, _ = antithetic_mc(S0, K, r, sigma, T, steps, n, option_type)
            prices.append(p)

        else:
            p, _, _, _ = control_variate_mc(S0, K, r, sigma, T, steps, n, option_type)
            prices.append(p)

    conv[method] = prices


# ================================
# 2x2 GRID LAYOUT
# ================================

col1, col2 = st.columns(2)
col3, col4 = st.columns(2)

# Price Comparison
with col1:
    st.subheader("Option Prices")
    fig1 = plot_price_comparison({
        "Asian MC": asian_price,
        "European MC": euro_mc_price,
        "Black-Scholes": bs_price,
        "Market": market_price if market_price else 0
    })
    st.pyplot(fig1)


with col2:
    st.subheader("Variance Reduction Comparison")
    fig2 = plot_variance_comparison(std_dev_dict)
    st.pyplot(fig2)


# Convergence Plot
with col3:
    st.subheader("Convergence")

    fig3 = plt.figure(figsize=(5, 4))

    for label, prices in conv.items():
        plt.plot(sim_points, prices, label=label)

    plt.xlabel("Number of Simulations")  
    plt.ylabel("Estimated Price")
    plt.title("Convergence Comparison")
    plt.legend()

    st.pyplot(fig3)


# Payoff Distribution 
with col4:
    st.subheader("Payoff Distribution")

    fig4 = plt.figure(figsize=(5, 4))

    for label, payoffs in payoffs_dict.items():

        payoffs = np.maximum(payoffs, 0)

        max_val = np.percentile(payoffs, 99)

        # Histogram data
        counts, bins = np.histogram(payoffs, bins=50)

        # Convert to line plot (midpoints)
        bin_centers = 0.5 * (bins[1:] + bins[:-1])

        plt.plot(bin_centers, counts, label=label)

        plt.xlim(0, max_val)

    plt.xlabel("Payoff")
    plt.ylabel("Number of Simulations")
    plt.title("Payoff Distribution")
    plt.legend()

    st.pyplot(fig4)

