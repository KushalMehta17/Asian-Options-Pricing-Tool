import numpy as np


# ================================
# 1. GBM SIMULATION
# ================================

def simulate_gbm(S0, r, sigma, T, steps, simulations):
    """
    Generate stock price paths using Geometric Brownian Motion (GBM)
    """

    dt = T / steps  # size of each time step

    # Matrix to store all simulated paths
    paths = np.zeros((simulations, steps + 1))

    # Set initial price
    paths[:, 0] = S0

    # Random normal values for simulation
    Z = np.random.standard_normal((simulations, steps))

    # Generate paths step by step
    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    return paths


# ================================
# 2. PAYOFF FUNCTIONS
# ================================

def european_call_payoff(ST, K):
    """
    Payoff for European Call Option
    Uses only final price
    """
    return np.maximum(ST - K, 0)


def asian_call_payoff(paths, K):
    """
    Payoff for Arithmetic Asian Call Option
    Uses average price over the path
    """
    average_price = np.mean(paths, axis=1)  # average across time steps
    return np.maximum(average_price - K, 0)

def european_put_payoff(ST, K):
    return np.maximum(K - ST, 0)


def asian_put_payoff(paths, K):
    avg_price = np.mean(paths, axis=1)
    return np.maximum(K - avg_price, 0)


# ================================
# 3. MONTE CARLO PRICING
# ================================

def price_european_mc(S0, K, r, sigma, T, steps, simulations, option_type="call"):
    """
    Price European Option using Monte Carlo
    """

    paths = simulate_gbm(S0, r, sigma, T, steps, simulations)

    ST = paths[:, -1]

    if option_type == "call":
        payoffs = european_call_payoff(ST, K)
    else:
        payoffs = european_put_payoff(ST, K)

    price = np.exp(-r * T) * np.mean(payoffs)

    return price, payoffs


def price_asian_mc(S0, K, r, sigma, T, steps, simulations, option_type="call"):
    
    paths = simulate_gbm(S0, r, sigma, T, steps, simulations)

    if option_type == "call":
        payoffs = asian_call_payoff(paths, K)
    else:
        payoffs = asian_put_payoff(paths, K)

    price = np.exp(-r * T) * np.mean(payoffs)

    return price, payoffs

# ================================
# 4. RETURN STATS
# ================================

def price_with_stats(S0, K, r, sigma, T, steps, simulations, option_type="asian"):
    """
    Extended function that returns:
    - price
    - standard deviation
    - standard error
    - raw payoffs (for plotting)
    """

    paths = simulate_gbm(S0, r, sigma, T, steps, simulations)

    if option_type == "asian":
        payoffs = asian_call_payoff(paths, K)
    else:
        payoffs = european_call_payoff(paths[:, -1], K)

    # Discount payoffs
    discounted_payoffs = np.exp(-r * T) * payoffs

    price = np.mean(discounted_payoffs)

    # Measure spread of results (variance)
    std_dev = np.std(discounted_payoffs)

    # ✅ NEW: standard error (important for MC accuracy)
    std_error = std_dev / np.sqrt(simulations)

    # ✅ NEW: return payoffs for histogram / comparison
    return price, std_dev, std_error, discounted_payoffs


# ================================
# 5. HELPER (OPTIONAL)
# ================================

def compute_price_from_payoffs(payoffs, r, T):
    """
    Utility to compute discounted price from payoffs
    Useful when variance reduction modifies payoffs
    """
    return np.exp(-r * T) * np.mean(payoffs)