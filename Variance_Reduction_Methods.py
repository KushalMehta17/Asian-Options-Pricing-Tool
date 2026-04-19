import numpy as np
from scipy.stats import norm


# ================================
# 1. BLACK-SCHOLES (for control variate)
# ================================

def black_scholes_call(S0, K, T, r, sigma):
    if T <= 0:
        return max(S0 - K, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S0, K, T, r, sigma):
    if T <= 0:
        return max(K - S0, 0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# Optional wrapper (clean usage later)
def black_scholes_price(S0, K, T, r, sigma, option_type="call"):
    if option_type == "call":
        return black_scholes_call(S0, K, T, r, sigma)
    else:
        return black_scholes_put(S0, K, T, r, sigma)


# ================================
# 2. ANTITHETIC VARIATES
# ================================

def antithetic_mc(S0, K, r, sigma, T, steps, simulations, option_type="call"):

    dt = T / steps
    half = simulations // 2

    Z = np.random.standard_normal((half, steps))
    Z_full = np.vstack((Z, -Z))

    paths = np.zeros((2 * half, steps + 1))
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z_full[:, t - 1]
        )

    avg_price = np.mean(paths, axis=1)

    if option_type == "call":
        pay = np.maximum(avg_price - K, 0)
    else:
        pay = np.maximum(K - avg_price, 0)

    # Pair averaging
    pay = 0.5 * (pay[:half] + pay[half:])

    discounted = np.exp(-r * T) * pay

    price = np.mean(discounted)
    std_dev = np.std(discounted)
    std_error = std_dev / np.sqrt(half)

    return price, std_dev, std_error, discounted

# ================================
# 3. CONTROL VARIATE METHOD
# ================================

def control_variate_mc(S0, K, r, sigma, T, steps, simulations, option_type="call"):

    dt = T / steps

    # Generate paths
    Z = np.random.standard_normal((simulations, steps))

    paths = np.zeros((simulations, steps + 1))
    paths[:, 0] = S0

    for t in range(1, steps + 1):
        paths[:, t] = paths[:, t - 1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t - 1]
        )

    avg_price = np.mean(paths, axis=1)
    ST = paths[:, -1]

    # Payoffs
    if option_type == "call":
        asian = np.maximum(avg_price - K, 0)
        euro = np.maximum(ST - K, 0)
    else:
        asian = np.maximum(K - avg_price, 0)
        euro = np.maximum(K - ST, 0)

    asian = np.exp(-r * T) * asian
    euro = np.exp(-r * T) * euro

    # BS price (already discounted)
    bs_price = black_scholes_price(S0, K, T, r, sigma, option_type)

    cov = np.mean((asian - np.mean(asian)) * (euro - np.mean(euro)))
    var = np.var(euro)

    c = -cov / var if var != 0 else 0

    adjusted = asian + c * (euro - bs_price)

    price = np.mean(adjusted)
    std_dev = np.std(adjusted)
    std_error = std_dev / np.sqrt(simulations)

    return price, std_dev, std_error, adjusted