import matplotlib.pyplot as plt
import numpy as np


# ================================
# 1. PLOT STOCK PATHS
# ================================

def plot_paths(paths, num_paths=10):
    """
    Plot a subset of simulated stock paths
    """

    plt.figure()

    # Plot only a few paths for clarity
    for i in range(min(num_paths, len(paths))):
        plt.plot(paths[i])

    plt.title("Simulated Stock Price Paths")
    plt.xlabel("Time Steps")
    plt.ylabel("Stock Price")

    return plt


# ================================
# 2. CONVERGENCE PLOT (MULTI-METHOD)
# ================================

def plot_convergence(multi_prices_dict):
    """
    Plot convergence for multiple methods

    multi_prices_dict = {
        "Standard MC": [...],
        "Antithetic": [...],
        "Control Variate": [...]
    }
    """

    plt.figure()

    for label, prices in multi_prices_dict.items():
        plt.plot(prices, label=label)

    plt.title("Monte Carlo Convergence Comparison")
    plt.xlabel("Simulation Batches")
    plt.ylabel("Estimated Price")

    plt.legend()

    return plt


# ================================
# 3. PAYOFF DISTRIBUTION (MULTI)
# ================================

def plot_payoff_distribution(payoffs_dict):
    """
    Plot payoff distributions for different methods

    payoffs_dict = {
        "Standard MC": [...],
        "Antithetic": [...],
        "Control Variate": [...]
    }
    """

    plt.figure()

    for label, payoffs in payoffs_dict.items():
        plt.hist(payoffs, bins=50, alpha=0.5, label=label)

    plt.title("Payoff Distribution Comparison")
    plt.xlabel("Payoff")
    plt.ylabel("Frequency")

    plt.legend()

    return plt


# ================================
# 4. PRICE COMPARISON BAR CHART
# ================================

def plot_price_comparison(price_dict):
    """
    Compare different pricing methods

    price_dict = {
        "Asian MC": ...,
        "European MC": ...,
        "European BS": ...,
        "Market": ...
    }
    """

    labels = list(price_dict.keys())
    values = list(price_dict.values())

    plt.figure()

    plt.bar(labels, values)

    plt.title("Option Price Comparison")
    plt.ylabel("Price")

    return plt


# ================================
# 5. VARIANCE COMPARISON BAR CHART
# ================================

def plot_variance_comparison(std_dev_dict):
    """
    Compare standard deviation across methods
    """
    
    ordered_methods = ["Standard MC", "Control Variate", "Antithetic"]

    labels = []
    values = []

    for method in ordered_methods:
        if method in std_dev_dict:
            labels.append(method)
            values.append(std_dev_dict[method])

    plt.figure()

    plt.bar(labels, values)

    plt.title("Variance Reduction Comparison (Std Dev)")
    plt.ylabel("Standard Deviation")

    return plt