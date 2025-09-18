import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# 1. Download Data
# -----------------------------
tickers = ["AAPL", "MSFT", "TSLA"]

# Download daily data since 2020
data = yf.download(tickers, start="2020-01-01")

# Extract just the 'Close' prices
prices = data["Close"]

# Save raw prices to CSV
prices.to_csv("prices.csv")

print("Price data (first 5 rows):")
print(prices.head())

# Plot example chart
prices["AAPL"].plot(title="AAPL Price History", figsize=(10,6))
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.show()

# -----------------------------
# 2. Calculate Returns
# -----------------------------
# Daily log returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

print("\nDaily returns (first 5 rows):")
print(returns.head())

# Mean daily returns and covariance
mean_returns = returns.mean()
cov_matrix = returns.cov()

print("\nMean daily returns:")
print(mean_returns)

print("\nCovariance matrix:")
print(cov_matrix)

# -----------------------------
# 3. Portfolio Performance Function
# -----------------------------
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free=0.0):
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - risk_free) / port_vol
    return port_return, port_vol, sharpe

# Test equal weights
num_assets = len(mean_returns)
weights = np.array([1/num_assets] * num_assets)

port_return, port_vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix)
print("\nExample Portfolio (Equal Weights):")
print(f"Return: {port_return:.2%}")
print(f"Volatility: {port_vol:.2%}")
print(f"Sharpe Ratio: {sharpe:.2f}")

# -----------------------------
# 4. Monte Carlo Simulation
# -----------------------------
def monte_carlo_simulation(num_portfolios, mean_returns, cov_matrix, risk_free=0.0):
    results = []
    num_assets = len(mean_returns)

    for _ in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)

        port_return, port_vol, sharpe = portfolio_performance(weights, mean_returns, cov_matrix, risk_free)
        results.append([port_return, port_vol, sharpe, weights])

    results_df = pd.DataFrame(results, columns=["Return", "Volatility", "Sharpe", "Weights"])
    return results_df

results_df = monte_carlo_simulation(5000, mean_returns, cov_matrix)

print("\nSample simulated portfolios:")
print(results_df.head())

# Plot Monte Carlo portfolios
plt.figure(figsize=(10,6))
plt.scatter(results_df["Volatility"], results_df["Return"],
            c=results_df["Sharpe"], cmap="viridis", alpha=0.7)
plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.title("Monte Carlo Portfolio Simulation")
plt.show()

# -----------------------------
# 5. Optimization with SciPy
# -----------------------------
def neg_sharpe(weights, mean_returns, cov_matrix, risk_free=0.0):
    return -portfolio_performance(weights, mean_returns, cov_matrix, risk_free)[2]

def portfolio_volatility(weights, mean_returns, cov_matrix):
    return portfolio_performance(weights, mean_returns, cov_matrix)[1]

def optimize_portfolio(mean_returns, cov_matrix, risk_free=0.0):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for asset in range(num_assets))
    init_guess = num_assets * [1. / num_assets]

    # Max Sharpe
    opt_sharpe = minimize(neg_sharpe, init_guess, args=args,
                          method='SLSQP', bounds=bounds, constraints=constraints)

    # Min Volatility
    opt_min_vol = minimize(portfolio_volatility, init_guess, args=(mean_returns, cov_matrix),
                           method='SLSQP', bounds=bounds, constraints=constraints)

    return opt_sharpe, opt_min_vol

opt_sharpe, opt_min_vol = optimize_portfolio(mean_returns, cov_matrix)

print("\nMax Sharpe Portfolio:")
print("Weights:", opt_sharpe.x)
print("Performance:", portfolio_performance(opt_sharpe.x, mean_returns, cov_matrix))

print("\nMin Volatility Portfolio:")
print("Weights:", opt_min_vol.x)
print("Performance:", portfolio_performance(opt_min_vol.x, mean_returns, cov_matrix))
