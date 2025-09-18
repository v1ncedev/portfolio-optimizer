import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -----------------------------
# User Input
# -----------------------------
tickers = input("Enter tickers separated by spaces (e.g. AAPL MSFT TSLA): ").split()
start_date = input("Enter start date (YYYY-MM-DD): ") or "2020-01-01"

print(f"\nDownloading data for {tickers} since {start_date}...")

# -----------------------------
# 1. Download Data
# -----------------------------
data = yf.download(tickers, start=start_date)

# Extract just the 'Close' prices
prices = data["Close"]

# Save raw prices to CSV
prices.to_csv("prices.csv")

print("\nPrice data (first 5 rows):")
print(prices.head())

# -----------------------------
# 2. Calculate Returns
# -----------------------------
# Daily log returns
returns = np.log(prices / prices.shift(1))
returns = returns.dropna()

mean_returns = returns.mean()
cov_matrix = returns.cov()

print("\nMean daily returns:")
print(mean_returns)

# -----------------------------
# 3. Portfolio Performance Function
# -----------------------------
def portfolio_performance(weights, mean_returns, cov_matrix, risk_free=0.0):
    port_return = np.sum(mean_returns * weights) * 252
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (port_return - risk_free) / port_vol
    return port_return, port_vol, sharpe

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

# Generate 20k portfolios for a denser cloud
results_df = monte_carlo_simulation(20000, mean_returns, cov_matrix)

print(f"\nSimulated {len(results_df)} random portfolios.")

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

# -----------------------------
# 6. Results
# -----------------------------
sharpe_ret, sharpe_vol, sharpe_ratio = portfolio_performance(opt_sharpe.x, mean_returns, cov_matrix)
minvol_ret, minvol_vol, minvol_ratio = portfolio_performance(opt_min_vol.x, mean_returns, cov_matrix)

print("\nMax Sharpe Portfolio:")
print("Weights:", opt_sharpe.x.round(3))
print(f"Return: {sharpe_ret:.2%}, Volatility: {sharpe_vol:.2%}, Sharpe: {sharpe_ratio:.2f}")

print("\nMin Volatility Portfolio:")
print("Weights:", opt_min_vol.x.round(3))
print(f"Return: {minvol_ret:.2%}, Volatility: {minvol_vol:.2%}, Sharpe: {minvol_ratio:.2f}")

# -----------------------------
# 7. Plot Efficient Frontier
# -----------------------------
plt.figure(figsize=(10,6))

# Monte Carlo scatter
plt.scatter(results_df["Volatility"], results_df["Return"],
            c=results_df["Sharpe"], cmap="viridis", alpha=0.7)

# Highlight Max Sharpe (gold star)
plt.scatter(sharpe_vol, sharpe_ret, c="gold", marker="*", s=300, label="Max Sharpe")

# Highlight Min Volatility (red star)
plt.scatter(minvol_vol, minvol_ret, c="red", marker="*", s=300, label="Min Volatility")

plt.colorbar(label="Sharpe Ratio")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.title("Efficient Frontier with Optimal Portfolios")
plt.legend()
plt.show()
