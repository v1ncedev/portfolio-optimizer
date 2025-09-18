# 📈 Portfolio Optimizer

A Python tool that optimizes stock/ETF portfolios using **Modern Portfolio Theory (MPT)**.  
It calculates portfolio performance, runs Monte Carlo simulations, and finds the **optimal portfolios** (maximum Sharpe ratio and minimum volatility).  

---

## 🚀 Features
- Fetch historical stock/ETF price data from Yahoo Finance
- Compute log returns, mean returns, and covariance matrix
- Evaluate portfolio performance (return, volatility, Sharpe ratio)
- Monte Carlo simulation of thousands of random portfolios
- Optimization using **SciPy**:
  - 📌 Max Sharpe Portfolio (best risk-adjusted return)
  - 📌 Min Volatility Portfolio (lowest risk)
- Visualize the **Efficient Frontier** with optimal portfolios highlighted

---

## 🛠 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/YOURUSERNAME/portfolio-optimizer.git
cd portfolio-optimizer
pip install -r requirements.txt
