# Predict2Optimize: Market Dynamics & Portfolio Optimization

This repository documents my progress through the **Predict2Optimize** project, which aims to build a data-driven pipeline for financial forecasting and portfolio management.

## Project Overview
The project follows a structured path:
**Data Collection → Feature Engineering → Return Prediction → Portfolio Optimization → Backtesting**

Currently, I have completed the tasks for **Week 1** and **Week 2**.

---

## Week 1: Financial Data & Feature Extraction
The goal of this week was to build a foundation in handling financial time-series data and understanding its statistical properties.

### Tasks & Implementation
1.  **Data Acquisition**:
    *   Utilized `yfinance` to download historical `Adjusted Close` prices for a basket of tech stocks: `AAPL`, `MSFT`, `GOOG`, `AMZN`, `TSLA`, and `NVDA`.
    *   Managed data quality by handling missing values with forward and backward filling.
2.  **Exploratory Data Analysis**:
    *   Calculated simple and **log returns** over various horizons (1, 5, and 20 days).
    *   Visualized price trends alongside 20-day moving averages and rolling volatilities.
    *   Observed "Volatility Clustering" — where high-volatility periods (like the March 2020 COVID crash) tend to persist.
3.  **Statistical Testing**:
    *   Performed the **Augmented Dickey-Fuller (ADF) Test** on log returns, confirming they are stationary (p-value ≈ 0), unlike raw prices.
4.  **Volatility Modeling**:
    *   Compared **Rolling Window Volatility** with **Exponentially Weighted Moving Average (EWMA)** ($\lambda = 0.94$).
    *   Implemented a regime detection system by shading "High Volatility" regions where EWMA exceeded the 60th percentile.
5.  **The "Normal" Illusion**:
    *   Analyzed **Skewness** and **Kurtosis** across Daily, Weekly, and Monthly horizons.
    *   Verified the principle of **Aggregational Gaussianity**: while daily returns have "fat tails" (high kurtosis), returns over longer horizons tend to look more normally distributed.
6.  **Smart Investing (Bonus)**:
    *   Calculated the potential profit from a long-term investment in NVIDIA ($NVDA) and translated it into "purchasing power" for modern hardware (RTX 4090s).

### Learning Outcomes
*   **Stationarity is Key**: Learned that raw prices are non-stationary and cannot be used directly in many statistical models. The ADF test is a crucial tool for verifying that returns are suitable for modeling.
*   **Volatility isn't Constant**: Gained an intuition for "Volatility Clustering" and learned how EWMA provides a more responsive risk measure than simple rolling averages.
*   **The Normality Myth**: Discovered that financial returns often violate the "Normal Distribution" assumption at daily scales (excess kurtosis/fat tails), which is vital for accurate risk assessment.
*   **Data Handling Proficiency**: Mastered the use of `pandas` and `yfinance` for cleaning and transforming messy financial time-series data.

---

## Week 2: Baseline Prediction Models & Evaluation
In the second week, I shifted focus to predictive modeling, emphasizing rigorous evaluation to avoid common pitfalls like look-ahead bias.

### Tasks & Implementation
1.  **Feature Construction**:
    *   Developed a feature matrix for predicting next-day returns ($r_{t+1}$):
        *   **Lags**: $r_t$ and $r_{t-1}$.
        *   **Rolling Stats**: 20-day mean and standard deviation.
        *   **Momentum**: 5-day cumulative return.
2.  **Model Development**:
    *   **Naive Baselines**: Implemented a **Zero Predictor** and a **Rolling Mean Predictor** to serve as benchmarks.
    *   **Statistical Models**: Built an **Ordinary Least Squares (OLS)** regression model.
    *   **Machine Learning**: Explored **Random Forest Regressors** to capture non-linearities, noting their tendency to overfit in noisy financial data.
3.  **Walk-Forward Evaluation**:
    *   Instead of traditional random splits, I used **TimeSeriesSplit** (100 folds) for walk-forward validation.
    *   This ensures that the model is always tested on data that chronologically follows the training set, mimicking real-world trading.
4.  **Trading Strategy Simulation (Bonus)**:
    *   Developed a simple "all-in" long/short strategy based on the sign of the predicted returns.
    *   Compared the cumulative returns of this strategy across different models (Zero, Rolling Mean, OLS, and RF).

### Learning Outcomes
1.  **Evaluation Discipline**: Recognized that random train/test splits are disastrous for time-series data as they lead to "look-ahead bias." Walk-forward validation (via `TimeSeriesSplit`) is the only way to get a realistic estimate of performance.
2.  **The Baseline Hurdle**: Learned that beating simple benchmarks (like predicting zero) is exceptionally difficult in finance due to the low signal-to-noise ratio. A model that doesn't beat these baselines is effectively learning noise.
3.  **Model Overfitting**: Observed how high-capacity models like Random Forests can easily overfit historical noise, reinforcing the value of simpler, more robust models like OLS in certain regimes.
4.  **Prediction vs. Strategy**: Realized that even a model with decent RMSE might not translate into a profitable trading strategy, highlighting the gap between statistical accuracy and economic utility.

---

## Technical Stack
*   **Data**: `yfinance`
*   **Analysis**: `pandas`, `numpy`, `statsmodels`
*   **Visualization**: `matplotlib`, `seaborn`
*   **Modeling**: `scikit-learn`
