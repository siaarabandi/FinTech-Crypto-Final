import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
import os


START_DATE = "2018-01-01"
END_DATE = "2025-11-01"
ROLLING_WINDOW = 365  

# # fetching data from yahoo finance
# tickers = ["BTC-USD", "ETH-USD", "^GSPC"]  
# data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]
# data = data.dropna()
# data.head()


CACHE_FILE = "yfinance_data.csv"  
TICKERS = ["BTC-USD", "ETH-USD", "^GSPC"]

# if local file exists, load it
if os.path.exists(CACHE_FILE):
    print("📂 Loading data from local cache...")
    data = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)
else:
    print("🌐 Downloading data from Yahoo Finance...")
    data = yf.download(TICKERS, start=START_DATE, end=END_DATE)["Close"]
    data = data.dropna()
    # save to CSV for next time
    data.to_csv(CACHE_FILE)
    print("✅ Data saved to", CACHE_FILE)


print(data.head())

# normalization
norm_data = data / data.iloc[0] 

# daily returns
returns = data.pct_change().dropna()

# rolling correlations 
btc_sp_corr = returns["BTC-USD"].rolling(ROLLING_WINDOW).corr(returns["^GSPC"])
eth_sp_corr = returns["ETH-USD"].rolling(ROLLING_WINDOW).corr(returns["^GSPC"])

corr_df = pd.DataFrame({
    "BTC-S&P500": btc_sp_corr,
    "ETH-S&P500": eth_sp_corr
}).dropna()

# statistical test of correlations past vs. present 
early_period = corr_df.loc["2018":"2021"]
late_period = corr_df.loc["2022":"2025"]

t_stat_btc, p_btc = ttest_ind(
    early_period["BTC-S&P500"].dropna(),
    late_period["BTC-S&P500"].dropna(),
    equal_var=False
)
t_stat_eth, p_eth = ttest_ind(
    early_period["ETH-S&P500"].dropna(),
    late_period["ETH-S&P500"].dropna(),
    equal_var=False
)

print(" Statistical Test Results (Independent Two Sample t-Test): ")
print(f"BTC-S&P500: p-value = {p_btc:.4f}")
print(f"ETH-S&P500: p-value = {p_eth:.4f}")
if p_btc < 0.05 or p_eth < 0.05:
    print("Correlation has increased significantly since 2018 (p < 0.05)")
else:
    print("No statistically significant increase in correlation.")

# visualizations
plt.style.use("seaborn-v0_8-whitegrid")

# normalized price trends
plt.figure(figsize=(10,5))
plt.plot(norm_data.index, norm_data["BTC-USD"], label="Bitcoin", color="orange")
plt.plot(norm_data.index, norm_data["ETH-USD"], label="Ethereum", color="blue")
plt.plot(norm_data.index, norm_data["^GSPC"], label="S&P 500", color="gray")
plt.title("Normalized Price Trends (2018–2025)")
plt.xlabel("Date")
plt.ylabel("Normalized Price (Start = 1)")
plt.legend()
plt.tight_layout()
plt.show()


# dual axis market price trends

fig, ax1 = plt.subplots(figsize=(12,6))

ax1.plot(data.index, data["^GSPC"], color="purple", label="S&P 500 Price")
ax1.set_ylabel("S&P 500 Price", color="purple")
ax1.tick_params(axis="y", labelcolor="purple")

ax2 = ax1.twinx()
ax2.plot(data.index, data["BTC-USD"], color="orange", label="Bitcoin Price")
ax2.set_ylabel("Bitcoin Price (USD)", color="orange")
ax2.tick_params(axis="y", labelcolor="orange")

plt.title("Bitcoin vs S&P 500 — Actual Prices (Dual Axis)")
fig.tight_layout()
plt.show()


# rolling correlations
plt.figure(figsize=(10,5))
plt.plot(corr_df.index, corr_df["BTC-S&P500"], label="BTC–S&P500", color="orange")
plt.plot(corr_df.index, corr_df["ETH-S&P500"], label="ETH–S&P500", color="blue")
plt.axhline(0, color="black", linestyle="--", linewidth=0.8)
plt.title(f"{ROLLING_WINDOW}-Day Rolling Correlation with S&P500")
plt.xlabel("Date")
plt.ylabel("Correlation")
plt.legend()
plt.tight_layout()
plt.show()

# boxplot of correlations before vs after 2021 
period_labels = ["2018–2021", "2022–2025"]
btc_data = [early_period["BTC-S&P500"], late_period["BTC-S&P500"]]
eth_data = [early_period["ETH-S&P500"], late_period["ETH-S&P500"]]

plt.figure(figsize=(8,5))
plt.boxplot(btc_data, positions=[1, 2], widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="orange", alpha=0.4))
plt.boxplot(eth_data, positions=[4, 5], widths=0.6, patch_artist=True,
            boxprops=dict(facecolor="blue", alpha=0.4))
plt.xticks([1.5, 4.5], ["BTC-S&P500", "ETH-S&P500"])
plt.ylabel("Correlation")
plt.title("Correlation Before vs After 2021")
plt.text(1.5, 0.9, f"p={p_btc:.3f}", ha="center")
plt.text(4.5, 0.9, f"p={p_eth:.3f}", ha="center")
plt.tight_layout()
plt.show()

# output summary
avg_corr_early = early_period.mean()
avg_corr_late = late_period.mean()
print("\n Average Rolling Correlations")
print(pd.DataFrame({
    "2018–2021": avg_corr_early,
    "2022–2025": avg_corr_late
}).round(3))

corr_df.to_csv("rolling_correlation.csv")
print("📁 Saved rolling_correlation.csv")

