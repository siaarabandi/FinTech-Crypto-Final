import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred
from scipy.stats import pearsonr
from dotenv import load_dotenv
import os

START_DATE = "2018-01-01"
END_DATE = "2025-11-01"
FRED_API_KEY = "FRED_API_KEY_HERE"   

# fetching inflation data from fred inflation 
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

# CPI inflatiion rate over the years (% change)
cpi = fred.get_series("CPIAUCSL")  
cpi = cpi.resample("M").last()  
inflation = cpi.pct_change(periods=12) * 100  
inflation = inflation.loc[START_DATE:END_DATE]
inflation.name = "Inflation (%)"
inflation = inflation.to_frame()

print("Inflation data sample:")
print(inflation.head())

# fetching financial data from yahoo
tickers = ["BTC-USD", "ETH-USD", "^GSPC"]
data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]
data = data.resample("M").last()  
returns = data.pct_change() * 100  
returns = returns.dropna()

# merging datasets
merged = pd.concat([returns, inflation], axis=1).dropna()

# correlation analysis
corr_btc, p_btc = pearsonr(merged["BTC-USD"], merged["Inflation (%)"])
corr_eth, p_eth = pearsonr(merged["ETH-USD"], merged["Inflation (%)"])
corr_sp, p_sp = pearsonr(merged["^GSPC"], merged["Inflation (%)"])

print("\n=== Correlation with Inflation ===")
print(f"Bitcoin vs Inflation:  {corr_btc:.3f} (p={p_btc:.3f})")
print(f"Ethereum vs Inflation: {corr_eth:.3f} (p={p_eth:.3f})")
print(f"S&P 500 vs Inflation:  {corr_sp:.3f} (p={p_sp:.3f})")

# visualizations
plt.style.use("seaborn-v0_8-whitegrid")

# inflation trends
plt.figure(figsize=(10,5))
plt.plot(inflation.index, inflation["Inflation (%)"], color="red", linewidth=2)
plt.title("U.S. Inflation Rate (YoY %) 2020–2025")
plt.xlabel("Date")
plt.ylabel("Inflation (%)")
plt.tight_layout()
plt.show()

# inflation vs. asset returns
fig, axs = plt.subplots(1, 3, figsize=(15,5))
axs[0].scatter(merged["Inflation (%)"], merged["BTC-USD"], color="orange")
axs[1].scatter(merged["Inflation (%)"], merged["ETH-USD"], color="blue")
axs[2].scatter(merged["Inflation (%)"], merged["^GSPC"], color="gray")

axs[0].set_title(f"BTC vs Inflation (r={corr_btc:.2f})")
axs[1].set_title(f"ETH vs Inflation (r={corr_eth:.2f})")
axs[2].set_title(f"S&P500 vs Inflation (r={corr_sp:.2f})")

for ax in axs:
    ax.set_xlabel("Inflation (%)")
    ax.set_ylabel("Monthly Return (%)")

plt.tight_layout()
plt.show()

# inflation vs. crypto+equity correlation
btc_sp_corr = returns["BTC-USD"].rolling(12).corr(returns["^GSPC"])  
merged_corr = pd.concat([btc_sp_corr, inflation["Inflation (%)"]], axis=1).dropna()

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(merged_corr.index, merged_corr["BTC-USD"], color="orange", label="BTC–S&P500 Correlation")
ax1.set_ylabel("BTC–S&P500 Correlation", color="orange")

ax2 = ax1.twinx()
ax2.plot(merged_corr.index, merged_corr["Inflation (%)"], color="red", linestyle="--", label="Inflation (%)")
ax2.set_ylabel("Inflation (%)", color="red")

plt.title("Inflation vs BTC–S&P500 Correlation (12-Month Rolling)")
plt.tight_layout()
plt.show()

# summary
summary = pd.DataFrame({
    "Asset": ["Bitcoin", "Ethereum", "S&P 500"],
    "Correlation with Inflation": [corr_btc, corr_eth, corr_sp],
    "p-value": [p_btc, p_eth, p_sp]
})
print("\n Summary")
print(summary.round(3))
