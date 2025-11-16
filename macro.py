import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from fredapi import Fred
from scipy.stats import pearsonr
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from dotenv import load_dotenv
import os

START_DATE = "2018-01-01"
END_DATE = "2025-11-01"

load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)

cpi = fred.get_series("CPIAUCSL")  
cpi = cpi.resample("M").last()
inflation = cpi.pct_change(periods=12) * 100  
inflation = inflation.loc[START_DATE:END_DATE]
inflation.name = "Inflation (%)"
inflation = inflation.to_frame()


fedfunds = fred.get_series("FEDFUNDS")  
fedfunds = fedfunds.resample("M").last()
fedfunds = fedfunds.loc[START_DATE:END_DATE]
fedfunds.name = "Federal Funds Rate (%)"
fedfunds = fedfunds.to_frame()


macro = pd.concat([inflation, fedfunds], axis=1).dropna()

print("Macro data sample:")
print(macro.head())


tickers = ["BTC-USD", "ETH-USD", "^GSPC"]
data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]
data = data.resample("M").last()
returns = data.pct_change() * 100
returns = returns.dropna()

CACHE_FILE = "fred_data.csv"  
TICKERS = ["BTC-USD", "ETH-USD", "^GSPC"]


if os.path.exists(CACHE_FILE):
    print("Loading data from local cache...")
    returns = pd.read_csv(CACHE_FILE, index_col=0, parse_dates=True)

else:
    print("Downloading data from Yahoo Finance...")
    data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]
    data = data.resample("M").last()
    returns = data.pct_change() * 100
    returns = returns.dropna()

    returns.to_csv(CACHE_FILE)
    print("Data saved to", CACHE_FILE)

merged = pd.concat([returns, macro], axis=1).dropna()

corr_btc_inf, p_btc_inf = pearsonr(merged["BTC-USD"], merged["Inflation (%)"])
corr_eth_inf, p_eth_inf = pearsonr(merged["ETH-USD"], merged["Inflation (%)"])
corr_sp_inf, p_sp_inf = pearsonr(merged["^GSPC"], merged["Inflation (%)"])

corr_btc_rate, p_btc_rate = pearsonr(merged["BTC-USD"], merged["Federal Funds Rate (%)"])
corr_eth_rate, p_eth_rate = pearsonr(merged["ETH-USD"], merged["Federal Funds Rate (%)"])
corr_sp_rate, p_sp_rate = pearsonr(merged["^GSPC"], merged["Federal Funds Rate (%)"])

print("\n Correlation with Inflation ")
print(f"Bitcoin vs Inflation:  {corr_btc_inf:.3f} (p={p_btc_inf:.3f})")
print(f"Ethereum vs Inflation: {corr_eth_inf:.3f} (p={p_eth_inf:.3f})")
print(f"S&P 500 vs Inflation:  {corr_sp_inf:.3f} (p={p_sp_inf:.3f})")

print("\n Correlation with Interest Rate ")
print(f"Bitcoin vs Fed Rate:  {corr_btc_rate:.3f} (p={p_btc_rate:.3f})")
print(f"Ethereum vs Fed Rate: {corr_eth_rate:.3f} (p={p_eth_rate:.3f})")
print(f"S&P 500 vs Fed Rate:  {corr_sp_rate:.3f} (p={p_sp_rate:.3f})")


plt.style.use("seaborn-v0_8-whitegrid")


fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(macro.index, macro["Inflation (%)"], color="red", linewidth=2, label="Inflation (%)")
ax1.set_ylabel("Inflation (%)", color="red")
ax2 = ax1.twinx()
ax2.plot(macro.index, macro["Federal Funds Rate (%)"], color="blue", linestyle="--", linewidth=2, label="Fed Funds Rate (%)")
ax2.set_ylabel("Fed Funds Rate (%)", color="blue")
plt.title("Inflation and Federal Funds Rate (2018–2025)")
plt.tight_layout()
plt.show()


btc_sp_corr = returns["BTC-USD"].rolling(12).corr(returns["^GSPC"])
btc_sp_corr.name = "BTC–S&P500 Correlation"
corr_m = btc_sp_corr.resample("M").last().dropna()

macro_timeline = pd.concat([corr_m, inflation, fedfunds], axis=1).dropna()

fig, ax1 = plt.subplots(figsize=(12,6))
ax1.plot(macro_timeline.index, macro_timeline["BTC–S&P500 Correlation"],
         linewidth=2.2, color="orange", label="BTC–S&P500 Corr")
ax1.set_ylabel("BTC–S&P500 Correlation", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

ax2 = ax1.twinx()
ax2.plot(macro_timeline.index, macro_timeline["Inflation (%)"],
         linestyle="--", linewidth=2, color="red", label="Inflation (YoY %)")
ax2.plot(macro_timeline.index, macro_timeline["Federal Funds Rate (%)"],
         linestyle=":", linewidth=2, color="blue", label="Fed Funds Rate (%)")
ax2.set_ylabel("Macro Indicators (%)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")


ax1.axvspan(pd.to_datetime("2020-02-15"), pd.to_datetime("2020-04-30"),
            color="gray", alpha=0.15, label="COVID Shock")
ax1.axvspan(pd.to_datetime("2020-04-01"), pd.to_datetime("2021-11-30"),
            color="green", alpha=0.08, label="Zero-Rate/QE Era")
ax1.axvspan(pd.to_datetime("2022-03-01"), pd.to_datetime("2023-07-31"),
            color="red", alpha=0.06, label="Rate-Hike Cycle")
ax1.axvline(pd.to_datetime("2023-03-10"), color="purple", alpha=0.35,
            linestyle="--", linewidth=1.2, label="Banking Stress (Mar 2023)")


legend_elements = [
    Line2D([0], [0], color="orange", lw=2.2, label="BTC–S&P500 Corr"),
    Line2D([0], [0], color="red", lw=2, ls="--", label="Inflation (YoY %)"),
    Line2D([0], [0], color="blue", lw=2, ls=":", label="Fed Funds Rate (%)"),
    Patch(facecolor="purple", alpha=0.15, label="COVID Shock"),
    Patch(facecolor="green", alpha=0.08, label="Zero-Rate/QE Era"),
    Patch(facecolor="red", alpha=0.06, label="Rate-Hike Cycle"),
    Line2D([0], [0], color="yellow", lw=1.2, ls="--", label="Mar 2023 Bank Stress")
]
ax1.legend(handles=legend_elements, loc="upper left", frameon=True)

plt.title("BTC–S&P500 Correlation vs Inflation & Fed Funds Rate (12-Month Rolling)")
plt.tight_layout()
plt.show()



btc_sp_corr = returns["BTC-USD"].rolling(12).corr(returns["^GSPC"])
btc_sp_corr.name = "BTC–S&P500 Correlation"

merged_corr = pd.concat([btc_sp_corr, inflation, fedfunds], axis=1).dropna()

fig, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(merged_corr.index, merged_corr["BTC–S&P500 Correlation"], color="orange", label="BTC–S&P500 Corr")
ax1.set_ylabel("BTC–S&P500 Correlation", color="orange")

ax2 = ax1.twinx()
ax2.plot(merged_corr.index, merged_corr["Inflation (%)"], color="red", linestyle="--", label="Inflation (%)")
ax2.plot(merged_corr.index, merged_corr["Federal Funds Rate (%)"], color="blue", linestyle=":", label="Fed Funds Rate (%)")
ax2.set_ylabel("Macro Indicators (%)", color="blue")

plt.title("BTC–S&P500 Correlation vs Inflation and Interest Rates (12-Month Rolling)")
plt.tight_layout()
plt.show()


summary = pd.DataFrame({
    "Asset": ["Bitcoin", "Ethereum", "S&P 500"],
    "Corr with Inflation": [corr_btc_inf, corr_eth_inf, corr_sp_inf],
    "p (Inflation)": [p_btc_inf, p_eth_inf, p_sp_inf],
    "Corr with Fed Rate": [corr_btc_rate, corr_eth_rate, corr_sp_rate],
    "p (Fed Rate)": [p_btc_rate, p_eth_rate, p_sp_rate]
})
print("\n Summary ")
print(summary.round(3))
