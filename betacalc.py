# ===============================================================
#   Mini-Thesis: Crypto–Equity Correlation and Inflation Analysis
#   Authors: Sia Arabandi & Nithya Ravula
# ===============================================================

import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from fredapi import Fred
from scipy.stats import pearsonr
from dotenv import load_dotenv
import os

START_DATE = "2018-01-01"
END_DATE = "2025-11-01"

plt.style.use("seaborn-v0_8-whitegrid")

#get data from FRED
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")
fred = Fred(api_key=FRED_API_KEY)


#get specific search
cpi = fred.get_series("CPIAUCSL")
cpi = cpi.resample("M").last()
inflation = cpi.pct_change(periods=12) * 100
inflation = inflation.loc[START_DATE:END_DATE]
inflation.name = "Inflation (%)"
inflation = inflation.to_frame()

print("Inflation data sample:")
print(inflation.head())

#get yahoo finance data
tickers = ["BTC-USD", "ETH-USD", "^GSPC"]
data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]
data = data.resample("M").last()

# percent returns
returns = data.pct_change() * 100
returns = returns.dropna()



import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

START_DATE = "2018-01-01"
END_DATE = "2025-11-01"

#get data
tickers = ["BTC-USD", "ETH-USD", "^GSPC"]
data = yf.download(tickers, start=START_DATE, end=END_DATE)["Close"]

#calculate daily returns
returns = data.pct_change().dropna()

# rolling beta calc
ROLLING = 365  

btc_beta = (
    returns["BTC-USD"].rolling(ROLLING).cov(returns["^GSPC"]) /
    returns["^GSPC"].rolling(ROLLING).var()
)

eth_beta = (
    returns["ETH-USD"].rolling(ROLLING).cov(returns["^GSPC"]) /
    returns["^GSPC"].rolling(ROLLING).var()
)


beta_df = pd.concat([btc_beta, eth_beta], axis=1)
beta_df.columns = ["BTC Beta", "ETH Beta"]

#plot betas 
plt.figure(figsize=(12,6))
plt.plot(beta_df.index, beta_df["BTC Beta"], color="orange", label="BTC Beta (365D Rolling)")
plt.plot(beta_df.index, beta_df["ETH Beta"], color="purple", label="ETH Beta (365D Rolling)")
plt.axhline(0, color="gray", linestyle="--")

plt.title("Rolling 365-Day Beta of Bitcoin and Ethereum vs S&P 500 (2018–2025)")
plt.xlabel("Date")
plt.ylabel("Beta (Sensitivity to S&P 500)")
plt.legend(loc="upper left")
plt.tight_layout()
plt.show()




