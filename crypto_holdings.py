import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from fredapi import Fred
import pandas as pd
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

import requests
import pandas as pd

url = "https://api.coingecko.com/api/v3/companies/public_treasury/bitcoin"
response = requests.get(url)
data = response.json()

companies = pd.DataFrame(data["companies"])
companies = companies[["name", "total_holdings", "country"]]

print(companies)
print("Total Corporate BTC:", companies["total_holdings"].sum())


inst = pd.read_csv("btc.csv")
corr_df = pd.read_csv("rolling_correlation.csv", index_col=0, parse_dates=True)

# crypto institutional holders (bar chart) 
plt.figure(figsize=(10,6))
plt.bar(inst["Issuer"], inst["BTC_Held"], color="gold")
plt.title("Top Institutional Bitcoin Issuers (ETF) (BTC Amount)")
plt.ylabel("BTC Holdings")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


total_btc = inst["BTC_Held"].sum()
fig, ax1 = plt.subplots(figsize=(12,6))

inst_series = pd.Series(
    np.linspace(200_000, total_btc, len(corr_df)),  
    index=corr_df.index,
    name="Institutional BTC Holdings"
)

# BTC + S&P 500 correlation (left axis)
ax1.plot(corr_df.index, corr_df["BTC-S&P500"], 
         color="orange", label="BTC–S&P500 Rolling Correlation")
ax1.set_ylabel("BTC–S&P500 Rolling Correlation", color="orange")
ax1.tick_params(axis="y", labelcolor="orange")

# BTC holdings (right axis)
ax2 = ax1.twinx()
ax2.plot(inst_series.index, inst_series, 
         color="blue", label="Institutional BTC Holdings")
ax2.set_ylabel("Institutional BTC Holdings (BTC)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

ax1.axhline(corr_df["BTC-S&P500"].mean(), 
            linestyle="--", color="blue", alpha=0.5)

plt.title("Institutional Bitcoin Holdings vs BTC–S&P500 Correlation (2018–2025)")

# correlation vs btc holdings plot
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
plt.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

plt.show()




