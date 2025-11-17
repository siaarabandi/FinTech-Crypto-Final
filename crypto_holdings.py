import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd
import matplotlib.pyplot as plt
import requests
import pandas as pd
import matplotlib.dates as mdates
from matplotlib.patches import Patch

inst = pd.read_csv("btc.csv")
corr_df = pd.read_csv("rolling_correlation.csv", index_col=0, parse_dates=True)

# crypto institutional holders (bar chart)
# two types: bitcoin holders and bitcoin ETF issuers
color_map = {"BTC Holder": "skyblue", "ETF Issuer": "gold"}


colors = inst["Type"].map(color_map)

plt.figure(figsize=(12, 6))
plt.bar(inst["Company"], inst["BTC_Held"], color=colors)

plt.title(
    "Top Institutional Bitcoin Owners (Corporate Holders + ETF Issuers)", fontsize=14
)
plt.ylabel("BTC Held (BTC)", fontsize=12)
plt.xticks(rotation=60, ha="right")

legend_elements = [
    Patch(facecolor="skyblue", label="Public Company"),
    Patch(facecolor="gold", label="ETF Issuer"),
]
plt.legend(handles=legend_elements, title="Institution Type")

plt.tight_layout()
plt.show()


# BTC + S&P500 correlation (weekly dot plot)

# limiting range to 2024 to account for when 
# blackrock + vanguard + fidelity started holding crypto
corr_2025 = corr_df.loc["2024-01-01":]
corr_2025_w = corr_2025["BTC-S&P500"].resample("W").last()

# dot plot per week to look at small scale fluctuations
plt.figure(figsize=(10, 5))
plt.scatter(corr_2025_w.index, corr_2025_w.values, color="orange", s=65)

plt.title("BTC–S&P 500 Correlation (Time Series Dot Plot, 2024–Present)", fontsize=14)
plt.xlabel("Month")
plt.ylabel("Correlation")
plt.grid(True, linestyle="--", alpha=0.5)
plt.ylim(0.05, 0.5)
plt.plot(corr_2025_w.index, corr_2025_w.values, color="orange", alpha=0.4)

ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator())  
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))  
plt.xticks(rotation=45, ha="right")

plt.tight_layout()
plt.show()


# FOR REFERENCE ONLY (NOT APPLICABLE TO PROJECT OUTPUT)
# we used coingecko api to access up to date bitcoin holdings of the top companies around the world
url = "https://api.coingecko.com/api/v3/companies/public_treasury/bitcoin"
response = requests.get(url)
data = response.json()

companies = pd.DataFrame(data["companies"])
companies = companies[["name", "total_holdings", "country"]]

print(companies)
print("Total Corporate BTC:", companies["total_holdings"].sum())
