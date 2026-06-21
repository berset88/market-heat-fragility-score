"""
build_score.py
FRED bulanan + Shiller CAPE -> percentile -> Market Heat Score 0-100.
Versi 0.2: Valuation memakai CAPE asli (bukan proxy).
"""
import os
import json
import pandas as pd
import numpy as np

df = pd.read_csv("data/fred_monthly_master.csv", parse_dates=["date"], index_col="date")
df = df.ffill()

# Gabungkan CAPE bila tersedia
cape_path = "data/cape.csv"
if os.path.exists(cape_path):
    cape = pd.read_csv(cape_path, parse_dates=["date"], index_col="date")
    df = df.join(cape, how="left")
    df["cape"] = df["cape"].ffill()
    HAS_CAPE = True
else:
    HAS_CAPE = False

sig = pd.DataFrame(index=df.index)

# Valuation: CAPE asli kalau ada (tinggi = panas); kalau tidak, proxy lama
if HAS_CAPE:
    sig["valuation"] = df["cape"]
else:
    sig["valuation"] = -df["us_10y_yield"]

# Momentum
sig["mom_3m"] = df["sp500"].pct_change(3)
sig["mom_12m"] = df["sp500"].pct_change(12)
sig["dist_12m_avg"] = df["sp500"] / df["sp500"].rolling(12).mean() - 1

# Sentiment (VIX rendah = panas)
sig["vix_inv"] = -df["vix"]

# Credit/Liquidity
sig["hy_inv"] = -df["hy_oas"]
sig["hy_chg_3m_inv"] = -df["hy_oas"].diff(3)
sig["nfci_inv"] = -df["nfci"]

# Macro Reality Gap
sig["macro_gap"] = df["sp500"].pct_change(12) - df["consumer_sentiment"].pct_change(12)

def expanding_percentile(s, min_periods=36):
    out = pd.Series(index=s.index, dtype="float64")
    for i in range(len(s)):
        window = s.iloc[: i + 1].dropna()
        if len(window) >= min_periods and not np.isnan(s.iloc[i]):
            out.iloc[i] = (window.rank(pct=True).iloc[-1]) * 100
    return out

pct = sig.apply(expanding_percentile)

pillar = pd.DataFrame(index=df.index)
pillar["valuation"] = pct[["valuation"]].mean(axis=1)
pillar["momentum"] = pct[["mom_3m", "mom_12m", "dist_12m_avg"]].mean(axis=1)
pillar["sentiment"] = pct[["vix_inv"]].mean(axis=1)
pillar["credit_liq"] = pct[["hy_inv", "hy_chg_3m_inv", "nfci_inv"]].mean(axis=1)
pillar["macro_gap"] = pct[["macro_gap"]].mean(axis=1)

WEIGHTS = {"valuation": 0.25, "momentum": 0.20, "sentiment": 0.20,
           "credit_liq": 0.20, "macro_gap": 0.15}
pillar["heat_score"] = sum(pillar[k] * w for k, w in WEIGHTS.items())

def regime(x):
    if pd.isna(x): return ""
    if x < 30: return "Cold"
    if x < 60: return "Normal"
    if x < 75: return "Warm"
    if x < 85: return "Overheated"
    return "Extreme"

pillar["regime"] = pillar["heat_score"].apply(regime)

out = pillar.round(1)
out = out[~out.index.duplicated(keep="last")].sort_index()   # <-- pengaman duplikat
out.to_csv("data/heat_score_history.csv")

os.makedirs("docs", exist_ok=True)
dash = out.reset_index()
dash["date"] = pd.to_datetime(dash["date"]).dt.strftime("%Y-%m-%d")
dash = dash.dropna(subset=["heat_score"])
records = dash.to_dict(orient="records")
with open("docs/data.js", "w") as f:
    f.write("const HEAT_DATA = " + json.dumps(records) + ";\n")
    f.write('const LAST_UPDATED = "' + dash["date"].iloc[-1] + '";\n')
    f.write('const HAS_CAPE = ' + ("true" if HAS_CAPE else "false") + ';\n')

print("Heat Score v0.2 selesai. CAPE dipakai:", HAS_CAPE)
print(out.tail(6))
