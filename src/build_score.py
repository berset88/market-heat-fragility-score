"""
build_score.py
FRED + Shiller CAPE + FINRA margin debt -> percentile -> Market Heat Score 0-100.
Versi 0.3: menambah leverage (margin debt YoY) ke pilar Credit/Liquidity.
"""
import os
import json
import pandas as pd
import numpy as np

df = pd.read_csv("data/fred_monthly_master.csv", parse_dates=["date"], index_col="date")
df = df.ffill()

# --- Gabungkan CAPE ---
if os.path.exists("data/cape.csv"):
    cape = pd.read_csv("data/cape.csv", parse_dates=["date"], index_col="date")
    df = df.join(cape, how="left")
    df["cape"] = df["cape"].ffill()
    HAS_CAPE = True
else:
    HAS_CAPE = False

# --- Gabungkan FINRA margin debt ---
if os.path.exists("data/margin_debt.csv"):
    md = pd.read_csv("data/margin_debt.csv", parse_dates=["date"], index_col="date")
    df = df.join(md, how="left")
    df["margin_debt"] = df["margin_debt"].ffill()
    HAS_FINRA = True
else:
    HAS_FINRA = False

sig = pd.DataFrame(index=df.index)

# Valuation
sig["valuation"] = df["cape"] if HAS_CAPE else -df["us_10y_yield"]

# Momentum
sig["mom_3m"] = df["sp500"].pct_change(3)
sig["mom_12m"] = df["sp500"].pct_change(12)
sig["dist_12m_avg"] = df["sp500"] / df["sp500"].rolling(12).mean() - 1

# Sentiment (VIX rendah = panas)
sig["vix_inv"] = -df["vix"]

# Credit / Liquidity / Leverage
sig["hy_inv"] = -df["hy_oas"]
sig["hy_chg_3m_inv"] = -df["hy_oas"].diff(3)
sig["nfci_inv"] = -df["nfci"]
if HAS_FINRA:
    sig["margin_yoy"] = df["margin_debt"].pct_change(12)  # lonjakan leverage = panas

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

credit_cols = ["hy_inv", "hy_chg_3m_inv", "nfci_inv"]
if HAS_FINRA:
    credit_cols.append("margin_yoy")
pillar["credit_liq"] = pct[credit_cols].mean(axis=1)

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
out = out[~out.index.duplicated(keep="last")].sort_index()
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
    f.write('const HAS_FINRA = ' + ("true" if HAS_FINRA else "false") + ';\n')

print("Heat Score v0.3 selesai. CAPE:", HAS_CAPE, "| FINRA:", HAS_FINRA)
print(out.tail(6))
