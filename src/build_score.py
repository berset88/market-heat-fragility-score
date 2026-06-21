"""
build_score.py
FRED + CAPE + FINRA margin + NAAIM -> percentile -> Market Heat Score 0-100.
Versi 0.4: menambah NAAIM (positioning manajer) ke pilar Sentiment.
"""
import os
import json
import pandas as pd
import numpy as np

df = pd.read_csv("data/fred_monthly_master.csv", parse_dates=["date"], index_col="date")
df = df.ffill()

def join_csv(path, col):
    global df
    if os.path.exists(path):
        extra = pd.read_csv(path, parse_dates=["date"], index_col="date")
        df = df.join(extra, how="left")
        df[col] = df[col].ffill()
        return True
    return False

HAS_CAPE = join_csv("data/cape.csv", "cape")
HAS_FINRA = join_csv("data/margin_debt.csv", "margin_debt")
HAS_NAAIM = join_csv("data/naaim.csv", "naaim")

sig = pd.DataFrame(index=df.index)

# Valuation
sig["valuation"] = df["cape"] if HAS_CAPE else -df["us_10y_yield"]

# Momentum
sig["mom_3m"] = df["sp500"].pct_change(3)
sig["mom_12m"] = df["sp500"].pct_change(12)
sig["dist_12m_avg"] = df["sp500"] / df["sp500"].rolling(12).mean() - 1

# Sentiment (VIX rendah = panas; NAAIM tinggi = crowded = panas)
sig["vix_inv"] = -df["vix"]
if HAS_NAAIM:
    sig["naaim"] = df["naaim"]

# Credit / Liquidity / Leverage
sig["hy_inv"] = -df["hy_oas"]
sig["hy_chg_3m_inv"] = -df["hy_oas"].diff(3)
sig["nfci_inv"] = -df["nfci"]
if HAS_FINRA:
    sig["margin_yoy"] = df["margin_debt"].pct_change(12)

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

sent_cols = ["vix_inv"] + (["naaim"] if HAS_NAAIM else [])
pillar["sentiment"] = pct[sent_cols].mean(axis=1)

credit_cols = ["hy_inv", "hy_chg_3m_inv", "nfci_inv"] + (["margin_yoy"] if HAS_FINRA else [])
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
    f.write('const HAS_NAAIM = ' + ("true" if HAS_NAAIM else "false") + ';\n')

print("Heat Score v0.4. CAPE:", HAS_CAPE, "| FINRA:", HAS_FINRA, "| NAAIM:", HAS_NAAIM)
print(out.tail(6))
