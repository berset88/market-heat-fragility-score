"""
build_score.py
Membaca data FRED bulanan -> menghitung percentile tiap indikator ->
menggabungkan jadi Market Heat Score 0-100 (versi 0.1, PARSIAL).
Tidak ada angka yang dikarang: semua dihitung dari CSV yang ada.
"""

import pandas as pd
import numpy as np

# 1) Baca data bulanan hasil fetch_fred.py
df = pd.read_csv("data/fred_monthly_master.csv", parse_dates=["date"], index_col="date")

# Isi bolong ringan (mis. bulan tanpa rilis) dengan nilai terakhir
df = df.ffill()

# 2) Bangun indikator turunan (raw signals)
sig = pd.DataFrame(index=df.index)

# --- Valuation (proxy ERP; makin RENDAH ERP makin PANAS) ---
# earnings yield kasar belum tersedia tanpa CAPE -> pakai inverse: -(10Y) sbg placeholder ERP
# Catatan: ini PROXY sementara, diganti CAPE di Tahap G.
sig["valuation_proxy"] = -df["us_10y_yield"]  # makin tinggi = makin panas (yield rendah)

# --- Momentum (makin TINGGI makin PANAS) ---
sig["mom_3m"] = df["sp500"].pct_change(3)
sig["mom_12m"] = df["sp500"].pct_change(12)
sig["dist_12m_avg"] = df["sp500"] / df["sp500"].rolling(12).mean() - 1

# --- Sentiment (VIX RENDAH = complacency = PANAS) ---
sig["vix_inv"] = -df["vix"]

# --- Credit/Liquidity (HY spread RENDAH = complacency = PANAS; NFCI longgar = PANAS) ---
sig["hy_inv"] = -df["hy_oas"]
sig["hy_chg_3m_inv"] = -df["hy_oas"].diff(3)  # spread menyempit = panas
sig["nfci_inv"] = -df["nfci"]                 # NFCI negatif = longgar = panas

# --- Macro Reality Gap (rally tinggi sementara sentimen turun = PANAS/rapuh) ---
sig["macro_gap"] = df["sp500"].pct_change(12) - df["consumer_sentiment"].pct_change(12)

# 3) Ubah tiap sinyal jadi PERCENTILE ekspanding (0-100), pakai data masa lalu saja
def expanding_percentile(s, min_periods=36):
    out = pd.Series(index=s.index, dtype="float64")
    for i in range(len(s)):
        window = s.iloc[: i + 1].dropna()
        if len(window) >= min_periods and not np.isnan(s.iloc[i]):
            out.iloc[i] = (window.rank(pct=True).iloc[-1]) * 100
    return out

pct = sig.apply(expanding_percentile)

# 4) Gabung jadi sub-skor per pilar (rata-rata indikator dalam pilar)
pillar = pd.DataFrame(index=df.index)
pillar["valuation"] = pct[["valuation_proxy"]].mean(axis=1)
pillar["momentum"] = pct[["mom_3m", "mom_12m", "dist_12m_avg"]].mean(axis=1)
pillar["sentiment"] = pct[["vix_inv"]].mean(axis=1)
pillar["credit_liq"] = pct[["hy_inv", "hy_chg_3m_inv", "nfci_inv"]].mean(axis=1)
pillar["macro_gap"] = pct[["macro_gap"]].mean(axis=1)

# 5) Bobot pilar sesuai framework MD
WEIGHTS = {
    "valuation": 0.25,
    "momentum": 0.20,
    "sentiment": 0.20,
    "credit_liq": 0.20,
    "macro_gap": 0.15,
}

heat = sum(pillar[k] * w for k, w in WEIGHTS.items())
pillar["heat_score"] = heat

# 6) Label regime
def regime(x):
    if pd.isna(x): return ""
    if x < 30: return "Cold / attractive"
    if x < 60: return "Normal"
    if x < 75: return "Warm"
    if x < 85: return "Overheated"
    return "Extreme overheat"

pillar["regime"] = pillar["heat_score"].apply(regime)

# 7) Simpan
out = pillar.round(1)
out.to_csv("data/heat_score_history.csv")

print("Heat Score v0.1 (PARSIAL) selesai.")
print(out.tail(6))
