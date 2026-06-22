"""
fetch_fred.py
Mengambil data ekonomi AS dari FRED API, menyimpannya sebagai CSV.
Dijalankan otomatis oleh GitHub Actions. Tidak menghitung skor apa pun.
"""

import os
import time
import requests
import pandas as pd

# Kunci API dibaca dari environment (dari GitHub Secrets), bukan dari file ini.
FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY tidak ditemukan. Set dulu di GitHub Secrets.")

BASE_URL = "https://api.stlouisfed.org/fred/series/observations"

# Seri FRED -> nama kolom yang rapi
SERIES = {
    "SP500": "sp500",
    "VIXCLS": "vix",
    "UMCSENT": "consumer_sentiment",
    "DGS10": "us_10y_yield",
    "DGS2": "us_2y_yield",
    "T10YIE": "us_10y_breakeven",
    "NFCI": "nfci",
    "BAMLH0A0HYM2": "hy_oas",
    "BAA10Y": "baa_spread",
}

START_DATE = "1990-01-01"


def fetch_series(series_id):
    """Ambil satu seri dari FRED, kembalikan DataFrame berindeks tanggal."""
    params = {
        "series_id": series_id,
        "api_key": FRED_API_KEY,
        "file_type": "json",
        "observation_start": START_DATE,
    }
    response = requests.get(BASE_URL, params=params, timeout=30)
    response.raise_for_status()
    observations = response.json()["observations"]

    df = pd.DataFrame(observations)
    df["date"] = pd.to_datetime(df["date"])
    # FRED memakai "." untuk data kosong -> ubah jadi NaN
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["date", "value"]].set_index("date")


def main():
    all_series = []
    for series_id, clean_name in SERIES.items():
        print(f"Mengambil {series_id} ...")
        df = fetch_series(series_id).rename(columns={"value": clean_name})
        all_series.append(df)
        time.sleep(0.5)  # jeda sopan ke API

    daily = pd.concat(all_series, axis=1).sort_index()
    monthly = daily.resample("ME").last()  # nilai akhir tiap bulan

    os.makedirs("data", exist_ok=True)
    daily.to_csv("data/fred_daily_master.csv")
    monthly.to_csv("data/fred_monthly_master.csv")

    print("Selesai.")
    print(monthly.tail())


if __name__ == "__main__":
    main()
