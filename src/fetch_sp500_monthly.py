"""
fetch_sp500_monthly.py
Harga S&P 500 bulanan (rata-rata penutupan, metodologi Shiller) sejak 1871
dari multpl.com -> data/sp500_monthly.csv.
Memperpanjang sejarah pilar Momentum & Macro Gap (FRED harian hanya ~10thn).
"""
import pandas as pd

URL = "https://www.multpl.com/s-p-500-historical-prices/table/by-month"

def main():
    tables = pd.read_html(URL)
    df = tables[0]
    df.columns = ["date", "sp500_long"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["sp500_long"] = (
        df["sp500_long"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([\d.]+)")[0]
        .astype(float)
    )
    df = df.dropna(subset=["date", "sp500_long"]).sort_values("date")

    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df = df.set_index("date")[["sp500_long"]]
    df = df[~df.index.duplicated(keep="last")].sort_index()  # buang baris 'current' kembar

    df.to_csv("data/sp500_monthly.csv")
    print("S&P 500 bulanan selesai. Baris terakhir:")
    print(df.tail(3))

if __name__ == "__main__":
    main()
