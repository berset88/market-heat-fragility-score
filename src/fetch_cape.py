"""
fetch_cape.py
Mengambil Shiller CAPE (PE10) bulanan dari multpl.com -> data/cape.csv
Sumber resmi data Robert Shiller. Tidak butuh API key.
"""
import pandas as pd

URL = "https://www.multpl.com/shiller-pe/table/by-month"

def main():
    tables = pd.read_html(URL)
    df = tables[0]
    df.columns = ["date", "cape"]

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cape"] = (
        df["cape"].astype(str)
        .str.replace(",", "", regex=False)
        .str.extract(r"([\d.]+)")[0]
        .astype(float)
    )
    df = df.dropna(subset=["date", "cape"]).sort_values("date")

    # Samakan ke akhir bulan agar cocok dengan data FRED bulanan
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    df = df.set_index("date")[["cape"]]

    # Buang tanggal kembar (baris "Current" vs baris bulan dari multpl)
    df = df[~df.index.duplicated(keep="last")].sort_index()

    df.to_csv("data/cape.csv")
    print("CAPE selesai. Baris terakhir:")
    print(df.tail(3))

if __name__ == "__main__":
    main()
