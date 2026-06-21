"""
fetch_aaii.py
Membaca spreadsheet AAII Sentiment Survey (nama file mengandung 'aaii')
-> data/aaii.csv bersih (date, aaii_spread = Bullish - Bearish),
diringkas mingguan -> rata-rata bulanan.
"""
import glob
import pandas as pd

OUT = "data/aaii.csv"
EXCLUDE = ["average", "avg", "mov", "spread", "dev", "8-week", "8 week"]

def find_file():
    pats = []
    for ext in ("*.xls", "*.xlsx"):
        for d in ("", "data/", "manual/", "data/manual/"):
            pats += glob.glob(d + ext)
    cands = [c for c in pats if "aaii" in c.lower()]
    return sorted(cands)[-1] if cands else None

def main():
    raw = find_file()
    if not raw:
        print("Tidak ada file AAII (nama harus mengandung 'aaii'). Lewati AAII.")
        return
    print("Membaca:", raw)

    sheets = pd.read_excel(raw, sheet_name=None, header=None)
    best = None
    for name, rd in sheets.items():
        for i in range(min(20, len(rd))):
            row = rd.iloc[i].astype(str).str.lower()
            if row.str.contains("bullish").any() and row.str.contains("bearish").any():
                best = pd.read_excel(raw, sheet_name=name, header=i)
                break
        if best is not None:
            break
    if best is None:
        raise ValueError("Header Bullish/Bearish tidak ketemu di file AAII.")

    date_col = best.columns[0]

    def pick(word):
        for c in best.columns:
            lc = str(c).lower()
            if word in lc and not any(x in lc for x in EXCLUDE):
                return c
        return None

    bull, bear = pick("bullish"), pick("bearish")
    if bull is None or bear is None:
        raise ValueError("Kolom Bullish/Bearish mentah tidak ketemu. Kolom: " + ", ".join(map(str, best.columns)))

    out = best[[date_col, bull, bear]].copy()
    out.columns = ["date", "bullish", "bearish"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["bullish"] = pd.to_numeric(out["bullish"], errors="coerce")
    out["bearish"] = pd.to_numeric(out["bearish"], errors="coerce")
    out = out.dropna(subset=["date", "bullish", "bearish"]).sort_values("date")

    out["aaii_spread"] = out["bullish"] - out["bearish"]
    out = out.set_index("date").resample("ME")["aaii_spread"].mean().to_frame().dropna()

    out.to_csv(OUT)
    print("AAII selesai. Baris terakhir:")
    print(out.tail(3))

if __name__ == "__main__":
    main()
