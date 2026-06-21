"""
fetch_naaim.py
Membaca Excel NAAIM Exposure Index (nama file mengandung 'naaim')
-> data/naaim.csv bersih (date, naaim), diringkas mingguan -> rata-rata bulanan.
"""
import glob
import pandas as pd

OUT = "data/naaim.csv"
EXCLUDE = ["bear", "bull", "quart", "median", "most", "s&p", "sp500", "date"]

def find_excel():
    cands = (glob.glob("*.xlsx") + glob.glob("data/*.xlsx")
             + glob.glob("manual/*.xlsx") + glob.glob("data/manual/*.xlsx"))
    cands = [c for c in cands if "naaim" in c.lower()]
    return sorted(cands)[-1] if cands else None

def main():
    raw = find_excel()
    if not raw:
        print("Tidak ada file NAAIM .xlsx (nama harus mengandung 'naaim'). Lewati NAAIM.")
        return
    print("Membaca:", raw)

    sheets = pd.read_excel(raw, sheet_name=None, header=None)
    best = None
    for name, rd in sheets.items():
        for i in range(min(15, len(rd))):
            row = rd.iloc[i].astype(str).str.lower()
            if row.str.contains("date").any() and \
               row.str.contains("exposure|mean|average|number", regex=True).any():
                best = pd.read_excel(raw, sheet_name=name, header=i)
                break
        if best is not None:
            break
    if best is None:
        best = pd.read_excel(raw, sheet_name=0)

    date_col = best.columns[0]
    exp_col = None
    for c in best.columns:
        lc = str(c).lower()
        if any(k in lc for k in ["exposure", "mean", "average", "number"]) \
           and not any(x in lc for x in EXCLUDE):
            exp_col = c
            break
    if exp_col is None:
        raise ValueError("Kolom exposure NAAIM tidak ketemu. Kolom: " + ", ".join(map(str, best.columns)))

    out = best[[date_col, exp_col]].copy()
    out.columns = ["date", "naaim"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["naaim"] = pd.to_numeric(out["naaim"], errors="coerce")
    out = out.dropna(subset=["date", "naaim"]).sort_values("date")

    # mingguan -> rata-rata bulanan (akhir bulan)
    out = out.set_index("date").resample("ME")["naaim"].mean().to_frame().dropna()

    out.to_csv(OUT)
    print("NAAIM selesai. Baris terakhir:")
    print(out.tail(3))

if __name__ == "__main__":
    main()
