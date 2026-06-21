"""
fetch_finra.py
Mencari Excel margin statistics FINRA (akar repo / data/ / manual/)
-> menulis data/margin_debt.csv yang bersih (date, margin_debt; USD juta).
FINRA tidak punya API: file Excel diunduh & di-commit manual.
"""
import glob
import pandas as pd

OUT = "data/margin_debt.csv"

def find_excel():
    candidates = (
        glob.glob("*.xlsx")
        + glob.glob("data/*.xlsx")
        + glob.glob("manual/*.xlsx")
        + glob.glob("data/manual/*.xlsx")
    )
    candidates = [c for c in candidates if "margin_debt" not in c]
    return sorted(candidates)[-1] if candidates else None

def main():
    raw_path = find_excel()
    if not raw_path:
        print("Tidak ada file .xlsx FINRA ditemukan. Lewati FINRA.")
        return
    print("Membaca:", raw_path)

    # Cari sheet & baris header yang memuat kata 'Debit'
    sheets = pd.read_excel(raw_path, sheet_name=None, header=None)
    best = None
    for name, raw in sheets.items():
        for i in range(min(15, len(raw))):
            row = raw.iloc[i].astype(str).str.lower()
            if row.str.contains("debit").any():
                best = pd.read_excel(raw_path, sheet_name=name, header=i)
                break
        if best is not None:
            break

    if best is None:
        raise ValueError("Kolom 'Debit' tidak ditemukan. Kirim daftar nama kolom Excel ke Claude.")

    date_col = best.columns[0]
    debit_col = next(c for c in best.columns if "debit" in str(c).lower())

    out = best[[date_col, debit_col]].copy()
    out.columns = ["date", "margin_debt"]
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["margin_debt"] = pd.to_numeric(
        out["margin_debt"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=["date", "margin_debt"]).sort_values("date")
    out["date"] = out["date"] + pd.offsets.MonthEnd(0)
    out = out.drop_duplicates(subset="date", keep="last").set_index("date")

    out.to_csv(OUT)
    print("FINRA margin debt selesai. Baris terakhir:")
    print(out.tail(3))

if __name__ == "__main__":
    main()
