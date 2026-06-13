# Market Heat & Fragility Score

Indikator makro komposit untuk menilai apakah pasar saham AS sedang
*overheated*, rapuh (*fragile*), atau masih sehat — bukan dari satu sinyal,
melainkan gabungan valuasi, momentum, sentimen/positioning, kredit/likuiditas,
dan kesenjangan makro.

## Arsitektur (two-speed)
- **Engine** (otomatis, bulanan): mengambil data → menghitung skor → menyimpan CSV.
- **Cockpit / Dashboard**: menampilkan skor terkini & historis agar mudah dibaca.

Prinsip inti: **LLM tidak pernah menghitung angka.** Semua skor dihitung oleh
kode deterministik yang bisa diaudit, untuk mencegah halusinasi.

## Struktur folder
- `src/` — kode Python (pengambilan & perhitungan data)
- `data/` — output CSV (dibuat otomatis oleh Engine)
- `docs/` — dashboard web (diterbitkan via GitHub Pages)
- `.github/workflows/` — penjadwal otomatis bulanan

## Sumber data — MVP (100% otomatis via FRED)
| Indikator | Seri FRED | Fungsi |
|---|---|---|
| S&P 500 | SP500 | Harga & momentum |
| VIX | VIXCLS | Volatilitas / complacency |
| Consumer Sentiment | UMCSENT | Macro reality gap |
| 10Y Treasury | DGS10 | ERP proxy |
| 2Y Treasury | DGS2 | Impuls kebijakan |
| 10Y Breakeven | T10YIE | Tekanan inflasi |
| Financial Conditions | NFCI | Kondisi finansial |
| HY Credit Spread | BAMLH0A0HYM2 | Risiko kredit |

## Roadmap data tambahan (menyusul)
- Shiller CAPE (valuasi jangka panjang)
- FINRA margin debt (leverage)
- NAAIM / AAII (positioning & sentimen)
- Tactical 5% Asymmetry model (overlay gaya UBS)

## Status
**Versi 0.1 — MVP FRED.** Dibangun bertahap.
