"""
build_tactical.py
Tactical 5% Asymmetry overlay (gaya UBS).
Prediksi P(rally >=+5%) - P(decline >=-5%) dalam 63 hari bursa ke depan,
memakai regresi logistik transparan + walk-forward validation (TimeSeriesSplit).

Keterbatasan jujur:
- Data harian S&P 500 FRED hanya ~10 tahun -> overlay demonstratif, bukan ramalan pasti.
- Fitur UBS yang tak tersedia gratis (ISM, recession prob, put/call, futures positioning)
  dihilangkan.
"""
import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

N = 63          # hari bursa ke depan (~3 bulan)
THRESH = 0.05   # ambang +/-5%

# 1) Data harian
d = pd.read_csv("data/fred_daily_master.csv", parse_dates=["date"], index_col="date")
d = d.sort_index().ffill()

# Gabungkan NAAIM & AAII (reindex harian + ffill)
for path, col in [("data/naaim.csv", "naaim"), ("data/aaii.csv", "aaii_spread")]:
    if os.path.exists(path):
        x = pd.read_csv(path, parse_dates=["date"], index_col="date")
        d = d.join(x.reindex(d.index, method="ffill"))

d = d.dropna(subset=["sp500"])

# 2) Fitur
f = pd.DataFrame(index=d.index)
f["vix"] = d["vix"]
f["hy_oas"] = d["hy_oas"]
f["hy_oas_3m_chg"] = d["hy_oas"] - d["hy_oas"].shift(N)
f["sp_3m_mom"] = d["sp500"] / d["sp500"].shift(N) - 1
f["dgs2_3m_chg"] = d["us_2y_yield"] - d["us_2y_yield"].shift(N)
f["breakeven"] = d["us_10y_breakeven"]
f["nfci"] = d["nfci"]
if "naaim" in d.columns:
    f["naaim"] = d["naaim"]
if "aaii_spread" in d.columns:
    f["aaii"] = d["aaii_spread"]

feat_cols = list(f.columns)

# 3) Label: max/min ke depan 63 hari
sp = d["sp500"].values
n = len(sp)
fwd_high = np.full(n, np.nan)
fwd_low = np.full(n, np.nan)
for i in range(n):
    w = sp[i + 1:i + 1 + N]
    if len(w) == N:
        fwd_high[i] = w.max()
        fwd_low[i] = w.min()

f["rally"] = ((fwd_high / sp - 1) >= THRESH).astype(float)
f["decline"] = ((fwd_low / sp - 1) <= -THRESH).astype(float)
f.loc[np.isnan(fwd_high), ["rally", "decline"]] = np.nan

# 4) Set latih (fitur+label) vs baris prediksi terkini (fitur saja)
train = f.dropna()
feat_now = f[feat_cols].dropna()

X = train[feat_cols].values
y_rally = train["rally"].values
y_decline = train["decline"].values

def walk_forward_auc(X, y):
    aucs = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        p = m.predict_proba(sc.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)) if aucs else float("nan")

auc_rally = walk_forward_auc(X, y_rally)
auc_decline = walk_forward_auc(X, y_decline)

# 5) Model final + prediksi terkini + kontribusi
scaler = StandardScaler().fit(X)
m_rally = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(X), y_rally)
m_decline = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(X), y_decline)

x_now = scaler.transform(feat_now.iloc[[-1]].values)[0]
p_rally = float(m_rally.predict_proba([x_now])[0, 1])
p_decline = float(m_decline.predict_proba([x_now])[0, 1])
net = p_rally - p_decline

contrib = {c: float((m_rally.coef_[0][j] - m_decline.coef_[0][j]) * x_now[j])
           for j, c in enumerate(feat_cols)}

out = {
    "as_of": str(feat_now.index[-1].date()),
    "p_rally": round(p_rally, 4),
    "p_decline": round(p_decline, 4),
    "net": round(net, 4),
    "auc_rally": round(auc_rally, 3),
    "auc_decline": round(auc_decline, 3),
    "contributions": {k: round(v, 4) for k, v in contrib.items()},
    "n_samples": int(len(train)),
    "history_start": str(train.index[0].date()),
}

os.makedirs("docs", exist_ok=True)
with open("data/tactical.json", "w") as fp:
    json.dump(out, fp, indent=2)
with open("docs/tactical.js", "w") as fp:
    fp.write("const TACTICAL = " + json.dumps(out) + ";\n")

print(json.dumps(out, indent=2))
