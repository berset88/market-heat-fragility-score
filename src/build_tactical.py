"""
build_tactical.py
Tactical 5% Asymmetry overlay (gaya UBS), FRED-only, ADAPTIF.
Fitur kredit memakai BAA10Y (Baa - 10Y, sejak 1986) -- pengganti HY OAS yang
kini dibatasi FRED ke 3 tahun. Prioritas: panjang sejarah untuk model 'decline'.
"""
import os, json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score

N, THRESH, MIN_YEARS = 63, 0.05, 7

d = pd.read_csv("data/fred_daily_master.csv", parse_dates=["date"], index_col="date")
d = d.sort_index().ffill().dropna(subset=["sp500"])

f = pd.DataFrame(index=d.index)
f["vix"] = d["vix"]
f["sp_3m_mom"] = d["sp500"] / d["sp500"].shift(N) - 1
f["dgs2_3m_chg"] = d["us_2y_yield"] - d["us_2y_yield"].shift(N)
f["breakeven"] = d["us_10y_breakeven"]
f["nfci"] = d["nfci"]
f["curve_10y2y"] = d["us_10y_yield"] - d["us_2y_yield"]

# Pilih seri kredit berhistori panjang: utamakan baa_spread, fallback hy_oas
credit_col = None
for cand in ["baa_spread", "hy_oas"]:
    if cand in d.columns and d[cand].notna().any():
        s = d[cand].dropna()
        years = (s.index.max() - s.index.min()).days / 365.25
        if years >= MIN_YEARS:
            credit_col = cand
            break
if credit_col:
    f["credit"] = d[credit_col]
    f["credit_3m_chg"] = d[credit_col] - d[credit_col].shift(N)

feat_cols = list(f.columns)

sp = d["sp500"].values; n = len(sp)
fwd_high = np.full(n, np.nan); fwd_low = np.full(n, np.nan)
for i in range(n):
    w = sp[i+1:i+1+N]
    if len(w) == N:
        fwd_high[i] = w.max(); fwd_low[i] = w.min()
f["rally"] = ((fwd_high/sp-1) >= THRESH).astype(float)
f["decline"] = ((fwd_low/sp-1) <= -THRESH).astype(float)
f.loc[np.isnan(fwd_high), ["rally","decline"]] = np.nan

train = f.dropna(); feat_now = f[feat_cols].dropna()
X = train[feat_cols].values
y_rally = train["rally"].values; y_decline = train["decline"].values

def wf_auc(X, y):
    a = []
    for tr, te in TimeSeriesSplit(n_splits=5).split(X):
        if len(np.unique(y[tr]))<2 or len(np.unique(y[te]))<2: continue
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(max_iter=1000, C=0.5).fit(sc.transform(X[tr]), y[tr])
        a.append(roc_auc_score(y[te], m.predict_proba(sc.transform(X[te]))[:,1]))
    return float(np.mean(a)) if a else float("nan")

auc_rally = wf_auc(X, y_rally); auc_decline = wf_auc(X, y_decline)

scaler = StandardScaler().fit(X)
m_r = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(X), y_rally)
m_d = LogisticRegression(max_iter=1000, C=0.5).fit(scaler.transform(X), y_decline)
x_now = scaler.transform(feat_now.iloc[[-1]].values)[0]
p_rally = float(m_r.predict_proba([x_now])[0,1])
p_decline = float(m_d.predict_proba([x_now])[0,1])
contrib = {c: float((m_r.coef_[0][j]-m_d.coef_[0][j])*x_now[j]) for j,c in enumerate(feat_cols)}

out = {
    "as_of": str(feat_now.index[-1].date()),
    "p_rally": round(p_rally,4), "p_decline": round(p_decline,4),
    "net": round(p_rally-p_decline,4),
    "auc_rally": round(auc_rally,3), "auc_decline": round(auc_decline,3),
    "contributions": {k: round(v,4) for k,v in contrib.items()},
    "n_samples": int(len(train)), "history_start": str(train.index[0].date()),
    "credit_series": credit_col,
}
os.makedirs("docs", exist_ok=True)
with open("data/tactical.json","w") as fp: json.dump(out, fp, indent=2)
with open("docs/tactical.js","w") as fp: fp.write("const TACTICAL = "+json.dumps(out)+";\n")
print(json.dumps(out, indent=2))
