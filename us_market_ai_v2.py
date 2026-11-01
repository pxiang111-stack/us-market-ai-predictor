"""
US Market AI Opportunity Scanner — V2
Ranks liquid US equities by six-month opportunity with risk controls.
Research prototype; not investment advice. No guarantee of predictive edge.
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
OUT = Path("output")
OUT.mkdir(exist_ok=True)
HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
ETFS = ["SPY","QQQ","IWM","DIA","VTI","RSP","XLF","XLK","XLI","XLY","XLP","XLE",
        "XLV","XLB","XLU","SMH","SOXX","TLT","IEF","HYG","LQD","GLD","UUP"]
EXCLUDE_ETFS_FROM_STOCK_RANK = True

def universe():
    tickers = set(ETFS)
    for url in [
        "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
        "https://en.wikipedia.org/wiki/Nasdaq-100",
        "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average",
    ]:
        try:
            for table in pd.read_html(url):
                for col in table.columns:
                    if str(col).lower() in ("symbol", "ticker"):
                        tickers.update(table[col].astype(str).str.replace(".", "-", regex=False))
                        break
        except Exception as exc:
            print(f"[WARN] Cannot load constituent list: {exc}")
    return sorted(t for t in tickers if isinstance(t, str) and 1 <= len(t) <= 7)

def download(tickers, period):
    out = {}
    for i in range(0, len(tickers), 60):
        batch = tickers[i:i+60]
        try:
            raw = yf.download(batch, period=period, auto_adjust=True, group_by="ticker",
                              threads=True, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                # yfinance can return either (ticker, field) or (field, ticker)
                for t in batch:
                    try:
                        if t in raw.columns.get_level_values(0):
                            d = raw[t]
                        elif t in raw.columns.get_level_values(1):
                            d = raw.xs(t, axis=1, level=1)
                        else:
                            continue
                        if {"Close","High","Low","Volume"}.issubset(d.columns):
                            out[t] = d[["Close","High","Low","Volume"]].dropna()
                    except Exception:
                        continue
            elif len(batch) == 1 and {"Close","High","Low","Volume"}.issubset(raw.columns):
                out[batch[0]] = raw[["Close","High","Low","Volume"]].dropna()
        except Exception as exc:
            print(f"[WARN] Batch download issue: {exc}")
    return out

def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100/(1+rs)

def sentiment_for(ticker, idx, news_path):
    # Optional sentiment CSV: date,ticker,sentiment; use only timestamped historical values.
    p = Path(news_path)
    if not p.exists():
        return pd.Series(0.0, index=idx)
    n = pd.read_csv(p)
    if not {"date","ticker","sentiment"}.issubset(n.columns):
        raise ValueError("news_sentiment.csv requires date,ticker,sentiment")
    n["date"] = pd.to_datetime(n["date"]).dt.tz_localize(None)
    n["ticker"] = n["ticker"].astype(str).str.upper()
    n["sentiment"] = pd.to_numeric(n["sentiment"], errors="coerce").clip(-1,1)
    n = n[n.ticker.isin([ticker.upper(), "MARKET", "SPY"])]
    if n.empty:
        return pd.Series(0.0, index=idx)
    s = n.groupby("date").sentiment.mean().reindex(idx).fillna(0)
    return s.shift(1).fillna(0)

def features_for(t, d, spy_close, news_path):
    c,h,l,v = d.Close, d.High, d.Low, d.Volume
    r = c.pct_change()
    x = pd.DataFrame(index=d.index)
    x["ret_1d"] = r
    for n in [5,21,63,126,252]:
        x[f"mom_{n}"] = c.pct_change(n)
        x[f"vol_{n}"] = r.rolling(n).std()*np.sqrt(252)
        x[f"ma_gap_{n}"] = c/c.rolling(n).mean()-1
    x["rsi_14"] = rsi(c)
    e12,e26 = c.ewm(span=12).mean(),c.ewm(span=26).mean()
    m = e12-e26
    x["macd_norm"] = m/c
    x["macd_signal_gap"] = (m-m.ewm(span=9).mean())/c
    x["volume_z_21"] = (v-v.rolling(21).mean())/v.rolling(21).std()
    x["range_21"] = ((h-l)/c).rolling(21).mean()
    x["drawdown_126"] = c/c.rolling(126).max()-1
    sp = spy_close.reindex(x.index).ffill()
    x["relative_mom_21"] = c.pct_change(21)-sp.pct_change(21)
    x["spy_mom_21"] = sp.pct_change(21)
    x["spy_vol_21"] = sp.pct_change().rolling(21).std()*np.sqrt(252)
    x["news_sentiment"] = sentiment_for(t,x.index,news_path)
    x["ticker"] = t
    for name,days in HORIZONS.items():
        x[f"target_ret_{name}"] = c.shift(-days)/c-1
        x[f"target_up_{name}"] = (x[f"target_ret_{name}"]>0).astype(float)
        x.loc[x.index[-days:], [f"target_ret_{name}",f"target_up_{name}"]] = np.nan
    return x.replace([np.inf,-np.inf],np.nan)

FEATURES = ["ret_1d"] + [f"{k}_{n}" for k in ["mom","vol","ma_gap"] for n in [5,21,63,126,252]] + [
    "rsi_14","macd_norm","macd_signal_gap","volume_z_21","range_21",
    "drawdown_126","relative_mom_21","spy_mom_21","spy_vol_21","news_sentiment"
]

def validation(data, horizon):
    ycol=f"target_up_{horizon}"
    z=data.dropna(subset=[ycol,f"target_ret_{horizon}"]).sort_index()
    dates=np.array(sorted(z.index.unique()))
    if len(dates)<500: return {"horizon":horizon,"status":"insufficient_history"}
    cut=int(len(dates)*.8)
    tr=z[z.index<dates[cut]]
    te=z[z.index>=dates[cut]]
    if tr[ycol].nunique()<2 or te[ycol].nunique()<2:
        return {"horizon":horizon,"status":"single_class_split"}
    lr=Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                 ("m",LogisticRegression(max_iter=1200,class_weight="balanced"))])
    rf=Pipeline([("imp",SimpleImputer(strategy="median")),
                 ("m",RandomForestClassifier(n_estimators=220,min_samples_leaf=20,
                     max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=7))])
    lr.fit(tr[FEATURES],tr[ycol].astype(int)); rf.fit(tr[FEATURES],tr[ycol].astype(int))
    p=(lr.predict_proba(te[FEATURES])[:,1]+rf.predict_proba(te[FEATURES])[:,1])/2
    return {"horizon":horizon,"status":"ok","test_start":str(te.index.min().date()),
            "test_end":str(te.index.max().date()),"n_test":len(te),
            "auc":roc_auc_score(te[ycol].astype(int),p),
            "brier":brier_score_loss(te[ycol].astype(int),p),
            "actual_up_rate":float(te[ycol].mean())}

def train_predict(data):
    latest_date=data.index.max()
    latest=(data.reset_index(names="date").sort_values("date")
            .groupby("ticker").tail(1).set_index("date"))
    out=[]
    for h in HORIZONS:
        ycol=f"target_up_{h}"; rcol=f"target_ret_{h}"
        tr=data.dropna(subset=[ycol,rcol])
        if tr.empty or tr[ycol].nunique()<2: continue
        lr=Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler()),
                     ("m",LogisticRegression(max_iter=1200,class_weight="balanced"))])
        rf=Pipeline([("imp",SimpleImputer(strategy="median")),
                     ("m",RandomForestClassifier(n_estimators=220,min_samples_leaf=20,
                         max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=7))])
        reg=Pipeline([("imp",SimpleImputer(strategy="median")),
                      ("m",HistGradientBoostingRegressor(max_iter=140,learning_rate=.05,
                         l2_regularization=2,random_state=7))])
        lr.fit(tr[FEATURES],tr[ycol].astype(int)); rf.fit(tr[FEATURES],tr[ycol].astype(int))
        reg.fit(tr[FEATURES],tr[rcol])
        valid=latest.dropna(subset=FEATURES)
        if valid.empty: continue
        prob=(lr.predict_proba(valid[FEATURES])[:,1]+rf.predict_proba(valid[FEATURES])[:,1])/2
        exp=reg.predict(valid[FEATURES])
        for i,(_,row) in enumerate(valid.iterrows()):
            vol=float(row.get("vol_126",np.nan))
            dd=float(row.get("drawdown_126",np.nan))
            # Transparent ranking score: expected return penalized for volatility and deep drawdown.
            risk_penalty=(0.20*max(vol-0.35,0) if np.isfinite(vol) else 0)
            dd_penalty=0.10*abs(min(dd,0)) if np.isfinite(dd) else 0
            score=100*(0.60*prob[i]+0.40*(1/(1+np.exp(-np.clip(exp[i]*4,-20,20))))) - 100*risk_penalty - 100*dd_penalty
            out.append({"as_of":str(latest_date.date()),"ticker":row.ticker,"horizon":h,
                        "prob_up":prob[i],"expected_return":exp[i],
                        "volatility_6m":vol,"drawdown_6m":dd,
                        "opportunity_score":score,
                        "signal":"High potential" if prob[i]>=.60 and exp[i]>0 else
                                 ("Avoid/weak" if prob[i]<=.40 or exp[i]<0 else "Watch")})
    pred=pd.DataFrame(out)
    if pred.empty: return pred
    # Add a rank within each horizon. Avoid ETF rows in stock leaderboard.
    pred["rank"]=pred.groupby("horizon")["opportunity_score"].rank(method="first",ascending=False).astype(int)
    return pred

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--period",default="10y")
    ap.add_argument("--news",default="news_sentiment.csv")
    args=ap.parse_args()
    ts=universe()
    print("Universe candidates:",len(ts))
    px=download(ts,args.period)
    print("Price histories downloaded:",len(px))
    if "SPY" not in px: raise RuntimeError("SPY data unavailable.")
    rows=[]
    for t,d in px.items():
        if len(d)<300: continue
        rows.append(features_for(t,d,px["SPY"].Close,args.news))
    if not rows: raise RuntimeError("No sufficiently long price histories.")
    data=pd.concat(rows).replace([np.inf,-np.inf],np.nan)
    pd.DataFrame([validation(data,h) for h in HORIZONS]).to_csv(OUT/"validation_report.csv",index=False)
    pred=train_predict(data)
    if pred.empty: raise RuntimeError("No predictions created.")
    pred.to_csv(OUT/"all_predictions.csv",index=False)
    six=pred[pred.horizon=="6m"].copy()
    if EXCLUDE_ETFS_FROM_STOCK_RANK:
        six=six[~six.ticker.isin(ETFS)]
    six=six.sort_values(["opportunity_score","prob_up","expected_return"],ascending=False)
    six.to_csv(OUT/"top_6m_opportunities.csv",index=False)
    print("\nTop 25 six-month candidates:")
    print(six.head(25).to_string(index=False))
    print("\nValidation:")
    print(pd.read_csv(OUT/"validation_report.csv").to_string(index=False))
    print("\nFiles saved under output/")

if __name__=="__main__": main()
