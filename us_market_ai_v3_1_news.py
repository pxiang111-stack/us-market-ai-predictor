"""
US Market AI Opportunity Scanner — V3
Research prototype only. Not investment advice; no guaranteed predictive edge.
"""
from pathlib import Path
from io import StringIO
import argparse, warnings, time
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, brier_score_loss
from quant_research import add_factor_scores, evaluate_factor_portfolio, factor_information_coefficient

warnings.filterwarnings("ignore")
OUT = Path("output")
OUT.mkdir(exist_ok=True)
HORIZONS = {"1m": 21, "3m": 63, "6m": 126}
TRAIN_SAMPLE_LIMIT = 100000
ETFS = set("""SPY QQQ IWM DIA VTI RSP XLF XLK XLI XLY XLP XLE XLV XLB XLU SMH SOXX
TLT IEF HYG LQD GLD UUP ARKK IBIT VNQ EEM EFA""".split())
FALLBACK = set("""AAPL MSFT NVDA AMZN GOOGL GOOG META AVGO BRK-B LLY JPM WMT XOM V MA COST
NFLX ORCL PG JNJ ABBV HD BAC KO PLTR CRM CVX AMD CAT GE MRK CSCO IBM MCD ACN TMO LIN
ADBE QCOM TXN AMAT NOW DIS PM UNH GS MS RTX ISRG INTU VZ T DHR SPGI BLK C AMGN PEP
UPS LOW HON COP MU LRCX KLAC PANW CRWD ANET APP SHOP UBER ABNB DASH COIN HOOD SOFI RBLX
SNOW DDOG NET ZS MDB TEAM SQ PYPL TGT NKE SBUX TSLA INTC SMCI ARM DE BA ETN SCHW PFE
CVS BMY GILD REGN BSX SYK ADP PAYX FI ICE CME CBOE MCO MMC AON PGR CB TRV""".split())
FEATURES = ["ret_1d"] + [f"{k}_{n}" for k in ("mom","vol","ma_gap") for n in (5,21,63,126,252)] + [
    "rsi_14","macd_norm","macd_signal_gap","volume_z_21","range_21","drawdown_126",
    "relative_mom_21","spy_mom_21","spy_vol_21","news_sentiment","factor_score"]

def clean_ticker(v):
    t = str(v).strip().upper().replace(".", "-")
    if not t or t in {"NAN","NONE"} or "?" in t: return None
    return t if len(t) <= 8 and all(c.isalnum() or c in "-^" for c in t) else None

def universe(max_symbols=1200):
    """Build a broad US-listed equity universe with official directory fallback."""
    symbols = set(ETFS)
    source = None
    try:
        url = "https://api.nasdaq.com/api/screener/stocks"
        params = {"tableonly":"true","limit":"6000","offset":"0","exchange":"nasdaq,nyse,amex"}
        headers = {"User-Agent":"Mozilla/5.0","Accept":"application/json, text/plain, */*",
                   "Origin":"https://www.nasdaq.com","Referer":"https://www.nasdaq.com/"}
        r = requests.get(url, params=params, headers=headers, timeout=25)
        r.raise_for_status()
        payload = r.json()
        table = (payload.get("data") or {}).get("table") or {}
        rows = table.get("rows") or []
        if not isinstance(rows, list) or len(rows) < 100:
            raise RuntimeError(f"Nasdaq screener returned {len(rows) if isinstance(rows, list) else 'invalid'} rows")
        from_api = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            ticker = clean_ticker(row.get("symbol", ""))
            if ticker and str(row.get("is_etf", row.get("etf", "false"))).lower() not in {"true", "1", "yes"}:
                raw_cap = str(row.get("marketCap", row.get("market_cap", "0"))).replace(",", "").replace("$", "").strip().upper()
                try:
                    scale = {"K": 1e3, "M": 1e6, "B": 1e9, "T": 1e12}
                    multiplier = scale.get(raw_cap[-1:], 1.0)
                    numeric_cap = raw_cap[:-1] if multiplier != 1.0 else raw_cap
                    market_cap = float(numeric_cap) * multiplier
                except (TypeError, ValueError, IndexError):
                    market_cap = 0.0
                from_api[ticker] = market_cap
        if len(from_api) < 100:
            raise RuntimeError(f"Only {len(from_api)} usable equity symbols")
        symbols.update(from_api)
        source = "Nasdaq public screener (market-cap ranked universe)"
    except Exception as e:
        print(f"[WARN] Nasdaq screener unavailable: {e}; trying Nasdaq Trader directories")
        headers = {"User-Agent":"Mozilla/5.0"}
        directory_symbols = set()
        directory_errors = []
        directories = [
            ("https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt", "nasdaq"),
            ("https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt", "other"),
        ]
        for url, kind in directories:
            try:
                response = requests.get(url, headers=headers, timeout=25)
                response.raise_for_status()
                lines = response.text.splitlines()
                if not lines or ("Symbol" not in lines[0] and "ACT Symbol" not in lines[0]):
                    raise RuntimeError("Unexpected symbol-directory format")
                header = lines[0].split("|")
                for line in lines[1:]:
                    if not line or line.startswith("File Creation Time"):
                        continue
                    row = dict(zip(header, line.split("|")))
                    symbol = row.get("Symbol") if kind == "nasdaq" else row.get("ACT Symbol")
                    if row.get("Test Issue", "N") != "N" or row.get("ETF", "N") == "Y":
                        continue
                    ticker = clean_ticker(symbol or "")
                    if ticker:
                        directory_symbols.add(ticker)
            except Exception as exc:
                directory_errors.append(f"{kind}: {exc}")
        if len(directory_symbols) >= 500:
            symbols.update(directory_symbols)
            source = "Nasdaq Trader official symbol directories"
        else:
            print(f"[WARN] Nasdaq Trader yielded {len(directory_symbols)} symbols; {directory_errors}")
            print("[WARN] Falling back to index constituents and curated liquid-stock list")
            headers = {"User-Agent":"Mozilla/5.0"}
            for url in ["https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
                        "https://en.wikipedia.org/wiki/Nasdaq-100"]:
                try:
                    response = requests.get(url, headers=headers, timeout=20)
                    response.raise_for_status()
                    for table in pd.read_html(StringIO(response.text)):
                        for col in table.columns:
                            if str(col).strip().lower() in {"symbol", "ticker"}:
                                symbols.update(filter(None, (clean_ticker(x) for x in table[col])))
                                break
                except Exception as exc:
                    print(f"[WARN] Constituent list unavailable: {exc}")
            symbols.update(FALLBACK)
            source = "index constituents + curated fallback (limited universe)"
    if source and "market-cap ranked" in source:
        equities = sorted((x for x in symbols if x not in ETFS),
                          key=lambda x: from_api.get(x, 0.0), reverse=True)
    else:
        # Keep curated liquid names near the front when no market-cap data exists.
        equities = sorted((x for x in symbols if x not in ETFS),
                          key=lambda x: (x not in FALLBACK, x))
    result = sorted(ETFS) + equities[:max(0, max_symbols - len(ETFS))]
    print(f"Universe source: {source}; symbols: {len(result)}", flush=True)
    return result

def download(tickers, period, batch_size=40):
    out = {}
    for start in range(0,len(tickers),batch_size):
        print(f"[INFO] Download batch {start//batch_size+1}/{(len(tickers)+batch_size-1)//batch_size}", flush=True)
        batch = tickers[start:start+batch_size]
        try:
            raw = yf.download(batch,period=period,auto_adjust=True,group_by="ticker",
                              threads=True,progress=False,timeout=30)
            if raw is None or raw.empty: continue
            if isinstance(raw.columns,pd.MultiIndex):
                for t in batch:
                    try:
                        if t in raw.columns.get_level_values(0): d = raw[t]
                        elif t in raw.columns.get_level_values(1): d = raw.xs(t,axis=1,level=1)
                        else: continue
                        if {"Close","High","Low","Volume"}.issubset(d.columns):
                            d = d[["Close","High","Low","Volume"]].dropna()
                            if len(d) >= 260: out[t] = d
                    except Exception: pass
            elif len(batch)==1 and {"Close","High","Low","Volume"}.issubset(raw.columns):
                d = raw[["Close","High","Low","Volume"]].dropna()
                if len(d)>=260: out[batch[0]]=d
        except Exception as e: print(f"[WARN] Download batch {start//batch_size+1}: {e}")
        time.sleep(.2)
    return out

def rsi(s,n=14):
    d=s.diff(); up=d.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    down=(-d.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/down.replace(0,np.nan)
    return 100-100/(1+rs)

def sentiment_for(ticker, idx, news_path):
    p=Path(news_path)
    if not p.exists(): return pd.Series(np.nan,index=idx,dtype=float)
    n=pd.read_csv(p)
    if not {"date","ticker","sentiment"}.issubset(n.columns):
        raise ValueError("news CSV must contain date,ticker,sentiment")
    n["date"]=pd.to_datetime(n["date"],errors="coerce").dt.tz_localize(None)
    n["ticker"]=n["ticker"].astype(str).str.upper()
    n["sentiment"]=pd.to_numeric(n["sentiment"],errors="coerce").clip(-1,1)
    n=n.dropna(subset=["date","sentiment"])
    n=n[n.ticker.isin([ticker.upper(),"MARKET","SPY"])]
    if n.empty: return pd.Series(np.nan,index=idx,dtype=float)
    daily=n.groupby("date")["sentiment"].mean().sort_index()
    return daily.reindex(idx,method="ffill").shift(1)

def features_for(t,d,spy,news_path):
    c,h,l,v=d.Close,d.High,d.Low,d.Volume
    r=c.pct_change(); x=pd.DataFrame(index=d.index); x["ret_1d"]=r
    for n in (5,21,63,126,252):
        x[f"mom_{n}"]=c.pct_change(n)
        x[f"vol_{n}"]=r.rolling(n).std()*np.sqrt(252)
        x[f"ma_gap_{n}"]=c/c.rolling(n).mean()-1
    x["rsi_14"]=rsi(c)
    e12,e26=c.ewm(span=12,adjust=False).mean(),c.ewm(span=26,adjust=False).mean()
    m=e12-e26; x["macd_norm"]=m/c
    x["macd_signal_gap"]=(m-m.ewm(span=9,adjust=False).mean())/c
    x["volume_z_21"]=(v-v.rolling(21).mean())/v.rolling(21).std().replace(0,np.nan)
    x["range_21"]=((h-l)/c).rolling(21).mean()
    x["drawdown_126"]=c/c.rolling(126).max()-1
    sp=spy.reindex(x.index).ffill()
    x["relative_mom_21"]=c.pct_change(21)-sp.pct_change(21)
    x["spy_mom_21"]=sp.pct_change(21)
    x["spy_vol_21"]=sp.pct_change().rolling(21).std()*np.sqrt(252)
    x["news_sentiment"]=sentiment_for(t,x.index,news_path)
    for name,days in HORIZONS.items():
        x[f"target_ret_{name}"]=c.shift(-days)/c-1
        x[f"target_up_{name}"]=(x[f"target_ret_{name}"]>0).astype(float)
        x.loc[x.index[-days:],[f"target_ret_{name}",f"target_up_{name}"]]=np.nan
    x["ticker"]=t
    return x.replace([np.inf,-np.inf],np.nan)


def fetch_gdelt_articles(query, max_records=30):
    """Fetch recent public news headlines from GDELT DOC API (no API key)."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "ArtList",
        "format": "json",
        "maxrecords": str(max_records),
        "timespan": "7d",
        "sort": "HybridRel"
    }
    headers = {"User-Agent": "Mozilla/5.0 (compatible; MarketSentimentResearch/1.0)"}
    try:
        response = requests.get(url, params=params, headers=headers, timeout=8)
        response.raise_for_status()
        payload = response.json()
        return payload.get("articles", []) or []
    except Exception as exc:
        print(f"[WARN] News request failed for query [{query}]: {exc}")
        return []

def score_headlines(articles):
    """VADER headline polarity; returns neutral-safe aggregate and article count."""
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError:
        return np.nan, 0
    analyzer = SentimentIntensityAnalyzer()
    scores = []
    for article in articles:
        title = str(article.get("title", "")).strip()
        if len(title) < 8:
            continue
        scores.append(analyzer.polarity_scores(title)["compound"])
    if not scores:
        return np.nan, 0
    # Clip extremes and use a modest recency-agnostic mean over the last 7 days.
    return float(np.mean(scores)), len(scores)

def add_news_overlay(predictions, max_tickers=15):
    """
    News overlay for the highest-ranked 6m candidates.
    GDELT headlines are current/recent only, so this is deliberately not used
    in historical model training or presented as a historical backtest feature.
    """
    if predictions.empty:
        return predictions, pd.DataFrame()
    six_mask = predictions["horizon"].eq("6m")
    candidates = (predictions.loc[six_mask]
                  .sort_values("opportunity_score", ascending=False)
                  .head(max_tickers))
    market_articles = fetch_gdelt_articles(
        '(US stock market OR Federal Reserve OR inflation OR Treasury yields OR '
        'artificial intelligence OR semiconductor OR earnings OR tariffs)',
        max_records=50
    )
    market_score, market_count = score_headlines(market_articles)
    if not np.isfinite(market_score):
        market_score = 0.0
    records = []
    ticker_list = candidates["ticker"].dropna().unique()
    for ticker_no, ticker in enumerate(ticker_list, start=1):
        print(f"[NEWS] Candidate {ticker_no}/{len(ticker_list)}: {ticker}", flush=True)
        # Quoted ticker and $ticker reduce (but do not eliminate) symbol ambiguity.
        articles = fetch_gdelt_articles(f'"{ticker}" OR "${ticker}"', max_records=30)
        stock_score, count = score_headlines(articles)
        if not np.isfinite(stock_score):
            stock_score = 0.0
        # Blend company headlines (70%) with broad market tone (30%).
        combined = 0.70 * stock_score + 0.30 * market_score
        # Confidence guard: with fewer than 3 headlines, shrink toward neutral.
        confidence = min(count / 5.0, 1.0)
        adjusted = combined * confidence
        records.append({
            "ticker": ticker,
            "news_score": float(stock_score),
            "news_count_7d": int(count),
            "market_news_score": float(market_score),
            "market_news_count_7d": int(market_count),
            "combined_news_score": float(adjusted),
            "news_confidence": float(confidence)
        })
        time.sleep(0.15)
    news = pd.DataFrame(records)
    if news.empty:
        predictions["news_score"] = np.nan
        predictions["news_count_7d"] = 0
        predictions["combined_news_score"] = np.nan
        predictions["news_adjusted_score"] = predictions["opportunity_score"]
        return predictions, news
    predictions = predictions.merge(
        news[["ticker","news_score","news_count_7d","market_news_score",
              "market_news_count_7d","combined_news_score","news_confidence"]],
        on="ticker", how="left"
    )
    # Only six-month rows get the news adjustment. Missing headlines are neutral,
    # explicitly marked as missing/zero-count rather than treated as positive.
    mask = predictions["horizon"].eq("6m")
    sentiment = predictions["combined_news_score"].fillna(0.0)
    news_component = 50.0 + 50.0 * sentiment.clip(-1, 1)
    predictions["news_adjusted_score"] = predictions["opportunity_score"]
    predictions.loc[mask, "news_adjusted_score"] = (
        0.80 * predictions.loc[mask, "opportunity_score"] +
        0.20 * news_component.loc[mask]
    )
    predictions.loc[mask, "opportunity_score"] = predictions.loc[mask, "news_adjusted_score"]
    return predictions, news

def training_sample(frame, target, limit=TRAIN_SAMPLE_LIMIT):
    """Bound fit cost while retaining a deterministic, broad sample."""
    if len(frame) <= limit:
        return frame
    sampled = frame.sample(n=limit, random_state=17)
    print(f"[INFO] Training sample capped: {len(frame)} -> {len(sampled)} rows", flush=True)
    return sampled


def classifier(forest=False):
    steps=[("imputer",SimpleImputer(strategy="median",add_indicator=True))]
    if not forest: steps.append(("scale",StandardScaler()))
    steps.append(("model",RandomForestClassifier(n_estimators=100,min_samples_leaf=30,
        max_features="sqrt",class_weight="balanced_subsample",n_jobs=-1,random_state=17)
        if forest else LogisticRegression(max_iter=1500,class_weight="balanced")))
    return Pipeline(steps)

def model_features(data):
    """Exclude optional historical news when no usable observations exist."""
    if "news_sentiment" not in data or not data["news_sentiment"].notna().any():
        return [feature for feature in FEATURES if feature != "news_sentiment"]
    return FEATURES


def validate(data,horizon):
    """
    One chronological holdout with a horizon-sized embargo.
    Top-decile metrics are evaluated only on the untouched test period.
    """
    ycol,rcol=f"target_up_{horizon}",f"target_ret_{horizon}"
    features=model_features(data)
    z=data.dropna(subset=[ycol,rcol]).sort_index()
    dates=np.array(sorted(z.index.unique()))
    if len(dates)<500:
        return {"horizon":horizon,"status":"insufficient_history","n_dates":len(dates)}
    split=int(len(dates)*0.8)
    test_start=dates[split]
    train_end=dates[max(0,split-HORIZONS[horizon])]
    tr=z[z.index<train_end]
    te=z[z.index>=test_start].copy()
    if len(tr)<1000 or tr[ycol].nunique()<2 or te[ycol].nunique()<2:
        return {"horizon":horizon,"status":"insufficient_or_single_class",
                "n_train":len(tr),"n_test":len(te)}
    fit_tr=training_sample(tr,ycol)
    a,b=classifier(),classifier(True)
    a.fit(fit_tr[features],fit_tr[ycol].astype(int))
    b.fit(fit_tr[features],fit_tr[ycol].astype(int))
    p=(a.predict_proba(te[features])[:,1]+b.predict_proba(te[features])[:,1])/2
    y=te[ycol].astype(int)
    te=te.assign(_probability=p)
    cutoff=max(1,int(np.ceil(len(te)*0.10)))
    top=te.nlargest(cutoff,"_probability")
    spy=te.loc[te["ticker"].eq("SPY"),rcol]
    return {
        "horizon":horizon,"status":"ok",
        "train_end":str(pd.Timestamp(tr.index.max()).date()),
        "test_start":str(pd.Timestamp(te.index.min()).date()),
        "test_end":str(pd.Timestamp(te.index.max()).date()),
        "n_train":len(tr),"n_test":len(te),
        "auc":float(roc_auc_score(y,p)),
        "brier":float(brier_score_loss(y,p)),
        "actual_up_rate":float(y.mean()),
        "top_decile_n":int(len(top)),
        "top_decile_up_rate":float(top[ycol].mean()),
        "top_decile_mean_return":float(top[rcol].mean()),
        "test_universe_mean_return":float(te[rcol].mean()),
        "top_decile_return_lift":float(top[rcol].mean()-te[rcol].mean()),
        "spy_test_mean_return":float(spy.mean()) if len(spy) else np.nan,
        "news_feature_nonmissing_test_pct":float(te["news_sentiment"].notna().mean()*100)
            if "news_sentiment" in te else 0.0
    }

def predict(data):
    features=model_features(data)
    latest=(data.reset_index(names="date").sort_values("date").groupby("ticker",as_index=False)
            .tail(1).set_index("date"))
    results=[]
    for horizon in HORIZONS:
        yc,rc=f"target_up_{horizon}",f"target_ret_{horizon}"
        tr=data.dropna(subset=[yc,rc]); current=latest.copy()
        if len(tr)<1000 or tr[yc].nunique()<2 or current.empty:
            print(f"[WARN] Skipping {horizon}: train_rows={len(tr)}, train_classes={tr[yc].nunique()}, current_rows={len(current)}")
            continue
        fit_tr=training_sample(tr,yc)
        a,b=classifier(),classifier(True)
        reg=Pipeline([("imputer",SimpleImputer(strategy="median",add_indicator=True)),
            ("model",HistGradientBoostingRegressor(max_iter=150,learning_rate=.05,
                                                    l2_regularization=2,random_state=17))])
        a.fit(fit_tr[features],fit_tr[yc].astype(int)); b.fit(fit_tr[features],fit_tr[yc].astype(int))
        reg.fit(fit_tr[features],fit_tr[rc])
        p=(a.predict_proba(current[features])[:,1]+b.predict_proba(current[features])[:,1])/2
        exp=reg.predict(current[features])
        for i,(as_of,row) in enumerate(current.iterrows()):
            vol=float(row.get("vol_126",np.nan)); dd=float(row.get("drawdown_126",np.nan))
            vp=.15*max(vol-.35,0) if np.isfinite(vol) else 0
            dp=.08*abs(min(dd,0)) if np.isfinite(dd) else 0
            score=100*(.6*p[i]+.4/(1+np.exp(-np.clip(exp[i]*4,-20,20))))-100*(vp+dp)
            results.append({"as_of":str(pd.Timestamp(as_of).date()),"ticker":row.ticker,
                "horizon":horizon,"prob_up":float(p[i]),"expected_return_model_estimate":float(exp[i]),
                "volatility_6m_annualized":vol,"drawdown_6m":dd,"opportunity_score":float(score)})
    out=pd.DataFrame(results)
    if not out.empty:
        out["rank"]=out.groupby("horizon")["opportunity_score"].rank(method="first",ascending=False).astype(int)
        out["signal"]=np.select([(out.prob_up>=.60)&(out.expected_return_model_estimate>0),
            (out.prob_up<=.40)|(out.expected_return_model_estimate<0)],
            ["Higher model score; verify risk","Weak model outlook"],default="Watch")
    return out

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--period",default="10y")
    ap.add_argument("--news",default="news_sentiment.csv")
    ap.add_argument("--max-symbols",type=int,default=1200)
    args=ap.parse_args()
    tickers=universe(args.max_symbols)
    prices=download(tickers,args.period)
    print(f"Price histories downloaded: {len(prices)}", flush=True)
    if "SPY" not in prices: raise RuntimeError("SPY data unavailable")
    rows=[features_for(t,d,prices["SPY"].Close,args.news) for t,d in prices.items() if len(d)>=300]
    if not rows: raise RuntimeError("No sufficiently long price histories")
    data=pd.concat(rows).sort_index().replace([np.inf,-np.inf],np.nan)
    data=add_factor_scores(data)
    modeled = sorted(data["ticker"].dropna().unique())
    print(f"Model data: rows={len(data)}, symbols={len(modeled)}, "
          f"date_range={data.index.min().date()}..{data.index.max().date()}")
    for horizon in HORIZONS:
        yc, rc = f"target_up_{horizon}", f"target_ret_{horizon}"
        eligible = data.dropna(subset=[yc, rc])
        print(f"Training sample {horizon}: rows={len(eligible)}, "
              f"dates={eligible.index.nunique()}, classes={eligible[yc].nunique()}")
    report=pd.DataFrame([validate(data,h) for h in HORIZONS])
    report.to_csv(OUT/"validation_report.csv",index=False)
    factor_rows=[]
    factor_details=[]
    for h in HORIZONS:
        summary, detail=evaluate_factor_portfolio(data,h,etfs=ETFS)
        summary.update(factor_information_coefficient(data,h,etfs=ETFS))
        factor_rows.append(summary)
        if not detail.empty:
            factor_details.append(detail)
    pd.DataFrame(factor_rows).to_csv(OUT/"factor_validation_report.csv",index=False)
    if factor_details:
        pd.concat(factor_details,ignore_index=True).to_csv(OUT/"factor_backtest_detail.csv",index=False)
    data.reset_index(names="date")[["date","ticker","factor_score","factor_momentum",
        "factor_relative_strength","factor_trend","factor_low_volatility",
        "factor_drawdown_control"]].to_csv(OUT/"factor_scores.csv",index=False)
    pred=predict(data)
    if pred.empty: raise RuntimeError("No predictions generated; check data download/training sample")
    print("Fetching recent news sentiment for leading six-month candidates...")
    pred, news_report = add_news_overlay(pred, max_tickers=15)
    pred.to_csv(OUT/"all_predictions.csv",index=False)
    news_report.to_csv(OUT/"news_sentiment_report.csv",index=False)
    six=pred[(pred.horizon=="6m")&(~pred.ticker.isin(ETFS))].sort_values(
        ["opportunity_score","prob_up","expected_return_model_estimate"],ascending=False)
    six.to_csv(OUT/"top_6m_opportunities.csv",index=False)
    six.head(25).to_csv(OUT/"top_25_6m.csv",index=False)
    pd.DataFrame([{"run_time_utc":pd.Timestamp.now(tz="UTC").isoformat(),
        "universe_symbols":len(tickers),"downloaded_symbols":len(prices),
        "modeled_symbols":len(modeled),
        "six_month_stock_candidates":len(six),"news_csv_present":Path(args.news).exists(),
        "factor_engine":"cross-sectional momentum/relative strength/trend/low-volatility/drawdown/RSI composite",
        "note":"Research prototype; scores are not guaranteed predictions or investment advice."}]
    ).to_csv(OUT/"run_summary.csv",index=False)
    print("\\nTop 25 six-month candidates:")
    print(six.head(25).to_string(index=False) if not six.empty else "No stock candidates.")
    print("\\nValidation report:")
    print(report.to_string(index=False))
    print("\\nOutputs saved in output/")

if __name__=="__main__": main()
