"""V4 short-horizon NDX/SOXL forecasts and AI/credit risk dashboard.
Research prototype only. Missing data is reported, never fabricated.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore")
OUT=Path("output"); OUT.mkdir(exist_ok=True)
HORIZONS={"5d":5,"10d":10,"20d":20}
TARGETS={"NQ_futures":"NQ=F","SOXL":"SOXL"}
SYMBOLS=sorted(set(["NQ=F","^NDX","SOXL","QQQ","^SOX","SOXX","SMH","NVDA","AMD","AVGO",
 "MSFT","AMZN","GOOGL","META","CRWV","^VIX","HYG","LQD","TLT","IEF","UUP"]))
COMPANIES=["NVDA","MSFT","AMZN","GOOGL","META","ORCL","CRWV"]

def download_prices(symbols,period="10y"):
    data={}
    for ticker in symbols:
        try:
            d=yf.download(ticker,period=period,auto_adjust=True,progress=False,threads=False,timeout=25)
            if d is None or d.empty: continue
            if isinstance(d.columns,pd.MultiIndex): d.columns=d.columns.get_level_values(0)
            if "Close" not in d: continue
            s=pd.to_numeric(d["Close"],errors="coerce").dropna()
            s.index=pd.to_datetime(s.index).tz_localize(None)
            if len(s)>=260: data[ticker]=s[~s.index.duplicated(keep="last")].sort_index()
        except Exception as e: print(f"[WARN] {ticker}: {e}",flush=True)
        time.sleep(.1)
    return data

def rsi(s,n=14):
    delta=s.diff(); up=delta.clip(lower=0).ewm(alpha=1/n,adjust=False).mean()
    dn=(-delta.clip(upper=0)).ewm(alpha=1/n,adjust=False).mean()
    rs=up/dn.replace(0,np.nan)
    return 100-100/(1+rs)

def make_features(prices,ticker):
    c=prices[ticker].dropna(); r=c.pct_change()
    x=pd.DataFrame(index=c.index); x["ret_1d"]=r
    for n in (2,5,10,20,50,100,200):
        x[f"mom_{n}"]=c.pct_change(n); x[f"ma_gap_{n}"]=c/c.rolling(n).mean()-1
    for n in (5,10,20,50): x[f"vol_{n}"]=r.rolling(n).std()*np.sqrt(252)
    x["rsi_14"]=rsi(c); x["dd_50"]=c/c.rolling(50).max()-1
    x["dd_200"]=c/c.rolling(200).max()-1; x["absret_20"]=r.abs().rolling(20).mean()
    for name,s in prices.items():
        if name==ticker: continue
        a=s.reindex(x.index).ffill(); key=name.replace("^","").replace("-","_")
        x[f"{key}_ret1"]=a.pct_change(); x[f"{key}_mom5"]=a.pct_change(5); x[f"{key}_mom20"]=a.pct_change(20)
    if "^VIX" in prices:
        v=prices["^VIX"].reindex(x.index).ffill(); x["vix_level"]=v; x["vix_chg5"]=v.pct_change(5)
    if "HYG" in prices and "LQD" in prices:
        cr=prices["HYG"].reindex(x.index).ffill()/prices["LQD"].reindex(x.index).ffill()
        x["credit_ratio"]=cr; x["credit_mom20"]=cr.pct_change(20)
    return x.replace([np.inf,-np.inf],np.nan)

def labeled(x,close,horizon):
    n=HORIZONS[horizon]; y=x.copy()
    y["forward_return"]=close.shift(-n).reindex(x.index)/close.reindex(x.index)-1
    y["up"]=(y["forward_return"]>0).astype(float)
    y.loc[y.index[-n:],["up","forward_return"]]=np.nan
    return y

def models():
    a=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),
      ("scale",StandardScaler()),("lr",LogisticRegression(max_iter=1500,class_weight="balanced"))])
    b=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),
      ("rf",RandomForestClassifier(n_estimators=250,min_samples_leaf=15,max_features="sqrt",
       class_weight="balanced_subsample",n_jobs=-1,random_state=29))])
    reg=Pipeline([("imp",SimpleImputer(strategy="median",add_indicator=True)),
      ("gb",HistGradientBoostingRegressor(max_iter=150,learning_rate=.04,l2_regularization=3,random_state=29))])
    return a,b,reg

def run_forecast(asset,ticker,prices,x,horizon):
    z=labeled(x,prices[ticker],horizon).dropna(subset=["up","forward_return"])
    current=x.dropna(how="all").tail(1); cols=list(x.columns); n=HORIZONS[horizon]
    if len(z)<500 or z.up.nunique()<2 or current.empty:
        return None,{"asset":asset,"horizon":horizon,"status":"insufficient_history","n":len(z)}
    split=int(len(z)*.8); fit=z.iloc[:max(1,split-n)]; test=z.iloc[split:]
    if len(fit)<250 or len(test)<30 or test.up.nunique()<2:
        return None,{"asset":asset,"horizon":horizon,"status":"insufficient_test_sample"}
    a,b,g=models(); a.fit(fit[cols],fit.up.astype(int)); b.fit(fit[cols],fit.up.astype(int))
    p=(a.predict_proba(test[cols])[:,1]+b.predict_proba(test[cols])[:,1])/2
    diag={"asset":asset,"horizon":horizon,"status":"ok","train_end":str(fit.index.max().date()),
      "test_start":str(test.index.min().date()),"test_end":str(test.index.max().date()),
      "n_train":len(fit),"n_test":len(test),"auc":float(roc_auc_score(test.up.astype(int),p)),
      "brier":float(brier_score_loss(test.up.astype(int),p)),
      "directional_accuracy":float(((p>=.5)==(test.up.astype(int)==1)).mean()),
      "test_up_rate":float(test.up.mean()),"test_mean_forward_return":float(test.forward_return.mean())}
    a.fit(z[cols],z.up.astype(int)); b.fit(z[cols],z.up.astype(int)); g.fit(z[cols],z.forward_return)
    prob=float((a.predict_proba(current[cols])[0,1]+b.predict_proba(current[cols])[0,1])/2)
    exp=float(g.predict(current[cols])[0]); hist=z.forward_return
    row={"as_of":str(current.index[-1].date()),"asset":asset,"ticker":ticker,"horizon":horizon,
      "probability_up":prob,"expected_return":exp,
      "historical_q10":float(hist.quantile(.1)),"historical_median":float(hist.median()),
      "historical_q90":float(hist.quantile(.9)),
      "signal":"bullish_bias" if prob>=.6 and exp>0 else "bearish_bias" if prob<=.4 and exp<0 else "mixed_or_neutral",
      "note":"Empirical historical return quantiles, not confidence intervals."}
    return row,diag

def market_stress(prices):
    rows=[]; points=[]
    if "^VIX" in prices:
        s=prices["^VIX"]; val=float(s.iloc[-1]); pct=float((s.tail(756)<=val).mean())
        rows.append({"metric":"VIX","value":val,"percentile_3y":pct})
        if val>=25: points.append(25)
    if "HYG" in prices and "LQD" in prices:
        ratio=(prices["HYG"]/prices["LQD"]).dropna(); val=float(ratio.pct_change(20).iloc[-1])
        rows.append({"metric":"HYG_LQD_20d_change","value":val,"percentile_3y":float((ratio.pct_change(20).dropna().tail(756)<=val).mean())})
        if val<-.01: points.append(25)
    if "^NDX" in prices:
        s=prices["^NDX"]; g50=float(s.iloc[-1]/s.rolling(50).mean().iloc[-1]-1); g200=float(s.iloc[-1]/s.rolling(200).mean().iloc[-1]-1)
        rows.extend([{"metric":"NDX_vs_50dma","value":g50},{"metric":"NDX_vs_200dma","value":g200}])
        if g50<0: points.append(20)
        if g200<0: points.append(30)
    return rows,{"risk_score":min(100,sum(points)),"coverage_items":len(rows),
      "method":"Rule-based market-stress proxy, not a crash probability."}

def statement_value(df,patterns):
    if df is None or df.empty: return None
    for label in df.index:
        if any(p in str(label).lower() for p in patterns):
            vals=pd.to_numeric(df.loc[label],errors="coerce").dropna()
            if len(vals): return float(vals.iloc[0])
    return None

def fundamentals(ticker):
    r={"ticker":ticker,"status":"missing_or_partial"}
    try:
        t=yf.Ticker(ticker); inc=t.quarterly_income_stmt; cf=t.quarterly_cashflow; bs=t.quarterly_balance_sheet
        r.update({"revenue_latest":statement_value(inc,["total revenue","operating revenue"]),
          "interest_expense_latest":statement_value(inc,["interest expense"]),
          "operating_cash_flow_latest":statement_value(cf,["operating cash flow"]),
          "capex_latest":abs(statement_value(cf,["capital expenditure"])) if statement_value(cf,["capital expenditure"]) is not None else None,
          "cash_latest":statement_value(bs,["cash and cash equivalents"]),
          "total_debt_latest":statement_value(bs,["total debt"])})
        if r["revenue_latest"] is not None: r["status"]="partial"
        r["note"]="Statement line items differ by issuer; null means unavailable, not zero."
    except Exception as e: r["error"]=str(e)
    return r

def ai_infrastructure_risk(company_rows):
    """Transparent company-level capex/financing stress screen; missing fields stay missing."""
    df=pd.DataFrame(company_rows)
    if df.empty:
        return [],{"score":None,"covered_dimensions":0,"companies":0}
    revenue=pd.to_numeric(df["revenue_latest"],errors="coerce")
    capex=pd.to_numeric(df["capex_latest"],errors="coerce")
    interest=pd.to_numeric(df["interest_expense_latest"],errors="coerce")
    ocf=pd.to_numeric(df["operating_cash_flow_latest"],errors="coerce")
    debt=pd.to_numeric(df["total_debt_latest"],errors="coerce")
    cash=pd.to_numeric(df["cash_latest"],errors="coerce")
    df["capex_to_revenue"]=capex/revenue.where(revenue>0)
    df["interest_to_revenue"]=interest/revenue.where(revenue>0)
    df["ocf_after_capex"]=ocf-capex
    df["debt_to_cash"]=debt/cash.where(cash>0)
    rules=[
      ("capex_to_revenue",lambda x:x>0.50,25,"Capex exceeds 50% of quarterly revenue"),
      ("interest_to_revenue",lambda x:x>0.05,25,"Interest expense exceeds 5% of quarterly revenue"),
      ("ocf_after_capex",lambda x:x<0,25,"Operating cash flow does not cover capex"),
      ("debt_to_cash",lambda x:x>2.0,25,"Debt exceeds twice reported cash")
    ]
    details=[]; points=0.0; covered=0
    for col,condition,weight,description in rules:
        values=pd.to_numeric(df[col],errors="coerce").dropna()
        if len(values):
            covered+=1; triggered=int(condition(values).sum())
            contribution=weight*triggered/len(values); points+=contribution
            details.append({"metric":col,"companies_with_data":len(values),
              "companies_triggered":triggered,"risk_points":round(contribution,1),
              "rule":description})
        else:
            details.append({"metric":col,"companies_with_data":0,"companies_triggered":None,
              "risk_points":None,"rule":"Insufficient data; excluded"})
    return details,{"score":round(points,1) if covered else None,
      "covered_dimensions":covered,"companies":len(df),
      "method":"Equal-weight rule-based screen across selected issuers; not a default probability or calibrated index."}

def headlines():
    try:
        u="https://api.gdeltproject.org/api/v2/doc/doc"
        p={"query":"(artificial intelligence OR data center OR GPU OR semiconductor OR cloud capex OR CoreWeave OR NVIDIA)",
           "mode":"ArtList","format":"json","maxrecords":"60","timespan":"7d","sort":"HybridRel"}
        r=requests.get(u,params=p,headers={"User-Agent":"Mozilla/5.0"},timeout=20); r.raise_for_status()
        return [{"date":a.get("seendate"),"title":a.get("title"),"domain":a.get("domain"),"url":a.get("url")} for a in r.json().get("articles",[])]
    except Exception as e:
        print(f"[WARN] GDELT unavailable: {e}"); return []

def main():
    print("[INFO] Downloading 10y market data...",flush=True)
    prices=download_prices(SYMBOLS)
    if "NQ=F" not in prices and "^NDX" in prices:
        target_map={"NQ_proxy_NDX":"^NDX","SOXL":"SOXL"}
        print("[WARN] NQ futures feed unavailable; using ^NDX cash-index proxy.",flush=True)
    else:
        target_map=TARGETS
    if "NQ=F" not in prices and "^NDX" not in prices:
        raise RuntimeError("Neither NQ futures nor Nasdaq-100 cash-index data is available")
    forecasts=[]; diagnostics=[]
    for asset,ticker in target_map.items():
        if ticker not in prices:
            diagnostics.append({"asset":asset,"status":"price_data_missing"}); continue
        x=make_features(prices,ticker)
        for h in HORIZONS:
            try:
                row,diag=run_forecast(asset,ticker,prices,x,h); diagnostics.append(diag)
                if row: forecasts.append(row)
            except Exception as e: diagnostics.append({"asset":asset,"horizon":h,"status":"error","error":str(e)})
    stress,stress_summary=market_stress(prices)
    company_data=[fundamentals(t) for t in COMPANIES]
    ai_risk_details,ai_risk_summary=ai_infrastructure_risk(company_data)
    news=headlines(); now=pd.Timestamp.now(tz="UTC").isoformat()
    pd.DataFrame(forecasts).to_csv(OUT/"market_forecasts_v4.csv",index=False)
    pd.DataFrame(diagnostics).to_csv(OUT/"walkforward_validation_v4.csv",index=False)
    pd.DataFrame(stress).to_csv(OUT/"market_stress_proxies_v4.csv",index=False)
    pd.DataFrame([stress_summary]).to_csv(OUT/"market_stress_summary_v4.csv",index=False)
    pd.DataFrame(company_data).to_csv(OUT/"ai_company_fundamentals_v4.csv",index=False)
    pd.DataFrame(ai_risk_details).to_csv(OUT/"ai_infrastructure_risk_details_v4.csv",index=False)
    pd.DataFrame([ai_risk_summary]).to_csv(OUT/"ai_infrastructure_risk_summary_v4.csv",index=False)
    pd.DataFrame(news).to_csv(OUT/"ai_news_v4.csv",index=False)
    summary={"run_time_utc":now,"downloaded_price_series":len(prices),"forecast_rows":len(forecasts),
      "news_articles":len(news),"fundamental_companies":len(company_data),
      "market_stress_proxy_score":stress_summary["risk_score"],"stress_metrics":stress_summary["coverage_items"],
      "ai_infrastructure_risk_score":ai_risk_summary["score"],
      "ai_risk_covered_dimensions":ai_risk_summary["covered_dimensions"],
      "options_chain_data":"not_connected"}
    (OUT/"run_summary_v4.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    lines=["# Market Forecast & AI Risk V4","",f"- Run time UTC: {now}",
      "- Sources: Yahoo Finance market prices/company statements; GDELT recent headlines.",
      "- Forecasts are research estimates, not investment advice or guaranteed outcomes.",
      "- GEX, Call Wall, Put Wall and dealer positioning are not connected; no synthetic values are shown.","",
      "## NQ / SOXL forecasts",""]
    if forecasts: lines.append(pd.DataFrame(forecasts).to_markdown(index=False))
    else: lines.append("No forecast rows generated.")
    lines += ["","## Walk-forward holdout diagnostics","",pd.DataFrame(diagnostics).to_markdown(index=False) if diagnostics else "No diagnostics.",
      "","## Market stress proxy","",f"- Score: {stress_summary['risk_score']}/100 (rule-based, not probability).",
      pd.DataFrame(stress).to_markdown(index=False) if stress else "No market stress metrics.",
      "","## AI infrastructure financing-risk screen","",
      f"- Score: {ai_risk_summary['score']}/100; covered dimensions: {ai_risk_summary['covered_dimensions']}/4.",
      f"- Method: {ai_risk_summary['method']}",
      pd.DataFrame(ai_risk_details).to_markdown(index=False),
      "","## AI company fundamentals","",pd.DataFrame(company_data).to_markdown(index=False),
      "","## Recent AI headlines",""]
    lines += [f"- {a.get('date')}: {a.get('title')} ({a.get('domain')})" for a in news[:30]] or ["No headlines retrieved."]
    lines += ["","## Limitations","",
      "- Short-horizon probability outputs are not yet calibrated; inspect AUC/Brier and sample counts.",
      "- Historical q10/q50/q90 are empirical forward-return quantiles, not confidence intervals.",
      "- Financial statement line items vary; missing values are not interpreted as zero.",
      "- Credit ETF ratios are market proxies, not CCC spreads or CDS.",
      "- Current headlines are not inserted into historical training, avoiding false historical news backtests.",
      "- Universe is limited to the explicitly listed index, ETF, cloud and semiconductor symbols; not a whole-market scan.",""]
    (OUT/"market_dashboard_v4.md").write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps(summary,indent=2),flush=True)

if __name__=="__main__": main()
