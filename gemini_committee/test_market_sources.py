#!/usr/bin/env python3
"""Public market-data diagnostics; response success is not proof of realtime entitlement."""
import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

WATCHLIST=["SPY","QQQ","IWM","SMH","NVDA","AAPL","MSFT","AMZN","META","GOOGL","TSLA","AMD","AVGO","MU","ARM","TSM","PLTR","SMCI","NFLX","CRM","ORCL","APP","UBER","COIN","MSTR","HOOD","GLD","SLV","TLT","XLE"]
ETF={"SPY","QQQ","IWM","SMH","GLD","SLV","TLT","XLE"}
OUT=Path(__file__).resolve().parent
H={"User-Agent":"Mozilla/5.0 (compatible; market-source-diagnostic/1.1)","Accept":"application/json"}

def age(ts):
    return round((datetime.now(timezone.utc)-ts).total_seconds()/60,1) if ts else None

def probe_yahoo():
    rows=[]; errors=[]
    for s in WATCHLIST:
        try:
            r=requests.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{s}",params={"range":"1d","interval":"1m","includePrePost":"true"},headers=H,timeout=12);r.raise_for_status()
            c=r.json()["chart"]
            if c.get("error") or not c.get("result"): raise RuntimeError(str(c.get("error") or "empty result"))
            z=c["result"][0];ts=z.get("timestamp") or [];q=z["indicators"]["quote"][0]
            valid=[i for i,t in enumerate(ts) if i<len(q.get("close",[])) and q["close"][i] is not None]
            if not valid: raise RuntimeError("no intraday bars")
            i=valid[-1];stamp=datetime.fromtimestamp(ts[i],timezone.utc);vol=(q.get("volume") or [None]*len(ts))[i]
            rows.append({"ticker":s,"timestamp":stamp.isoformat(),"age_min":age(stamp),"price":q["close"][i],"volume":vol})
        except Exception as e:errors.append({"ticker":s,"error":str(e)[:180]})
    fresh3=sum(x["age_min"] is not None and x["age_min"]<=3 for x in rows)
    fresh5=sum(x["age_min"] is not None and x["age_min"]<=5 for x in rows)
    nonzero=sum(x["volume"] not in (None,0) for x in rows)
    return {"name":"Yahoo chart 1m includePrePost","ok":len(rows),"total":len(WATCHLIST),"fresh_le_3m":fresh3,"fresh_le_5m":fresh5,"nonzero_latest_bar_volume":nonzero,"rows":rows,"errors":errors,"note":"Unofficial endpoint; latest bars can have zero volume, and extended-hours coverage/entitlement are not guaranteed."}

def probe_tradingview():
    # Explicit exchange prefixes; try a second exchange for symbols absent in the first response.
    primary=["NASDAQ:" + s if s not in {"SPY","IWM","GLD","SLV","TLT","XLE","SMH"} else "AMEX:" + s for s in WATCHLIST]
    primary=[("NASDAQ:QQQ" if x=="AMEX:QQQ" else x) for x in primary]
    alternatives=["AMEX:SMH","NASDAQ:SMH","NASDAQ:TSM","NYSE:TSM","NASDAQ:QQQ"]
    tickers=list(dict.fromkeys(primary+alternatives))
    payload={"symbols":{"tickers":tickers,"query":{"types":[]}},"columns":["name","close","volume","change","Recommend.All"]}
    try:
        r=requests.post("https://scanner.tradingview.com/america/scan",json=payload,headers=H,timeout=20);r.raise_for_status()
        rows=[{"symbol":x.get("s"),"values":x.get("d")} for x in r.json().get("data",[])]
        unique={x["symbol"].split(":")[-1] for x in rows}
        return {"name":"TradingView scanner","ok":len(unique & set(WATCHLIST)),"total":len(WATCHLIST),"symbols_returned":sorted(unique),"rows":rows,"note":"No quote timestamp or exchange entitlement proof in this scanner response."}
    except Exception as e:return {"name":"TradingView scanner","ok":0,"total":len(WATCHLIST),"rows":[],"error":str(e)[:300],"note":"Endpoint may block or change."}

def probe_nasdaq():
    rows=[];errors=[]
    for s in WATCHLIST:
        asset="etf" if s in ETF else "stocks"
        try:
            u=f"https://api.nasdaq.com/api/quote/{s}/info?assetclass={asset}"
            r=requests.get(u,headers=H,timeout=10);r.raise_for_status();data=r.json().get("data")
            if not data:raise RuntimeError("empty response")
            p=data.get("primaryData") or {}
            rows.append({"ticker":s,"assetclass":asset,"marketStatus":data.get("marketStatus"),"primaryData":p,"secondaryData":data.get("secondaryData")})
        except Exception as e:errors.append({"ticker":s,"assetclass":asset,"error":str(e)[:150]})
    rt=sum(bool((x.get("primaryData") or {}).get("isRealTime")) for x in rows)
    stamps=sum(bool((x.get("primaryData") or {}).get("lastTradeTimestamp")) for x in rows)
    return {"name":"Nasdaq quote info (stocks/etf)","ok":len(rows),"total":len(WATCHLIST),"primary_isRealTime_true":rt,"primary_timestamp_present":stamps,"rows":rows,"errors":errors,"note":"Public quote endpoint; isRealTime is provider metadata, not proof of licensed use or continuous 1-minute bars."}

def main():
    report={"generated_utc":datetime.now(timezone.utc).isoformat(),"session_et":datetime.now(ZoneInfo("America/New_York")).isoformat(),"watchlist":WATCHLIST,"sources":[]}
    for fn in [probe_yahoo,probe_tradingview,probe_nasdaq]:
        print("PROBING",fn.__name__,flush=True);s=fn();report["sources"].append(s);print(s["name"],s.get("ok"),"/",s.get("total"),flush=True)
    y,t,n=report["sources"]
    report["assessment"]={"yahoo_fresh_le_3m":y.get("fresh_le_3m",0),"yahoo_nonzero_volume":y.get("nonzero_latest_bar_volume",0),"tradingview_coverage":t.get("ok",0),"nasdaq_coverage":n.get("ok",0),"nasdaq_realtime_flag":n.get("primary_isRealTime_true",0),"trade_signal_ready":False,"reason":"No source has yet passed repeatability, extended-hours and timestamp/volume validation."}
    (OUT/"source_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Market Data Source Probe","",f"Generated UTC: {report['generated_utc']}",f"Session ET: {report['session_et']}","","> Diagnostic only; not an execution-grade or licensed feed assessment.","","## Summary",f"- Yahoo: {y['ok']}/30 responses; {y.get('fresh_le_3m',0)} within 3m; latest bars with nonzero volume {y.get('nonzero_latest_bar_volume',0)}/30.",f"- TradingView: {t.get('ok',0)}/30 unique symbols.",f"- Nasdaq: {n.get('ok',0)}/30 responses; realtime flag true {n.get('primary_isRealTime_true',0)}.", "- Trade-signal readiness: NO (repeatability and extended-hours behavior still unverified).",""]
    for s in report["sources"]:
        lines += [f"## {s['name']}",f"- Coverage: {s.get('ok',0)}/{s.get('total',30)}",f"- Limitation: {s.get('note','')}"]
        if s.get("errors"):lines.append("- Failed tickers: "+", ".join(x["ticker"] for x in s["errors"]))
        lines.append("")
    (OUT/"source_probe.md").write_text("\n".join(lines),encoding="utf-8")
    print("REPORT WRITTEN",flush=True)
if __name__=="__main__":main()
