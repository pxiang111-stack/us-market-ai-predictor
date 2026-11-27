#!/usr/bin/env python3
"""Probe candidate public market-data sources. No credentials; diagnostic only."""
import json, time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo
import requests

WATCHLIST=["SPY","QQQ","IWM","SMH","NVDA","AAPL","MSFT","AMZN","META","GOOGL","TSLA","AMD","AVGO","MU","ARM","TSM","PLTR","SMCI","NFLX","CRM","ORCL","APP","UBER","COIN","MSTR","HOOD","GLD","SLV","TLT","XLE"]
OUT=Path(__file__).resolve().parent
HEADERS={"User-Agent":"Mozilla/5.0 (compatible; market-source-diagnostic/1.0)"}

def freshness(ts):
    if not ts: return None
    return round((datetime.now(timezone.utc)-ts).total_seconds()/60,1)

def probe_yahoo():
    rows=[]; ok=0; errors=[]
    for s in WATCHLIST:
        try:
            u=f"https://query1.finance.yahoo.com/v8/finance/chart/{s}"
            r=requests.get(u,params={"range":"1d","interval":"1m","includePrePost":"true"},headers=HEADERS,timeout=12)
            r.raise_for_status(); j=r.json()["chart"]
            if j.get("error"): raise RuntimeError(str(j["error"]))
            res=j["result"][0]; ts=res.get("timestamp") or []
            q=res["indicators"]["quote"][0]
            valid=[i for i,t in enumerate(ts) if i<len(q.get("close",[])) and q["close"][i] is not None]
            if not valid: raise RuntimeError("no intraday bars")
            i=valid[-1]; stamp=datetime.fromtimestamp(ts[i],timezone.utc)
            ok+=1; rows.append({"ticker":s,"timestamp":stamp.isoformat(),"age_min":freshness(stamp),"price":q["close"][i],"volume":q.get("volume",[None])[i],"source":"Yahoo chart 1m"})
        except Exception as e: errors.append({"ticker":s,"error":str(e)[:180]})
    return {"name":"Yahoo chart 1m includePrePost","ok":ok,"total":len(WATCHLIST),"rows":rows,"errors":errors,"note":"Unofficial endpoint; availability and extended-hours data are not guaranteed."}

def probe_tradingview():
    payload={"symbols":{"tickers":["NASDAQ:"+s if s not in ["SPY","QQQ","IWM","GLD","SLV","TLT"] else "AMEX:"+s for s in WATCHLIST],"query":{"types":[]}},"columns":["name","close","volume","change","Recommend.All"]}
    try:
        r=requests.post("https://scanner.tradingview.com/america/scan",json=payload,headers=HEADERS,timeout=20); r.raise_for_status()
        data=r.json(); rows=[]
        for x in data.get("data",[]): rows.append({"symbol":x.get("s"),"values":x.get("d")})
        return {"name":"TradingView scanner","ok":len(rows),"total":len(WATCHLIST),"rows":rows,"note":"Scanner data fields do not prove exchange real-time entitlement or timestamp freshness."}
    except Exception as e:return {"name":"TradingView scanner","ok":0,"total":len(WATCHLIST),"rows":[],"error":str(e)[:300],"note":"Endpoint may block or change."}

def probe_nasdaq():
    rows=[]; errors=[]
    for s in WATCHLIST:
        try:
            u=f"https://api.nasdaq.com/api/quote/{s}/info?assetclass=stocks"
            r=requests.get(u,headers={**HEADERS,"Accept":"application/json"},timeout=10); r.raise_for_status()
            j=r.json(); data=j.get("data")
            if not data: raise RuntimeError("empty response")
            rows.append({"ticker":s,"data":data})
        except Exception as e: errors.append({"ticker":s,"error":str(e)[:150]})
    return {"name":"Nasdaq quote info endpoint","ok":len(rows),"total":len(WATCHLIST),"rows":rows,"errors":errors,"note":"Public endpoint not a licensed real-time feed; payload timestamp/fields are evaluated but not assumed."}

def main():
    report={"generated_utc":datetime.now(timezone.utc).isoformat(),"session_et":datetime.now(ZoneInfo("America/New_York")).isoformat(),"watchlist":WATCHLIST,"sources":[]}
    for fn in [probe_yahoo,probe_tradingview,probe_nasdaq]:
        print("PROBING",fn.__name__,flush=True)
        report["sources"].append(fn())
    for s in report["sources"]:
        print(s["name"],s.get("ok"),"/",s.get("total"),flush=True)
    (OUT/"source_probe.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8")
    lines=["# Market Data Source Probe","",f"Generated UTC: {report['generated_utc']}","","> Diagnostic only. Success/coverage is not proof of licensed real-time data or actionable freshness.",""]
    for s in report["sources"]:
        lines += [f"## {s['name']}",f"- Coverage/response: {s.get('ok',0)}/{s.get('total',len(WATCHLIST))}",f"- Limitation: {s.get('note','')}"]
        if s.get("error"): lines.append(f"- Error: {s['error']}")
        if s.get("errors"): lines.append(f"- Failed tickers: {', '.join(x['ticker'] for x in s['errors'])}")
        lines.append("")
    (OUT/"source_probe.md").write_text("\n".join(lines),encoding="utf-8")
    print("REPORT WRITTEN",flush=True)
if __name__=="__main__":main()
