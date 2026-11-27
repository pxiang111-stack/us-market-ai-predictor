#!/usr/bin/env python3
"""每5分钟轻量监控；只有显著异动或委员会报告超过60分钟才调用DeepSeek。"""
import json, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import yfinance as yf

ROOT=Path(__file__).resolve().parent
WATCHLIST=["NVDA","MSFT","AAPL","META","AMZN","TSLA","QQQ","SPY"]
MOVE_THRESHOLD_PCT=0.75
MAX_QUOTE_AGE_MIN=15
MAX_COMMITTEE_AGE_MIN=60

def utc_now():
    return datetime.now(timezone.utc)

def latest_quote(symbol):
    df=yf.download(symbol,period="1d",interval="1m",prepost=True,auto_adjust=True,progress=False,threads=False)
    if df is None or df.empty:
        return None
    if isinstance(df.columns,pd.MultiIndex):
        df.columns=[str(x[0]) for x in df.columns]
    close=pd.to_numeric(df["Close"],errors="coerce").dropna()
    if close.empty:
        return None
    ts=pd.Timestamp(close.index[-1])
    if ts.tzinfo is None:
        ts=ts.tz_localize("America/New_York")
    ts=ts.tz_convert("UTC")
    age=(pd.Timestamp.now(tz="UTC")-ts).total_seconds()/60
    return {"price":float(close.iloc[-1]),"timestamp_utc":ts.isoformat(),
            "age_minutes":round(age,2),"fresh":0<=age<=MAX_QUOTE_AGE_MIN,
            "source":"Yahoo Finance 1-minute bars; extended hours; not guaranteed real-time"}

def main():
    snapshot={"checked_at_utc":utc_now().isoformat(),"source":"Yahoo Finance 1-minute bars (非保证实时)","quotes":{}}
    for symbol in WATCHLIST:
        try:
            q=latest_quote(symbol)
            if q: snapshot["quotes"][symbol]=q
        except Exception as exc:
            print(f"{symbol}: quote fetch failed: {type(exc).__name__}: {exc}")
    fresh={s:q for s,q in snapshot["quotes"].items() if q["fresh"]}
    snapshot["fresh_quote_count"]=len(fresh)
    (ROOT/"intraday_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if not fresh:
        print("没有足够新鲜的分钟行情；跳过委员会，避免用过期数据生成交易结论。")
        return
    report_path=ROOT/"daily_report.json"
    old={}
    if report_path.exists():
        try: old=json.loads(report_path.read_text(encoding="utf-8"))
        except Exception: old={}
    old_prices={x.get("ticker"):x.get("market",{}).get("current_price") for x in old.get("results",[])}
    age_min=None
    try:
        generated=datetime.fromisoformat(old["generated_at_utc"].replace("Z","+00:00"))
        age_min=(utc_now()-generated).total_seconds()/60
    except Exception:
        pass
    moves={}
    for symbol,q in fresh.items():
        previous=old_prices.get(symbol)
        if isinstance(previous,(int,float)) and previous>0:
            moves[symbol]=round((q["price"]/previous-1)*100,3)
    max_move=max((abs(v) for v in moves.values()),default=0)
    reason=None
    if not old_prices:
        reason="首次运行或缺少基准报告"
    elif max_move>=MOVE_THRESHOLD_PCT:
        reason=f"至少一只标的相对上次委员会基准异动达到{max_move:.3f}%"
    elif age_min is None or age_min>=MAX_COMMITTEE_AGE_MIN:
        reason=f"距离上次委员会报告已{round(age_min,1) if age_min is not None else '未知'}分钟"
    snapshot["change_vs_last_committee_pct"]=moves
    snapshot["committee_trigger_reason"]=reason
    (ROOT/"intraday_snapshot.json").write_text(json.dumps(snapshot,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    if reason:
        print("触发DeepSeek委员会："+reason)
        result=subprocess.run([sys.executable,str(ROOT/"quant_committee.py")],cwd=str(ROOT))
        if result.returncode:
            raise SystemExit(result.returncode)
    else:
        print(f"仅更新分钟行情快照；最大异动{max_move:.3f}%，委员会报告年龄{age_min:.1f}分钟，暂不调用DeepSeek。")

if __name__=="__main__":
    main()
