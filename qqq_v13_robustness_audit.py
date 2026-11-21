"""V13 cost and regime robustness audit for fixed V12 QQQ strategies."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import qqq_v11_raw_price_dividends as cashbt

OUT = Path("output/qqq_v13_robustness")
OUT.mkdir(parents=True, exist_ok=True)

def rsi(series, period):
    delta=series.diff()
    gain=delta.clip(lower=0).rolling(period).mean()
    loss=(-delta.clip(upper=0)).rolling(period).mean()
    rs=gain/loss.replace(0,np.nan)
    return 100-100/(1+rs)

def timed_position(entry, hold_days):
    active=0; values=[]
    for signal in entry.fillna(False).astype(bool):
        if active>0:
            values.append(1.0); active-=1
        elif signal:
            values.append(1.0); active=hold_days-1
        else:
            values.append(0.0)
    return pd.Series(values,index=entry.index,dtype=float)

def make_entries(bars):
    close,high=bars["Close"],bars["High"]
    sma20=close.rolling(20).mean(); sma50=close.rolling(50).mean(); sma200=close.rolling(200).mean()
    mom3=close.pct_change(3); mom5=close.pct_change(5); mom10=close.pct_change(10)
    prior_high20=high.rolling(20).max().shift(1); rsi2=rsi(close,2)
    bull=(close>sma200)&(sma50>sma200)
    return {
      "momentum_3d_hold3":timed_position(bull&(mom3>0.015)&(close>sma20),3),
      "momentum_5d_hold5":timed_position(bull&(mom5>0.025)&(close>sma20),5),
      "momentum_10d_hold10":timed_position(bull&(mom10>0.04)&(close>sma20),10),
      "breakout20_hold5":timed_position(bull&(close>prior_high20),5),
      "pullback_reversal_hold3":timed_position(bull&(mom3< -0.03)&(rsi2<10),3)
    }

def period_stats(curve, start, end, initial):
    dates=pd.to_datetime(curve["date"])
    mask=(dates>=pd.Timestamp(start))&(dates<=pd.Timestamp(end))
    x=curve.loc[mask].copy()
    if x.empty: return None
    first=x.index[0]
    start_equity=float(curve.loc[first-1,"equity"]) if first>0 else float(initial)
    vals=np.r_[start_equity,x["equity"].astype(float).values]
    peak=np.maximum.accumulate(vals)
    mdd=float(np.min(vals/peak-1)*100)
    ret=float(vals[-1]/vals[0]-1)*100
    return {"period":f"{start} to {end}","return_pct":ret,"max_drawdown_pct":mdd,
            "start_equity":start_equity,"end_equity":float(vals[-1])}

def main():
    bars=cashbt.download_raw_bars()
    entries=make_entries(bars)
    entries["QQQ_buy_and_hold_cash_dividends"]=pd.Series(1.0,index=bars.index)
    cost_rows=[]; regime_rows=[]
    periods=[
      ("1999-03-10","2009-12-31"),
      ("2010-01-01","2019-12-31"),
      ("2020-01-01","2026-11-20"),
      ("2022-01-01","2026-11-20")
    ]
    for bps in [5.0,10.0,20.0,30.0]:
        cashbt.COST_BPS=bps
        for name,signal in entries.items():
            bh=name=="QQQ_buy_and_hold_cash_dividends"
            summary,curve,ledger=cashbt.simulate(name,bars,signal,buy_at_start=bh)
            summary["cost_bps_per_order"]=bps
            cost_rows.append(summary)
            for start,end in periods:
                ps=period_stats(curve,start,end,10000.0)
                if ps:
                    regime_rows.append({"strategy":name,"cost_bps_per_order":bps,**ps})
    pd.DataFrame(cost_rows).to_csv(OUT/"cost_sensitivity.csv",index=False)
    pd.DataFrame(regime_rows).to_csv(OUT/"regime_performance.csv",index=False)
    meta={"status":"completed","ticker":"QQQ","initial_cash_usd":10000,
      "cost_grid_bps_per_order":[5,10,20,30],
      "strategies":"Same frozen V12 rules; no parameter changes during audit.",
      "periods":periods,"execution":"signal after close t; trade at close t+1",
      "account":"whole shares, cash only, long only, no leverage",
      "limitations":["Historical subperiods are diagnostics, not a substitute for a genuinely untouched future test.",
        "Close-price execution approximation; no intraday spread, gaps or market impact.",
        "Cash interest and taxes excluded; dividends paid in cash, not reinvested.",
        "Only QQQ tested."]}
    (OUT/"experiment_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(pd.DataFrame(cost_rows)[["strategy","cost_bps_per_order","ending_equity_usd","cagr_pct","max_drawdown_pct","sharpe_rf0","trade_orders","total_trading_cost_usd"]].to_string(index=False),flush=True)

if __name__=="__main__": main()
