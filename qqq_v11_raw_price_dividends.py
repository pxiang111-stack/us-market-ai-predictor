"""V11: $10k QQQ cash-account backtest using raw executable prices + cash dividends.

Signals are generated from raw OHLC prices after close t and traded at close t+1.
Cash dividends are credited on ex-dates to shares held at the prior close.
Whole shares, long-only, no leverage, no shorting, no fractional shares.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf
from qqq_v8_trend_breakout import build_signals, state_machine

INITIAL_CASH = 10_000.0
COST_BPS = 10.0
OUT = Path("output/qqq_v11_raw_price_dividends")
OUT.mkdir(parents=True, exist_ok=True)

def download_raw_bars() -> pd.DataFrame:
    raw = yf.download("QQQ", period="max", interval="1d", auto_adjust=False,
                      actions=True, progress=False, threads=False, timeout=30)
    if raw is None or raw.empty:
        raise RuntimeError("QQQ raw daily data unavailable")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ["High", "Low", "Close", "Adj Close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"Missing raw OHLC columns: {missing}")
    for col in ["Dividends", "Stock Splits"]:
        if col not in raw.columns:
            raw[col] = 0.0
    raw = raw[required + ["Dividends", "Stock Splits"]].apply(pd.to_numeric, errors="coerce")
    raw = raw.dropna(subset=["Close", "High", "Low"])
    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx.normalize()
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()
    return raw

def simulate(name: str, bars: pd.DataFrame, target: pd.Series, buy_at_start=False):
    cash, shares, costs, dividends = INITIAL_CASH, 0, 0.0, 0.0
    trades, rows = [], []
    executable = target.reindex(bars.index).shift(1).fillna(0.0)
    if buy_at_start:
        executable.iloc[0] = 1.0
    peak = INITIAL_CASH
    for i, (date, row) in enumerate(bars.iterrows()):
        price = float(row["Close"])
        div_per_share = float(row["Dividends"]) if pd.notna(row["Dividends"]) else 0.0
        # Ex-date entitlement belongs to positions held at the previous close.
        if div_per_share > 0 and shares:
            div_cash = shares * div_per_share
            cash += div_cash
            dividends += div_cash
        # Yahoo OHLC is already on a split-adjusted share basis. Do NOT multiply
        # the share count again; that would double-count QQQ's 2000 2-for-1 split.
        desired = bool(executable.loc[date] > 0)
        target_shares = int(cash // (price * (1.0 + COST_BPS/10000.0))) if desired and shares == 0 else (shares if desired else 0)
        if target_shares > shares:
            qty = target_shares - shares
            notional = qty * price
            fee = notional * COST_BPS / 10000.0
            cash -= notional + fee
            shares += qty
            costs += fee
            trades.append({"date":str(date.date()),"side":"BUY","shares":qty,"price":price,"notional":notional,"cost":fee,"cash_after":cash,"shares_after":shares})
        elif target_shares < shares:
            qty = shares - target_shares
            notional = qty * price
            fee = notional * COST_BPS / 10000.0
            cash += notional - fee
            shares -= qty
            costs += fee
            trades.append({"date":str(date.date()),"side":"SELL","shares":qty,"price":price,"notional":notional,"cost":fee,"cash_after":cash,"shares_after":shares})
        equity = cash + shares * price
        peak = max(peak, equity)
        rows.append({"date":str(date.date()),"raw_close":price,"adj_close":float(row["Adj Close"]),"dividend_per_share":div_per_share,"cash":cash,"shares":shares,"market_value":shares*price,"equity":equity,"drawdown_pct":(equity/peak-1)*100,"dividends_cumulative":dividends,"costs_cumulative":costs,"signal_from_prior_close":float(executable.loc[date])})
    curve = pd.DataFrame(rows)
    ledger = pd.DataFrame(trades, columns=["date","side","shares","price","notional","cost","cash_after","shares_after"])
    daily = curve.equity.pct_change().dropna()
    years = max((pd.to_datetime(curve.date.iloc[-1])-pd.to_datetime(curve.date.iloc[0])).days/365.25, 1/365.25)
    end = float(curve.equity.iloc[-1])
    summary = {
      "strategy":name,"initial_cash_usd":INITIAL_CASH,"ending_equity_usd":end,
      "profit_loss_usd":end-INITIAL_CASH,"total_return_pct":(end/INITIAL_CASH-1)*100,
      "cagr_pct":((end/INITIAL_CASH)**(1/years)-1)*100 if end>0 else None,
      "max_drawdown_pct":float((curve.equity/curve.equity.cummax()-1).min()*100),
      "sharpe_rf0":float(daily.mean()/daily.std(ddof=1)*np.sqrt(252)) if len(daily)>1 and daily.std(ddof=1)>0 else None,
      "trade_orders":len(ledger),"buy_orders":int((ledger.side=="BUY").sum()) if len(ledger) else 0,
      "sell_orders":int((ledger.side=="SELL").sum()) if len(ledger) else 0,
      "total_trading_cost_usd":costs,"dividends_received_usd":dividends,
      "ending_cash_usd":float(curve.cash.iloc[-1]),"ending_shares":int(curve.shares.iloc[-1]),
      "average_invested_pct":float((curve.market_value/curve.equity).replace([np.inf,-np.inf],np.nan).fillna(0).mean()*100),
      "start_date":curve.date.iloc[0],"end_date":curve.date.iloc[-1]
    }
    return summary,curve,ledger

def main():
    bars=download_raw_bars()
    f=build_signals(bars)
    configs=[("trend_long_only_cash","trend"),("breakout_long_only_cash","breakout_long")]
    summaries,curves,ledgers=[],[],[]
    for name,mode in configs:
        sig=state_machine(f,mode)
        s,c,l=simulate(name,bars,(sig>0).astype(float))
        summaries.append(s); c.insert(1,"strategy",name)
        if len(l): l.insert(1,"strategy",name)
        curves.append(c); ledgers.append(l)
    bh=pd.Series(1.0,index=bars.index)
    s,c,l=simulate("QQQ_buy_and_hold_cash_dividends",bars,bh,buy_at_start=True)
    summaries.insert(0,s); c.insert(1,"strategy","QQQ_buy_and_hold_cash_dividends")
    if len(l): l.insert(1,"strategy","QQQ_buy_and_hold_cash_dividends")
    curves.insert(0,c); ledgers.insert(0,l)
    pd.DataFrame(summaries).to_csv(OUT/"account_summary.csv",index=False)
    pd.concat(curves,ignore_index=True).to_csv(OUT/"equity_cash_share_curves.csv",index=False)
    pd.concat(ledgers,ignore_index=True).to_csv(OUT/"trade_ledger.csv",index=False)
    split_checks=[]
    for date, row in bars[bars["Stock Splits"].fillna(0) > 0].iterrows():
        loc=bars.index.get_loc(date)
        if loc == 0:
            continue
        prev=bars.iloc[loc-1]
        split_checks.append({"date":str(date.date()),"split_ratio":float(row["Stock Splits"]),
          "previous_close":float(prev["Close"]),"event_close":float(row["Close"]),
          "previous_adj_close":float(prev["Adj Close"]),"event_adj_close":float(row["Adj Close"]),
          "close_ratio_event_vs_previous":float(row["Close"]/prev["Close"]),
          "adj_close_ratio_event_vs_previous":float(row["Adj Close"]/prev["Adj Close"]),
          "close_to_adj_close_previous":float(prev["Close"]/prev["Adj Close"]),
          "close_to_adj_close_event":float(row["Close"]/row["Adj Close"])})
    pd.DataFrame(split_checks).to_csv(OUT/"split_diagnostics.csv",index=False)
    meta={"status":"completed","ticker":"QQQ","initial_cash_usd":INITIAL_CASH,
      "price_source":"Yahoo Finance OHLC + Adj Close; auto_adjust=False. OHLC is already split-adjusted; Adj Close additionally reflects dividend adjustments. Share count is not multiplied on split dates.",
      "dividends":"Cash distributions credited on ex-date to shares held at prior close; not reinvested",
      "execution":"signal after close t; execute at raw close t+1; buy-and-hold enters first close",
      "account":"whole shares, cash only, long only, no leverage, no shorting, no fractional shares",
      "transaction_cost_bps_per_order":COST_BPS,"cash_interest":0,
      "limitations":["Daily close execution approximation; opening gaps and intraday fills not modeled.",
      "10bp is assumed combined friction, not broker-specific verified fees.",
      "Cash dividends are not reinvested; taxes, cash interest and market impact excluded.",
      "Only QQQ is tested; this is not proof of whole-market stock selection skill.",
      "Historical data vendor revisions may alter past bars.",
      "split_diagnostics.csv audits Close and Adj Close around each reported split. Yahoo OHLC already uses split-adjusted share units, so the simulator must not multiply shares again."]}
    (OUT/"experiment_metadata.json").write_text(json.dumps(meta,indent=2),encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False),flush=True)

if __name__=="__main__": main()
