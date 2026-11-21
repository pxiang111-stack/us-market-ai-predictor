"""V12 short-horizon cash-account experiments on QQQ.

Predeclared 3/5/10-session holding-period tests. Long-only, whole shares,
no leverage, signals at close t and execution at close t+1.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from qqq_v11_raw_price_dividends import download_raw_bars, simulate, OUT as V11_OUT

OUT = Path("output/qqq_v12_short_horizon")
OUT.mkdir(parents=True, exist_ok=True)

def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)

def timed_position(entry: pd.Series, hold_days: int) -> pd.Series:
    active = 0
    values = []
    for signal in entry.fillna(False).astype(bool):
        if active > 0:
            values.append(1.0)
            active -= 1
        elif signal:
            values.append(1.0)
            active = hold_days - 1
        else:
            values.append(0.0)
    return pd.Series(values, index=entry.index, dtype=float)

def main():
    bars = download_raw_bars()
    close, high = bars["Close"], bars["High"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    mom3 = close.pct_change(3)
    mom5 = close.pct_change(5)
    mom10 = close.pct_change(10)
    prior_high20 = high.rolling(20).max().shift(1)
    rsi2 = rsi(close, 2)
    bull = (close > sma200) & (sma50 > sma200)

    entries = {
      "momentum_3d_hold3": (bull & (mom3 > 0.015) & (close > sma20), 3),
      "momentum_5d_hold5": (bull & (mom5 > 0.025) & (close > sma20), 5),
      "momentum_10d_hold10": (bull & (mom10 > 0.04) & (close > sma20), 10),
      "breakout20_hold5": (bull & (close > prior_high20), 5),
      "pullback_reversal_hold3": (bull & (mom3 < -0.03) & (rsi2 < 10), 3),
    }
    summaries, curves, ledgers = [], [], []
    for name, (entry, days) in entries.items():
        signal = timed_position(entry, days)
        summary, curve, ledger = simulate(name, bars, signal)
        summaries.append(summary)
        curve.insert(1, "strategy", name)
        if len(ledger):
            ledger.insert(1, "strategy", name)
        curves.append(curve)
        ledgers.append(ledger)

    pd.DataFrame(summaries).to_csv(OUT / "account_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "equity_curves.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(OUT / "trade_ledger.csv", index=False)
    meta = {
      "status":"completed", "ticker":"QQQ", "initial_cash_usd":10000,
      "period":"1999-03-10 through latest available daily bar",
      "execution":"signal after close t; execute at close t+1",
      "account":"whole shares, cash only, long only, no leverage, no shorting",
      "transaction_cost_bps_per_order":10,
      "dividends":"Cash dividends on ex-date to prior-close shares; not reinvested",
      "holding_periods_sessions":[3,5,10],
      "entries":{
        "momentum_3d_hold3":"bull regime; 3-session return >1.5%; close>SMA20; hold 3 sessions",
        "momentum_5d_hold5":"bull regime; 5-session return >2.5%; close>SMA20; hold 5 sessions",
        "momentum_10d_hold10":"bull regime; 10-session return >4%; close>SMA20; hold 10 sessions",
        "breakout20_hold5":"bull regime; close exceeds prior 20-session high; hold 5 sessions",
        "pullback_reversal_hold3":"bull regime; 3-session return <-3%; RSI(2)<10; hold 3 sessions"
      },
      "limitations":[
        "Exploratory fixed-rule comparison, not optimized or proven out-of-sample.",
        "Close-price fills approximate execution; spread, intraday gaps and market impact are not modeled.",
        "10bp is an assumed combined per-order cost, not broker-specific fees.",
        "Cash interest, taxes and dividend reinvestment are excluded.",
        "Only QQQ is tested; no whole-market stock selection yet."
      ]
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
