"""V14: QQQ trend-core + tactical overlay portfolio test.

Frozen components from V12; tests whether a trend core plus short tactical
entries improves risk-adjusted returns without excessive turnover.
Whole shares, cash-only, long-only, no leverage/shorts. Signals at close t,
rebalance at close t+1. Target weight changes trigger rebalances only.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from qqq_v11_raw_price_dividends import download_raw_bars

INITIAL_CASH = 10_000.0
OUT = Path("output/qqq_v14_portfolio")
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

def simulate_weight(name: str, bars: pd.DataFrame, target_weight: pd.Series,
                    cost_bps: float, buy_at_start: bool = False):
    cash, shares, costs, dividends = INITIAL_CASH, 0, 0.0, 0.0
    rows, trades = [], []
    weights = target_weight.reindex(bars.index).clip(0, 1).fillna(0.0).shift(1).fillna(0.0)
    if buy_at_start:
        weights.iloc[0] = 1.0
    prior_weight = 0.0
    peak = INITIAL_CASH
    for i, (date, row) in enumerate(bars.iterrows()):
        price = float(row["Close"])
        div = float(row["Dividends"]) if pd.notna(row["Dividends"]) else 0.0
        if div > 0 and shares:
            amount = shares * div
            cash += amount
            dividends += amount
        desired_weight = float(weights.iloc[i])
        if i == 0 and buy_at_start:
            prior_weight = -1.0
        if abs(desired_weight - prior_weight) > 1e-10:
            equity_before = cash + shares * price
            desired_shares = int((equity_before * desired_weight) // price) if desired_weight > 0 else 0
            if desired_shares > shares:
                qty = desired_shares - shares
                max_affordable = int(cash // (price * (1.0 + cost_bps / 10000.0)))
                qty = min(qty, max_affordable)
                if qty > 0:
                    notional = qty * price
                    fee = notional * cost_bps / 10000.0
                    cash -= notional + fee
                    shares += qty
                    costs += fee
                    trades.append({"date": str(date.date()), "side": "BUY", "shares": qty,
                                   "price": price, "notional": notional, "cost": fee,
                                   "target_weight": desired_weight})
            elif desired_shares < shares:
                qty = shares - desired_shares
                notional = qty * price
                fee = notional * cost_bps / 10000.0
                cash += notional - fee
                shares -= qty
                costs += fee
                trades.append({"date": str(date.date()), "side": "SELL", "shares": qty,
                               "price": price, "notional": notional, "cost": fee,
                               "target_weight": desired_weight})
            prior_weight = desired_weight
        equity = cash + shares * price
        peak = max(peak, equity)
        rows.append({"date": str(date.date()), "close": price, "target_weight": desired_weight,
                     "cash": cash, "shares": shares, "market_value": shares * price,
                     "equity": equity, "drawdown_pct": (equity / peak - 1) * 100,
                     "dividends_cumulative": dividends, "costs_cumulative": costs,
                     "actual_invested_pct": (shares * price / equity * 100) if equity else 0})
    curve = pd.DataFrame(rows)
    ledger = pd.DataFrame(trades)
    daily = curve.equity.pct_change().dropna()
    years = max((pd.to_datetime(curve.date.iloc[-1]) -
                 pd.to_datetime(curve.date.iloc[0])).days / 365.25, 1 / 365.25)
    end = float(curve.equity.iloc[-1])
    summary = {
        "strategy": name, "cost_bps_per_order": cost_bps,
        "initial_cash_usd": INITIAL_CASH, "ending_equity_usd": end,
        "total_return_pct": (end / INITIAL_CASH - 1) * 100,
        "cagr_pct": ((end / INITIAL_CASH) ** (1 / years) - 1) * 100 if end > 0 else None,
        "max_drawdown_pct": float((curve.equity / curve.equity.cummax() - 1).min() * 100),
        "sharpe_rf0": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252))
            if len(daily) > 1 and daily.std(ddof=1) > 0 else None,
        "orders": len(ledger), "buy_orders": int((ledger.side == "BUY").sum()) if len(ledger) else 0,
        "sell_orders": int((ledger.side == "SELL").sum()) if len(ledger) else 0,
        "trading_cost_usd": costs, "cash_dividends_usd": dividends,
        "average_invested_pct": float(curve.actual_invested_pct.mean()),
        "ending_cash_usd": float(curve.cash.iloc[-1]), "ending_shares": int(curve.shares.iloc[-1]),
        "start_date": curve.date.iloc[0], "end_date": curve.date.iloc[-1]
    }
    return summary, curve, ledger

def main():
    bars = download_raw_bars()
    close = bars["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()
    sma200 = close.rolling(200).mean()
    bull = (close > sma200) & (sma50 > sma200)
    mom3 = close.pct_change(3)
    pullback = bull & (mom3 < -0.03) & (rsi(close, 2) < 10)
    momentum = bull & (mom3 > 0.015) & (close > sma20)
    pullback_active = timed_position(pullback, 3)
    momentum_active = timed_position(momentum, 3)
    trend = bull.astype(float)

    strategies = {
        "QQQ_buy_and_hold": (pd.Series(1.0, index=bars.index), True),
        "trend_only_100": (trend, False),
        "trend_core_70_pullback_30": ((0.70 * trend + 0.30 * pullback_active).clip(0, 1), False),
        "trend_core_50_pullback_50": ((0.50 * trend + 0.50 * pullback_active).clip(0, 1), False),
        "trend_core_70_momentum_30": ((0.70 * trend + 0.30 * momentum_active).clip(0, 1), False),
        "trend_core_50_momentum_50": ((0.50 * trend + 0.50 * momentum_active).clip(0, 1), False),
    }
    all_summaries, all_curves, all_ledgers = [], [], []
    for cost in [5.0, 10.0, 20.0, 30.0]:
        for name, (weight, is_bh) in strategies.items():
            summary, curve, ledger = simulate_weight(name, bars, weight, cost, is_bh)
            all_summaries.append(summary)
            curve.insert(1, "strategy", name)
            all_curves.append(curve)
            if len(ledger):
                ledger.insert(1, "strategy", name)
                all_ledgers.append(ledger)
    summaries = pd.DataFrame(all_summaries)
    summaries.to_csv(OUT / "cost_sensitivity.csv", index=False)
    pd.concat(all_curves, ignore_index=True).to_csv(OUT / "equity_curves.csv", index=False)
    pd.concat(all_ledgers, ignore_index=True).to_csv(OUT / "trade_ledger.csv", index=False)
    meta = {
        "status": "completed", "ticker": "QQQ", "initial_cash_usd": INITIAL_CASH,
        "cost_grid_bps_per_order": [5, 10, 20, 30],
        "execution": "Signal after close t; rebalance at close t+1; target-weight changes only.",
        "account": "Whole shares, cash only, long only, no leverage, no shorting.",
        "dividends": "Cash dividends credited to prior-close shares; not reinvested.",
        "strategy_design": "Trend core uses close>SMA200 and SMA50>SMA200. Tactical pullback uses frozen V12 bull-regime, 3-day return<-3%, RSI(2)<10 entry and 3-session duration. Tactical momentum uses frozen V12 bull-regime, 3-day return>1.5%, close>SMA20 and 3-session duration.",
        "limitations": [
            "Historical regime results are not a genuinely untouched future test.",
            "Close fills approximate execution; spreads, gaps, intraday fills and market impact excluded.",
            "Cash interest, taxes and dividend reinvestment excluded.",
            "Only QQQ tested; no whole-market stock selection.",
            "Target weights rebalance only when the desired target changes, not daily to maintain exact weights."
        ]
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    cols = ["strategy", "cost_bps_per_order", "ending_equity_usd", "cagr_pct",
            "max_drawdown_pct", "sharpe_rf0", "orders", "trading_cost_usd", "average_invested_pct"]
    print(summaries[cols].to_string(index=False), flush=True)

if __name__ == "__main__":
    main()
