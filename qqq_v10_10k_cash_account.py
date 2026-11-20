"""V10: realistic $10,000 cash-account simulation for QQQ long-only signals.

Uses frozen V8 indicators/state machine without parameter tuning. Signals formed
at close t are executed at close t+1 (one-session lag). Whole shares only,
no borrowing, no leverage, no shorting, no fractional shares, no interest on
cash. Commissions/slippage are modeled as a configurable bps charge on traded
notional. This is a daily-close approximation, not broker-grade execution.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from qqq_v8_trend_breakout import download_ohlcv, build_signals, state_machine

INITIAL_CASH = 10_000.0
COST_BPS = 10.0
OUT = Path("output/qqq_v10_10k_cash_account")
OUT.mkdir(parents=True, exist_ok=True)


def simulate_cash_account(name: str, bars: pd.DataFrame, target: pd.Series):
    cash = INITIAL_CASH
    shares = 0
    total_cost = 0.0
    trades = []
    rows = []
    peak = INITIAL_CASH

    # target.shift(1): a signal known at yesterday's close is traded today.
    executable = target.reindex(bars.index).shift(1).fillna(0.0)
    for date, price in bars["Close"].items():
        price = float(price)
        desired_long = bool(executable.loc[date] > 0)
        equity_before_trade = cash + shares * price
        target_shares = int(cash // (price * (1.0 + COST_BPS / 10000.0))) if desired_long and shares == 0 else (shares if desired_long else 0)

        if target_shares > shares:
            qty = target_shares - shares
            notional = qty * price
            fee = notional * COST_BPS / 10000.0
            cash -= notional + fee
            shares += qty
            total_cost += fee
            trades.append({
                "date": str(date.date()), "side": "BUY", "shares": qty,
                "price": price, "notional": notional, "cost": fee,
                "cash_after": cash, "shares_after": shares,
            })
        elif target_shares < shares:
            qty = shares - target_shares
            notional = qty * price
            fee = notional * COST_BPS / 10000.0
            cash += notional - fee
            shares -= qty
            total_cost += fee
            trades.append({
                "date": str(date.date()), "side": "SELL", "shares": qty,
                "price": price, "notional": notional, "cost": fee,
                "cash_after": cash, "shares_after": shares,
            })

        equity = cash + shares * price
        peak = max(peak, equity)
        rows.append({
            "date": str(date.date()), "close": price, "signal_from_prior_close": float(executable.loc[date]),
            "cash": cash, "shares": shares, "market_value": shares * price,
            "equity": equity, "drawdown_pct": (equity / peak - 1.0) * 100.0,
            "costs_cumulative": total_cost,
        })

    curve = pd.DataFrame(rows)
    trade_df = pd.DataFrame(trades, columns=[
        "date", "side", "shares", "price", "notional", "cost", "cash_after", "shares_after"
    ])
    daily = curve["equity"].pct_change().dropna()
    years = max(len(daily) / 252.0, 1 / 252.0)
    end_equity = float(curve["equity"].iloc[-1])
    peak_curve = curve["equity"].cummax()
    summary = {
        "strategy": name,
        "initial_cash_usd": INITIAL_CASH,
        "ending_equity_usd": end_equity,
        "profit_loss_usd": end_equity - INITIAL_CASH,
        "total_return_pct": (end_equity / INITIAL_CASH - 1.0) * 100.0,
        "cagr_pct": ((end_equity / INITIAL_CASH) ** (1.0 / years) - 1.0) * 100.0 if end_equity > 0 else None,
        "max_drawdown_pct": float((curve["equity"] / peak_curve - 1.0).min() * 100.0),
        "sharpe_rf0": float(daily.mean() / daily.std(ddof=1) * np.sqrt(252)) if len(daily) > 1 and daily.std(ddof=1) > 0 else None,
        "trade_orders": int(len(trade_df)),
        "buy_orders": int((trade_df["side"] == "BUY").sum()) if len(trade_df) else 0,
        "sell_orders": int((trade_df["side"] == "SELL").sum()) if len(trade_df) else 0,
        "total_trading_cost_usd": total_cost,
        "ending_cash_usd": float(curve["cash"].iloc[-1]),
        "ending_shares": int(curve["shares"].iloc[-1]),
        "average_invested_pct": float((curve["market_value"] / curve["equity"]).replace([np.inf, -np.inf], np.nan).fillna(0).mean() * 100.0),
        "start_date": curve["date"].iloc[0],
        "end_date": curve["date"].iloc[-1],
    }
    return summary, curve, trade_df


def main():
    bars = download_ohlcv()
    f = build_signals(bars)
    # Strategies are frozen V8 definitions; shorts become cash, not short positions.
    configs = [
        ("trend_long_only_cash", "trend"),
        ("breakout_long_only_cash", "breakout_long"),
    ]
    summaries, curves, ledgers = [], [], []
    for name, mode in configs:
        signal = state_machine(f, mode)
        long_only_target = (signal > 0).astype(float)
        summary, curve, ledger = simulate_cash_account(name, bars, long_only_target)
        summaries.append(summary)
        curve.insert(1, "strategy", name)
        if len(ledger):
            ledger.insert(1, "strategy", name)
        curves.append(curve)
        ledgers.append(ledger)

    # Buy-and-hold with the same initial cash, whole shares, and same transaction cost.
    bh_target = pd.Series(1.0, index=bars.index)
    bh_summary, bh_curve, bh_ledger = simulate_cash_account("QQQ_buy_and_hold", bars, bh_target)
    summaries.insert(0, bh_summary)
    bh_curve.insert(1, "strategy", "QQQ_buy_and_hold")
    if len(bh_ledger):
        bh_ledger.insert(1, "strategy", "QQQ_buy_and_hold")
    curves.insert(0, bh_curve)
    ledgers.insert(0, bh_ledger)

    pd.DataFrame(summaries).to_csv(OUT / "account_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "equity_cash_share_curves.csv", index=False)
    pd.concat(ledgers, ignore_index=True).to_csv(OUT / "trade_ledger.csv", index=False)
    metadata = {
        "status": "completed", "ticker": "QQQ", "initial_cash_usd": INITIAL_CASH,
        "execution": "signal at close t is executed at close t+1; daily close data",
        "account_type": "cash-only; whole shares; long-only; no leverage; no shorting; no fractional shares",
        "transaction_cost_bps_per_order_notional": COST_BPS,
        "cash_interest": 0, "parameter_optimization": False,
        "strategies": [x[0] for x in configs],
        "benchmark": "QQQ buy-and-hold with same $10,000, whole shares and transaction cost",
        "limitations": [
            "Daily close execution is an approximation; actual fills, opening gaps and bid-ask spread may differ.",
            "10 bps per order is an assumed combined friction, not verified broker-specific commission.",
            "No taxes, cash interest, ETF tracking error or market impact beyond the assumed cost.",
            "Only QQQ is tested; results do not establish whole-market stock selection skill.",
            "No options, leverage, margin, shorting or fractional shares.",
        ],
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(pd.DataFrame(summaries).to_string(index=False), flush=True)
    print("Saved account summary, daily equity/cash/share curves and complete trade ledger.")


if __name__ == "__main__":
    main()
