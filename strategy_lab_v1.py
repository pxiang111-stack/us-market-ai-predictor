"""Cash-only, whole-share QQQ strategy lab.

Signals are observed after a session close and executed at the next session open.
The simulator permits long QQQ or cash only: no shorts, leverage, or fractional
shares. Dividends are credited to cash on ex-dates and are not reinvested unless
a later signal buys shares. Yahoo OHLC is requested with auto_adjust=False:
the Close series is split-adjusted but not dividend-adjusted; dividend cash
flows are read from the Dividends action column.

This is a transparent rules-based research baseline, not a live trading system.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("output/strategy_lab_v1")
INITIAL_EQUITY = 10_000.0
COST_CASES_BPS = (5.0, 10.0, 20.0, 30.0)
HOLDOUT_FRACTION = 0.30
BACKTEST_YEARS = 3
WARMUP_SESSIONS = 252


def _flatten_yfinance_columns(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    d = frame.copy()
    if isinstance(d.columns, pd.MultiIndex):
        if ticker in d.columns.get_level_values(-1):
            d = d.xs(ticker, axis=1, level=-1)
        elif ticker in d.columns.get_level_values(0):
            d = d.xs(ticker, axis=1, level=0)
        else:
            d.columns = d.columns.get_level_values(0)
    return d


def download_qqq(period: str = "4y") -> pd.DataFrame:
    raw = yf.download(
        "QQQ", period=period, interval="1d", auto_adjust=False,
        actions=True, progress=False, threads=False, timeout=30,
    )
    if raw is None or raw.empty:
        raise RuntimeError("Yahoo Finance returned no QQQ history.")
    d = _flatten_yfinance_columns(raw, "QQQ")
    required = ["Open", "High", "Low", "Close"]
    missing = [name for name in required if name not in d.columns]
    if missing:
        raise RuntimeError(f"QQQ history is missing required fields: {missing}")
    if "Dividends" not in d.columns:
        d["Dividends"] = 0.0
    d = d[required + ["Dividends"]].copy()
    idx = pd.to_datetime(d.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    d.index = idx.normalize()
    d = d[~d.index.duplicated(keep="last")].sort_index()
    for col in required + ["Dividends"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d = d.dropna(subset=required)
    d["Dividends"] = d["Dividends"].fillna(0.0)
    return d


def build_strategy_signals(bars: pd.DataFrame) -> pd.DataFrame:
    """Return predeclared long/cash rules; all rolling levels exclude today's bar."""
    close = pd.to_numeric(bars["Close"], errors="coerce")
    high = pd.to_numeric(bars["High"], errors="coerce")
    low = pd.to_numeric(bars["Low"], errors="coerce")

    sma200 = close.rolling(200, min_periods=200).mean()
    momentum_12_1 = close.shift(21) / close.shift(252) - 1.0

    donchian_entry = close > high.rolling(55, min_periods=55).max().shift(1)
    donchian_exit = close < low.rolling(20, min_periods=20).min().shift(1)
    donchian_state = []
    invested = False
    for entry, exit_ in zip(donchian_entry.fillna(False), donchian_exit.fillna(False)):
        if not invested and bool(entry):
            invested = True
        elif invested and bool(exit_):
            invested = False
        donchian_state.append(invested)

    signals = pd.DataFrame(index=bars.index)
    signals["QQQ_buy_hold"] = True
    signals["SMA200_trend"] = close > sma200
    signals["MOM_12_1_plus_SMA200"] = (close > sma200) & (momentum_12_1 > 0)
    signals["Donchian_55_20"] = donchian_state
    # Keep warm-up rows so the first scored session can use a valid prior-close signal.
    return signals.fillna(False).astype(bool)


def _curve_metrics(curve: pd.DataFrame, starting_equity: float | None = None) -> dict:
    if curve.empty or len(curve) < 2:
        return {"status": "insufficient_curve"}
    equity = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    if len(equity) < 2 or (equity <= 0).any():
        return {"status": "invalid_equity"}
    returns = equity.pct_change().dropna()
    base_equity = float(starting_equity) if starting_equity is not None else float(equity.iloc[0])
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    drawdown = equity / equity.cummax() - 1.0
    vol = returns.std(ddof=1) if len(returns) > 1 else np.nan
    sharpe = returns.mean() / vol * np.sqrt(252) if pd.notna(vol) and vol > 0 else np.nan
    return {
        "status": "ok",
        "start_date": str(equity.index[0].date()),
        "end_date": str(equity.index[-1].date()),
        "start_equity": base_equity,
        "ending_equity": float(equity.iloc[-1]),
        "total_return_pct": float((equity.iloc[-1] / base_equity - 1.0) * 100.0),
        "cagr_pct": float(((equity.iloc[-1] / base_equity) ** (1.0 / years) - 1.0) * 100.0),
        "max_drawdown_pct": float(drawdown.min() * 100.0),
        "sharpe_daily": float(sharpe) if pd.notna(sharpe) else None,
    }


def simulate_cash_account(
    bars: pd.DataFrame,
    target_long: pd.Series,
    *,
    initial_equity: float = INITIAL_EQUITY,
    cost_bps_per_order: float = 10.0,
    start_date: pd.Timestamp | str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Integer-share cash account; signal at close t fills at open t+1."""
    required = {"Open", "Close", "Dividends"}
    if not required.issubset(bars.columns):
        raise ValueError(f"bars missing fields: {sorted(required - set(bars.columns))}")
    if initial_equity <= 0 or cost_bps_per_order < 0:
        raise ValueError("initial_equity must be positive and costs cannot be negative")
    b = bars.sort_index().copy()
    s = target_long.reindex(b.index).fillna(False).astype(bool)
    cash, shares = float(initial_equity), 0
    cost_rate = cost_bps_per_order / 10000.0
    ledger, curve = [], []
    dividend_cash = 0.0
    start_pos = 0 if start_date is None else int(b.index.searchsorted(pd.Timestamp(start_date)))
    if start_pos >= len(b):
        raise ValueError('start_date is after the available price history')

    for i in range(start_pos, len(b)):
        date, bar = b.iloc[i].name, b.iloc[i]
        # Shares held at the prior close receive the ex-date cash dividend.
        dividend_per_share = float(bar["Dividends"]) if pd.notna(bar["Dividends"]) else 0.0
        if i > 0 and dividend_per_share > 0 and shares:
            credit = shares * dividend_per_share
            cash += credit
            dividend_cash += credit

        # Execute the previous session close signal at today's open, including
        # the valid warm-up signal immediately before the scored start date.
        target = False if i == 0 else bool(s.iloc[i - 1])
        open_price = float(bar["Open"])
        if not np.isfinite(open_price) or open_price <= 0:
            raise ValueError(f"Invalid open price on {date}")
        if target and shares == 0:
            qty = int(math.floor(cash / (open_price * (1.0 + cost_rate))))
            if qty > 0:
                notional = qty * open_price
                fee = notional * cost_rate
                cash -= notional + fee
                shares += qty
                ledger.append({
                    "date": date, "side": "BUY", "shares": qty,
                    "price": open_price, "notional": notional, "cost": fee,
                    "cash_after": cash, "shares_after": shares,
                })
        elif not target and shares > 0:
            qty = shares
            notional = qty * open_price
            fee = notional * cost_rate
            cash += notional - fee
            shares = 0
            ledger.append({
                "date": date, "side": "SELL", "shares": qty,
                "price": open_price, "notional": notional, "cost": fee,
                "cash_after": cash, "shares_after": shares,
            })

        close_price = float(bar["Close"])
        equity = cash + shares * close_price
        if cash < -1e-7 or shares < 0 or shares != int(shares):
            raise AssertionError("Cash-only / whole-share invariant violated.")
        curve.append({
            "date": date, "equity": equity, "cash": cash,
            "shares": shares, "close": close_price,
            "invested_fraction": (shares * close_price / equity) if equity > 0 else 0.0,
            "dividend_cash_cumulative": dividend_cash,
        })

    ledger_df = pd.DataFrame(ledger)
    curve_df = pd.DataFrame(curve).set_index("date")
    costs = float(ledger_df["cost"].sum()) if not ledger_df.empty else 0.0
    stats = _curve_metrics(curve_df, starting_equity=initial_equity)
    stats.update({
        "initial_equity_usd": float(initial_equity),
        "ending_cash_usd": float(cash),
        "ending_shares": int(shares),
        "orders": int(len(ledger_df)),
        "round_trips_completed": int((ledger_df["side"] == "SELL").sum()) if not ledger_df.empty else 0,
        "transaction_costs_usd": costs,
        "dividend_cash_received_usd": float(dividend_cash),
        "average_invested_fraction_pct": float(curve_df["invested_fraction"].mean() * 100.0),
        "whole_shares_only": True,
        "shorts_enabled": False,
        "leverage_enabled": False,
    })
    return ledger_df, curve_df, stats


def run_research(bars: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    signals = build_strategy_signals(bars)
    if len(bars) < WARMUP_SESSIONS + 500:
        raise RuntimeError("Not enough price history for warm-up and a 3-year strategy comparison.")
    backtest_start = bars.index[-1] - pd.DateOffset(years=BACKTEST_YEARS)
    scored = bars.loc[bars.index >= backtest_start].copy()
    if len(scored) < 500:
        raise RuntimeError("Not enough sessions in the requested 3-year backtest window.")
    split_pos = int(len(scored) * (1.0 - HOLDOUT_FRACTION))
    holdout_start = scored.index[split_pos]
    summaries, curves, ledgers = [], [], []

    for cost_bps in COST_CASES_BPS:
        for strategy in signals.columns:
            ledger, curve, stats = simulate_cash_account(
                bars, signals[strategy], initial_equity=INITIAL_EQUITY,
                cost_bps_per_order=cost_bps, start_date=backtest_start,
            )
            curve = curve.copy()
            curve["strategy"] = strategy
            curve["cost_bps_per_order"] = cost_bps
            curves.append(curve.reset_index())
            if not ledger.empty:
                ledger = ledger.copy()
                ledger["strategy"] = strategy
                ledger["cost_bps_per_order"] = cost_bps
                ledgers.append(ledger)
            holdout_curve = curve.loc[curve.index >= holdout_start]
            holdout = _curve_metrics(holdout_curve)
            summaries.append({
                "strategy": strategy,
                "cost_bps_per_order": cost_bps,
                "backtest_start": str(backtest_start.date()),
                "backtest_end": str(bars.index[-1].date()),
                "holdout_start": str(holdout_start.date()),
                **stats,
                "holdout_total_return_pct": holdout.get("total_return_pct"),
                "holdout_cagr_pct": holdout.get("cagr_pct"),
                "holdout_max_drawdown_pct": holdout.get("max_drawdown_pct"),
            })

    return pd.DataFrame(summaries), pd.concat(curves, ignore_index=True), (
        pd.concat(ledgers, ignore_index=True) if ledgers else pd.DataFrame()
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    print("[INFO] Downloading 3 scored years plus warm-up history of raw-price QQQ OHLC and dividend actions...", flush=True)
    bars = download_qqq()
    summary, curves, ledger = run_research(bars)
    summary.to_csv(OUT / "strategy_summary.csv", index=False)
    curves.to_csv(OUT / "equity_curves.csv", index=False)
    ledger.to_csv(OUT / "order_ledger.csv", index=False)
    payload = {
        "status": "completed",
        "ticker": "QQQ",
        "initial_equity_usd": INITIAL_EQUITY,
        "scored_backtest_years": BACKTEST_YEARS,
        "warmup_sessions_excluded_from_performance": WARMUP_SESSIONS,
        "cost_cases_bps_per_order": list(COST_CASES_BPS),
        "holdout_fraction": HOLDOUT_FRACTION,
        "strategies": ["QQQ_buy_hold", "SMA200_trend", "MOM_12_1_plus_SMA200", "Donchian_55_20"],
        "execution": "close signal t, next session open fill; whole shares; cash-only; long or cash",
        "dividends": "cash credited on ex-date for shares held at prior close; not reinvested automatically",
        "splits": "Yahoo Finance OHLC history is requested with auto_adjust=False; split-adjusted price history is used",
        "limitations": [
            "Yahoo Finance corporate-action fields may be revised; dividend split-adjustment consistency must be audited.",
            "No taxes, cash interest, bid/ask quote replay, market impact, or broker-specific commissions.",
            "Daily OHLC cannot determine intraday order sequencing; strategies use next-open market fills.",
            "Fixed rule parameters are research hypotheses, not optimized or proven profitable.",
            "Only the latest 3 calendar years are scored; prior sessions are indicator warm-up only.",
            "The final chronological 30% of the 3-year window is reported separately as a holdout; repeated inspection can still create research overfitting.",
        ],
        "summary_rows": int(len(summary)),
    }
    (OUT / "strategy_lab_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
