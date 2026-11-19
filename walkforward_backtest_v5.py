"""End-to-end purged walk-forward backtest for NQ, SOXL, QQQ and SPY.

This is a daily-bar research experiment. Forecasts are produced after close t,
trade plans are formed from information through t, and fills begin at t+1 open.
It does not model overnight futures sessions, bid/ask delta, OI or options gamma.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from market_risk_v4 import download_prices, make_features, merge_market_structure_features
from trade_decision_v5 import build_trade_decision
from walkforward_forecast_v5 import generate_walkforward_forecasts
from walkforward_execution_v5 import simulate_trades, performance_summary

OUT = Path("output/walkforward_v5")
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {"NQ=F": "NQ_futures", "SOXL": "SOXL", "QQQ": "QQQ", "SPY": "SPY"}
HORIZONS = (5, 10, 20)
COST_CASES_BPS = (10.0, 20.0, 30.0)


def fetch_bars(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="10y", interval="1d", auto_adjust=False,
                      progress=False, threads=False, timeout=30)
    if raw is None or raw.empty:
        raise RuntimeError(f"No OHLCV data for {ticker}")
    if isinstance(raw.columns, pd.MultiIndex):
        if ticker in raw.columns.get_level_values(-1):
            raw = raw.xs(ticker, axis=1, level=-1)
        elif ticker in raw.columns.get_level_values(0):
            raw = raw.xs(ticker, axis=1, level=0)
        else:
            raw.columns = raw.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"{ticker} missing OHLCV columns: {missing}")
    bars = raw[required].dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    idx = pd.to_datetime(bars.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    bars.index = idx.normalize()
    return bars[~bars.index.duplicated(keep="last")]


def atr_series(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    prev = bars["Close"].shift(1)
    tr = pd.concat([bars["High"] - bars["Low"],
                    (bars["High"] - prev).abs(),
                    (bars["Low"] - prev).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def main():
    print("[INFO] Downloading cross-market close series...", flush=True)
    price_symbols = ["NQ=F", "SOXL", "QQQ", "SPY", "^VIX", "HYG", "LQD", "SMH", "NVDA"]
    prices = download_prices(price_symbols, period="10y")
    all_ledgers, all_curves, all_forecasts, summaries = [], [], [], []

    for ticker, asset in TARGETS.items():
        if ticker not in prices:
            summaries.append({"ticker": ticker, "status": "missing_close_series"})
            continue
        print(f"[INFO] Preparing {ticker} OHLCV/features...", flush=True)
        try:
            bars = fetch_bars(ticker)
            x = make_features(prices, ticker)
            x, metadata = merge_market_structure_features(x, bars, ticker)
            close = prices[ticker].reindex(x.index)
            atr = atr_series(bars).reindex(x.index)
        except Exception as exc:
            summaries.append({"ticker": ticker, "status": "data_error", "error": str(exc)})
            continue

        for horizon in HORIZONS:
            print(f"[INFO] {ticker} {horizon}d walk-forward forecasts...", flush=True)
            try:
                forecasts = generate_walkforward_forecasts(
                    x, close, horizon, min_train_rows=500,
                    retrain_every=63, max_train_rows=1260,
                )
                if forecasts.empty:
                    summaries.append({"ticker": ticker, "horizon": horizon,
                                      "status": "insufficient_walkforward_history"})
                    continue
                plan_rows = []
                for date, pred in forecasts.iterrows():
                    if date not in bars.index or date not in x.index:
                        continue
                    row = build_trade_decision(
                        close=float(close.loc[date]), atr=float(atr.loc[date]),
                        probability_up=float(pred["probability_up"]),
                        expected_return=float(pred["expected_return"]),
                        features=x.loc[date].to_dict(),
                        allow_short=(ticker == "NQ=F"),
                    )
                    row.update({"as_of": date, "ticker": ticker, "asset": asset,
                                "horizon_sessions": horizon,
                                "train_end": pred["train_end"],
                                "n_train": int(pred["n_train"]),
                                "probability_calibrated": False})
                    plan_rows.append(row)
                plans = pd.DataFrame(plan_rows).set_index("as_of") if plan_rows else pd.DataFrame()
                if plans.empty:
                    summaries.append({"ticker": ticker, "horizon": horizon, "status": "no_plans"})
                    continue
                forecasts = forecasts.join(plans[["decision"]], how="left")
                forecasts["ticker"] = ticker
                forecasts["horizon_sessions"] = horizon
                all_forecasts.append(forecasts.reset_index())
                plans["decision"] = plans["decision"].fillna("FLAT")

                for cost_bps in COST_CASES_BPS:
                    ledger, curve = simulate_trades(
                        bars, plans[["decision", "stop_loss", "take_profit_1r", "take_profit_2r"]],
                        initial_equity=100_000.0, risk_budget_pct=0.005,
                        max_gross_exposure=1.0, round_trip_cost_bps=cost_bps,
                        slippage_bps_per_side=5.0, max_holding_sessions=horizon,
                    )
                    stats = performance_summary(ledger, curve)
                    tag = f"{ticker}_{horizon}d_cost{int(cost_bps)}bps"
                    if not ledger.empty:
                        ledger = ledger.copy()
                        ledger["ticker"], ledger["horizon_sessions"], ledger["round_trip_cost_bps"] = ticker, horizon, cost_bps
                        ledger["strategy_id"] = tag
                        all_ledgers.append(ledger)
                    if not curve.empty:
                        curve = curve.copy()
                        curve["ticker"], curve["horizon_sessions"], curve["round_trip_cost_bps"] = ticker, horizon, cost_bps
                        curve["strategy_id"] = tag
                        all_curves.append(curve)
                    summaries.append({
                        "strategy_id": tag, "ticker": ticker, "horizon_sessions": horizon,
                        "round_trip_cost_bps": cost_bps, "slippage_bps_per_side": 5.0,
                        "forecast_count": int(len(forecasts)),
                        "long_signals": int((plans["decision"] == "LONG").sum()),
                        "short_signals": int((plans["decision"] == "SHORT").sum()),
                        "flat_signals": int((plans["decision"] == "FLAT").sum()),
                        "probability_calibrated": False,
                        "feed": "Yahoo Finance daily OHLCV",
                        **stats,
                    })
            except Exception as exc:
                summaries.append({"ticker": ticker, "horizon": horizon,
                                  "status": "backtest_error", "error": str(exc)})
            time.sleep(0.2)

    pd.DataFrame(all_forecasts).to_csv(OUT / "oos_forecasts.csv", index=False)
    pd.DataFrame(all_ledgers).to_csv(OUT / "trade_ledger.csv", index=False)
    pd.DataFrame(all_curves).to_csv(OUT / "daily_equity_curves.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "performance_summary.csv", index=False)
    payload = {
        "status": "completed",
        "targets": list(TARGETS),
        "horizons": list(HORIZONS),
        "cost_cases_round_trip_bps": list(COST_CASES_BPS),
        "risk_budget_pct": 0.5,
        "max_gross_exposure": 1.0,
        "shorts_enabled_only_for": ["NQ=F"],
        "probabilities_calibrated": False,
        "data_scope": "daily OHLCV only; not intraday, overnight-session-separated or order-book data",
        "summary_rows": len(summary_df),
        "successful_rows": int((summary_df.get("status", pd.Series(index=summary_df.index, dtype=object)) == "ok").sum()),
        "limitations": [
            "This is an experimental historical simulation, not a live trading system.",
            "Yahoo daily bars do not separate regular, pre-market, after-hours and overnight futures sessions.",
            "No true bid/ask delta, reliable historical futures OI, options GEX or dealer positioning is connected.",
            "Model probabilities are not calibrated; a probability threshold is not a measured win probability.",
            "Transaction costs and slippage are sensitivity assumptions, not verified broker fills.",
            "NQ=F continuous futures series may have roll and vendor adjustment limitations.",
        ],
    }
    (OUT / "backtest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
