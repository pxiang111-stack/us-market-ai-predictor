"""QQQ-only out-of-sample trade-frequency sensitivity experiment.

Uses identical purged walk-forward forecasts, next-session-open execution,
risk budget and cost assumptions across predeclared gate profiles. It is a
research comparison, not a live signal or a claim that looser gates are better.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yfinance as yf

from market_risk_v4 import download_prices, make_features, merge_market_structure_features
from trade_decision_v5 import build_trade_decision
from walkforward_backtest_v5 import atr_series, fetch_bars
from walkforward_forecast_v5 import generate_walkforward_forecasts
from walkforward_execution_v5 import simulate_trades, performance_summary

OUT = Path("output/frequency_sweep_v1")
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (5, 10, 20)
COSTS_BPS = (10.0, 20.0, 30.0)
SLIPPAGE_BPS_PER_SIDE = 5.0
DECISION_COST_FLOOR = 0.004
PROFILES = {
    "baseline": {
        "p_long": 0.62, "p_short": 0.38, "rvol": 1.20,
        "clv_long": 0.65, "clv_short": 0.35, "flow": 0.05,
    },
    "active": {
        "p_long": 0.58, "p_short": 0.42, "rvol": 1.05,
        "clv_long": 0.60, "clv_short": 0.40, "flow": 0.03,
    },
    "aggressive": {
        "p_long": 0.54, "p_short": 0.46, "rvol": 1.00,
        "clv_long": 0.55, "clv_short": 0.45, "flow": 0.02,
    },
}


def main() -> None:
    print("[INFO] Downloading QQQ cross-market features...", flush=True)
    symbols = ["QQQ", "SPY", "NQ=F", "ES=F", "^NDX", "^GSPC", "^SOX", "^VIX",
               "HYG", "LQD", "SMH", "NVDA"]
    prices = download_prices(symbols, period="10y")
    if "QQQ" not in prices:
        raise RuntimeError("QQQ close series unavailable")
    bars = fetch_bars("QQQ")
    features = make_features(prices, "QQQ")
    features, _ = merge_market_structure_features(features, bars, "QQQ")
    close = prices["QQQ"].reindex(features.index)
    atr = atr_series(bars).reindex(features.index)
    rows, ledgers, curves = [], [], []

    for horizon in HORIZONS:
        print(f"[INFO] QQQ {horizon}-session purged walk-forward forecasts...", flush=True)
        forecasts = generate_walkforward_forecasts(
            features, close, horizon, min_train_rows=500,
            retrain_every=63, max_train_rows=1260,
        )
        if forecasts.empty:
            rows.append({"horizon_sessions": horizon, "status": "insufficient_forecasts"})
            continue
        for profile_name, gates in PROFILES.items():
            plans = []
            for date, pred in forecasts.iterrows():
                if date not in bars.index or pd.isna(atr.get(date)):
                    continue
                plan = build_trade_decision(
                    close=float(close.loc[date]), atr=float(atr.loc[date]),
                    probability_up=float(pred["probability_up"]),
                    expected_return=float(pred["expected_return"]),
                    features=features.loc[date].to_dict(),
                    allow_short=True, round_trip_cost=DECISION_COST_FLOOR,
                    strategy_mode="confluence",
                    probability_threshold_long=gates["p_long"],
                    probability_threshold_short=gates["p_short"],
                    structure_relative_volume_min=gates["rvol"],
                    structure_close_location_long=gates["clv_long"],
                    structure_close_location_short=gates["clv_short"],
                    flow_pressure_threshold=gates["flow"],
                )
                plan.update({"as_of": date, "profile": profile_name,
                             "horizon_sessions": horizon})
                plans.append(plan)
            plan_frame = pd.DataFrame(plans).set_index("as_of").sort_index()
            plan_frame["decision"] = plan_frame["decision"].fillna("FLAT")
            signal_count = int((plan_frame["decision"] != "FLAT").sum())
            long_count = int((plan_frame["decision"] == "LONG").sum())
            short_count = int((plan_frame["decision"] == "SHORT").sum())
            for cost in COSTS_BPS:
                ledger, curve = simulate_trades(
                    bars,
                    plan_frame[["decision", "stop_loss", "take_profit_1r", "take_profit_2r"]],
                    initial_equity=100_000.0, risk_budget_pct=0.005,
                    max_gross_exposure=1.0, round_trip_cost_bps=cost,
                    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
                    max_holding_sessions=horizon,
                )
                stats = performance_summary(ledger, curve)
                tag = f"QQQ_{horizon}d_{profile_name}_cost{int(cost)}bps"
                rows.append({
                    "strategy_id": tag, "ticker": "QQQ", "profile": profile_name,
                    "horizon_sessions": horizon, "cost_bps": cost,
                    "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
                    "forecast_count": len(forecasts), "signal_count": signal_count,
                    "long_signals": long_count, "short_signals": short_count,
                    "flat_signals": len(plan_frame) - signal_count,
                    "probability_calibrated": False,
                    "decision_cost_floor_bps": DECISION_COST_FLOOR * 10000,
                    **stats,
                })
                if not ledger.empty:
                    ledger = ledger.copy()
                    ledger["strategy_id"] = tag
                    ledger["profile"] = profile_name
                    ledger["horizon_sessions"] = horizon
                    ledger["cost_bps"] = cost
                    ledgers.append(ledger)
                if not curve.empty:
                    curve = curve.copy()
                    curve["strategy_id"] = tag
                    curve["profile"] = profile_name
                    curve["horizon_sessions"] = horizon
                    curve["cost_bps"] = cost
                    curves.append(curve)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "frequency_sweep_summary.csv", index=False)
    pd.concat(ledgers, ignore_index=True, sort=False).to_csv(
        OUT / "frequency_sweep_trades.csv", index=False
    ) if ledgers else pd.DataFrame().to_csv(OUT / "frequency_sweep_trades.csv", index=False)
    pd.concat(curves, ignore_index=True, sort=False).to_csv(
        OUT / "frequency_sweep_equity.csv", index=False
    ) if curves else pd.DataFrame().to_csv(OUT / "frequency_sweep_equity.csv", index=False)
    payload = {
        "status": "completed", "ticker": "QQQ", "horizons": list(HORIZONS),
        "profiles": PROFILES, "cost_cases_round_trip_bps": list(COSTS_BPS),
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "risk_budget_pct": 0.5, "max_gross_exposure": 1.0,
        "data_scope": "Yahoo Finance daily OHLCV; no intraday or order-book data",
        "probabilities_calibrated": False,
        "notes": [
            "Profiles are predeclared sensitivity tests, not optimized winners.",
            "All profiles share identical walk-forward forecasts and execution rules.",
            "Higher signal count is not evidence of higher profitability.",
            "Results require adequate trade counts and stable cost-adjusted OOS performance.",
        ],
        "outputs": ["frequency_sweep_summary.csv", "frequency_sweep_trades.csv",
                    "frequency_sweep_equity.csv"],
    }
    (OUT / "frequency_sweep_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
