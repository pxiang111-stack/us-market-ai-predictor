"""QQQ-only out-of-sample trade-frequency sensitivity experiment.

Uses identical purged walk-forward forecasts, next-session-open execution,
risk budget and cost assumptions across predeclared gate profiles. It is a
research comparison, not a live signal or a claim that looser gates are better.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
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
    scored_start = bars.index[-1] - pd.DateOffset(years=3)
    eval_bars = bars.loc[bars.index >= scored_start].copy()
    rows, ledgers, curves = [], [], []

    # Matched-date buy-and-hold benchmark on the latest three-year score window.
    # as every strategy curve, so active profiles are not judged in isolation.
    benchmark_close = pd.to_numeric(close.reindex(eval_bars.index), errors="coerce").dropna()
    if len(benchmark_close) > 1:
        benchmark_curve = pd.DataFrame({
            "date": benchmark_close.index,
            "equity": 10_000.0 * benchmark_close / float(benchmark_close.iloc[0]),
        })
        benchmark_stats = performance_summary(pd.DataFrame(), benchmark_curve)
        for horizon in HORIZONS:
            rows.append({
                "strategy_id": f"QQQ_{horizon}d_buy_and_hold",
                "ticker": "QQQ", "profile": "buy_and_hold",
                "horizon_sessions": horizon, "cost_bps": 0.0,
                "slippage_bps_per_side": 0.0,
                "forecast_count": 0, "signal_count": 0,
                "long_signals": 0, "short_signals": 0,
                "flat_signals": 0, "probability_calibrated": False,
                "decision_cost_floor_bps": 0.0,
                "benchmark": "matched_date_buy_and_hold",
                **benchmark_stats,
            })
            curve_copy = benchmark_curve.copy()
            curve_copy["strategy_id"] = f"QQQ_{horizon}d_buy_and_hold"
            curve_copy["profile"] = "buy_and_hold"
            curve_copy["horizon_sessions"] = horizon
            curve_copy["cost_bps"] = 0.0
            curves.append(curve_copy)

    for horizon in HORIZONS:
        print(f"[INFO] QQQ {horizon}-session purged walk-forward forecasts...", flush=True)
        forecasts = generate_walkforward_forecasts(
            features, close, horizon, min_train_rows=500,
            retrain_every=63, max_train_rows=1260, start_date=scored_start,
        )
        forecasts = forecasts.loc[forecasts.index >= scored_start].copy()
        if forecasts.empty:
            rows.append({"horizon_sessions": horizon, "status": "insufficient_forecasts"})
            continue
        for profile_name, gates in PROFILES.items():
            plans = []
            for date, pred in forecasts.iterrows():
                if date not in eval_bars.index or pd.isna(atr.get(date)):
                    continue
                plan = build_trade_decision(
                    close=float(close.loc[date]), atr=float(atr.loc[date]),
                    probability_up=float(pred["probability_up"]),
                    expected_return=float(pred["expected_return"]),
                    features=features.loc[date].to_dict(),
                    allow_short=False, round_trip_cost=DECISION_COST_FLOOR,
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
                    eval_bars,
                    plan_frame[["decision", "stop_loss", "take_profit_1r", "take_profit_2r"]],
                    initial_equity=10_000.0, risk_budget_pct=0.005,
                    max_gross_exposure=1.0, round_trip_cost_bps=cost,
                    slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
                    max_holding_sessions=horizon, whole_shares_only=True,
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
        "scored_backtest_years": 3, "scored_start": str(scored_start.date()),
        "initial_equity_usd": 10000.0, "whole_shares_only": True, "shorts_enabled": False,
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
