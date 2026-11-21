"""QQQ-only OOS frequency and cost-aware decision-gate experiment.

Compares predeclared signal profiles under both a fixed 40bp decision edge
floor and a cost-aware edge floor equal to modeled round-trip costs including
both sides of slippage. Execution assumptions and forecast dates are matched.
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

OUT = Path("output/frequency_sweep_v2")
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (5, 10, 20)
COSTS_BPS = (10.0, 20.0, 30.0)
SLIPPAGE_BPS_PER_SIDE = 5.0
FIXED_DECISION_COST_FLOOR_BPS = 40.0
PROFILES = {
    "baseline": {"p_long": .62, "p_short": .38, "rvol": 1.20, "clv_long": .65, "clv_short": .35, "flow": .05},
    "active": {"p_long": .58, "p_short": .42, "rvol": 1.05, "clv_long": .60, "clv_short": .40, "flow": .03},
    "aggressive": {"p_long": .54, "p_short": .46, "rvol": 1.00, "clv_long": .55, "clv_short": .45, "flow": .02},
}


def make_plans(forecasts, eval_bars, close, atr, features, gates, edge_floor_bps, horizon, profile_name):
    plans = []
    for date, pred in forecasts.iterrows():
        if date not in eval_bars.index or pd.isna(atr.get(date)):
            continue
        plan = build_trade_decision(
            close=float(close.loc[date]), atr=float(atr.loc[date]),
            probability_up=float(pred["probability_up"]),
            expected_return=float(pred["expected_return"]),
            features=features.loc[date].to_dict(), allow_short=False,
            round_trip_cost=edge_floor_bps / 10000.0,
            strategy_mode="confluence",
            probability_threshold_long=gates["p_long"],
            probability_threshold_short=gates["p_short"],
            structure_relative_volume_min=gates["rvol"],
            structure_close_location_long=gates["clv_long"],
            structure_close_location_short=gates["clv_short"],
            flow_pressure_threshold=gates["flow"],
        )
        plan.update({"as_of": date, "profile": profile_name, "horizon_sessions": horizon})
        plans.append(plan)
    if not plans:
        return pd.DataFrame(columns=["decision", "stop_loss", "take_profit_1r", "take_profit_2r"])
    frame = pd.DataFrame(plans).set_index("as_of").sort_index()
    frame["decision"] = frame["decision"].fillna("FLAT")
    return frame


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

    benchmark_close = pd.to_numeric(close.reindex(eval_bars.index), errors="coerce").dropna()
    if len(benchmark_close) > 1:
        benchmark_curve = pd.DataFrame({
            "date": benchmark_close.index,
            "equity": 10_000.0 * benchmark_close / float(benchmark_close.iloc[0]),
        })
        benchmark_stats = performance_summary(pd.DataFrame(), benchmark_curve)
        for horizon in HORIZONS:
            tag = f"QQQ_{horizon}d_buy_and_hold"
            rows.append({
                "strategy_id": tag, "ticker": "QQQ", "profile": "buy_and_hold",
                "gate_mode": "benchmark", "horizon_sessions": horizon,
                "cost_bps": 0.0, "slippage_bps_per_side": 0.0,
                "forecast_count": 0, "signal_count": 0, "long_signals": 0,
                "short_signals": 0, "flat_signals": 0, "probability_calibrated": False,
                "decision_cost_floor_bps": 0.0, "benchmark": "matched_date_buy_and_hold",
                **benchmark_stats,
            })
            c = benchmark_curve.copy()
            c["strategy_id"], c["profile"], c["gate_mode"] = tag, "buy_and_hold", "benchmark"
            c["horizon_sessions"], c["cost_bps"] = horizon, 0.0
            curves.append(c)

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
            for cost in COSTS_BPS:
                modeled_total_cost_bps = cost + 2 * SLIPPAGE_BPS_PER_SIDE
                gate_modes = {
                    "fixed_40bps": FIXED_DECISION_COST_FLOOR_BPS,
                    "cost_aware": modeled_total_cost_bps,
                }
                for gate_mode, edge_floor_bps in gate_modes.items():
                    plan_frame = make_plans(
                        forecasts, eval_bars, close, atr, features, gates,
                        edge_floor_bps, horizon, profile_name,
                    )
                    signal_count = int((plan_frame["decision"] != "FLAT").sum())
                    long_count = int((plan_frame["decision"] == "LONG").sum())
                    short_count = int((plan_frame["decision"] == "SHORT").sum())
                    tag = f"QQQ_{horizon}d_{profile_name}_{gate_mode}_cost{int(cost)}bps"
                    ledger, curve, diag = simulate_trades(
                        eval_bars,
                        plan_frame[["decision", "stop_loss", "take_profit_1r", "take_profit_2r"]],
                        initial_equity=10_000.0, risk_budget_pct=.005,
                        max_gross_exposure=1.0, round_trip_cost_bps=cost,
                        slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE,
                        max_holding_sessions=horizon, whole_shares_only=True, return_diagnostics=True,
                    )
                    stats = performance_summary(ledger, curve)
                    rows.append({
                        "strategy_id": tag, "ticker": "QQQ", "profile": profile_name,
                        "gate_mode": gate_mode, "horizon_sessions": horizon,
                        "cost_bps": cost, "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
                        "forecast_count": len(forecasts), "signal_count": signal_count,
                        "long_signals": long_count, "short_signals": short_count,
                        "flat_signals": len(plan_frame) - signal_count,
                        "decision_cost_floor_bps": edge_floor_bps,
                        "probability_calibrated": False,
                        "signals_received_by_executor": diag["signals_received"],
                        "executed_trades": diag["executed_trades"],
                        "skipped_total": diag["skipped_total"],
                        "skipped_existing_position": diag["skipped_existing_position"],
                        "skipped_no_next_session": diag["skipped_no_next_session"],
                        "skipped_invalid_entry_or_stop": diag["skipped_invalid_entry_or_stop"],
                        "skipped_invalid_risk_geometry": diag["skipped_invalid_risk_geometry"],
                        "skipped_zero_units": diag["skipped_zero_units"],
                        **stats,
                    })
                    if not ledger.empty:
                        ledger = ledger.copy()
                        ledger["strategy_id"], ledger["profile"] = tag, profile_name
                        ledger["gate_mode"], ledger["horizon_sessions"] = gate_mode, horizon
                        ledger["cost_bps"] = cost
                        ledgers.append(ledger)
                    if not curve.empty:
                        curve = curve.copy()
                        curve["strategy_id"], curve["profile"] = tag, profile_name
                        curve["gate_mode"], curve["horizon_sessions"] = gate_mode, horizon
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
        "decision_gate_modes": {
            "fixed_40bps": "constant 40bp expected-edge floor",
            "cost_aware": "round-trip cost plus two sides of slippage",
        },
        "risk_budget_pct": .5, "max_gross_exposure": 1.0,
        "data_scope": "Yahoo Finance daily OHLCV; no intraday or order-book data",
        "probabilities_calibrated": False,
        "limitations": [
            "Profiles are predeclared sensitivity tests, not optimized winners.",
            "One position at a time; skipped signals are explicitly counted.",
            "Cost-aware edge floor is a sensitivity assumption, not proof of executable edge.",
            "Daily OHLC cannot establish intraday fill order; stop-first execution remains conservative.",
            "Buy-and-hold is a zero-cost reference, not a cost-matched strategy benchmark.",
        ],
    }
    (OUT / "frequency_sweep_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
