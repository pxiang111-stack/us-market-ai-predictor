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
from sklearn.metrics import roc_auc_score, brier_score_loss
from qqq_drawdown_diagnostics_v1 import evaluate_drawdown_warnings

OUT = Path("output/walkforward_v5")
OUT.mkdir(parents=True, exist_ok=True)
TARGETS = {"SOXL": "SOXL", "QQQ": "QQQ", "SPY": "SPY"}
HORIZONS = (5, 10, 20)
COST_CASES_BPS = (10.0, 20.0, 30.0)
SLIPPAGE_BPS_PER_SIDE = 5.0
# Keep signal eligibility conservative across the full execution-cost sensitivity range.
def decision_cost_floor(cost_bps: float) -> float:
    """Per-scenario minimum expected edge: round-trip cost plus both-side slippage."""
    return (float(cost_bps) + 2 * SLIPPAGE_BPS_PER_SIDE) / 10000.0


# Retain the worst-case value for reporting/backward-compatible tests only.
DECISION_COST_FLOOR = decision_cost_floor(max(COST_CASES_BPS))


def fetch_bars(ticker: str) -> pd.DataFrame:
    raw = yf.download(ticker, period="10y", interval="1d", auto_adjust=True,
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


def build_benchmark_curve(close: pd.Series, dates, initial_equity: float = 10_000.0) -> pd.DataFrame:
    """Build a close-to-close benchmark on the strategy's exact evaluation dates."""
    if close is None or close.empty:
        return pd.DataFrame(columns=["date", "equity"])
    series = pd.to_numeric(close.copy(), errors="coerce")
    idx = pd.to_datetime(series.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    series.index = idx.normalize()
    series = series[~series.index.duplicated(keep="last")].sort_index()
    target_dates = pd.DatetimeIndex(pd.to_datetime(list(dates)))
    if getattr(target_dates, "tz", None) is not None:
        target_dates = target_dates.tz_localize(None)
    target_dates = target_dates.normalize()
    aligned = series.reindex(target_dates).ffill().dropna()
    if len(aligned) < 2 or float(aligned.iloc[0]) <= 0:
        return pd.DataFrame(columns=["date", "equity"])
    return pd.DataFrame({"date": aligned.index, "equity": initial_equity * aligned / float(aligned.iloc[0])})


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Safely combine per-symbol/horizon frames with non-identical columns."""
    nonempty = [frame for frame in frames if isinstance(frame, pd.DataFrame) and not frame.empty]
    return pd.concat(nonempty, ignore_index=True, sort=False) if nonempty else pd.DataFrame()


def forecast_quality_summary(
    forecasts: pd.DataFrame, close: pd.Series,
    next_session_open: pd.Series, horizon: int
) -> dict:
    """Evaluate forecasts on the same next-open-to-horizon-close return traded."""
    if forecasts is None or forecasts.empty:
        return {"forecast_eval_n": 0, "forecast_auc": None, "forecast_brier": None}
    idx = forecasts.index
    entry = next_session_open.reindex(close.index).shift(-1).reindex(idx)
    exit_price = close.shift(-horizon).reindex(idx)
    realized = exit_price / entry - 1.0
    p = pd.to_numeric(forecasts["probability_up"], errors="coerce")
    expected = pd.to_numeric(forecasts["expected_return"], errors="coerce")
    valid = pd.DataFrame({"realized": realized, "p": p, "expected": expected}).replace(
        [np.inf, -np.inf], np.nan
    ).dropna()
    if valid.empty:
        return {"forecast_eval_n": 0, "forecast_auc": None, "forecast_brier": None}
    y = (valid["realized"] > 0).astype(int)
    pred = (valid["p"] >= 0.5).astype(int)
    brier = float(brier_score_loss(y, valid["p"]))
    base_rate = float(y.mean())
    climatology_brier = float(brier_score_loss(y, np.full(len(y), base_rate)))
    brier_50 = float(brier_score_loss(y, np.full(len(y), 0.5)))
    train_rate = pd.to_numeric(forecasts.get("train_up_rate", pd.Series(index=forecasts.index, dtype=float)), errors="coerce").reindex(valid.index)
    train_rate = train_rate.clip(0.0, 1.0)
    train_rate_brier = float(((train_rate - y) ** 2).mean()) if train_rate.notna().all() else None
    train_rate_accuracy = float(((train_rate >= 0.5).astype(int) == y).mean() * 100.0) if train_rate.notna().all() else None
    # Expected calibration error: weighted absolute gap between mean forecast
    # probability and realized frequency in fixed, predeclared probability bins.
    calibration = pd.DataFrame({"p": valid["p"].clip(0.0, 1.0), "y": y})
    calibration["bin"] = pd.cut(
        calibration["p"], bins=np.linspace(0.0, 1.0, 11),
        include_lowest=True, labels=False,
    )
    ece = 0.0
    for _, group in calibration.groupby("bin", observed=False):
        if len(group):
            ece += len(group) / len(calibration) * abs(float(group["p"].mean()) - float(group["y"].mean()))
    error = valid["expected"] - valid["realized"]
    auc = float(roc_auc_score(y, valid["p"])) if y.nunique() == 2 else None
    return {
        "forecast_eval_n": int(len(valid)),
        "forecast_auc": auc,
        "forecast_brier": brier,
        "forecast_brier_skill_vs_50": float(1.0 - brier / brier_50) if brier_50 > 0 else None,
        "forecast_brier_vs_training_base_rate": train_rate_brier,
        "forecast_brier_skill_vs_training_base_rate": (
            float(1.0 - brier / train_rate_brier) if train_rate_brier is not None and train_rate_brier > 0 else None
        ),
        "forecast_training_base_rate_accuracy_pct": train_rate_accuracy,
        # This test-period climatology is a diagnostic reference, not a deployable forecast.
        "forecast_brier_skill_vs_test_climatology": (
            float(1.0 - brier / climatology_brier) if climatology_brier > 0 else None
        ),
        "forecast_ece_10": float(ece),
        "forecast_mean_probability_pct": float(valid["p"].mean() * 100.0),
        "forecast_observed_up_rate_pct": base_rate * 100.0,
        "forecast_calibration_gap_pct": float((valid["p"].mean() - base_rate) * 100.0),
        "forecast_directional_accuracy_50_pct": float((pred == y).mean() * 100.0),
        "forecast_base_rate_accuracy_pct": float(max(base_rate, 1.0 - base_rate) * 100.0),
        "forecast_up_rate_pct": base_rate * 100.0,
        "predicted_return_mean_pct": float(valid["expected"].mean() * 100.0),
        "realized_return_mean_pct": float(valid["realized"].mean() * 100.0),
        "predicted_return_mae_pct": float(error.abs().mean() * 100.0),
        "predicted_return_rmse_pct": float(np.sqrt((error ** 2).mean()) * 100.0),
        "predicted_vs_realized_spearman_ic": (
            float(valid["expected"].corr(valid["realized"], method="spearman"))
            if valid["expected"].nunique() > 1 and valid["realized"].nunique() > 1 else None
        ),
    }


def main():
    print("[INFO] Downloading cross-market close series...", flush=True)
    price_symbols = ["NQ=F", "ES=F", "SOXL", "QQQ", "SPY", "^NDX", "^GSPC", "^SOX", "^VIX", "HYG", "LQD", "SMH", "NVDA"]
    prices = download_prices(price_symbols, period="10y")
    all_ledgers, all_curves, all_forecasts, all_plans, summaries = [], [], [], [], []

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
            # Train on the longer available history, but score trades only on
            # the most recent three calendar years requested for this project.
            scored_start = bars.index[-1] - pd.DateOffset(years=3)
            eval_bars = bars.loc[bars.index >= scored_start].copy()
        except Exception as exc:
            summaries.append({"ticker": ticker, "status": "data_error", "error": str(exc)})
            continue

        # Buy-and-hold benchmarks use the same close history and starting equity.
        bh_close = close.loc[close.index >= scored_start].dropna()
        if len(bh_close) > 1:
            bh_curve = pd.DataFrame({"date": bh_close.index,
                                     "equity": 10_000.0 * bh_close / float(bh_close.iloc[0])})
            bh_stats = performance_summary(pd.DataFrame(), bh_curve)
            summaries.append({"strategy_id": f"{ticker}_buy_and_hold", "status": "ok",
                              "ticker": ticker, "horizon_sessions": None,
                              "round_trip_cost_bps": 0.0, "slippage_bps_per_side": 0.0,
                              "forecast_count": 0, "long_signals": 0,
                              "short_signals": 0, "flat_signals": 0,
                              "benchmark": "buy_and_hold_close_to_close", **bh_stats})
            bh_curve["ticker"], bh_curve["strategy_id"] = ticker, f"{ticker}_buy_and_hold"
            all_curves.append(bh_curve)

        for horizon in HORIZONS:
            print(f"[INFO] {ticker} {horizon}d walk-forward forecasts...", flush=True)
            try:
                forecasts = generate_walkforward_forecasts(
                    x, close, horizon,
                    entry_open=bars["Open"].reindex(x.index),
                    min_train_rows=500, retrain_every=63,
                    max_train_rows=1260, start_date=scored_start,
                )
                forecasts = forecasts.loc[forecasts.index >= scored_start].copy()
                if forecasts.empty:
                    summaries.append({"ticker": ticker, "horizon": horizon,
                                      "status": "insufficient_walkforward_history_in_scored_window",
                                      "scored_start": str(scored_start.date())})
                    continue
                strategy_modes = ("confluence", "trend_breakout", "reversal_absorption")
                plans_by_cost_mode = {}
                for cost_bps in COST_CASES_BPS:
                    plan_rows = []
                    scenario_cost_floor = decision_cost_floor(cost_bps)
                    for date, pred in forecasts.iterrows():
                        if date not in bars.index or date not in x.index:
                            continue
                        for strategy_mode in strategy_modes:
                            row = build_trade_decision(
                                close=float(close.loc[date]), atr=float(atr.loc[date]),
                                probability_up=float(pred["probability_up"]),
                                expected_return=float(pred["expected_return"]),
                                features=x.loc[date].to_dict(),
                                allow_short=False,  # $10k cash-account comparison: no shorting or leverage
                                round_trip_cost=scenario_cost_floor,
                                strategy_mode=strategy_mode,
                            )
                            row.update({"as_of": date, "ticker": ticker, "asset": asset,
                                        "horizon_sessions": horizon,
                                        "round_trip_cost_bps": cost_bps,
                                        "decision_cost_floor_bps": scenario_cost_floor * 10000.0,
                                        "train_end": pred["train_end"],
                                        "n_train": int(pred["n_train"]),
                                        "probability_calibrated": False,
                                        "market_regime": x.loc[date].get("ms_market_regime", None)})
                            plan_rows.append(row)
                    plans_all = pd.DataFrame(plan_rows)
                    if not plans_all.empty:
                        plans_by_cost_mode[cost_bps] = {
                            mode: plans_all[plans_all["strategy_mode"] == mode].set_index("as_of").sort_index()
                            for mode in strategy_modes
                        }
                if not plans_by_cost_mode:
                    summaries.append({"ticker": ticker, "horizon": horizon, "status": "no_plans"})
                    continue
                # Forecast report carries the least-cost (10bp) confluence decision;
                # each backtest below uses its own matching cost hurdle.
                baseline_plans = plans_by_cost_mode[COST_CASES_BPS[0]]["confluence"]
                forecasts = forecasts.join(
                    baseline_plans[["decision", "reason"]].rename(columns={"reason": "decision_reason"}),
                    how="left",
                )
                forecasts["ticker"] = ticker
                forecasts["horizon_sessions"] = horizon
                # Realized labels are written only for post-hoc evaluation, never used by the signal.
                entry_open = bars["Open"].reindex(close.index).shift(-1)
                forecasts["realized_forward_return_eval_only"] = (
                    close.shift(-horizon).reindex(forecasts.index)
                    / entry_open.reindex(forecasts.index) - 1.0
                )
                forecasts["realized_up_eval_only"] = (
                    forecasts["realized_forward_return_eval_only"] > 0
                ).where(forecasts["realized_forward_return_eval_only"].notna())
                all_forecasts.append(forecasts.reset_index())
                quality = forecast_quality_summary(forecasts, close, bars["Open"], horizon)
                summaries.append({
                    "strategy_id": f"{ticker}_{horizon}d_forecast_quality",
                    "status": "ok", "ticker": ticker, "horizon_sessions": horizon,
                    "benchmark": "out_of_sample_forecast_diagnostics_only",
                    **quality,
                })

                for strategy_mode in strategy_modes:
                    for cost_bps in COST_CASES_BPS:
                        plans = plans_by_cost_mode[cost_bps][strategy_mode].copy()
                        plans["decision"] = plans["decision"].fillna("FLAT")
                        all_plans.append(plans.reset_index())
                        ledger, curve = simulate_trades(
                            eval_bars, plans[["decision", "stop_loss", "take_profit_1r", "take_profit_2r"]],
                            initial_equity=10_000.0, risk_budget_pct=0.005,
                            max_gross_exposure=1.0, round_trip_cost_bps=cost_bps,
                            slippage_bps_per_side=SLIPPAGE_BPS_PER_SIDE, max_holding_sessions=horizon,
                            whole_shares_only=True,
                        )
                        stats = performance_summary(ledger, curve)
                        tag = f"{ticker}_{horizon}d_{strategy_mode}_cost{int(cost_bps)}bps"
                        if not ledger.empty:
                            ledger = ledger.copy()
                            ledger["ticker"], ledger["horizon_sessions"], ledger["round_trip_cost_bps"] = ticker, horizon, cost_bps
                            ledger["strategy_mode"] = strategy_mode
                            regime_by_date = plans["market_regime"].to_dict()
                            ledger["market_regime"] = ledger["signal_date"].map(regime_by_date)
                            ledger["strategy_id"] = tag
                            all_ledgers.append(ledger)
                        if not curve.empty:
                            curve = curve.copy()
                            curve["ticker"], curve["horizon_sessions"], curve["round_trip_cost_bps"] = ticker, horizon, cost_bps
                            curve["strategy_mode"] = strategy_mode
                            curve["strategy_id"] = tag
                            all_curves.append(curve)
                        summaries.append({
                            "strategy_id": tag, "strategy_mode": strategy_mode,
                            "status": "ok", "ticker": ticker, "horizon_sessions": horizon,
                            "round_trip_cost_bps": cost_bps, "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
                            "decision_cost_floor_bps": decision_cost_floor(cost_bps) * 10000.0,
                            "forecast_count": int(len(forecasts)),
                            "long_signals": int((plans["decision"] == "LONG").sum()),
                            "short_signals": int((plans["decision"] == "SHORT").sum()),
                            "flat_signals": int((plans["decision"] == "FLAT").sum()),
                            "probability_long_gate_count": int(plans["gate_probability_long"].sum()),
                            "probability_short_gate_count": int(plans["gate_probability_short"].sum()),
                            "edge_long_gate_count": int(plans["gate_edge_long"].sum()),
                            "edge_short_gate_count": int(plans["gate_edge_short"].sum()),
                            "bull_trend_gate_count": int(plans["gate_bull_trend"].sum()),
                            "bear_trend_gate_count": int(plans["gate_bear_trend"].sum()),
                            "long_structure_gate_count": int(plans["gate_long_structure"].sum()),
                            "short_structure_gate_count": int(plans["gate_short_structure"].sum()),
                            "probability_calibrated": False,
                            "feed": "Yahoo Finance daily OHLCV",
                            **stats,
                        })
                        if not curve.empty:
                            for benchmark_symbol in ("SPY", "QQQ"):
                                if benchmark_symbol not in prices:
                                    continue
                                benchmark_curve = build_benchmark_curve(
                                    prices[benchmark_symbol], curve["date"], initial_equity=10_000.0
                                )
                                if benchmark_curve.empty:
                                    continue
                                benchmark_tag = f"{tag}_benchmark_{benchmark_symbol}"
                                benchmark_curve["ticker"] = benchmark_symbol
                                benchmark_curve["strategy_id"] = benchmark_tag
                                benchmark_curve["benchmark_for"] = tag
                                all_curves.append(benchmark_curve)
                                benchmark_stats = performance_summary(pd.DataFrame(), benchmark_curve)
                                summaries.append({
                                    "strategy_id": benchmark_tag, "status": "ok",
                                    "ticker": ticker, "benchmark_ticker": benchmark_symbol,
                                    "horizon_sessions": horizon,
                                    "round_trip_cost_bps": 0.0, "slippage_bps_per_side": 0.0,
                                    "benchmark": "buy_and_hold_close_to_close_same_dates",
                                    **benchmark_stats,
                                })
            except Exception as exc:
                summaries.append({"ticker": ticker, "horizon": horizon,
                                  "status": "backtest_error", "error": str(exc)})
            time.sleep(0.2)

    forecast_frame = concat_frames(all_forecasts)
    forecast_frame.to_csv(OUT / "oos_forecasts.csv", index=False)
    # QQQ-specific early-warning audit: score bearish forecasts before future
    # 3/5/8% downside excursions, separately from confirmed SHORT trade entries.
    qqq_forecasts = forecast_frame[forecast_frame.get("ticker", pd.Series(dtype=str)) == "QQQ"].copy()
    if not qqq_forecasts.empty:
        qqq_bars = fetch_bars("QQQ")
        drawdown_detail, drawdown_summary = evaluate_drawdown_warnings(
            qqq_forecasts, qqq_bars, thresholds=(0.03, 0.05, 0.08)
        )
        drawdown_detail.to_csv(OUT / "qqq_drawdown_warning_detail.csv", index=False)
        drawdown_summary.to_csv(OUT / "qqq_drawdown_warning_summary.csv", index=False)
    else:
        pd.DataFrame().to_csv(OUT / "qqq_drawdown_warning_detail.csv", index=False)
        pd.DataFrame().to_csv(OUT / "qqq_drawdown_warning_summary.csv", index=False)
    concat_frames(all_plans).to_csv(OUT / "trade_decisions.csv", index=False)
    concat_frames(all_ledgers).to_csv(OUT / "trade_ledger.csv", index=False)
    concat_frames(all_curves).to_csv(OUT / "daily_equity_curves.csv", index=False)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "performance_summary.csv", index=False)

    # Direction-specific OOS trade diagnostics: a profitable combined result can
    # hide a losing short book (or vice versa), so report each side separately.
    ledger_all = concat_frames(all_ledgers)
    side_rows = []
    if not ledger_all.empty:
        for keys, group in ledger_all.groupby(
            ["ticker", "horizon_sessions", "strategy_mode", "round_trip_cost_bps", "side"],
            dropna=False,
        ):
            ticker_name, horizon_value, mode, cost_value, side = keys
            pnl = pd.to_numeric(group["net_pnl"], errors="coerce").dropna()
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]
            risk_cash = pd.to_numeric(group.get("equity_before"), errors="coerce") * 0.005
            r_multiple = (pnl / risk_cash.reindex(pnl.index)).replace([np.inf, -np.inf], np.nan).dropna()
            side_rows.append({
                "ticker": ticker_name,
                "horizon_sessions": int(horizon_value),
                "strategy_mode": mode,
                "round_trip_cost_bps": float(cost_value),
                "side": side,
                "trade_count": int(len(pnl)),
                "win_rate_pct": float((pnl > 0).mean() * 100.0) if len(pnl) else None,
                "profit_factor": float(wins.sum() / abs(losses.sum())) if len(losses) and abs(losses.sum()) > 0 else (float("inf") if len(wins) else None),
                "expectancy_net_pnl": float(pnl.mean()) if len(pnl) else None,
                "median_net_pnl": float(pnl.median()) if len(pnl) else None,
                "mean_net_r_multiple": float(r_multiple.mean()) if len(r_multiple) else None,
                "max_single_trade_loss": float(pnl.min()) if len(pnl) else None,
            })
    pd.DataFrame(side_rows).to_csv(OUT / "side_performance_summary.csv", index=False)

    # Trade-level conditional diagnostics by regime. Regime labels are attached
    # from the signal date only; this is descriptive, not an optimized regime filter.
    regime_summary = pd.DataFrame()
    if not ledger_all.empty and "market_regime" in ledger_all:
        valid_regime = ledger_all.dropna(subset=["market_regime"]).copy()
        if not valid_regime.empty:
            regime_summary = valid_regime.groupby(
                ["ticker", "horizon_sessions", "round_trip_cost_bps",
                 "strategy_mode", "market_regime"], dropna=False
            ).agg(
                trade_count=("net_pnl", "size"),
                wins=("net_pnl", lambda z: int((z > 0).sum())),
                net_pnl_sum=("net_pnl", "sum"),
                expectancy_per_trade=("net_pnl", "mean"),
                median_net_pnl=("net_pnl", "median"),
            ).reset_index()
            regime_summary["win_rate_pct"] = (
                regime_summary["wins"] / regime_summary["trade_count"] * 100.0
            )
            pf = valid_regime.groupby(
                ["ticker", "horizon_sessions", "round_trip_cost_bps",
                 "strategy_mode", "market_regime"]
            )["net_pnl"].apply(
                lambda z: (z[z > 0].sum() / abs(z[z < 0].sum()))
                if (z < 0).any() else (float("inf") if (z > 0).any() else 0.0)
            ).rename("profit_factor").reset_index()
            regime_summary = regime_summary.merge(
                pf, on=["ticker", "horizon_sessions", "round_trip_cost_bps",
                        "strategy_mode", "market_regime"], how="left"
            )
    regime_summary.to_csv(OUT / "regime_trade_summary.csv", index=False)

    payload = {
        "status": "completed",
        "targets": list(TARGETS),
        "market_context_only": ["NQ=F"],
        "scored_backtest_years": 3,
        "scored_window_start_rule": "latest available bar date minus 3 calendar years",
        "horizons": list(HORIZONS),
        "cost_cases_round_trip_bps": list(COST_CASES_BPS),
        "slippage_bps_per_side": SLIPPAGE_BPS_PER_SIDE,
        "decision_cost_floor_bps": DECISION_COST_FLOOR * 10000.0,
        "initial_equity_usd": 10000.0,
        "risk_budget_pct": 0.5,
        "max_gross_exposure": 1.0,
        "position_units_are_not_exchange_contracts": True,
        "fractional_units_used": False,
        "cash_account_whole_share_constraint": True,
        "position_sizing": "whole-share floor; entry costs reserved; long-only; max gross exposure 100%; no borrowing",
        "shorts_enabled": False,
        "probabilities_calibrated": False,
        "forecast_target": "close(t+horizon_sessions) / open(t+1) - 1; signal after close t, enter next session open, exit at the horizon close.",
        "data_scope": "daily OHLCV only; not intraday, overnight-session-separated or order-book data",
        "strategy_modes": ["confluence", "trend_breakout", "reversal_absorption"],
        "outputs": ["oos_forecasts.csv", "qqq_drawdown_warning_detail.csv", "qqq_drawdown_warning_summary.csv", "trade_decisions.csv", "trade_ledger.csv", "daily_equity_curves.csv", "performance_summary.csv", "side_performance_summary.csv", "regime_trade_summary.csv"],
        "summary_rows": len(summary_df),
        "successful_rows": int((summary_df.get("status", pd.Series(index=summary_df.index, dtype=object)) == "ok").sum()),
        "limitations": [
            "This is an experimental historical simulation, not a live trading system.",
            "Short selling is disabled for the $10,000 cash-account comparison.",
            "Initial capital is USD 10,000; the trade simulator now floors position size to whole shares and reserves estimated entry costs. One-share positions exit fully at 1R because they cannot be split.",
            "OHLCV bars use adjusted prices in this version; cash dividends are not separately ledgered.",
            "Yahoo daily bars do not separate regular, pre-market, after-hours and overnight futures sessions.",
            "No true bid/ask delta, reliable historical futures OI, options GEX or dealer positioning is connected.",
            "Model probabilities are not calibrated; a probability threshold is not a measured win probability.",
            "Transaction costs and slippage are sensitivity assumptions, not verified broker fills.",
            "NQ=F continuous futures series may have roll and vendor adjustment limitations.",
            "Position sizing uses normalized price units; NQ=F units are not integer CME contracts and do not model contract multipliers or futures margin.",
        ],
    }
    (OUT / "backtest_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
