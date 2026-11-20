"""QQQ matched-window downside-risk overlay experiment.

A risk score observed at close t controls exposure only for the next session's
close-to-close return. All overlays and buy-and-hold use identical dates,
initial equity, and return observations. Thresholds are predeclared sensitivity
cases, not optimized winners. Probabilities are explicitly uncalibrated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from market_risk_v4 import (
    download_market_structure_bars,
    download_prices,
    make_features,
    merge_market_structure_features,
)
from qqq_drawdown_risk_v1 import generate_drawdown_forecasts, evaluate_drawdown_forecasts

OUT = Path("output/qqq_matched_risk_overlay_v1")
OUT.mkdir(parents=True, exist_ok=True)

HORIZON = 5
DRAWDOWN_THRESHOLD = 0.03
ALERT_THRESHOLDS = (0.20, 0.30, 0.50)
REDUCED_EXPOSURE = 0.50
INITIAL_EQUITY = 100_000.0
REBALANCE_COST_BPS = 5.0


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    return float((equity / peak - 1.0).min())


def summarize_returns(returns: pd.Series, equity: pd.Series, exposure: pd.Series,
                      turnover: pd.Series) -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    years = len(r) / 252.0
    total = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    cagr = float((equity.iloc[-1] / INITIAL_EQUITY) ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else None
    vol = float(r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 else None
    sharpe = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if len(r) > 1 and r.std(ddof=1) > 0 else None
    return {
        "observations": int(len(r)),
        "start_date": str(r.index[0].date()),
        "end_date": str(r.index[-1].date()),
        "total_return_pct": total * 100,
        "cagr_pct": cagr * 100 if cagr is not None else None,
        "max_drawdown_pct": max_drawdown(equity) * 100,
        "annualized_volatility_pct": vol * 100 if vol is not None else None,
        "sharpe_rf0": sharpe,
        "average_exposure_pct": float(exposure.mean() * 100),
        "total_turnover_exposure_units": float(turnover.sum()),
        "rebalance_cost_bps": REBALANCE_COST_BPS,
    }


def main() -> None:
    print("[INFO] Downloading QQQ and cross-market daily data...", flush=True)
    symbols = ["QQQ", "SPY", "NQ=F", "ES=F", "^NDX", "^GSPC", "^SOX", "^VIX",
               "HYG", "LQD", "TLT", "IEF", "UUP", "SMH", "SOXX", "NVDA", "AMD", "AVGO"]
    prices = download_prices(symbols, period="10y")
    if "QQQ" not in prices:
        raise RuntimeError("QQQ close series unavailable")
    bars = download_market_structure_bars("QQQ", period="10y")
    features = make_features(prices, "QQQ")
    features, _ = merge_market_structure_features(features, bars, "QQQ")

    forecasts = generate_drawdown_forecasts(
        features, bars, HORIZON, DRAWDOWN_THRESHOLD,
        min_train_rows=500, retrain_every=63, max_train_rows=1260,
    )
    if forecasts.empty:
        raise RuntimeError("No QQQ downside forecasts generated")

    risk_details, risk_metrics = evaluate_drawdown_forecasts(forecasts, bars)
    close = pd.to_numeric(bars["Close"], errors="coerce")
    next_session_return = close.shift(-1).div(close).sub(1.0)
    scores = forecasts.set_index("as_of")["drawdown_risk_score"]
    frame = pd.concat([scores.rename("risk_score"), next_session_return.rename("next_session_return")], axis=1)
    frame = frame.dropna().sort_index()
    if len(frame) < 100:
        raise RuntimeError(f"Insufficient matched observations: {len(frame)}")

    # The forecast made after close t is applied only to the t -> t+1 return.
    benchmark_returns = frame["next_session_return"].astype(float)
    benchmark_equity = INITIAL_EQUITY * (1.0 + benchmark_returns).cumprod()
    rows, curves = [], []
    benchmark_exposure = pd.Series(1.0, index=frame.index)
    benchmark_turnover = pd.Series(0.0, index=frame.index)
    rows.append({
        "strategy": "QQQ_buy_and_hold_matched",
        "alert_threshold": None,
        "reduced_exposure": 1.0,
        "probabilities_calibrated": False,
        **summarize_returns(benchmark_returns, benchmark_equity, benchmark_exposure, benchmark_turnover),
    })
    curves.append(pd.DataFrame({
        "date": frame.index, "strategy": "QQQ_buy_and_hold_matched",
        "risk_score": frame["risk_score"].values, "exposure": 1.0,
        "gross_return": benchmark_returns.values, "net_return": benchmark_returns.values,
        "equity": benchmark_equity.values,
    }))

    for threshold in ALERT_THRESHOLDS:
        exposure = pd.Series(
            np.where(frame["risk_score"] >= threshold, REDUCED_EXPOSURE, 1.0),
            index=frame.index, dtype=float,
        )
        turnover = exposure.diff().abs().fillna(0.0)
        cost = turnover * (REBALANCE_COST_BPS / 10_000.0)
        net_returns = (exposure * benchmark_returns - cost).astype(float)
        equity = INITIAL_EQUITY * (1.0 + net_returns).cumprod()
        name = f"risk_overlay_alert_{threshold:.2f}_to_50pct"
        rows.append({
            "strategy": name,
            "alert_threshold": threshold,
            "reduced_exposure": REDUCED_EXPOSURE,
            "probabilities_calibrated": False,
            **summarize_returns(net_returns, equity, exposure, turnover),
        })
        curves.append(pd.DataFrame({
            "date": frame.index, "strategy": name,
            "risk_score": frame["risk_score"].values, "exposure": exposure.values,
            "gross_return": (exposure * benchmark_returns).values,
            "net_return": net_returns.values, "equity": equity.values,
            "turnover": turnover.values, "cost_return": cost.values,
        }))

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "matched_overlay_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "matched_overlay_equity.csv", index=False)
    risk_details.to_csv(OUT / "downside_forecasts.csv", index=False)
    risk_metrics.to_csv(OUT / "downside_warning_metrics.csv", index=False)
    payload = {
        "status": "completed",
        "ticker": "QQQ",
        "risk_model": "rolling logistic regression",
        "forecast_horizon_sessions": HORIZON,
        "event": f"future intrahorizon low at least {DRAWDOWN_THRESHOLD:.0%} below signal-date close",
        "thresholds_predeclared": list(ALERT_THRESHOLDS),
        "alert_exposure": REDUCED_EXPOSURE,
        "rebalance_cost_bps_per_one_unit_exposure_change": REBALANCE_COST_BPS,
        "alignment": "risk score at close t controls exposure for next session close-to-close return",
        "benchmark": "QQQ buy-and-hold on exact same return dates and initial equity",
        "probabilities_calibrated": False,
        "limitations": [
            "Overlapping 5-session event labels are not independent.",
            "Risk scores are not calibrated probabilities.",
            "Thresholds are sensitivity cases and must not be selected on the final test period.",
            "This is a daily-bar research overlay, not a live trading signal.",
            "The experiment tests drawdown reduction versus opportunity cost; it does not establish alpha.",
        ],
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(risk_metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
