"""Walk-forward audit of incremental OHLCV structure features.

This is a descriptive research diagnostic, not a strategy backtest. It compares
a price/trend baseline with the same model plus daily-volume/candle proxies.
Labels are purged so every training forward-return horizon has matured before
the forecast date. Overlapping test labels remain dependent and are not treated
as independent statistical evidence.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE_FEATURES = [
    "return_1bar", "trend_above_sma20", "sma20_above_sma50",
    "sma50_above_sma200", "atr_pct_14", "range_position_20",
]
STRUCTURE_FEATURES = [
    "relative_volume", "volume_zscore", "close_location", "body_fraction",
    "signed_volume_proxy_ratio", "close_location_volume_pressure",
    "effort_result_absorption_proxy", "volume_weighted_close_location_20",
    "range_compression_14_50", "spring_reclaim_20", "upthrust_reject_20",
    "breakout_up_volume_confirmed", "breakout_down_volume_confirmed",
    "failed_breakout_up_20", "failed_breakout_down_20",
]
HORIZONS = (5, 10, 20)


def forward_returns(one_bar_returns: pd.Series, horizon: int) -> pd.Series:
    """Return close-to-close t+1...t+h returns aligned to forecast date t."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    r = pd.to_numeric(one_bar_returns, errors="coerce")
    gross = (1.0 + r).shift(-1).rolling(horizon, min_periods=horizon).apply(
        np.prod, raw=True
    )
    return gross.shift(-(horizon - 1)) - 1.0


def _model():
    return make_pipeline(
        SimpleImputer(strategy="median", add_indicator=True),
        StandardScaler(),
        LogisticRegression(max_iter=1200, C=0.5),
    )


def evaluate_symbol(
    frame: pd.DataFrame,
    ticker: str,
    *,
    horizons=HORIZONS,
    first_test_index: int = 260,
    retrain_every: int = 20,
    min_train_rows: int = 180,
    max_train_rows: int = 1260,
) -> list[dict]:
    """Expanding walk-forward OOS comparison; no future labels enter training."""
    d = frame.sort_values("bar_date").reset_index(drop=True).copy()
    d = d.replace([np.inf, -np.inf], np.nan)
    rows = []
    feature_sets = {
        "price_trend_baseline": BASE_FEATURES,
        "baseline_plus_structure": BASE_FEATURES + STRUCTURE_FEATURES,
    }
    for horizon in horizons:
        d["forward_return"] = forward_returns(d["return_1bar"], horizon)
        d["target_up"] = (d["forward_return"] > 0).astype(float)
        d.loc[d.index[-horizon:], ["forward_return", "target_up"]] = np.nan
        predictions = {name: [] for name in feature_sets}
        truth = []
        forecast_dates = []
        for start in range(first_test_index, len(d) - horizon, retrain_every):
            # At forecast row start, the latest eligible label is start-horizon-1.
            train = d.iloc[: max(0, start - horizon)].dropna(
                subset=["target_up", "forward_return"]
            )
            test = d.iloc[start : min(start + retrain_every, len(d) - horizon)].dropna(
                subset=["target_up"]
            )
            if len(train) < min_train_rows or train["target_up"].nunique() < 2 or test.empty:
                continue
            train = train.tail(max_train_rows)
            truth.extend(test["target_up"].astype(int).tolist())
            forecast_dates.extend(test["bar_date"].astype(str).tolist())
            for name, cols in feature_sets.items():
                available = [c for c in cols if c in d.columns]
                if not available:
                    continue
                model = _model()
                x_train = train[available].apply(pd.to_numeric, errors="coerce")
                x_test = test[available].apply(pd.to_numeric, errors="coerce")
                model.fit(x_train, train["target_up"].astype(int))
                predictions[name].extend(model.predict_proba(x_test)[:, 1].tolist())
        if len(truth) < 30 or len(set(truth)) < 2:
            for name in feature_sets:
                rows.append({
                    "ticker": ticker, "horizon": horizon, "model": name,
                    "n_oos": len(truth), "auc": np.nan, "brier": np.nan,
                    "directional_accuracy": np.nan, "status": "insufficient_oos",
                })
            continue
        y = np.asarray(truth, dtype=int)
        for name, pred in predictions.items():
            if len(pred) != len(y):
                continue
            p = np.asarray(pred, dtype=float)
            rows.append({
                "ticker": ticker, "horizon": horizon, "model": name,
                "n_oos": len(y), "auc": float(roc_auc_score(y, p)),
                "brier": float(brier_score_loss(y, p)),
                "directional_accuracy": float(((p >= 0.5) == y).mean()),
                "up_rate": float(y.mean()), "status": "descriptive_oos",
            })
    return rows


def main():
    source = Path("output/market_structure_features_v5.csv")
    if not source.exists():
        raise FileNotFoundError(source)
    data = pd.read_csv(source, parse_dates=["bar_date"])
    required = {"ticker", "bar_date", "return_1bar"}
    if not required.issubset(data.columns):
        raise ValueError(f"Missing required columns: {sorted(required - set(data.columns))}")
    rows = []
    for ticker, frame in data.groupby("ticker"):
        if ticker in {"^NDX", "^GSPC"}:
            continue
        rows.extend(evaluate_symbol(frame, ticker))
    result = pd.DataFrame(rows)
    out = Path("output")
    result.to_csv(out / "market_structure_validation_v5.csv", index=False)
    summary = {
        "symbols": sorted(result["ticker"].unique().tolist()) if not result.empty else [],
        "horizons": list(HORIZONS),
        "rows": int(len(result)),
        "method": "Expanding walk-forward; horizon-purged training labels; 20-session refit; overlapping OOS labels are dependent.",
        "caveat": "Daily OHLCV candle/volume features are proxies, not institutional order flow. Results are descriptive and not a trading strategy backtest.",
    }
    (out / "market_structure_validation_summary_v5.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))
    print("[OK] Wrote market-structure incremental validation report")


if __name__ == "__main__":
    main()
