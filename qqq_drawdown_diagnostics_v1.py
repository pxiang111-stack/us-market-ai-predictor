"""Out-of-sample QQQ early-warning diagnostics for future downside excursions.

Warnings are evaluated only with information available at the forecast timestamp.
Future lows are used solely to label subsequent drawdowns, never as model inputs.
This is an event-detection diagnostic, not a claim that a drawdown is predictable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def evaluate_drawdown_warnings(
    forecasts: pd.DataFrame,
    bars: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = (0.03, 0.05, 0.08),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-forecast future drawdown labels and warning precision/recall.

    A bearish model warning is predeclared as probability_up <= 0.40 and
    expected_return < 0. A trade warning is an actual SHORT decision. These
    are reported separately because trade filters may confirm a trend late.
    Future drawdown is the minimum subsequent Low / as-of Close - 1 over the
    forecast horizon. The as-of bar itself is excluded.
    """
    required_bars = {"Low", "Close"}
    if not required_bars.issubset(bars.columns):
        raise ValueError(f"bars missing: {sorted(required_bars - set(bars.columns))}")
    required_forecasts = {"as_of", "horizon_sessions", "probability_up", "expected_return"}
    if not required_forecasts.issubset(forecasts.columns):
        raise ValueError(f"forecasts missing: {sorted(required_forecasts - set(forecasts.columns))}")
    if any((not np.isfinite(t) or t <= 0 or t >= 1) for t in thresholds):
        raise ValueError("drawdown thresholds must be fractions between 0 and 1")

    b = bars.copy().sort_index()
    b.index = pd.to_datetime(b.index).tz_localize(None).normalize() if getattr(pd.to_datetime(b.index), "tz", None) is not None else pd.to_datetime(b.index).normalize()
    b = b[~b.index.duplicated(keep="last")]
    rows = []
    f = forecasts.copy()
    f["as_of"] = pd.to_datetime(f["as_of"]).dt.tz_localize(None).dt.normalize() if getattr(pd.to_datetime(f["as_of"]).dt, "tz", None) is not None else pd.to_datetime(f["as_of"]).dt.normalize()
    for _, row in f.iterrows():
        date = row["as_of"]
        if date not in b.index:
            continue
        pos = b.index.get_loc(date)
        if not isinstance(pos, (int, np.integer)):
            continue
        horizon = int(row["horizon_sessions"])
        if horizon < 1:
            continue
        future = b.iloc[pos + 1: pos + 1 + horizon]
        if future.empty:
            continue
        close = float(b.iloc[pos]["Close"])
        if not np.isfinite(close) or close <= 0:
            continue
        future_drawdown = float((pd.to_numeric(future["Low"], errors="coerce") / close - 1.0).min())
        bearish = bool(float(row["probability_up"]) <= 0.40 and float(row["expected_return"]) < 0)
        short_trade = str(row.get("decision", "")).upper() == "SHORT"
        record = {
            "as_of": date, "horizon_sessions": horizon,
            "probability_up": float(row["probability_up"]),
            "expected_return": float(row["expected_return"]),
            "decision": row.get("decision", "UNAVAILABLE"),
            "bearish_model_warning": bearish,
            "short_trade_signal": short_trade,
            "future_max_drawdown_pct": future_drawdown * 100.0,
            "future_bars_observed": int(len(future)),
        }
        for threshold in thresholds:
            label = f"drawdown_{int(threshold * 100)}pct"
            record[label] = bool(future_drawdown <= -threshold)
            hits = np.flatnonzero((pd.to_numeric(future["Low"], errors="coerce").to_numpy() / close - 1.0) <= -threshold)
            record[f"{label}_first_hit_session"] = int(hits[0] + 1) if len(hits) else None
        rows.append(record)

    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    summaries = []
    for horizon, group in detail.groupby("horizon_sessions"):
        for threshold in thresholds:
            label = f"drawdown_{int(threshold * 100)}pct"
            actual = group[label].astype(bool)
            for warning_name in ("bearish_model_warning", "short_trade_signal"):
                warned = group[warning_name].astype(bool)
                tp = int((warned & actual).sum())
                fp = int((warned & ~actual).sum())
                fn = int((~warned & actual).sum())
                summaries.append({
                    "horizon_sessions": int(horizon),
                    "drawdown_threshold_pct": int(threshold * 100),
                    "warning_type": warning_name,
                    "forecast_count": int(len(group)),
                    "drawdown_event_count": int(actual.sum()),
                    "warning_count": int(warned.sum()),
                    "precision_pct": (tp / (tp + fp) * 100.0) if tp + fp else None,
                    "recall_pct": (tp / (tp + fn) * 100.0) if tp + fn else None,
                    "false_warning_count": fp,
                    "missed_event_count": fn,
                    "median_lead_sessions_on_detected_events": (
                        float(group.loc[warned & actual, f"{label}_first_hit_session"].dropna().median())
                        if tp else None
                    ),
                    "warning_rule": (
                        "probability_up<=0.40 and expected_return<0"
                        if warning_name == "bearish_model_warning"
                        else "decision==SHORT"
                    ),
                    "caveat": "Overlapping forward horizons; descriptive OOS event diagnostic, not independent event samples.",
                })
    return detail, pd.DataFrame(summaries)
