"""QQQ walk-forward downside-risk model.

This is a separate risk model, not a trade signal. It predicts whether the
future intrahorizon low will breach a predeclared drawdown threshold. Training
labels are purged by the forecast horizon; all reported metrics are OOS.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_risk_v4 import (
    download_market_structure_bars,
    download_prices,
    make_features,
    merge_market_structure_features,
)

OUT = Path("output/qqq_drawdown_risk_v1")
OUT.mkdir(parents=True, exist_ok=True)
HORIZONS = (5, 10, 20)
DRAWDOWN_THRESHOLDS = (0.03, 0.05, 0.08)
ALERT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50)


def future_drawdown_labels(bars: pd.DataFrame, horizon: int, threshold: float):
    """Label future minimum Low relative to today's Close; today's bar excluded."""
    if not {"Low", "Close"}.issubset(bars.columns):
        raise ValueError("bars must contain Low and Close")
    if horizon < 1 or not 0 < threshold < 1:
        raise ValueError("invalid horizon or drawdown threshold")
    low = pd.to_numeric(bars["Low"], errors="coerce")
    close = pd.to_numeric(bars["Close"], errors="coerce")
    future_lows = pd.concat([low.shift(-step) for step in range(1, horizon + 1)], axis=1)
    future_min = future_lows.min(axis=1)
    drawdown = future_min / close - 1.0
    label = (drawdown <= -threshold).astype(float)
    label.iloc[-horizon:] = np.nan
    drawdown.iloc[-horizon:] = np.nan
    return label, drawdown


def _new_model():
    # A transparent linear baseline. No class balancing is used because it
    # distorts raw probability outputs; probabilities remain uncalibrated.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scale", StandardScaler()),
        ("logit", LogisticRegression(max_iter=2000, C=0.5)),
    ])


def generate_drawdown_forecasts(
    features: pd.DataFrame,
    bars: pd.DataFrame,
    horizon: int,
    drawdown_threshold: float,
    *,
    min_train_rows: int = 500,
    retrain_every: int = 63,
    max_train_rows: int = 1260,
) -> pd.DataFrame:
    """Purged expanding/rolling OOS forecasts for one horizon and threshold."""
    if horizon < 1 or min_train_rows < 100 or retrain_every < 1:
        raise ValueError("invalid forecast configuration")
    x = features.sort_index().copy()
    b = bars.copy().sort_index()
    x.index = pd.to_datetime(x.index).tz_localize(None).normalize()
    b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
    b = b[~b.index.duplicated(keep="last")]
    x = x[~x.index.duplicated(keep="last")]
    b = b.reindex(x.index)
    labels, future_dd = future_drawdown_labels(b, horizon, drawdown_threshold)
    cols = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if not cols:
        raise ValueError("no numeric features")
    model = None
    last_fit_i = -10**9
    outputs = []
    for i, date in enumerate(x.index):
        if i < horizon:
            continue
        cutoff = x.index[i - horizon]
        eligible = labels.loc[labels.index <= cutoff].dropna()
        if len(eligible) < min_train_rows or eligible.nunique() < 2:
            continue
        if model is None or i - last_fit_i >= retrain_every:
            train_y = eligible.tail(max_train_rows).astype(int)
            model = _new_model()
            model.fit(x.loc[train_y.index, cols], train_y)
            last_fit_i = i
            train_end = str(train_y.index.max().date())
            train_n = int(len(train_y))
            event_rate = float(train_y.mean())
        try:
            probability = float(model.predict_proba(x.iloc[[i]][cols])[0, 1])
        except Exception:
            continue
        outputs.append({
            "as_of": date,
            "horizon_sessions": horizon,
            "drawdown_threshold_pct": int(drawdown_threshold * 100),
            "drawdown_risk_score": probability,
            "train_end": train_end,
            "n_train": train_n,
            "train_event_rate": event_rate,
            "retrained": i == last_fit_i,
            "probability_calibrated": False,
            # Evaluation-only target; never passed to the model.
            "future_drawdown_eval_only_pct": float(future_dd.loc[date] * 100)
                if pd.notna(future_dd.loc[date]) else np.nan,
            "future_event_eval_only": bool(labels.loc[date] == 1)
                if pd.notna(labels.loc[date]) else np.nan,
        })
    return pd.DataFrame(outputs)


def _metric_row(group: pd.DataFrame, model_name: str, alert_threshold, warning):
    actual = group["future_event_eval_only"].astype(bool)
    warned = warning.astype(bool)
    tp = int((actual & warned).sum())
    fp = int((~actual & warned).sum())
    fn = int((actual & ~warned).sum())
    hits = group.loc[actual & warned, "first_hit_session"].dropna() if "first_hit_session" in group else pd.Series(dtype=float)
    return {
        "model": model_name,
        "horizon_sessions": int(group["horizon_sessions"].iloc[0]),
        "drawdown_threshold_pct": int(group["drawdown_threshold_pct"].iloc[0]),
        "alert_threshold": alert_threshold,
        "forecast_count": int(len(group)),
        "event_count": int(actual.sum()),
        "warning_count": int(warned.sum()),
        "precision_pct": tp / (tp + fp) * 100 if tp + fp else None,
        "recall_pct": tp / (tp + fn) * 100 if tp + fn else None,
        "false_warning_count": fp,
        "missed_event_count": fn,
        "median_first_hit_session_on_detected_events": float(hits.median()) if len(hits) else None,
    }


def evaluate_drawdown_forecasts(
    forecasts: pd.DataFrame,
    bars: pd.DataFrame,
    directional_baseline: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate OOS alerts across fixed thresholds and compare current rule."""
    if forecasts.empty:
        return forecasts.copy(), pd.DataFrame()
    b = bars.copy().sort_index()
    b.index = pd.to_datetime(b.index).tz_localize(None).normalize()
    details = forecasts.copy()
    details["as_of"] = pd.to_datetime(details["as_of"]).dt.tz_localize(None).dt.normalize()
    for idx, row in details.iterrows():
        date = row["as_of"]
        if date not in b.index:
            continue
        pos = b.index.get_loc(date)
        future = b.iloc[pos + 1:pos + 1 + int(row["horizon_sessions"])]
        close = float(b.iloc[pos]["Close"])
        hit = np.flatnonzero(
            (pd.to_numeric(future["Low"], errors="coerce").to_numpy() / close - 1.0)
            <= -float(row["drawdown_threshold_pct"]) / 100.0
        )
        details.loc[idx, "first_hit_session"] = int(hit[0] + 1) if len(hit) else np.nan

    rows = []
    # The most recent forecast rows have no fully observed future horizon yet.
    # Keep them in the detail file as live scores, but exclude them from OOS metrics.
    eval_details = details[details["future_event_eval_only"].notna()].copy()
    if eval_details.empty:
        return details, pd.DataFrame()
    keys = ["horizon_sessions", "drawdown_threshold_pct"]
    for _, group in eval_details.groupby(keys):
        probabilities = pd.to_numeric(group["drawdown_risk_score"], errors="coerce")
        y = group["future_event_eval_only"].astype(int)
        auc = float(roc_auc_score(y, probabilities)) if y.nunique() == 2 else None
        ap = float(average_precision_score(y, probabilities)) if y.nunique() == 2 else None
        brier = float(brier_score_loss(y, probabilities))
        for threshold in ALERT_THRESHOLDS:
            row = _metric_row(group, "dedicated_drawdown_logistic", threshold, probabilities >= threshold)
            row.update({"roc_auc": auc, "pr_auc": ap, "brier_score": brier})
            rows.append(row)

        if directional_baseline is not None and not directional_baseline.empty:
            base = directional_baseline.copy()
            base["as_of"] = pd.to_datetime(base["as_of"]).dt.tz_localize(None).dt.normalize()
            base = base[
                (pd.to_numeric(base["horizon_sessions"], errors="coerce") == int(group["horizon_sessions"].iloc[0]))
                & (base["ticker"] == "QQQ")
            ]
            merged = group[["as_of", "future_event_eval_only", "first_hit_session"]].merge(
                base[["as_of", "probability_up", "expected_return"]], on="as_of", how="inner"
            )
            if not merged.empty:
                baseline_group = merged.copy()
                baseline_group["horizon_sessions"] = group["horizon_sessions"].iloc[0]
                baseline_group["drawdown_threshold_pct"] = group["drawdown_threshold_pct"].iloc[0]
                warning = (
                    pd.to_numeric(baseline_group["probability_up"], errors="coerce") <= 0.40
                ) & (pd.to_numeric(baseline_group["expected_return"], errors="coerce") < 0)
                rows.append(_metric_row(
                    baseline_group, "existing_directional_bearish_rule",
                    "p_up<=0.40_and_expected_return<0", warning
                ))
    return details, pd.DataFrame(rows)


def main():
    print("[INFO] Downloading QQQ cross-market daily data for dedicated downside model...", flush=True)
    symbols = ["QQQ", "SPY", "NQ=F", "ES=F", "^NDX", "^GSPC", "^SOX", "^VIX",
               "HYG", "LQD", "TLT", "IEF", "UUP", "SMH", "SOXX", "NVDA", "AMD", "AVGO"]
    prices = download_prices(symbols, period="10y")
    if "QQQ" not in prices:
        raise RuntimeError("QQQ close series unavailable")
    bars = download_market_structure_bars("QQQ", period="10y")
    features = make_features(prices, "QQQ")
    features, _ = merge_market_structure_features(features, bars, "QQQ")
    forecasts = []
    for horizon in HORIZONS:
        for threshold in DRAWDOWN_THRESHOLDS:
            print(f"[INFO] QQQ downside risk: horizon={horizon}, drawdown={threshold:.0%}", flush=True)
            part = generate_drawdown_forecasts(features, bars, horizon, threshold)
            if not part.empty:
                forecasts.append(part)
    if not forecasts:
        raise RuntimeError("No out-of-sample drawdown forecasts generated")
    forecast_frame = pd.concat(forecasts, ignore_index=True)
    baseline_path = Path("output/walkforward_v5/oos_forecasts.csv")
    baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
    detail, summary = evaluate_drawdown_forecasts(forecast_frame, bars, baseline)
    detail.to_csv(OUT / "qqq_drawdown_risk_forecasts.csv", index=False)
    summary.to_csv(OUT / "qqq_drawdown_risk_summary.csv", index=False)
    payload = {
        "status": "completed",
        "ticker": "QQQ",
        "horizons_sessions": list(HORIZONS),
        "future_low_drawdown_thresholds_pct": [int(x * 100) for x in DRAWDOWN_THRESHOLDS],
        "alert_thresholds": list(ALERT_THRESHOLDS),
        "model": "rolling logistic regression with median imputation and standardization",
        "retrain_every_sessions": 63,
        "min_train_rows": 500,
        "max_train_rows": 1260,
        "probabilities_calibrated": False,
        "data_scope": "Yahoo Finance daily OHLCV and cross-market daily close features",
        "limitations": [
            "Future-low event labels overlap and are not independent observations.",
            "Daily bars cannot determine intraday event order or pre-market/overnight timing.",
            "This is a risk-warning score, not a probability claim until calibration is validated.",
            "Alert thresholds must be selected on a separate validation period, not the final OOS test.",
        ],
    }
    (OUT / "qqq_drawdown_risk_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False), flush=True)
    print(f"[OK] Wrote dedicated QQQ downside-risk outputs to {OUT}", flush=True)


if __name__ == "__main__":
    main()
