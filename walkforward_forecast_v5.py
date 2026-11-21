"""Strict expanding/rolling walk-forward forecasts with horizon label purging.

Each forecast is made after close t. Training labels are restricted to samples
whose full forward-return horizon has ended by t. This module creates forecasts,
not fills; execution is handled by walkforward_execution_v5.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from market_risk_v4 import models


def build_execution_aligned_labels(
    close: pd.Series, next_session_open: pd.Series, horizon_sessions: int
) -> pd.DataFrame:
    """Label the return actually available to a next-open, fixed-horizon trade.

    A signal is formed after close t, entered at open t+1, and marked at
    close t+h. The label therefore uses close[t+h] / open[t+1] - 1.
    """
    if horizon_sessions < 1:
        raise ValueError("horizon_sessions must be >= 1")
    px = pd.to_numeric(close, errors="coerce")
    op = pd.to_numeric(next_session_open.reindex(px.index), errors="coerce")
    entry = op.shift(-1)
    exit_price = px.shift(-horizon_sessions)
    forward_return = exit_price / entry - 1.0
    valid = entry.notna() & exit_price.notna() & entry.gt(0) & exit_price.gt(0)
    out = pd.DataFrame({"forward_return": forward_return.where(valid)}, index=px.index)
    out["up"] = (out["forward_return"] > 0).astype(float).where(valid)
    return out


def generate_walkforward_forecasts(
    features: pd.DataFrame,
    close: pd.Series,
    horizon_sessions: int,
    *,
    entry_open: pd.Series,
    min_train_rows: int = 500,
    retrain_every: int = 20,
    max_train_rows: int | None = 1260,
    start_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if horizon_sessions < 1 or min_train_rows < 100 or retrain_every < 1:
        raise ValueError("Invalid horizon, minimum training size, or retrain interval")
    x = features.sort_index().copy()
    px = close.reindex(x.index)
    z = x.join(build_execution_aligned_labels(px, entry_open.reindex(x.index), horizon_sessions))
    # Regime labels are categorical diagnostics used by the decision layer and
    # post-trade attribution. The sklearn pipelines use median imputation, so
    # only numeric/bool columns may enter the model feature matrix.
    feature_cols = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    if not feature_cols:
        raise ValueError("No numeric model features available for walk-forward training")
    outputs = []
    fitted_at = None
    fitted_n = 0
    fitted_up_rate = float("nan")
    classifier_a = classifier_b = regressor = None
    last_fit_position = -10**9

    score_start = pd.Timestamp(start_date).normalize() if start_date is not None else None
    for i in range(len(x)):
        if score_start is not None and pd.Timestamp(x.index[i]).normalize() < score_start:
            continue
        if i < horizon_sessions:
            continue
        cutoff = x.index[i - horizon_sessions]
        train = z.loc[(z.index <= cutoff) & z["up"].notna() & z["forward_return"].notna()]
        if len(train) < min_train_rows or train["up"].nunique() < 2:
            continue
        if classifier_a is None or i - last_fit_position >= retrain_every:
            if max_train_rows is not None:
                train = train.tail(max_train_rows)
            classifier_a, classifier_b, regressor = models()
            classifier_a.fit(train[feature_cols], train["up"].astype(int))
            classifier_b.fit(train[feature_cols], train["up"].astype(int))
            regressor.fit(train[feature_cols], train["forward_return"].astype(float))
            last_fit_position = i
            fitted_at = str(train.index.max().date())
            fitted_n = int(len(train))
            fitted_up_rate = float(train["up"].mean())
        current = x.iloc[[i]]
        try:
            p = float((classifier_a.predict_proba(current[feature_cols])[0, 1]
                       + classifier_b.predict_proba(current[feature_cols])[0, 1]) / 2)
            expected = float(regressor.predict(current[feature_cols])[0])
        except Exception:
            continue
        outputs.append({
            "as_of": x.index[i], "probability_up": p,
            "expected_return": expected, "train_end": fitted_at,
            "n_train": fitted_n,
            "train_up_rate": fitted_up_rate,
            "horizon_sessions": horizon_sessions,
            "retrained": i == last_fit_position,
            "probability_calibrated": False,
        })
    return pd.DataFrame(outputs).set_index("as_of") if outputs else pd.DataFrame(
        columns=["probability_up", "expected_return", "train_end", "n_train",
                 "train_up_rate", "horizon_sessions", "retrained", "probability_calibrated"]
    )
