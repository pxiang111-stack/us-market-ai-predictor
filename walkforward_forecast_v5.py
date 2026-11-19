"""Strict expanding/rolling walk-forward forecasts with horizon label purging.

Each forecast is made after close t. Training labels are restricted to samples
whose full forward-return horizon has ended by t. This module creates forecasts,
not fills; execution is handled by walkforward_execution_v5.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from market_risk_v4 import labeled, models


def generate_walkforward_forecasts(
    features: pd.DataFrame,
    close: pd.Series,
    horizon_sessions: int,
    *,
    min_train_rows: int = 500,
    retrain_every: int = 20,
    max_train_rows: int | None = 1260,
) -> pd.DataFrame:
    if horizon_sessions < 1 or min_train_rows < 100 or retrain_every < 1:
        raise ValueError("Invalid horizon, minimum training size, or retrain interval")
    x = features.sort_index().copy()
    px = close.reindex(x.index)
    z = labeled(x, px, f"{horizon_sessions}d") if f"{horizon_sessions}d" in {"5d", "10d", "20d"} else None
    if z is None:
        z = x.copy()
        z["forward_return"] = px.shift(-horizon_sessions) / px - 1
        z["up"] = (z["forward_return"] > 0).astype(float)
        z.iloc[-horizon_sessions:, z.columns.get_loc("up")] = np.nan
        z.iloc[-horizon_sessions:, z.columns.get_loc("forward_return")] = np.nan
    feature_cols = list(x.columns)
    outputs = []
    fitted_at = None
    fitted_n = 0
    classifier_a = classifier_b = regressor = None
    last_fit_position = -10**9

    for i in range(len(x)):
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
            "n_train": fitted_n, "horizon_sessions": horizon_sessions,
            "retrained": i == last_fit_position,
            "probability_calibrated": False,
        })
    return pd.DataFrame(outputs).set_index("as_of") if outputs else pd.DataFrame(
        columns=["probability_up", "expected_return", "train_end", "n_train",
                 "horizon_sessions", "retrained", "probability_calibrated"]
    )
