"""Descriptive forward-return study for Regime-First v2 states.

Daily observations overlap. State-entry events are also reported separately.
All rows with fewer than 30 observations are marked RECORD_ONLY; no inference
or parameter selection is performed here.
"""
from __future__ import annotations

import pandas as pd


def _summarize(frame: pd.DataFrame, return_col: str, benchmark_mean: float,
               denominator: int, label: str, horizon: int) -> dict:
    values = pd.to_numeric(frame[return_col], errors="coerce").dropna()
    n = len(values)
    mean = float(values.mean()) if n else float("nan")
    median = float(values.median()) if n else float("nan")
    return {
        "sample_type": label,
        "horizon_days": horizon,
        "n": n,
        "status": "RECORD_ONLY" if n < 30 else "DESCRIPTIVE_ONLY",
        "coverage_pct": float(n / denominator * 100) if denominator else 0.0,
        "win_rate_pct": float((values > 0).mean() * 100) if n else float("nan"),
        "mean_forward_return_pct": mean * 100,
        "median_forward_return_pct": median * 100,
        "unconditional_mean_pct": benchmark_mean * 100,
        "excess_vs_unconditional_mean_pct": (mean - benchmark_mean) * 100,
    }


def build_event_study(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"date", "state", "NDX_close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing event-study fields: {sorted(missing)}")
    z = frame.copy().sort_values("date").reset_index(drop=True)
    z["NDX_close"] = pd.to_numeric(z["NDX_close"], errors="coerce")
    z["state"] = z["state"].astype(str)
    z["is_state_entry"] = z["state"].ne(z["state"].shift(1))
    rows = []
    for horizon in (5, 20):
        col = f"forward_{horizon}d"
        z[col] = z["NDX_close"].shift(-horizon) / z["NDX_close"] - 1
        valid = z.dropna(subset=[col])
        unconditional = pd.to_numeric(valid[col], errors="coerce").mean()
        for state in ("RISK_OFF", "NEUTRAL", "RISK_ON"):
            daily = valid.loc[valid["state"].eq(state)]
            entries = daily.loc[daily["is_state_entry"]]
            rows.append(_summarize(daily, col, unconditional, len(valid),
                                   f"DAILY_{state}", horizon))
            rows.append(_summarize(entries, col, unconditional, len(valid),
                                   f"ENTRY_{state}", horizon))
    return pd.DataFrame(rows)
