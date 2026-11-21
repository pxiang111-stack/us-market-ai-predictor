"""Transparent daily regime and cash-risk helpers for Simple NQ Trader V1."""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from simple_nq_support_resistance import build_support_resistance


def simple_moving_average(values: Sequence[float], period: int) -> float:
    if period < 1 or len(values) < period:
        raise ValueError("Not enough values for moving average")
    return sum(float(x) for x in values[-period:]) / period


def classify_regime(closes: Sequence[float]) -> dict[str, Any]:
    """Classify using completed daily closes only; no forward-looking inputs."""
    if len(closes) < 205:
        raise ValueError("Need at least 205 daily closes for stable SMA200 slope context")
    c = [float(x) for x in closes]
    sma20 = simple_moving_average(c, 20)
    sma50 = simple_moving_average(c, 50)
    sma200 = simple_moving_average(c, 200)
    sma20_prev = sum(c[-25:-5]) / 20
    if c[-1] > sma50 and sma50 > sma200 and sma20 > sma20_prev:
        regime = "UPTREND"
    elif c[-1] < sma50 and sma50 < sma200 and sma20 < sma20_prev:
        regime = "DOWNTREND"
    else:
        regime = "RANGE_TRANSITION"
    return {
        "regime": regime,
        "close": c[-1],
        "sma20": sma20,
        "sma50": sma50,
        "sma200": sma200,
        "sma20_rising_5d": sma20 > sma20_prev,
    }


def risk_sized_shares(equity: float, entry: float, stop: float,
                      risk_fraction: float = 0.005,
                      max_notional_fraction: float = 1.0) -> dict[str, Any]:
    """Whole-share, cash-only sizing. Returns zero shares for invalid/unaffordable risk."""
    if equity <= 0 or entry <= 0 or stop <= 0 or not 0 < risk_fraction <= 0.02:
        raise ValueError("Invalid equity, price, stop, or risk fraction")
    if stop >= entry:
        raise ValueError("For a long trade, stop must be below entry")
    risk_budget = equity * risk_fraction
    per_share_risk = entry - stop
    by_risk = int(risk_budget // per_share_risk)
    by_cash = int((equity * max_notional_fraction) // entry)
    shares = max(0, min(by_risk, by_cash))
    return {
        "shares": shares,
        "risk_budget": risk_budget,
        "planned_dollar_risk": shares * per_share_risk,
        "notional": shares * entry,
        "cash_remaining": equity - shares * entry,
    }


def daily_snapshot(bars: Sequence[dict[str, float]], equity: float = 10_000.0) -> dict[str, Any]:
    """One end-of-day analysis snapshot; this function never places orders."""
    closes = [float(b["close"]) for b in bars]
    regime = classify_regime(closes)
    levels = build_support_resistance(bars)
    return {
        "action": "WAIT",
        "regime": regime,
        "levels": levels,
        "paper_equity": float(equity),
        "note": "V1 snapshot only: setup confirmation and next-session execution are not yet implemented.",
    }
