"""Conservative end-of-day setup filter; candidates only, never execution."""
from __future__ import annotations

from typing import Any


def evaluate_setups(
    bars: list[dict[str, float]],
    regime: str,
    levels: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate support bounce / resistance rejection on the latest completed bar."""
    if not bars:
        raise ValueError("At least one completed bar is required")
    bar = bars[-1]
    high, low, close = (float(bar[k]) for k in ("high", "low", "close"))
    atr = float(levels["atr"])
    if atr <= 0:
        raise ValueError("ATR must be positive")

    # Only zones close enough to have been genuinely tested are eligible.
    support_candidates = [
        z for z in levels.get("supports", [])
        if low <= float(z["high"]) + 0.10 * atr
        and close > float(z["mid"])
        and regime in {"UPTREND", "RANGE_TRANSITION"}
    ]
    if support_candidates:
        zone = min(support_candidates, key=lambda z: abs(close - float(z["mid"])))
        return {
            "setup": "SUPPORT_BOUNCE",
            "action": "LONG_CANDIDATE",
            "zone": zone,
            "confirmation": "Completed bar tested support area and closed above its midpoint",
            "execution": "NEXT_SESSION_ONLY",
        }

    resistance_candidates = [
        z for z in levels.get("resistances", [])
        if high >= float(z["low"]) - 0.10 * atr
        and close < float(z["mid"])
    ]
    if resistance_candidates:
        zone = min(resistance_candidates, key=lambda z: abs(close - float(z["mid"])))
        if regime == "DOWNTREND":
            action = "SHORT_SETUP_RESEARCH_ONLY"
        else:
            action = "WAIT_FOR_CONFIRMATION"
        return {
            "setup": "RESISTANCE_REJECTION",
            "action": action,
            "zone": zone,
            "confirmation": "Completed bar tested resistance area and closed below its midpoint",
            "execution": "NO_V1_SHORT_ORDERS",
        }

    return {
        "setup": "NONE",
        "action": "WAIT",
        "zone": None,
        "confirmation": "No confirmed support bounce or resistance rejection",
        "execution": "NO_TRADE",
    }
