"""Small, point-in-time support/resistance engine for Simple NQ Trader V1.

Input bars are chronological dictionaries with high, low, close (and optionally open).
A pivot is not usable until its right-hand confirmation bars have closed.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def calculate_atr(bars: Sequence[dict[str, float]], period: int = 14) -> float:
    if period < 1 or len(bars) < period + 1:
        raise ValueError("ATR requires at least period + 1 bars")
    true_ranges = []
    for i in range(1, len(bars)):
        current = bars[i]
        previous_close = float(bars[i - 1]["close"])
        true_ranges.append(max(
            float(current["high"]) - float(current["low"]),
            abs(float(current["high"]) - previous_close),
            abs(float(current["low"]) - previous_close),
        ))
    window = true_ranges[-period:]
    return sum(window) / len(window)


def confirmed_pivots(
    bars: Sequence[dict[str, float]], left: int = 3, right: int = 3
) -> list[dict[str, Any]]:
    """Return pivots whose right-side confirmation bars are already present."""
    if left < 1 or right < 1:
        raise ValueError("left and right must be positive")
    pivots: list[dict[str, Any]] = []
    last_confirmable = len(bars) - right
    for i in range(left, last_confirmable):
        high = float(bars[i]["high"])
        low = float(bars[i]["low"])
        left_highs = [float(bars[j]["high"]) for j in range(i - left, i)]
        right_highs = [float(bars[j]["high"]) for j in range(i + 1, i + right + 1)]
        left_lows = [float(bars[j]["low"]) for j in range(i - left, i)]
        right_lows = [float(bars[j]["low"]) for j in range(i + 1, i + right + 1)]
        if high > max(left_highs) and high >= max(right_highs):
            pivots.append({"kind": "resistance", "level": high, "pivot_index": i,
                           "confirmed_index": i + right, "source": "confirmed_swing_high"})
        if low < min(left_lows) and low <= min(right_lows):
            pivots.append({"kind": "support", "level": low, "pivot_index": i,
                           "confirmed_index": i + right, "source": "confirmed_swing_low"})
    return pivots


def _cluster_levels(levels: list[dict[str, Any]], tolerance: float) -> list[dict[str, Any]]:
    if not levels:
        return []
    levels = sorted(levels, key=lambda x: float(x["level"]))
    clusters: list[list[dict[str, Any]]] = []
    for item in levels:
        if not clusters or float(item["level"]) - float(clusters[-1][0]["level"]) > tolerance:
            clusters.append([item])
        else:
            clusters[-1].append(item)

    zones = []
    for cluster in clusters:
        prices = [float(x["level"]) for x in cluster]
        zones.append({
            "low": min(prices),
            "high": max(prices),
            "mid": sum(prices) / len(prices),
            "touches": len(cluster),
            "sources": sorted({str(x["source"]) for x in cluster}),
            "strength": "major" if len(cluster) >= 2 else "minor",
        })
    return zones


def build_support_resistance(
    bars: Sequence[dict[str, float]],
    atr_period: int = 14,
    left: int = 3,
    right: int = 3,
    lookback: int = 120,
) -> dict[str, Any]:
    """Build support/resistance zones from confirmed pivots and prior range extremes."""
    if len(bars) < max(atr_period + 1, right + left + 2, 21):
        raise ValueError("Not enough bars to build support/resistance")
    recent = list(bars[-lookback:])
    atr = calculate_atr(bars, atr_period)
    last_close = float(bars[-1]["close"])
    raw = [
        {**p, "source": p["source"]}
        for p in confirmed_pivots(recent, left=left, right=right)
    ]

    # Prior completed-session range levels: deliberately exclude the current bar.
    prior = list(bars[:-1])
    for period in (20, 60):
        window = prior[-period:]
        if len(window) == period:
            raw.append({"kind": "resistance", "level": max(float(x["high"]) for x in window),
                        "source": f"prior_{period}d_high"})
            raw.append({"kind": "support", "level": min(float(x["low"]) for x in window),
                        "source": f"prior_{period}d_low"})

    tolerance = 0.5 * atr
    supports = [x for x in raw if float(x["level"]) <= last_close]
    resistances = [x for x in raw if float(x["level"]) > last_close]
    support_zones = _cluster_levels(supports, tolerance)
    resistance_zones = _cluster_levels(resistances, tolerance)

    # Keep the three nearest zones on each side; strongest zones win ties.
    support_zones.sort(key=lambda z: (last_close - z["mid"], -z["touches"]))
    resistance_zones.sort(key=lambda z: (z["mid"] - last_close, -z["touches"]))
    return {
        "last_close": last_close,
        "atr": atr,
        "tolerance": tolerance,
        "supports": support_zones[:3],
        "resistances": resistance_zones[:3],
        "as_of_bar_index": len(bars) - 1,
        "pivot_confirmation": f"{left} left / {right} right bars",
    }
