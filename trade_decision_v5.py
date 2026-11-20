"""Rule-based execution plan layer. Research only; not an order router."""
from __future__ import annotations
import numpy as np
import pandas as pd


def atr_value(bars: pd.DataFrame, period: int = 14) -> float:
    required = {"High", "Low", "Close"}
    if bars is None or bars.empty or not required.issubset(bars.columns):
        return float("nan")
    d = bars.sort_index()
    prev = d["Close"].shift(1)
    tr = pd.concat([
        d["High"] - d["Low"],
        (d["High"] - prev).abs(),
        (d["Low"] - prev).abs(),
    ], axis=1).max(axis=1)
    value = tr.rolling(period, min_periods=period).mean().iloc[-1]
    return float(value) if pd.notna(value) else float("nan")


def build_trade_decision(
    *, close: float, atr: float, probability_up: float, expected_return: float,
    features: dict, allow_short: bool = False, round_trip_cost: float = 0.0015,
    risk_budget_pct: float = 0.005, max_gross_exposure: float = 1.0,
    strategy_mode: str = "confluence",
    probability_threshold_long: float | None = None,
    probability_threshold_short: float | None = None,
    structure_relative_volume_min: float = 1.2,
    structure_close_location_long: float = 0.65,
    structure_close_location_short: float = 0.35,
) -> dict:
    """Convert forecast + structure into a conservative, explainable trade plan.

    Expected return is a model estimate, not a calibrated guarantee. Position
    fraction is a normalized equity exposure cap, not broker buying power.
    """
    result = {
        "decision": "FLAT", "entry_reference": None, "stop_loss": None,
        "take_profit_1r": None, "take_profit_2r": None,
        "stop_distance_pct": None, "risk_budget_pct": risk_budget_pct,
        "max_position_fraction": 0.0, "probability_up": probability_up,
        "expected_return": expected_return, "round_trip_cost_assumption": round_trip_cost,
        "reason": "No sufficiently strong, cost-aware confluence.",
        "execution_status": "research_signal_only",
    }
    vals = [close, atr, probability_up, expected_return]
    if not all(np.isfinite(v) for v in vals) or close <= 0 or atr <= 0:
        result["reason"] = "Missing or invalid price, ATR, or model output; stand aside."
        return result

    if strategy_mode not in {"confluence", "trend_breakout", "reversal_absorption"}:
        raise ValueError(f"Unknown strategy_mode: {strategy_mode}")
    bull_trend = all(features.get(k) == 1 for k in (
        "ms_trend_above_sma20", "ms_sma20_above_sma50", "ms_sma50_above_sma200"))
    bear_trend = (
        features.get("ms_trend_above_sma20") == 0
        and features.get("ms_sma20_above_sma50") == 0
        and features.get("ms_sma50_above_sma200") == 0
    )
    # Sustained volume-weighted close location is a directional effort/result
    # proxy. Spring/upthrust patterns can independently confirm a reclaim/rejection.
    flow_pressure = features.get("ms_volume_weighted_close_location_20", np.nan)
    bullish_effort = (
        (np.isfinite(flow_pressure) and flow_pressure >= 0.05)
        or features.get("ms_spring_reclaim_20") == 1
    )
    bearish_effort = (
        (np.isfinite(flow_pressure) and flow_pressure <= -0.05)
        or features.get("ms_upthrust_reject_20") == 1
    )
    long_structure = (
        bull_trend and features.get("ms_failed_breakout_up_20") != 1
        and features.get("ms_upthrust_reject_20") != 1
        and bullish_effort
        and (features.get("ms_breakout_up_volume_confirmed") == 1
             or (features.get("ms_relative_volume", 0) >= structure_relative_volume_min
                 and features.get("ms_close_location", 0) >= structure_close_location_long))
    )
    short_structure = (
        bear_trend and features.get("ms_failed_breakout_down_20") != 1
        and features.get("ms_spring_reclaim_20") != 1
        and bearish_effort
        and (features.get("ms_breakout_down_volume_confirmed") == 1
             or (features.get("ms_relative_volume", 0) >= structure_relative_volume_min
                 and features.get("ms_close_location", 1) <= structure_close_location_short))
    )

    # Two deliberately distinct hypotheses. Breakout trades require a confirmed
    # range break; reversal trades require a spring/upthrust reclaim/rejection.
    # Neither is treated as proof of institutional accumulation/distribution.
    if strategy_mode == "trend_breakout":
        long_structure = (
            bull_trend and features.get("ms_failed_breakout_up_20") != 1
            and features.get("ms_breakout_up_volume_confirmed") == 1
            and features.get("ms_close_location", 0) >= 0.65
        )
        short_structure = (
            bear_trend and features.get("ms_failed_breakout_down_20") != 1
            and features.get("ms_breakout_down_volume_confirmed") == 1
            and features.get("ms_close_location", 1) <= 0.35
        )
        long_probability_gate, short_probability_gate = probability_up >= (probability_threshold_long if probability_threshold_long is not None else 0.62), probability_up <= (probability_threshold_short if probability_threshold_short is not None else 0.38)
    elif strategy_mode == "reversal_absorption":
        long_structure = features.get("ms_spring_reclaim_20") == 1
        short_structure = features.get("ms_upthrust_reject_20") == 1
        long_probability_gate, short_probability_gate = probability_up >= (probability_threshold_long if probability_threshold_long is not None else 0.55), probability_up <= (probability_threshold_short if probability_threshold_short is not None else 0.45)
    else:
        long_probability_gate, short_probability_gate = probability_up >= (probability_threshold_long if probability_threshold_long is not None else 0.62), probability_up <= (probability_threshold_short if probability_threshold_short is not None else 0.38)

    net_edge = abs(expected_return) - round_trip_cost
    # Persist each gate independently so a FLAT decision can be diagnosed rather
    # than treated as an opaque failure. These are causal, as-of-date features.
    result.update({
        "strategy_mode": strategy_mode,
        "gate_probability_long": bool(long_probability_gate),
        "gate_probability_short": bool(short_probability_gate),
        "gate_edge_long": bool(expected_return > round_trip_cost),
        "gate_edge_short": bool(expected_return < -round_trip_cost),
        "gate_bull_trend": bool(bull_trend),
        "gate_bear_trend": bool(bear_trend),
        "gate_long_structure": bool(long_structure),
        "gate_short_structure": bool(short_structure),
        "gate_allow_short": bool(allow_short),
    })
    direction = None
    if long_probability_gate and expected_return > round_trip_cost and long_structure:
        direction = "LONG"
    elif allow_short and short_probability_gate and expected_return < -round_trip_cost and short_structure:
        direction = "SHORT"
    if direction is None:
        result["reason"] = (
            f"Confluence filter not met (bull_trend={bull_trend}, bear_trend={bear_trend}, "
            f"long_structure={long_structure}, short_structure={short_structure}, "
            f"net_edge_after_cost={net_edge:.4%})."
        )
        return result

    stop_distance = 2.0 * atr
    stop_pct = stop_distance / close
    if not np.isfinite(stop_pct) or stop_pct <= 0:
        result["reason"] = "Invalid ATR-based stop distance; stand aside."
        return result
    position_fraction = min(max_gross_exposure, risk_budget_pct / stop_pct)
    sign = 1 if direction == "LONG" else -1
    result.update({
        "decision": direction,
        "entry_reference": close,
        "stop_loss": close - sign * stop_distance,
        "take_profit_1r": close + sign * stop_distance,
        "take_profit_2r": close + sign * 2.0 * stop_distance,
        "stop_distance_pct": stop_pct,
        "max_position_fraction": position_fraction,
        "reason": (
            f"{direction}: model probability and expected-return filter agree with "
            f"multi-timeframe trend and volume/price structure; estimated edge after "
            f"round-trip cost is {net_edge:.4%}. Stop is 2x ATR; size targets "
            f"{risk_budget_pct:.2%} equity risk, capped at {max_gross_exposure:.0%} gross exposure."
        ),
    })
    return result
