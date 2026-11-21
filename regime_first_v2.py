"""Regime-First v2: deterministic market-state gate and risk-budget calculator.

Input rows are end-of-day observations. Rolling 20/60-day highs/lows must be
computed from prior completed sessions and must exclude the current session.
This module intentionally does not create directional short signals.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import pandas as pd

CRITICAL_FIELDS = (
    "NQ_close", "NQ_high", "NQ_low", "NDX_close",
    "BTC_close", "ETH_close", "VIX_close", "NQ_200ma",
    "NQ_20d_high", "NQ_20d_low", "NQ_60d_high", "NQ_60d_low",
    "BTC_20d_high", "BTC_20d_low", "ETH_20d_high", "ETH_20d_low",
    "CCC_spread", "rank_model_score",
)
STATE_CAP = {"RISK_OFF": 0.01, "NEUTRAL": 0.02, "RISK_ON": 0.04}


@dataclass
class RegimeResult:
    date: str
    state: str
    risk_cap_pct: float
    volatility_multiplier: float
    data_gap_multiplier: float
    monitor_veto_multiplier: float
    max_risk_pct: float
    flip_conditions: str
    flip_count: int
    hard_risk_off: bool
    data_gaps: str
    allowed_action: str
    notes: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _valid(row: pd.Series, fields: tuple[str, ...]) -> bool:
    return all(f in row.index and pd.notna(row[f]) for f in fields)


def _spread_widening_5d(history: pd.DataFrame) -> bool:
    fields = ("BBB_spread", "HY_spread")
    if len(history) < 6 or not all(f in history for f in fields):
        return False
    z = history[list(fields)].tail(6).apply(pd.to_numeric, errors="coerce")
    if z.isna().any().any():
        return False
    # Five consecutive positive daily changes for all three spreads.
    return bool((z.diff().iloc[1:] > 0).all().all())


def _raw_risk_off(history: pd.DataFrame, row: pd.Series) -> bool:
    # Evaluate independent triggers independently: missing ETH data must not
    # mask a confirmed BTC breakdown (or vice versa).
    btc_break = (_valid(row, ("BTC_close", "BTC_20d_low")) and
                 row.BTC_close < row.BTC_20d_low)
    eth_break = (_valid(row, ("ETH_close", "ETH_20d_low")) and
                 row.ETH_close < row.ETH_20d_low)
    nq_vol_break = (_valid(row, ("NQ_close", "NQ_200ma", "VIX_close")) and
                    row.NQ_close < row.NQ_200ma and row.VIX_close > 30)
    credit_break = (_valid(row, ("CCC_spread",)) and row.CCC_spread > 14 and
                    _spread_widening_5d(history))
    return bool(btc_break or eth_break or nq_vol_break or credit_break)


def _base_risk_on(row: pd.Series) -> bool:
    required = ("NQ_close", "NQ_200ma", "VIX_close",
                "BTC_close", "BTC_20d_low", "ETH_close", "ETH_20d_low",
                "rank_model_score")
    if not _valid(row, required):
        return False
    return bool(
        row.NQ_close > row.NQ_200ma
        and row.VIX_close < 20
        and row.BTC_close >= row.BTC_20d_low
        and row.ETH_close >= row.ETH_20d_low
        and row.rank_model_score >= 0
    )


def _flip_conditions(history: pd.DataFrame, i: int) -> list[bool]:
    row = history.iloc[i]
    if i < 2:
        return [False, False, False, False]

    # 1) Frozen initial NQ threshold; no adaptive recalibration in this module.
    c1 = pd.notna(row.get("NQ_close")) and row.NQ_close > 27480

    # 2) Same-session breakout by BTC, ETH and NQ vs prior 20 completed sessions.
    breakout_fields = ("BTC_close", "BTC_20d_high", "ETH_close",
                       "ETH_20d_high", "NQ_close", "NQ_20d_high")
    c2 = _valid(row, breakout_fields) and (
        row.BTC_close > row.BTC_20d_high
        and row.ETH_close > row.ETH_20d_high
        and row.NQ_close > row.NQ_20d_high
    )

    # 3) Operational interpretation of "recover from -2": start at or below
    # -2, rise by at least one score point on each of two consecutive sessions,
    # and finish at or above zero. Freeze this interpretation pending owner
    # confirmation; do not tune it on the holdout sample.
    scores = pd.to_numeric(history["rank_model_score"].iloc[i-2:i+1],
                           errors="coerce")
    c3 = bool(scores.notna().all() and
              scores.iloc[0] <= -2 and
              scores.iloc[1] - scores.iloc[0] >= 1 and
              scores.iloc[2] - scores.iloc[1] >= 1 and
              scores.iloc[2] >= 0)

    # 4) VIX closes below 20 for two consecutive sessions.
    vix = pd.to_numeric(history["VIX_close"].iloc[i-1:i+1],
                        errors="coerce")
    c4 = bool(vix.notna().all() and (vix < 20).all())
    return [bool(c1), bool(c2), bool(c3), bool(c4)]


def evaluate_history(data: pd.DataFrame, initial_state: str = "NEUTRAL") -> pd.DataFrame:
    """Evaluate rows chronologically and return one auditable result per session."""
    if initial_state not in STATE_CAP:
        raise ValueError(f"Unsupported initial_state: {initial_state}")
    if "date" not in data.columns:
        raise ValueError("Missing required column: date")

    z = data.copy()
    # Missing contract fields are recorded as gaps and halve the risk budget;
    # they must not crash the daily process or be silently treated as valid.
    for field in CRITICAL_FIELDS:
        if field not in z.columns:
            z[field] = float("nan")
    for field in ("BBB_spread", "HY_spread", "monitor_hard_veto"):
        if field not in z.columns:
            z[field] = False if field == "monitor_hard_veto" else float("nan")
    z["date"] = pd.to_datetime(z["date"], errors="coerce")
    z = z.sort_values("date").reset_index(drop=True)
    if z["date"].isna().any():
        raise ValueError("Invalid or missing date values")

    state = initial_state
    recovery_streak = 0
    outputs = []

    for i, row in z.iterrows():
        history = z.iloc[:i+1]
        hard_off = _raw_risk_off(history, row)
        base_on = _base_risk_on(row)

        recovery_fields = ("BTC_close", "BTC_20d_low", "ETH_close",
                           "ETH_20d_low", "NQ_close", "NQ_200ma", "VIX_close")
        recovery_ok = _valid(row, recovery_fields) and (
            row.BTC_close >= row.BTC_20d_low
            and row.ETH_close >= row.ETH_20d_low
            and (row.NQ_close > row.NQ_200ma or row.VIX_close < 30)
        )
        recovery_streak = recovery_streak + 1 if recovery_ok and not hard_off else 0

        if state == "RISK_OFF":
            if recovery_streak >= 2:
                state = "NEUTRAL"
        elif hard_off:
            state = "RISK_OFF"
        elif state == "RISK_ON":
            if not base_on:
                state = "NEUTRAL"
        elif state == "NEUTRAL" and base_on:
            flips = _flip_conditions(z, i)
            if all(flips):
                state = "RISK_ON"

        flips = _flip_conditions(z, i)
        gaps = [f for f in CRITICAL_FIELDS
                if f not in row.index or pd.isna(row[f])]
        data_gap_mult = 0.5 if gaps else 1.0
        vix = row.get("VIX_close")
        if pd.isna(vix):
            vol_mult = 1.0
        elif vix > 45:
            vol_mult = 0.25
        elif vix > 35:
            vol_mult = 0.5
        else:
            vol_mult = 1.0
        veto_value = row.get("monitor_hard_veto", False)
        veto_active = (pd.notna(veto_value) and
                       str(veto_value).strip().lower() in {"true", "1", "yes"})
        monitor_mult = 0.0 if veto_active else 1.0
        cap = STATE_CAP[state]
        budget = cap * vol_mult * data_gap_mult * monitor_mult

        if monitor_mult == 0.0 or state == "RISK_OFF":
            action = "只减仓/对冲"
        elif state == "RISK_ON":
            action = "趋势多"
        else:
            action = "不做"  # Event-confirmed small-long permission is a separate signal-layer decision.

        notes = []
        if (pd.isna(row.get("BBB_spread")) or pd.isna(row.get("HY_spread"))):
            notes.append("BBB/HY数据缺失，CCC>14%的信用风险触发器不可用")
        if gaps:
            notes.append("关键字段缺失，风险预算减半")
        if monitor_mult == 0.0:
            notes.append("监测硬否决生效，禁止新增风险")
        if state == "RISK_OFF":
            notes.append("不生成主动做空方向；保护性对冲须与组合风险合并计算")
        outputs.append(RegimeResult(
            date=row.date.date().isoformat(),
            state=state,
            risk_cap_pct=cap * 100,
            volatility_multiplier=vol_mult,
            data_gap_multiplier=data_gap_mult,
            monitor_veto_multiplier=monitor_mult,
            max_risk_pct=budget * 100,
            flip_conditions="/".join("1" if x else "0" for x in flips),
            flip_count=sum(flips),
            hard_risk_off=hard_off,
            data_gaps=";".join(gaps) if gaps else "无",
            allowed_action=action,
            notes="；".join(notes) if notes else "无",
        ).to_dict())

    return pd.DataFrame(outputs)
