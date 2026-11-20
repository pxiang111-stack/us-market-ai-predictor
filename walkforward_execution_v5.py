"""Conservative next-session OHLC execution simulator with two-stage exits.

Signals are generated after close t and enter at t+1 open. Half the position
exits at 1R and the remainder at 2R. If a stop and any target are touched in
the same OHLC bar, the stop is assumed to occur first. This is a daily-bar
research approximation, not an order-book fill simulator.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def simulate_trades(
    bars: pd.DataFrame,
    signals: pd.DataFrame,
    *,
    initial_equity: float = 100_000.0,
    risk_budget_pct: float = 0.005,
    max_gross_exposure: float = 1.0,
    round_trip_cost_bps: float = 15.0,
    slippage_bps_per_side: float = 5.0,
    max_holding_sessions: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate one position at a time; signals indexed by decision date."""
    required_bars = {"Open", "High", "Low", "Close"}
    required_signals = {"decision", "stop_loss", "take_profit_1r", "take_profit_2r"}
    if not required_bars.issubset(bars.columns):
        raise ValueError(f"bars missing: {sorted(required_bars - set(bars.columns))}")
    if not required_signals.issubset(signals.columns):
        raise ValueError(f"signals missing: {sorted(required_signals - set(signals.columns))}")
    if initial_equity <= 0 or max_holding_sessions < 1:
        raise ValueError("initial_equity must be positive and max_holding_sessions >= 1")
    if not 0 <= risk_budget_pct <= 1 or max_gross_exposure <= 0:
        raise ValueError("risk budget must be in [0,1] and gross exposure must be positive")
    if round_trip_cost_bps < 0 or slippage_bps_per_side < 0:
        raise ValueError("cost assumptions cannot be negative")

    b = bars.sort_index().copy()
    s = signals.sort_index().copy()
    b = b[~b.index.duplicated(keep="last")]
    s = s[~s.index.duplicated(keep="last")]
    dates = list(b.index)
    date_pos = {d: i for i, d in enumerate(dates)}
    equity = float(initial_equity)
    ledger = []
    occupied_until = -1
    cost_rate = round_trip_cost_bps / 20000.0 + slippage_bps_per_side / 10000.0

    for signal_date, sig in s.iterrows():
        if signal_date not in date_pos:
            continue
        i = date_pos[signal_date]
        entry_i = i + 1
        if entry_i >= len(dates) or entry_i <= occupied_until:
            continue
        side = str(sig["decision"]).upper()
        if side not in ("LONG", "SHORT"):
            continue
        entry_bar = b.iloc[entry_i]
        entry = float(entry_bar["Open"])
        stop, target1, target2 = map(float, (sig["stop_loss"], sig["take_profit_1r"], sig["take_profit_2r"]))
        if not all(np.isfinite(v) for v in (entry, stop, target1, target2)) or entry <= 0:
            continue
        direction = 1 if side == "LONG" else -1
        # The signal's stop is an absolute invalidation price from close t, but
        # the fill occurs at open t+1. Re-anchor 1R/2R to the actual fill so
        # target distances remain consistent with the risk actually sized.
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0:
            continue
        target1 = entry + direction * risk_per_unit
        target2 = entry + direction * 2.0 * risk_per_unit
        valid = (stop < entry < target1 < target2) if direction == 1 else (target2 < target1 < entry < stop)
        if not valid:
            continue

        risk_cash = equity * risk_budget_pct
        units = min(risk_cash / risk_per_unit, equity * max_gross_exposure / entry)
        if units <= 0:
            continue
        half_units = units * 0.5
        partial_date = partial_price = partial_cost = None
        active_stop = stop
        exit_i = exit_price = exit_reason = None
        last_i = min(len(dates) - 1, entry_i + max_holding_sessions - 1)

        for j in range(entry_i, last_i + 1):
            bar = b.iloc[j]
            o, h, l = map(float, (bar["Open"], bar["High"], bar["Low"]))
            if direction == 1:
                stop_gap, stop_hit = o <= active_stop, l <= active_stop
                target1_hit, target2_hit = h >= target1, h >= target2
            else:
                stop_gap, stop_hit = o >= active_stop, h >= active_stop
                target1_hit, target2_hit = l <= target1, l <= target2

            # Stop-first convention resolves ambiguous OHLC ordering conservatively.
            if stop_gap or stop_hit:
                exit_i, exit_price = j, (o if stop_gap else active_stop)
                exit_reason = "gap_stop" if stop_gap else ("stop_after_1r" if partial_date is not None else "stop")
                break

            if partial_date is None and target1_hit:
                partial_date, partial_price = dates[j], target1
                # After banking the first half at +1R, protect the remainder
                # at breakeven so a later reversal cannot erase the booked gain.
                active_stop = entry
                partial_cost = half_units * (entry + target1) * cost_rate
                if target2_hit:
                    exit_i, exit_price, exit_reason = j, target2, "target_2r"
                    break
            elif partial_date is not None and target2_hit:
                exit_i, exit_price, exit_reason = j, target2, "target_2r"
                break

        if exit_i is None:
            exit_i = last_i
            exit_price = float(b.iloc[last_i]["Close"])
            exit_reason = "time_exit_after_1r" if partial_date is not None else "time_exit"

        if partial_date is None:
            partial_units, remaining_units = 0.0, units
            partial_gross, partial_cost = 0.0, 0.0
            partial_exit_i = None
        else:
            partial_units, remaining_units = half_units, half_units
            partial_gross = partial_units * direction * (partial_price - entry)
            partial_exit_i = date_pos[partial_date]

        remaining_gross = remaining_units * direction * (exit_price - entry)
        remaining_cost = remaining_units * (entry + exit_price) * cost_rate
        gross_pnl = partial_gross + remaining_gross
        costs = partial_cost + remaining_cost
        net_pnl = gross_pnl - costs
        start_equity = equity
        equity += net_pnl
        occupied_until = exit_i

        ledger.append({
            "signal_date": signal_date, "entry_date": dates[entry_i], "exit_date": dates[exit_i],
            "side": side, "entry_price": entry, "exit_price": exit_price,
            "risk_per_unit_at_fill": risk_per_unit, "target_1r_at_fill": target1,
            "target_2r_at_fill": target2,
            "exit_reason": exit_reason, "units": units,
            "partial_exit_date": partial_date, "partial_exit_price": partial_price,
            "partial_units": partial_units, "partial_estimated_costs": partial_cost,
            "remaining_units": remaining_units, "remaining_estimated_costs": remaining_cost,
            "gross_return_pct": (gross_pnl / (units * entry)) * 100,
            "gross_pnl": gross_pnl, "estimated_costs": costs, "net_pnl": net_pnl,
            "net_return_on_equity_pct": net_pnl / start_equity * 100,
            "equity_before": start_equity, "equity_after": equity,
        })

    ledger_df = pd.DataFrame(ledger)
    daily = []
    for d in dates:
        realized = initial_equity
        active_value = 0.0
        for trade in ledger:
            if trade["exit_date"] < d:
                realized += trade["net_pnl"]
                continue
            if trade["entry_date"] <= d <= trade["exit_date"]:
                mark = trade["exit_price"] if d == trade["exit_date"] else float(b.loc[d, "Close"])
                direction = 1 if trade["side"] == "LONG" else -1
                if pd.notna(trade["partial_exit_date"]) and trade["partial_exit_date"] <= d:
                    partial_pnl = trade["partial_units"] * direction * (
                        trade["partial_exit_price"] - trade["entry_price"]
                    ) - trade["partial_estimated_costs"]
                    remaining_cost = trade["remaining_units"] * (
                        trade["entry_price"] + mark
                    ) * cost_rate
                    remaining_pnl = trade["remaining_units"] * direction * (
                        mark - trade["entry_price"]
                    ) - remaining_cost
                    active_value = partial_pnl + remaining_pnl
                else:
                    est_cost = trade["units"] * (trade["entry_price"] + mark) * cost_rate
                    active_value = trade["units"] * direction * (mark - trade["entry_price"]) - est_cost
                break
        daily.append({"date": d, "equity": realized + active_value})

    curve_df = pd.DataFrame(daily)
    if not curve_df.empty:
        curve_df["daily_return"] = curve_df["equity"].pct_change().fillna(0.0)
        curve_df["peak_equity"] = curve_df["equity"].cummax()
        curve_df["drawdown_pct"] = curve_df["equity"] / curve_df["peak_equity"] - 1.0
    return ledger_df, curve_df


def performance_summary(ledger: pd.DataFrame, curve: pd.DataFrame) -> dict:
    """Calculate basic portfolio and trade statistics from a daily equity curve."""
    if curve is None or curve.empty or "equity" not in curve:
        return {"status": "no_equity_curve", "trade_count": 0 if ledger is None else len(ledger)}
    values = pd.to_numeric(curve["equity"], errors="coerce").dropna()
    if len(values) < 2 or (values <= 0).any():
        return {"status": "insufficient_or_nonpositive_equity", "trade_count": 0 if ledger is None else len(ledger)}
    daily_ret = values.pct_change().dropna()
    years = max((len(values) - 1) / 252.0, 1 / 252.0)
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1
    vol = daily_ret.std(ddof=1) if len(daily_ret) > 1 else np.nan
    sharpe = (daily_ret.mean() / vol * np.sqrt(252)) if pd.notna(vol) and vol > 0 else np.nan
    downside = daily_ret[daily_ret < 0].std(ddof=1)
    sortino = (daily_ret.mean() / downside * np.sqrt(252)) if pd.notna(downside) and downside > 0 else np.nan
    drawdown = values / values.cummax() - 1
    trades = ledger if ledger is not None else pd.DataFrame()
    pnl = pd.to_numeric(trades.get("net_pnl", pd.Series(dtype=float)), errors="coerce").dropna()
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    gross_profit, gross_loss = wins.sum(), abs(losses.sum())
    return {
        "status": "ok", "trade_count": int(len(pnl)),
        "total_return_pct": float((values.iloc[-1] / values.iloc[0] - 1) * 100),
        "cagr_pct": float(cagr * 100), "max_drawdown_pct": float(drawdown.min() * 100),
        "sharpe_daily": float(sharpe) if pd.notna(sharpe) else None,
        "sortino_daily": float(sortino) if pd.notna(sortino) else None,
        "win_rate_pct": float((pnl > 0).mean() * 100) if len(pnl) else None,
        "profit_factor": float(gross_profit / gross_loss) if gross_loss > 0 else (None if gross_profit == 0 else float("inf")),
        "expectancy_per_trade": float(pnl.mean()) if len(pnl) else None,
        "average_win": float(wins.mean()) if len(wins) else None,
        "average_loss": float(losses.mean()) if len(losses) else None,
    }
