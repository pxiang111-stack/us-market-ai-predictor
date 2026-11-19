"""Conservative next-session OHLC execution simulator for timestamped research signals.

Signals must be generated using information available at the signal close.
Entries occur at the following session open. No same-close fills are allowed.
This module does not train models or create historical predictions.
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
    """Simulate one position at a time; signals indexed by decision date.

    Required bars: Open, High, Low, Close. Required signals: decision, stop_loss,
    take_profit_1r, take_profit_2r. Signal at t enters at t+1 open. If stop and
    target both touch in one bar, stop wins. A stop gap fills at the adverse open.
    Target fills conservatively at its limit. Time exit is at session close.
    """
    required_bars = {"Open", "High", "Low", "Close"}
    required_signals = {"decision", "stop_loss", "take_profit_1r", "take_profit_2r"}
    if not required_bars.issubset(bars.columns):
        raise ValueError(f"bars missing: {sorted(required_bars - set(bars.columns))}")
    if not required_signals.issubset(signals.columns):
        raise ValueError(f"signals missing: {sorted(required_signals - set(signals.columns))}")
    if initial_equity <= 0 or max_holding_sessions < 1:
        raise ValueError("initial_equity must be positive and max_holding_sessions >= 1")
    b = bars.sort_index().copy()
    s = signals.sort_index().copy()
    b = b[~b.index.duplicated(keep="last")]
    s = s[~s.index.duplicated(keep="last")]
    dates = list(b.index)
    date_pos = {d: i for i, d in enumerate(dates)}
    equity = float(initial_equity)
    ledger, curve = [], [{"date": dates[0] if dates else None, "equity": equity}]
    occupied_until = -1
    total_cost_rate = (round_trip_cost_bps + 2 * slippage_bps_per_side) / 10000.0

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
        stop = float(sig["stop_loss"])
        target = float(sig["take_profit_1r"])
        if not all(np.isfinite(x) for x in (entry, stop, target)) or entry <= 0:
            continue
        direction = 1 if side == "LONG" else -1
        risk_per_unit = abs(entry - stop)
        if risk_per_unit <= 0 or (direction == 1 and not stop < entry < target) or (direction == -1 and not target < entry < stop):
            continue
        risk_cash = equity * risk_budget_pct
        units = min(risk_cash / risk_per_unit, equity * max_gross_exposure / entry)
        if units <= 0:
            continue
        exit_i, exit_price, exit_reason = None, None, None
        last_i = min(len(dates) - 1, entry_i + max_holding_sessions - 1)
        for j in range(entry_i, last_i + 1):
            bar = b.iloc[j]
            o, h, l, c = map(float, (bar["Open"], bar["High"], bar["Low"], bar["Close"]))
            if direction == 1:
                if o <= stop:
                    exit_i, exit_price, exit_reason = j, o, "gap_stop"
                elif l <= stop:
                    exit_i, exit_price, exit_reason = j, stop, "stop"
                elif h >= target:
                    exit_i, exit_price, exit_reason = j, target, "target_1r"
            else:
                if o >= stop:
                    exit_i, exit_price, exit_reason = j, o, "gap_stop"
                elif h >= stop:
                    exit_i, exit_price, exit_reason = j, stop, "stop"
                elif l <= target:
                    exit_i, exit_price, exit_reason = j, target, "target_1r"
            if exit_i is not None:
                break
        if exit_i is None:
            exit_i = last_i
            exit_price = float(b.iloc[last_i]["Close"])
            exit_reason = "time_exit"
        gross_return = direction * (exit_price - entry) / entry
        gross_pnl = units * direction * (exit_price - entry)
        costs = units * (entry + exit_price) * (round_trip_cost_bps / 20000.0 + slippage_bps_per_side / 10000.0)
        net_pnl = gross_pnl - costs
        start_equity = equity
        equity += net_pnl
        occupied_until = exit_i
        ledger.append({
            "signal_date": signal_date, "entry_date": dates[entry_i], "exit_date": dates[exit_i],
            "side": side, "entry_price": entry, "exit_price": exit_price,
            "exit_reason": exit_reason, "units": units, "gross_return_pct": gross_return * 100,
            "gross_pnl": gross_pnl, "estimated_costs": costs, "net_pnl": net_pnl,
            "net_return_on_equity_pct": net_pnl / start_equity * 100,
            "equity_before": start_equity, "equity_after": equity,
        })
        curve.append({"date": dates[exit_i], "equity": equity})
    ledger_df = pd.DataFrame(ledger)
    # Build a complete daily mark-to-market curve, not merely a list of exit dates.
    daily = []
    for d in dates:
        realized = initial_equity
        active_value = 0.0
        for trade in ledger:
            if trade["exit_date"] < d:
                realized += trade["net_pnl"]
            elif trade["entry_date"] <= d <= trade["exit_date"]:
                mark = trade["exit_price"] if d == trade["exit_date"] else float(b.loc[d, "Close"])
                direction = 1 if trade["side"] == "LONG" else -1
                gross = trade["units"] * direction * (mark - trade["entry_price"])
                est_cost = trade["units"] * (trade["entry_price"] + mark) * (
                    round_trip_cost_bps / 20000.0 + slippage_bps_per_side / 10000.0
                )
                active_value = gross - est_cost
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
