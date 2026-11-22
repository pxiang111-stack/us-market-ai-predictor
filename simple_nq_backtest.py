"""Daily, cash-only QQQ backtest with split-adjusted OHLC and explicit actions.

Execution OHLC is split-adjusted but dividend-unadjusted. Dividends are credited
as cash to positions held through the prior close; split events adjust share counts
and price-based orders. Signals use completed closes and execute next session.
This is a research simulator, not brokerage execution.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from simple_nq_support_resistance import build_support_resistance
from simple_nq_trader_core import classify_regime
from simple_nq_setups import evaluate_setups


DEFAULT_COMMISSION_BPS = 1.0
DEFAULT_SLIPPAGE_BPS = 5.0
RISK_FRACTION = 0.005


def load_bars(path: str | Path) -> list[dict[str, Any]]:
    required = {"date", "open", "high", "low", "close", "volume"}
    bars = []
    with open(path, newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(set(reader.fieldnames)):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            bar = {"date": row["date"]}
            for key in ("open", "high", "low", "close", "volume"):
                bar[key] = float(row[key])
                if not math.isfinite(bar[key]):
                    raise ValueError(f"Non-finite {key} on {bar['date']}")
            bar["dividends"] = float(row.get("dividends") or 0.0)
            bar["stock_splits"] = float(row.get("stock_splits") or 0.0)
            if not math.isfinite(bar["dividends"]) or not math.isfinite(bar["stock_splits"]):
                raise ValueError(f"Non-finite corporate action on {bar['date']}")
            if bar["dividends"] < 0 or bar["stock_splits"] < 0:
                raise ValueError(f"Negative corporate action on {bar['date']}")
            if min(bar["open"], bar["high"], bar["low"], bar["close"]) <= 0:
                raise ValueError(f"Non-positive OHLC value on {bar['date']}")
            if bar["high"] < max(bar["open"], bar["close"], bar["low"]):
                raise ValueError(f"Invalid high on {bar['date']}")
            if bar["low"] > min(bar["open"], bar["close"], bar["high"]):
                raise ValueError(f"Invalid low on {bar['date']}")
            bars.append(bar)
    bars.sort(key=lambda x: x["date"])
    if len(bars) < 260:
        raise ValueError("Need at least 260 daily bars before running the backtest")
    if len({x["date"] for x in bars}) != len(bars):
        raise ValueError("Duplicate dates found in CSV")
    return bars


def _commission(notional: float, commission_bps: float) -> float:
    return notional * commission_bps / 10_000.0


def _slipped(price: float, side: str, slippage_bps: float) -> float:
    factor = slippage_bps / 10_000.0
    return price * (1.0 + factor if side == "buy" else 1.0 - factor)


def _max_drawdown(equity_curve: list[float]) -> float:
    peak = -math.inf
    worst = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            worst = min(worst, value / peak - 1.0)
    return worst


def _metrics(name: str, initial: float, curve: list[float],
             trades: list[dict[str, Any]], exposure: float) -> dict[str, Any]:
    final = curve[-1] if curve else initial
    total_return = final / initial - 1.0
    n = max(1, len(curve) - 1)
    cagr = (final / initial) ** (252.0 / n) - 1.0 if final > 0 else -1.0
    daily_returns = [
        curve[i] / curve[i - 1] - 1.0
        for i in range(1, len(curve)) if curve[i - 1] > 0
    ]
    if len(daily_returns) > 1:
        mean = sum(daily_returns) / len(daily_returns)
        variance = sum((x - mean) ** 2 for x in daily_returns) / (len(daily_returns) - 1)
        sharpe = mean / math.sqrt(variance) * math.sqrt(252.0) if variance > 0 else 0.0
    else:
        sharpe = 0.0
    wins = [t["net_pnl"] for t in trades if t["net_pnl"] > 0]
    losses = [t["net_pnl"] for t in trades if t["net_pnl"] < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {
        "strategy": name,
        "initial_equity": round(initial, 2),
        "final_equity": round(final, 2),
        "total_return_pct": round(total_return * 100, 2),
        "cagr_pct": round(cagr * 100, 2),
        "max_drawdown_pct": round(_max_drawdown(curve) * 100, 2),
        "sharpe_daily_annualized": round(sharpe, 3),
        "trades": len(trades),
        "win_rate_pct": round(100.0 * len(wins) / len(trades), 2) if trades else 0.0,
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss else (None if not gross_profit else "inf"),
        "average_win": round(sum(wins) / len(wins), 2) if wins else 0.0,
        "average_loss": round(sum(losses) / len(losses), 2) if losses else 0.0,
        "exposure_pct": round(exposure * 100, 2),
    }


def _trade_exit(bars: list[dict[str, Any]], start: int, stop: float, target: float,
                slippage_bps: float) -> tuple[int, float, str]:
    """Find exit; gaps through stops fill at open, ambiguous same-bar hits stop first."""
    for j in range(start, len(bars)):
        bar = bars[j]
        if bar["open"] <= stop:
            return j, _slipped(bar["open"], "sell", slippage_bps), "GAP_STOP"
        if bar["open"] >= target:
            return j, _slipped(target, "sell", slippage_bps), "TARGET_GAP"
        hit_stop = bar["low"] <= stop
        hit_target = bar["high"] >= target
        if hit_stop:
            return j, _slipped(stop, "sell", slippage_bps), "STOP" if not hit_target else "STOP_FIRST_AMBIGUOUS"
        if hit_target:
            return j, _slipped(target, "sell", slippage_bps), "TARGET"
    last = bars[-1]
    return len(bars) - 1, _slipped(last["close"], "sell", slippage_bps), "END_OF_DATA"


def _trade_exit_with_actions(
    bars: list[dict[str, Any]], start: int, stop: float, target: float,
    slippage_bps: float, initial_shares: float,
) -> tuple[int, float, str, float, float, list[tuple[int, float, float]]]:
    """Exit a long trade while applying split and dividend events at session open.

    A position entered at this session's open is not entitled to this session's
    dividend. Positions carried from the prior close are entitled on ex-date.
    """
    shares = float(initial_shares)
    dividend_cash = 0.0
    marks: list[tuple[int, float, float]] = []
    for j in range(start, len(bars)):
        bar = bars[j]
        if j > start:
            split = float(bar.get("stock_splits", 0.0))
            if split > 0:
                shares *= split
                stop /= split
                target /= split
            dividend_cash += shares * float(bar.get("dividends", 0.0))
        marks.append((j, shares, dividend_cash))
        if bar["open"] <= stop:
            return j, _slipped(bar["open"], "sell", slippage_bps), "GAP_STOP", shares, dividend_cash, marks
        if bar["open"] >= target:
            return j, _slipped(target, "sell", slippage_bps), "TARGET_GAP", shares, dividend_cash, marks
        hit_stop = bar["low"] <= stop
        hit_target = bar["high"] >= target
        if hit_stop:
            reason = "STOP" if not hit_target else "STOP_FIRST_AMBIGUOUS"
            return j, _slipped(stop, "sell", slippage_bps), reason, shares, dividend_cash, marks
        if hit_target:
            return j, _slipped(target, "sell", slippage_bps), "TARGET", shares, dividend_cash, marks
    last = bars[-1]
    return len(bars) - 1, _slipped(last["close"], "sell", slippage_bps), "END_OF_DATA", shares, dividend_cash, marks


def run_strategy(bars: list[dict[str, Any]], initial_equity: float = 10_000.0,
                 commission_bps: float = DEFAULT_COMMISSION_BPS,
                 slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
                 start_index: int = 205,
                 include_warmup: bool = True) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    if start_index < 205 or start_index >= len(bars) - 1:
        raise ValueError("start_index must be >= 205 and leave at least two bars")
    cash = float(initial_equity)
    curve: list[float] = [float(initial_equity)] * start_index if include_warmup else []
    trades: list[dict[str, Any]] = []
    invested_days = 0
    i = start_index
    while i < len(bars) - 1:
        history = bars[:i + 1]
        regime = classify_regime([x["close"] for x in history])["regime"]
        levels = build_support_resistance(history)
        setup = evaluate_setups(history, regime, levels)
        if setup["action"] != "LONG_CANDIDATE":
            curve.append(cash)
            i += 1
            continue

        zone = setup["zone"]
        atr = float(levels["atr"])
        stop = float(zone["low"]) - 0.25 * atr
        target_zones = [z for z in levels["resistances"] if float(z["low"]) > float(history[-1]["close"])]
        if not target_zones:
            curve.append(cash)
            i += 1
            continue
        target = min(target_zones, key=lambda z: float(z["low"]))["low"]
        next_bar = bars[i + 1]
        entry = _slipped(float(next_bar["open"]), "buy", slippage_bps)
        if not (stop < entry < target) or (target - entry) / (entry - stop) < 1.5:
            curve.append(cash)
            i += 1
            continue
        risk_budget = cash * RISK_FRACTION
        per_share_risk = entry - stop
        by_risk = int(risk_budget // per_share_risk)
        by_cash = int(cash // (entry * (1.0 + commission_bps / 10_000.0)))
        shares = max(0, min(by_risk, by_cash))
        if shares < 1:
            curve.append(cash)
            i += 1
            continue

        curve.append(cash)
        buy_notional = shares * entry
        buy_fee = _commission(buy_notional, commission_bps)
        cash_after_buy = cash - buy_notional - buy_fee
        exit_i, exit_price, reason, exit_shares, dividend_cash, marks = _trade_exit_with_actions(
            bars, i + 1, stop, target, slippage_bps, shares
        )
        sell_notional = exit_shares * exit_price
        sell_fee = _commission(sell_notional, commission_bps)
        cash = cash_after_buy + dividend_cash + sell_notional - sell_fee
        net_pnl = sell_notional - sell_fee - buy_notional - buy_fee + dividend_cash

        mark_by_index = {idx: (qty, divs) for idx, qty, divs in marks}
        for mark_i in range(i + 1, exit_i):
            qty, divs = mark_by_index[mark_i]
            curve.append(cash_after_buy + divs + qty * float(bars[mark_i]["close"]))
        curve.append(cash)
        invested_days += exit_i - i
        trades.append({
            "signal_date": history[-1]["date"],
            "entry_date": next_bar["date"],
            "exit_date": bars[exit_i]["date"],
            "shares": shares,
            "shares_at_exit": round(exit_shares, 6),
            "entry_price": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(float(target), 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": reason,
            "dividend_cash": round(dividend_cash, 4),
            "commission_total": round(buy_fee + sell_fee, 4),
            "net_pnl": round(net_pnl, 4),
            "net_return_pct": round(net_pnl / (buy_notional + buy_fee) * 100, 3),
        })
        i = exit_i + 1

    expected_length = len(bars) if include_warmup else len(bars) - start_index
    while len(curve) < expected_length:
        curve.append(cash)
    curve = curve[:expected_length]
    exposure = invested_days / max(1, len(bars) - start_index)
    return _metrics("SUPPORT_BOUNCE_V1_CORPORATE_ACTIONS", initial_equity, curve, trades, exposure), trades, curve

def run_buy_hold(bars: list[dict[str, Any]], initial_equity: float,
                 commission_bps: float, slippage_bps: float,
                 start_index: int = 0) -> dict[str, Any]:
    period_bars = bars[start_index:]
    if len(period_bars) < 2:
        raise ValueError("Buy-and-hold period must contain at least two bars")
    first = period_bars[0]
    entry = _slipped(first["open"], "buy", slippage_bps)
    shares = float(int(initial_equity // (entry * (1 + commission_bps / 10_000))))
    fee_in = _commission(shares * entry, commission_bps)
    cash = initial_equity - shares * entry - fee_in
    curve = [initial_equity]
    for k in range(1, len(period_bars)):
        bar = period_bars[k]
        split = float(bar.get("stock_splits", 0.0))
        if split > 0:
            shares *= split
        cash += shares * float(bar.get("dividends", 0.0))
        curve.append(cash + shares * bar["close"])
    last = period_bars[-1]
    exit_price = _slipped(last["close"], "sell", slippage_bps)
    fee_out = _commission(shares * exit_price, commission_bps)
    final = cash + shares * exit_price - fee_out
    curve[-1] = final
    return _metrics("BUY_AND_HOLD_TOTAL_RETURN", initial_equity, curve, [{
        "net_pnl": final - initial_equity
    }], 1.0)


def run_sma_baseline(bars: list[dict[str, Any]], initial_equity: float,
                     commission_bps: float, slippage_bps: float,
                     start_index: int = 0) -> dict[str, Any]:
    cash, shares = initial_equity, 0.0
    curve, trades = [], []
    entry_cost = 0.0
    invested_days = 0
    for i, bar in enumerate(bars):
        # Corporate actions occur at the session open; only prior-close holdings
        # receive the ex-date dividend.
        if shares:
            split = float(bar.get("stock_splits", 0.0))
            if split > 0:
                shares *= split
            cash += shares * float(bar.get("dividends", 0.0))
        if i >= max(200, start_index):
            closes = [x["close"] for x in bars[:i]]
            sma50 = sum(closes[-50:]) / 50
            sma200 = sum(closes[-200:]) / 200
            want_long = closes[-1] > sma50 and sma50 > sma200
            if want_long and shares == 0:
                price = _slipped(bar["open"], "buy", slippage_bps)
                qty = int(cash // (price * (1 + commission_bps / 10_000)))
                if qty:
                    fee = _commission(qty * price, commission_bps)
                    entry_cost = qty * price + fee
                    cash -= entry_cost
                    shares = float(qty)
            elif not want_long and shares:
                price = _slipped(bar["open"], "sell", slippage_bps)
                fee = _commission(shares * price, commission_bps)
                proceeds = shares * price - fee
                cash += proceeds
                trades.append({"net_pnl": proceeds - entry_cost})
                shares = 0.0
                entry_cost = 0.0
        if i >= start_index:
            if shares:
                invested_days += 1
            curve.append(cash + shares * bar["close"])
    if shares:
        price = _slipped(bars[-1]["close"], "sell", slippage_bps)
        fee = _commission(shares * price, commission_bps)
        proceeds = shares * price - fee
        cash += proceeds
        trades.append({"net_pnl": proceeds - entry_cost})
        shares = 0.0
        curve[-1] = cash
    return _metrics("SMA50_200_LONG_FLAT_TOTAL_RETURN", initial_equity, curve, trades,
                    invested_days / max(1, len(bars) - start_index))


def run_backtest(csv_path: str | Path, output_dir: str | Path = "output",
                 initial_equity: float = 10_000.0,
                 commission_bps: float = DEFAULT_COMMISSION_BPS,
                 slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> dict[str, Any]:
    bars = load_bars(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    strategy, trades, _ = run_strategy(
        bars, initial_equity, commission_bps, slippage_bps
    )
    buy_hold = run_buy_hold(bars, initial_equity, commission_bps, slippage_bps)
    sma = run_sma_baseline(bars, initial_equity, commission_bps, slippage_bps)
    cash = _metrics("CASH_NO_TRADE", initial_equity,
                    [initial_equity] * len(bars), [], 0.0)

    # Keep the last 20% untouched. Indicators see earlier history, but every
    # holdout strategy starts with fresh cash at the split boundary.
    split = max(205, int(len(bars) * 0.8))
    holdout_strategy, holdout_trades, _ = run_strategy(
        bars, initial_equity, commission_bps, slippage_bps,
        start_index=split, include_warmup=False,
    )
    holdout_buy_hold = run_buy_hold(
        bars, initial_equity, commission_bps, slippage_bps, start_index=split
    )
    holdout_sma = run_sma_baseline(
        bars, initial_equity, commission_bps, slippage_bps, start_index=split
    )
    holdout_cash = _metrics(
        "CASH_NO_TRADE", initial_equity,
        [initial_equity] * (len(bars) - split), [], 0.0,
    )
    summary = {
        "data_source": "Yahoo Finance via yfinance; auto_adjust=False; split-adjusted OHLC plus separate corporate actions",
        "data_file": str(csv_path),
        "first_date": bars[0]["date"],
        "last_date": bars[-1]["date"],
        "bars": len(bars),
        "initial_equity": initial_equity,
        "commission_bps_each_side": commission_bps,
        "slippage_bps_each_side": slippage_bps,
        "price_basis": "Split-adjusted, dividend-unadjusted OHLC; dividends credited as cash; splits adjust held share counts and price orders",
        "full_sample_strategies": [strategy, buy_hold, sma, cash],
        "holdout": {
            "start_date": bars[split]["date"],
            "end_date": bars[-1]["date"],
            "bars": len(bars) - split,
            "warmup_bars_available": split,
            "fresh_starting_equity": initial_equity,
            "strategies": [holdout_strategy, holdout_buy_hold, holdout_sma, holdout_cash],
            "trade_count": len(holdout_trades),
        },
        "limitations": [
            "Final 20% is a single frozen holdout, not repeated walk-forward validation.",
            "Daily OHLC cannot reveal intraday ordering; same-day stop and target hits assume stop first.",
            "Historical Yahoo Finance data may be revised; preserve the downloaded CSV with this report.",
            "Yahoo historical OHLC is split-adjusted and dividend-unadjusted; corporate actions are handled as cash dividends and split share/order adjustments. Provider data can be revised.",
        ],
        "note": "Full-sample results are diagnostic only. Holdout rules are unchanged and use fresh capital, but a single holdout does not guarantee future performance.",
    }
    (out / "simple_nq_backtest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    trade_fields = [
        "signal_date", "entry_date", "exit_date", "shares", "entry_price",
        "stop", "target", "exit_price", "exit_reason", "commission_total",
        "net_pnl", "net_return_pct",
    ]
    with (out / "simple_nq_backtest_trades.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=trade_fields)
        writer.writeheader()
        writer.writerows(trades)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Simple NQ / QQQ V1 daily backtest with corporate actions"
    )
    parser.add_argument("--csv", required=True, help="Split-adjusted, dividend-unadjusted daily OHLCV with dividends and stock_splits columns")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--commission-bps", type=float, default=DEFAULT_COMMISSION_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    args = parser.parse_args()
    result = run_backtest(
        args.csv, args.output_dir, args.capital,
        args.commission_bps, args.slippage_bps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
