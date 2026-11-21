"""Daily, cash-only QQQ backtest for Simple NQ Trader V1.

Input CSV must contain date, open, high, low, close, volume. OHLC prices must be
consistently split/dividend-adjusted (or consistently raw); do not mix bases.
Signals are evaluated after a completed close and orders execute no earlier than
the following session's open. This is a research simulator, not brokerage execution.
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


def run_strategy(bars: list[dict[str, Any]], initial_equity: float = 10_000.0,
                 commission_bps: float = DEFAULT_COMMISSION_BPS,
                 slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> tuple[dict[str, Any], list[dict[str, Any]], list[float]]:
    cash = float(initial_equity)
    curve: list[float] = [float(initial_equity)] * 205
    trades: list[dict[str, Any]] = []
    invested_days = 0
    i = 205
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
        if not (stop < entry < target):
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

        # Signal is known at today's close; entry is next session's open.
        curve.append(cash)
        buy_notional = shares * entry
        buy_fee = _commission(buy_notional, commission_bps)
        cash_after_buy = cash - buy_notional - buy_fee
        exit_i, exit_price, reason = _trade_exit(
            bars, i + 1, stop, target, slippage_bps
        )
        sell_notional = shares * exit_price
        sell_fee = _commission(sell_notional, commission_bps)
        cash = cash_after_buy + sell_notional - sell_fee
        net_pnl = sell_notional - sell_fee - buy_notional - buy_fee

        # Mark open risk daily, then record realized equity on the exit date.
        for mark_i in range(i + 1, exit_i):
            curve.append(cash_after_buy + shares * float(bars[mark_i]["close"]))
        curve.append(cash)
        invested_days += exit_i - i
        trades.append({
            "signal_date": history[-1]["date"],
            "entry_date": next_bar["date"],
            "exit_date": bars[exit_i]["date"],
            "shares": shares,
            "entry_price": round(entry, 4),
            "stop": round(stop, 4),
            "target": round(float(target), 4),
            "exit_price": round(exit_price, 4),
            "exit_reason": reason,
            "commission_total": round(buy_fee + sell_fee, 4),
            "net_pnl": round(net_pnl, 4),
            "net_return_pct": round(net_pnl / (buy_notional + buy_fee) * 100, 3),
        })
        i = exit_i + 1

    while len(curve) < len(bars):
        curve.append(cash)
    curve = curve[:len(bars)]
    exposure = invested_days / max(1, len(bars))
    return _metrics("SUPPORT_BOUNCE_V1", initial_equity, curve, trades, exposure), trades, curve

def run_buy_hold(bars: list[dict[str, Any]], initial_equity: float,
                 commission_bps: float, slippage_bps: float) -> dict[str, Any]:
    first = bars[0]
    entry = _slipped(first["open"], "buy", slippage_bps)
    shares = int(initial_equity // (entry * (1 + commission_bps / 10_000)))
    fee_in = _commission(shares * entry, commission_bps)
    last = bars[-1]
    exit_price = _slipped(last["close"], "sell", slippage_bps)
    fee_out = _commission(shares * exit_price, commission_bps)
    final = initial_equity - shares * entry - fee_in + shares * exit_price - fee_out
    curve = [initial_equity]
    for k in range(1, len(bars)):
        marked = initial_equity - shares * entry - fee_in + shares * bars[k]["close"]
        curve.append(marked)
    curve[-1] = final
    return _metrics("BUY_AND_HOLD", initial_equity, curve, [{
        "net_pnl": final - initial_equity
    }], 1.0)


def run_sma_baseline(bars: list[dict[str, Any]], initial_equity: float,
                     commission_bps: float, slippage_bps: float) -> dict[str, Any]:
    cash, shares = initial_equity, 0
    curve, trades = [], []
    for i, bar in enumerate(bars):
        if i >= 200:
            closes = [x["close"] for x in bars[:i]]
            want_long = closes[-1] > sum(closes[-50:]) / 50 and sum(closes[-50:]) / 50 > sum(closes[-200:]) / 200
            if want_long and shares == 0:
                price = _slipped(bar["open"], "buy", slippage_bps)
                qty = int(cash // (price * (1 + commission_bps / 10_000)))
                if qty:
                    fee = _commission(qty * price, commission_bps)
                    cash -= qty * price + fee
                    shares = qty
            elif not want_long and shares:
                price = _slipped(bar["open"], "sell", slippage_bps)
                fee = _commission(shares * price, commission_bps)
                cash += shares * price - fee
                trades.append({"net_pnl": 0.0})
                shares = 0
        curve.append(cash + shares * bar["close"])
    if shares:
        price = _slipped(bars[-1]["close"], "sell", slippage_bps)
        fee = _commission(shares * price, commission_bps)
        cash += shares * price - fee
        shares = 0
        curve[-1] = cash
    return _metrics("SMA50_200_LONG_FLAT", initial_equity, curve, trades, 0.5)


def run_backtest(csv_path: str | Path, output_dir: str | Path = "output",
                 initial_equity: float = 10_000.0,
                 commission_bps: float = DEFAULT_COMMISSION_BPS,
                 slippage_bps: float = DEFAULT_SLIPPAGE_BPS) -> dict[str, Any]:
    bars = load_bars(csv_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    strategy, trades, curve = run_strategy(bars, initial_equity, commission_bps, slippage_bps)
    benchmarks = [
        run_buy_hold(bars, initial_equity, commission_bps, slippage_bps),
        run_sma_baseline(bars, initial_equity, commission_bps, slippage_bps),
    ]
    summary = {
        "data_file": str(csv_path),
        "first_date": bars[0]["date"],
        "last_date": bars[-1]["date"],
        "bars": len(bars),
        "initial_equity": initial_equity,
        "commission_bps_each_side": commission_bps,
        "slippage_bps_each_side": slippage_bps,
        "price_basis_warning": "Verify OHLC is consistently adjusted; mixed adjusted/raw prices invalidate results.",
        "strategies": [strategy, *benchmarks],
        "note": "Daily OHLC cannot reveal intraday ordering; if stop and target both touch, stop is assumed first.",
    }
    (out / "simple_nq_backtest_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    with (out / "simple_nq_backtest_trades.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(trades[0]) if trades else [
            "signal_date", "entry_date", "exit_date", "shares", "entry_price",
            "stop", "target", "exit_price", "exit_reason", "commission_total",
            "net_pnl", "net_return_pct",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(trades)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Simple NQ / QQQ V1 daily backtest")
    parser.add_argument("--csv", required=True, help="Adjusted daily OHLCV CSV")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--commission-bps", type=float, default=DEFAULT_COMMISSION_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    args = parser.parse_args()
    print(json.dumps(run_backtest(
        args.csv, args.output_dir, args.capital, args.commission_bps, args.slippage_bps
    ), indent=2))


if __name__ == "__main__":
    main()
