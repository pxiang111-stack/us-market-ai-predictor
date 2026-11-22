#!/usr/bin/env python3
"""FlowFirst v0.1 research backtest: no look-ahead, next-bar-open entries.

Four variants: price-only, flow-only, price+flow, price+flow+OI participation filter.
Each symbol/variant starts with $10,000; max gross exposure is 1x equity.
This is a diagnostic backtest, not proof of future profitability.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/flowfirst")
OUT = Path("reports")
UNIVERSE = ["QQQUSDT", "NVDAUSDT", "SMHUSDT", "SOXLUSDT", "SPYUSDT"]
START_CASH = 10_000.0
RISK_FRACTION = 0.005
STOP_ATR = 1.5
TARGET_ATR = 3.0
MAX_HOLD_BARS = 16
FEE_PER_SIDE = 0.0005
SLIPPAGE_PER_SIDE = 0.0002
COST_PER_SIDE = FEE_PER_SIDE + SLIPPAGE_PER_SIDE


def read_kind(kind: str, period: str) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for path in sorted(ROOT.glob(f"*_{kind}_{period}.jsonl")):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                    result.setdefault(row["symbol"], []).append(row)
                except Exception:
                    continue
    for rows in result.values():
        rows.sort(key=lambda r: int(r["timestamp"]))
    return result


def ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    value = sum(values[:period]) / period
    alpha = 2.0 / (period + 1.0)
    for x in values[period:]:
        value = alpha * x + (1.0 - alpha) * value
    return value


def trend(rows: list[dict]) -> str:
    closes = [float(x["close"]) for x in rows]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    if e20 is None or e50 is None:
        return "unknown"
    if closes[-1] > e20 > e50:
        return "up"
    if closes[-1] < e20 < e50:
        return "down"
    return "mixed"


def atr(rows: list[dict], period: int = 14) -> float | None:
    if len(rows) < period + 1:
        return None
    tr = []
    for i in range(1, len(rows)):
        h, l, prev = float(rows[i]["high"]), float(rows[i]["low"]), float(rows[i-1]["close"])
        tr.append(max(h-l, abs(h-prev), abs(l-prev)))
    return sum(tr[-period:]) / period if len(tr) >= period else None


def flow_direction(rows: list[dict]) -> tuple[int, float, float] | None:
    ratios = []
    for x in rows:
        buy, sell = float(x["buyVol"]), float(x["sellVol"])
        ratios.append((buy-sell)/(buy+sell) if buy+sell else 0.0)
    if len(ratios) < 52:
        return None
    recent = sum(ratios[-4:]) / 4
    base = ratios[-52:-4]
    mean = sum(base) / len(base)
    var = sum((x-mean)**2 for x in base) / len(base)
    sd = math.sqrt(var)
    z = (recent-mean)/sd if sd > 1e-9 else 0.0
    if recent >= 0.06 and z >= 0.75:
        return 1, recent, z
    if recent <= -0.06 and z <= -0.75:
        return -1, recent, z
    return 0, recent, z


def crowding_state(rows: list[dict]) -> dict | None:
    if len(rows) < 49:
        return None
    rows = sorted(rows, key=lambda r: int(r["timestamp"]))
    long_now = float(rows[-1]["longAccount"])
    short_now = float(rows[-1]["shortAccount"])
    long_base = [float(x["longAccount"]) for x in rows[-49:-1]]
    short_base = [float(x["shortAccount"]) for x in rows[-49:-1]]
    def zscore(current: float, baseline: list[float]) -> float:
        mean = sum(baseline) / len(baseline)
        variance = sum((x-mean)**2 for x in baseline) / len(baseline)
        sd = math.sqrt(variance)
        return (current-mean)/sd if sd > 1e-9 else 0.0
    long_z, short_z = zscore(long_now, long_base), zscore(short_now, short_base)
    return {
        "crowded_long": long_now >= 0.80 and long_z >= 1.0,
        "crowded_short": short_now >= 0.80 and short_z >= 1.0,
        "long_account": long_now,
        "long_z": long_z,
        "short_account": short_now,
        "short_z": short_z
    }


def holding_window_end(signal_index: int, total_bars: int) -> int:
    """Last eligible bar index for a trade entered on signal_index + 1."""
    return min(total_bars - 1, signal_index + MAX_HOLD_BARS)


def oi_expanding(rows: list[dict]) -> bool:
    if len(rows) < 5:
        return False
    old = float(rows[-5]["sumOpenInterestValue"])
    new = float(rows[-1]["sumOpenInterestValue"])
    return old > 0 and (new / old - 1.0) >= 0.001


def metrics(trades: list[dict], equity_curve: list[float]) -> dict:
    pnl = [x["net_pnl"] for x in trades]
    wins = [x for x in pnl if x > 0]
    losses = [x for x in pnl if x < 0]
    peak, max_dd = START_CASH, 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak > 0:
            max_dd = max(max_dd, (peak-value)/peak)
    gross_win, gross_loss = sum(wins), abs(sum(losses))
    return {
        "trades": len(trades),
        "win_rate_pct": round(100*len(wins)/len(trades), 2) if trades else None,
        "net_pnl_usd": round(sum(pnl), 2),
        "return_pct": round(100*(equity_curve[-1]/START_CASH-1), 2) if equity_curve else 0.0,
        "profit_factor": round(gross_win/gross_loss, 3) if gross_loss else (None if not gross_win else "inf"),
        "expectancy_usd_per_trade": round(sum(pnl)/len(pnl), 2) if pnl else None,
        "max_drawdown_pct": round(max_dd*100, 2),
        "avg_r_multiple": round(sum(x["r_multiple"] for x in trades)/len(trades), 3) if trades else None,
        "avg_mae_r": round(sum(x["mae_r"] for x in trades)/len(trades), 3) if trades else None,
        "avg_mfe_r": round(sum(x["mfe_r"] for x in trades)/len(trades), 3) if trades else None,
        "cost_per_side_bps": round(COST_PER_SIDE*10000, 1),
    }


def run_variant(symbol: str, bars: list[dict], hourly: list[dict],
                flow_rows: list[dict], oi_rows: list[dict],
                longshort_rows: list[dict], variant: str,
                trade_start_ts: int | None = None,
                trade_end_ts: int | None = None) -> dict:
    if len(bars) < 80 or (variant != "flow_only" and len(hourly) < 55):
        return {"status": "INSUFFICIENT_DATA", "bars_15m": len(bars), "bars_1h": len(hourly)}
    flow_by_ts = {int(x["timestamp"]): x for x in flow_rows}
    oi_by_ts = {int(x["timestamp"]): x for x in oi_rows}
    hourly_close = [(int(x["close_time"]), x) for x in hourly if "close_time" in x]
    equity = START_CASH
    trades: list[dict] = []
    curve = [equity]
    i = 60
    while i < len(bars)-1:
        bar = bars[i]
        ts = int(bar["timestamp"])
        if trade_start_ts is not None and ts < trade_start_ts:
            i += 1
            continue
        if trade_end_ts is not None and ts >= trade_end_ts:
            break
        # Only hourly bars fully closed by this 15m signal bar are visible.
        hrows = [x for close_time, x in hourly_close if close_time <= int(bar.get("close_time", ts+899999))] if variant != "flow_only" else []
        if variant != "flow_only" and len(hrows) < 55:
            i += 1
            continue
        t15 = trend(bars[:i+1]) if variant != "flow_only" else "unknown"
        t1h = trend(hrows) if variant != "flow_only" else "unknown"
        flow_hist = [x for x in flow_rows if int(x["timestamp"]) <= ts]
        flow_aligned = bool(flow_hist) and int(flow_hist[-1]["timestamp"]) == ts
        f = flow_direction(flow_hist) if flow_aligned else None
        oi_hist = [x for x in oi_rows if int(x["timestamp"]) <= ts]
        oi_aligned = bool(oi_hist) and int(oi_hist[-1]["timestamp"]) == ts
        longshort_hist = [x for x in longshort_rows if int(x["timestamp"]) <= ts]
        a = atr(bars[:i+1])
        if a is None or a <= 0:
            i += 1
            continue
        direction = 0
        if variant == "price_only":
            if t15 == "up" and t1h == "up":
                direction = 1
            elif t15 == "down" and t1h == "down":
                direction = -1
        elif variant == "flow_only":
            direction = f[0] if f else 0
        else:
            if f and t15 == "up" and t1h == "up" and f[0] == 1:
                direction = 1
            elif f and t15 == "down" and t1h == "down" and f[0] == -1:
                direction = -1
            if variant == "combo_oi" and direction and (not oi_aligned or not oi_expanding(oi_hist)):
                direction = 0
            if variant == "combo_crowding" and direction:
                crowd = crowding_state(longshort_hist)
                aligned = bool(longshort_hist) and int(longshort_hist[-1]["timestamp"]) == ts
                if not aligned or crowd is None:
                    direction = 0
                elif direction == 1 and crowd["crowded_long"]:
                    direction = 0
                elif direction == -1 and crowd["crowded_short"]:
                    direction = 0
        if direction == 0:
            i += 1
            continue

        entry_bar = bars[i+1]
        if trade_end_ts is not None and int(entry_bar["timestamp"]) >= trade_end_ts:
            break
        entry = float(entry_bar["open"])
        stop_dist = STOP_ATR*a
        risk_budget = equity*RISK_FRACTION
        qty = min(risk_budget/stop_dist, equity/entry) if entry > 0 else 0.0
        if qty <= 0:
            i += 1
            continue
        stop = entry-direction*stop_dist
        target = entry+direction*TARGET_ATR*a
        exit_price, exit_index, exit_reason = None, None, "time"
        last_j = holding_window_end(i, len(bars))
        if trade_end_ts is not None:
            while last_j > i+1 and int(bars[last_j]["timestamp"]) >= trade_end_ts:
                last_j -= 1
        max_adverse, max_favorable = 0.0, 0.0
        for j in range(i+1, last_j+1):
            high, low = float(bars[j]["high"]), float(bars[j]["low"])
            if direction == 1:
                max_adverse = max(max_adverse, entry-low)
                max_favorable = max(max_favorable, high-entry)
                stop_hit, target_hit = low <= stop, high >= target
            else:
                max_adverse = max(max_adverse, high-entry)
                max_favorable = max(max_favorable, entry-low)
                stop_hit, target_hit = high >= stop, low <= target
            # Conservative tie-break: if both levels trade in one bar, assume stop first.
            if stop_hit:
                exit_price, exit_index, exit_reason = stop, j, "stop"
                break
            if target_hit:
                exit_price, exit_index, exit_reason = target, j, "target"
                break
        if exit_price is None:
            exit_index = last_j
            exit_price = float(bars[last_j]["close"])
        gross = qty*direction*(exit_price-entry)
        costs = qty*(entry+exit_price)*COST_PER_SIDE
        net = gross-costs
        risk_dollars = qty*stop_dist
        equity += net
        trades.append({
            "entry_ts": int(entry_bar["timestamp"]), "exit_ts": int(bars[exit_index]["timestamp"]),
            "direction": direction, "entry": round(entry, 5), "exit": round(exit_price, 5),
            "exit_reason": exit_reason, "holding_bars": exit_index - i,\n            "qty": round(qty, 6), "net_pnl": round(net, 2),
            "r_multiple": round(net/risk_dollars, 3) if risk_dollars else 0.0,
            "mae_r": round(max(0.0, max_adverse)/stop_dist, 3),
            "mfe_r": round(max(0.0, max_favorable)/stop_dist, 3),
        })
        curve.append(equity)
        i = exit_index
    result = metrics(trades, curve)
    result.update({"status": "OK", "symbol": symbol, "variant": variant,
                   "start_cash_usd": START_CASH, "end_equity_usd": round(equity, 2),
                   "max_gross_exposure": "1.0x"})
    start_index = next((n for n, b in enumerate(bars)
                        if trade_start_ts is None or int(b["timestamp"]) >= trade_start_ts), 0)
    end_index = next((n for n, b in enumerate(bars)
                      if trade_end_ts is not None and int(b["timestamp"]) >= trade_end_ts), len(bars))
    if end_index - start_index >= 2:
        buy_hold = START_CASH*(float(bars[end_index-1]["close"])/float(bars[start_index]["open"])-1.0)
        result["buy_hold_pnl_usd"] = round(buy_hold, 2)
        result["buy_hold_return_pct"] = round(100*buy_hold/START_CASH, 2)
    return result


def main() -> None:
    if not ROOT.exists():
        raise SystemExit("No accumulated FlowFirst data found.")
    k15, k1h = read_kind("klines", "15m"), read_kind("klines", "1h")
    flows, oi = read_kind("taker", "15m"), read_kind("oi", "15m")
    longshort = read_kind("longshort", "15m")
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "FlowFirst v0.1 backtest",
        "assumptions": {
            "entry": "next 15m bar open after signal bar close",
            "exit": "1.5 ATR stop, 3 ATR target, or 16 completed 15m bars maximum (entry bar counts as bar 1)",
            "same_bar_stop_target": "stop assumed first (conservative)",
            "risk_per_trade": "0.5% current equity",
            "gross_exposure_cap": "1.0x equity, no leverage",
            "costs": f"{COST_PER_SIDE*10000:.1f} bps per side (fee + slippage assumption)",
            "oi_filter": "4-interval OI value expansion >= 0.1%; participation filter only, not direction",
            "crowding_filter": "Block same-direction entry when long/short account share >= 80% and >= 1 standard deviation above its prior 48-bar mean",
            "limitations": "Binance TradFi venue data only; short sample and no validated win probability. MAE/MFE use full OHLC extremes of each held bar, whose intrabar sequence is unknown."
        },
        "results": []
    }
    for symbol in UNIVERSE:
        bars = k15.get(symbol, [])
        hourly = k1h.get(symbol, [])
        flow_rows = flows.get(symbol, [])
        oi_rows = oi.get(symbol, [])
        split_index = int(len(bars) * 0.70)
        split_ts = int(bars[split_index]["timestamp"]) if len(bars) > 120 else None
        longshort_rows = longshort.get(symbol, [])
        for variant in ("price_only", "flow_only", "combo", "combo_crowding", "combo_oi"):
            full = run_variant(symbol, bars, hourly, flow_rows, oi_rows, longshort_rows, variant)
            if split_ts is not None:
                train = run_variant(symbol, bars, hourly, flow_rows, oi_rows, longshort_rows,
                                    variant, trade_end_ts=split_ts)
                test = run_variant(symbol, bars, hourly, flow_rows, oi_rows, longshort_rows,
                                   variant, trade_start_ts=split_ts)
                full["time_split"] = {
                    "train_fraction": 0.70,
                    "test_fraction": 0.30,
                    "split_timestamp": split_ts,
                    "train": train,
                    "out_of_sample_test": test
                }
            else:
                full["time_split"] = {"status": "INSUFFICIENT_DATA",
                                      "reason": "Need at least 120 completed 15m bars"}
            report["results"].append(full)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flowfirst_backtest.json").write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    lines = ["# FlowFirst v0.1 backtest", "", f"Generated UTC: {report['generated_at_utc']}", "",
             "| Symbol | Variant | Trades | Win rate | Net P&L | Return | Max DD | OOS trades | OOS return | OOS max DD | Profit factor | Buy & hold | Status |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for x in report["results"]:
        if x.get("status") != "OK":
            lines.append(f"| {x.get('symbol','-')} | {x.get('variant','-')} | - | - | - | - | - | - | - | - | - | - | {x.get('status')} |")
        else:
            oos = x.get("time_split", {}).get("out_of_sample_test", {})
            lines.append("| " + " | ".join([
                x["symbol"], x["variant"], str(x["trades"]),
                str(x["win_rate_pct"]), f"${x['net_pnl_usd']:.2f}",
                f"{x['return_pct']:.2f}%", f"{x['max_drawdown_pct']:.2f}%",
                str(oos.get("trades", "-")),
                f"{oos['return_pct']:.2f}%" if oos.get("return_pct") is not None else "-",
                f"{oos['max_drawdown_pct']:.2f}%" if oos.get("max_drawdown_pct") is not None else "-",
                str(x["profit_factor"]), f"${x.get('buy_hold_pnl_usd', 0):.2f}", x["status"]
            ]) + " |")
    lines += ["", "## Assumptions"] + [f"- **{k}:** {v}" for k,v in report["assumptions"].items()]
    (OUT / "flowfirst_backtest.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"results": report["results"],
                      "report": "reports/flowfirst_backtest.json"}, indent=2))


if __name__ == "__main__":
    main()
