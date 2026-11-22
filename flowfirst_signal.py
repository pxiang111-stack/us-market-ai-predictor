#!/usr/bin/env python3
"""FlowFirst v0.1: conservative, explainable signal layer over accumulated Binance data.

Signals are research alerts, not trade execution or guaranteed probabilities.
Binance TradFi perpetual flow is not consolidated US equity order flow.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/flowfirst")
OUT = Path("reports")
UNIVERSE = ["QQQUSDT", "NVDAUSDT", "SMHUSDT", "SOXLUSDT", "SPYUSDT"]
PERIOD_MS = {"15m": 900_000, "1h": 3_600_000}


def load_rows(kind: str, period: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for path in sorted(ROOT.glob(f"*_{kind}_{period}.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                    grouped.setdefault(row["symbol"], []).append(row)
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue
    for rows in grouped.values():
        rows.sort(key=lambda r: int(r["timestamp"]))
    return grouped


def ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    alpha = 2.0 / (period + 1.0)
    result = sum(values[:period]) / period
    for value in values[period:]:
        result = alpha * value + (1.0 - alpha) * result
    return result


def atr(rows: list[dict], period: int = 14) -> float | None:
    if len(rows) < period + 1:
        return None
    trs = []
    for i in range(1, len(rows)):
        high, low = float(rows[i]["high"]), float(rows[i]["low"])
        prev = float(rows[i - 1]["close"])
        trs.append(max(high - low, abs(high - prev), abs(low - prev)))
    return sum(trs[-period:]) / period if len(trs) >= period else None


def trend(rows: list[dict]) -> dict:
    closes = [float(r["close"]) for r in rows]
    e20, e50 = ema(closes, 20), ema(closes, 50)
    if e20 is None or e50 is None:
        return {"direction": "unknown", "ema20": e20, "ema50": e50}
    last = closes[-1]
    if last > e20 and e20 > e50:
        direction = "up"
    elif last < e20 and e20 < e50:
        direction = "down"
    else:
        direction = "mixed"
    return {"direction": direction, "ema20": round(e20, 5),
            "ema50": round(e50, 5), "close": last}


def flow_stats(rows: list[dict]) -> dict:
    ratios = []
    for r in rows:
        buy, sell = float(r["buyVol"]), float(r["sellVol"])
        total = buy + sell
        ratios.append((buy - sell) / total if total else 0.0)
    if len(ratios) < 8:
        return {"ready": False, "reason": "不足8根已收盘资金流K线"}
    recent = sum(ratios[-4:]) / 4.0
    baseline = ratios[-52:-4]
    if len(baseline) < 20:
        return {"ready": False, "reason": "资金流历史样本不足20根"}
    mean = sum(baseline) / len(baseline)
    variance = sum((x - mean) ** 2 for x in baseline) / len(baseline)
    std = math.sqrt(variance)
    z = (recent - mean) / std if std > 1e-9 else 0.0
    return {"ready": True, "recent_4bar_imbalance": round(recent, 4),
            "baseline_mean": round(mean, 4), "zscore": round(z, 2),
            "direction": "buy" if recent > 0 else "sell" if recent < 0 else "flat"}


def crowding_stats(rows: list[dict]) -> dict:
    if len(rows) < 49:
        return {"ready": False, "reason": "多空账户比历史不足49根"}
    rows = sorted(rows, key=lambda r: int(r["timestamp"]))
    current_long = float(rows[-1]["longAccount"])
    current_short = float(rows[-1]["shortAccount"])
    base_long = [float(x["longAccount"]) for x in rows[-49:-1]]
    base_short = [float(x["shortAccount"]) for x in rows[-49:-1]]
    def zscore(current: float, baseline: list[float]) -> float:
        mean = sum(baseline) / len(baseline)
        variance = sum((x-mean)**2 for x in baseline) / len(baseline)
        sd = math.sqrt(variance)
        return (current-mean)/sd if sd > 1e-9 else 0.0
    long_z = zscore(current_long, base_long)
    short_z = zscore(current_short, base_short)
    return {
        "ready": True,
        "long_account_pct": round(current_long * 100, 2),
        "short_account_pct": round(current_short * 100, 2),
        "long_account_zscore": round(long_z, 2),
        "short_account_zscore": round(short_z, 2),
        "crowded_long": current_long >= 0.80 and long_z >= 1.0,
        "crowded_short": current_short >= 0.80 and short_z >= 1.0,
        "interpretation": "账户数量比例，只用于拥挤度过滤，不代表机构持仓或资金金额"
    }


def oi_change(rows: list[dict]) -> float | None:
    if len(rows) < 5:
        return None
    old = float(rows[-5]["sumOpenInterestValue"])
    new = float(rows[-1]["sumOpenInterestValue"])
    return round((new / old - 1.0) * 100.0, 3) if old else None


def latest_age_ms(rows: list[dict], now_ms: int) -> int | None:
    return now_ms - int(rows[-1]["timestamp"]) if rows else None


def analyze_symbol(symbol: str, k15: list[dict], k1h: list[dict],
                   flows: list[dict], oi: list[dict],
                   longshort: list[dict], now_ms: int) -> dict:
    result = {"symbol": symbol, "signal": "WAIT", "reason": []}
    if len(k15) < 60 or len(k1h) < 55:
        result["reason"].append("价格历史不足：需要至少60根15m和55根1h已收盘K线")
        return result
    age15, age1h = latest_age_ms(k15, now_ms), latest_age_ms(k1h, now_ms)
    if age15 is None or age1h is None or age15 > 45 * 60_000 or age1h > 3 * 60 * 60_000:
        result["reason"].append("行情数据过期，禁止生成方向信号")
        result["data_age_minutes"] = round(age15 / 60_000, 1) if age15 is not None else None
        return result
    t15, t1h = trend(k15), trend(k1h)
    # Align flow to the exact four latest completed 15m price bars.
    # Never let an old flow response silently drive a fresh price signal.
    latest_bars = k15[-4:]
    latest_bar_ts = [int(x["timestamp"]) for x in latest_bars]
    flow_by_ts = {int(x["timestamp"]): x for x in flows}
    if any(ts not in flow_by_ts for ts in latest_bar_ts):
        result["reason"].append("最近4根已收盘价格K线缺少对应资金流数据，禁止生成方向信号")
        return result
    # Use only flow observations no later than the latest completed price bar.
    eligible_flows = [x for x in flows if int(x["timestamp"]) <= latest_bar_ts[-1]]
    f = flow_stats(eligible_flows)
    if not f["ready"]:
        result["reason"].append(f["reason"])
        return result
    volatility = atr(k15)
    if not volatility or volatility <= 0:
        result["reason"].append("ATR不可用")
        return result
    eligible_longshort = [x for x in longshort if int(x["timestamp"]) <= latest_bar_ts[-1]]
    longshort_by_ts = {int(x["timestamp"]): x for x in eligible_longshort}
    if latest_bar_ts[-1] not in longshort_by_ts:
        result["reason"].append("最新价格K线缺少对应多空账户比数据，禁止生成方向信号")
        return result
    crowding = crowding_stats(eligible_longshort)
    if not crowding["ready"]:
        result["reason"].append(crowding["reason"])
        return result
    oi_pct = oi_change(oi)
    last_close = float(k15[-1]["close"])
    result.update({"price": last_close, "trend_15m": t15, "trend_1h": t1h,
                   "flow": f, "crowding": crowding,
                   "oi_value_change_4bars_pct": oi_pct,
                   "atr_15m": round(volatility, 5)})
    bullish = (t15["direction"] == "up" and t1h["direction"] == "up"
               and f["recent_4bar_imbalance"] >= 0.06 and f["zscore"] >= 0.75)
    bearish = (t15["direction"] == "down" and t1h["direction"] == "down"
               and f["recent_4bar_imbalance"] <= -0.06 and f["zscore"] <= -0.75)
    if bullish and crowding["crowded_long"]:
        result["reason"].append("主动买入与价格偏强，但多头账户拥挤度异常，过滤做多")
    elif bearish and crowding["crowded_short"]:
        result["reason"].append("主动卖出与价格偏弱，但空头账户拥挤度异常，过滤做空")
    elif bullish:
        result["signal"] = "LONG"
        result["entry_reference"] = last_close
        result["stop_reference"] = round(last_close - 1.5 * volatility, 5)
        result["target_reference"] = round(last_close + 3.0 * volatility, 5)
        result["reason"].append("15m与1h趋势同向，最近4根主动买入占优且资金流Z分数达标")
    elif bearish:
        result["signal"] = "SHORT"
        result["entry_reference"] = last_close
        result["stop_reference"] = round(last_close + 1.5 * volatility, 5)
        result["target_reference"] = round(last_close - 3.0 * volatility, 5)
        result["reason"].append("15m与1h趋势同向，最近4根主动卖出占优且资金流Z分数达标")
    else:
        result["reason"].append("价格趋势与资金流未同时确认，暂不交易")
    result["execution_note"] = "参考价为信号K线收盘；回测/模拟须下一根K线开盘执行，不能按收盘价假定成交"
    return result


def main() -> None:
    if not ROOT.exists():
        raise SystemExit("No accumulated FlowFirst data found.")
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    k15, k1h = load_rows("klines", "15m"), load_rows("klines", "1h")
    flow15, oi15 = load_rows("taker", "15m"), load_rows("oi", "15m")
    longshort15 = load_rows("longshort", "15m")
    market = {}
    for symbol in UNIVERSE:
        market[symbol] = analyze_symbol(
            symbol, k15.get(symbol, []), k1h.get(symbol, []),
            flow15.get(symbol, []), oi15.get(symbol, []),
            longshort15.get(symbol, []), now_ms)
    core = [market.get(s, {}).get("trend_1h", {}).get("direction", "unknown")
            for s in ("QQQUSDT", "SPYUSDT", "SMHUSDT")]
    if all(x == "up" for x in core):
        regime = "BULL"
    elif all(x == "down" for x in core):
        regime = "BEAR"
    elif "unknown" in core:
        regime = "UNAVAILABLE"
    else:
        regime = "MIXED"
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "FlowFirst v0.1",
        "market_regime": regime,
        "regime_inputs": {"QQQ_SPY_SMH_1h": core},
        "signals": market,
        "limitations": [
            "Binance TradFi perpetual data is not consolidated US equity order flow.",
            "Flow history is accumulated from collector start; no fabricated historical flow.",
            "LONG/SHORT are research alerts, not a validated win probability or execution instruction.",
            "Signals require complete bars and are intended for next-bar execution in testing."
        ]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flowfirst_latest.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = [f"# FlowFirst v0.1", "", f"Generated UTC: {report['generated_at_utc']}",
             f"Market regime: **{regime}**", "", "| Symbol | Signal | 15m trend | 1h trend | Flow imbalance | Flow Z | Entry ref | Stop ref | Target ref |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for symbol, item in market.items():
        f = item.get("flow", {})
        lines.append("| " + " | ".join([
            symbol, item.get("signal", "WAIT"),
            item.get("trend_15m", {}).get("direction", "-"),
            item.get("trend_1h", {}).get("direction", "-"),
            str(f.get("recent_4bar_imbalance", "-")), str(f.get("zscore", "-")),
            str(item.get("entry_reference", "-")), str(item.get("stop_reference", "-")),
            str(item.get("target_reference", "-"))
        ]) + " |")
    lines += ["", "## Limitations", *[f"- {x}" for x in report["limitations"]]]
    (OUT / "flowfirst_latest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"market_regime": regime,
                      "signals": {s: v["signal"] for s, v in market.items()},
                      "report": "reports/flowfirst_latest.json"}, indent=2))


if __name__ == "__main__":
    main()
