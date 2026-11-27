#!/usr/bin/env python3
"""Descriptive flow-quality diagnostics; never a standalone trade signal."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("data/flowfirst")
OUT = Path("reports")
UNIVERSE = ["QQQUSDT", "NVDAUSDT", "SMHUSDT", "SOXLUSDT", "SPYUSDT"]


def diagnose_flow_quality(price_rows: list[dict], flow_rows: list[dict],
                          oi_rows: list[dict], window: int = 8) -> dict:
    """Analyze the latest fully timestamp-aligned window; no forward data is used."""
    if window < 2:
        raise ValueError("window must be at least 2")
    pmap = {int(r["timestamp"]): r for r in price_rows}
    fmap = {int(r["timestamp"]): r for r in flow_rows}
    omap = {int(r["timestamp"]): r for r in oi_rows}
    common = sorted(set(pmap) & set(fmap))
    if len(common) < window:
        return {"ready": False, "reason": f"对齐后的价格/资金流K线不足{window}根"}
    stamps = common[-window:]
    prices = [pmap[t] for t in stamps]
    flows = [fmap[t] for t in stamps]
    imbalances = []
    total_volumes = []
    for row in flows:
        buy, sell = float(row["buyVol"]), float(row["sellVol"])
        total = buy + sell
        imbalances.append((buy - sell) / total if total > 0 else 0.0)
        total_volumes.append(total)

    positive = sum(x > 0.02 for x in imbalances)
    negative = sum(x < -0.02 for x in imbalances)
    net_abs = [abs(x) for x in imbalances]
    abs_total = sum(net_abs)
    dominant_share = max(net_abs) / abs_total if abs_total > 1e-12 else 0.0
    weighted_flow = ((sum(float(r["buyVol"]) for r in flows)
                      - sum(float(r["sellVol"]) for r in flows))
                     / sum(total_volumes)) if sum(total_volumes) > 0 else 0.0
    first_open = float(prices[0]["open"])
    last_close = float(prices[-1]["close"])
    price_return = (last_close / first_open - 1.0) if first_open else 0.0

    if weighted_flow >= 0.03 and price_return <= 0:
        response = "BUY_ABSORPTION_OR_WEAK_UPSIDE_RESPONSE"
    elif weighted_flow <= -0.03 and price_return >= 0:
        response = "SELL_ABSORPTION_OR_STRONG_DOWNSIDE_DEFENSE"
    elif weighted_flow >= 0.03 and price_return > 0:
        response = "BUY_FLOW_CONFIRMED_BY_PRICE"
    elif weighted_flow <= -0.03 and price_return < 0:
        response = "SELL_FLOW_CONFIRMED_BY_PRICE"
    else:
        response = "NO_CLEAR_PRICE_FLOW_RESPONSE"

    outlier_dominated = dominant_share >= 0.45
    if outlier_dominated:
        persistence = "SINGLE_BAR_DOMINATED"
    elif positive >= window * 0.625:
        persistence = "PERSISTENT_BUYING"
    elif negative >= window * 0.625:
        persistence = "PERSISTENT_SELLING"
    else:
        persistence = "MIXED_OR_WEAK_FLOW"

    oi_stamps = [t for t in stamps if t in omap]
    oi_pct = None
    oi_context = "OI_UNAVAILABLE_OR_MISALIGNED"
    if len(oi_stamps) >= 2:
        old = float(omap[oi_stamps[0]]["sumOpenInterestValue"])
        new = float(omap[oi_stamps[-1]]["sumOpenInterestValue"])
        if old > 0:
            oi_pct = (new / old - 1.0) * 100.0
            if oi_pct >= 0.1:
                oi_context = "OI_EXPANDING"
            elif oi_pct <= -0.1:
                oi_context = "OI_CONTRACTING"
            else:
                oi_context = "OI_FLAT"

    return {
        "ready": True,
        "window_bars": window,
        "start_timestamp": stamps[0],
        "end_timestamp": stamps[-1],
        "weighted_imbalance": round(weighted_flow, 4),
        "positive_flow_bars": positive,
        "negative_flow_bars": negative,
        "dominant_bar_abs_imbalance_share": round(dominant_share, 3),
        "outlier_dominated": outlier_dominated,
        "persistence": persistence,
        "price_return_pct": round(price_return * 100.0, 3),
        "price_flow_response": response,
        "oi_value_change_pct": round(oi_pct, 3) if oi_pct is not None else None,
        "oi_context": oi_context,
        "interpretation_limit": "描述性诊断，不是独立交易信号；Binance场内衍生品数据不等于美股全市场资金流"
    }


def read_rows(kind: str, period: str = "15m") -> dict[str, list[dict]]:
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


def main() -> None:
    prices, flows, oi = (read_rows("klines"), read_rows("taker"), read_rows("oi"))
    diagnostics = {}
    for symbol in UNIVERSE:
        diagnostics[symbol] = diagnose_flow_quality(
            prices.get(symbol, []), flows.get(symbol, []), oi.get(symbol, []))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": "FlowFirst flow-quality diagnostics v0.1",
        "diagnostics": diagnostics,
        "limitations": [
            "仅使用价格与资金流时间戳交集中的最近8根已收盘15m K线。",
            "单根主导比例按每根K线不平衡率绝对值计算，是异常值提示，不是统计显著性检验。",
            "OI只作参与度背景，不单独决定多空方向。",
            "Binance TradFi perpetual数据不是美股全市场逐笔成交或机构资金流。"
        ]
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "flowfirst_quality.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["# FlowFirst flow-quality diagnostics", "",
             f"Generated UTC: {report['generated_at_utc']}", "",
             "| Symbol | Persistence | Price/flow response | Weighted imbalance | Price return | Dominant bar share | OI context |",
             "|---|---|---|---:|---:|---:|---|"]
    for symbol, item in diagnostics.items():
        if not item.get("ready"):
            lines.append(f"| {symbol} | NOT_READY | {item.get('reason')} | - | - | - | - |")
            continue
        lines.append("| " + " | ".join([
            symbol, item["persistence"], item["price_flow_response"],
            str(item["weighted_imbalance"]), str(item["price_return_pct"]),
            str(item["dominant_bar_abs_imbalance_share"]), item["oi_context"]
        ]) + " |")
    lines += ["", "## Limits", *[f"- {x}" for x in report["limitations"]]]
    (OUT / "flowfirst_quality.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": "reports/flowfirst_quality.json",
                      "ready": {s: x.get("ready", False) for s, x in diagnostics.items()}}, indent=2))


if __name__ == "__main__":
    main()
