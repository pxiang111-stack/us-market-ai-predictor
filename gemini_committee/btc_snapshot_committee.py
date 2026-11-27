#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "btc_snapshot_input.json"
OUT_JSON = ROOT / "btc_deepseek_report.json"
OUT_MD = ROOT / "btc_deepseek_report.md"
LEDGER = ROOT / "btc_forecast_ledger.json"
MODEL = "deepseek-flash"

def utc_now():
    return datetime.now(timezone.utc)

def load_and_validate_snapshot():
    if not DATA_PATH.exists():
        raise RuntimeError("Missing btc_snapshot_input.json")
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    required = ("generated_at_utc", "source", "symbol", "spot", "daily", "four_hour",
                "fifteen_minute", "futures", "candle_status")
    missing = [key for key in required if key not in data]
    if missing:
        raise RuntimeError(f"Snapshot missing required fields: {missing}")
    if data["symbol"] != "BTCUSDT":
        raise RuntimeError(f"Unexpected symbol: {data['symbol']}")
    ts = datetime.fromisoformat(data["generated_at_utc"].replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise RuntimeError("Snapshot timestamp must include timezone")
    age_minutes = (utc_now() - ts.astimezone(timezone.utc)).total_seconds() / 60
    if age_minutes < -5:
        raise RuntimeError("Snapshot timestamp is more than 5 minutes in the future")
    if age_minutes > 24 * 60:
        raise RuntimeError(f"Snapshot is stale ({age_minutes:.0f} minutes); refusing to generate a current signal")
    for name in ("daily", "four_hour", "fifteen_minute"):
        candles = data[name]
        if not isinstance(candles, list) or len(candles) < 3:
            raise RuntimeError(f"{name} must contain at least 3 candles")
        for i, candle in enumerate(candles):
            fields = ("open", "high", "low")
            if any(field not in candle for field in fields):
                raise RuntimeError(f"{name}[{i}] missing OHLC fields")
            high, low = float(candle["high"]), float(candle["low"])
            op = float(candle["open"])
            close = candle.get("close", candle.get("last"))
            if close is None:
                raise RuntimeError(f"{name}[{i}] missing close/last")
            close = float(close)
            if high < max(op, close) or low > min(op, close) or high < low:
                raise RuntimeError(f"{name}[{i}] has inconsistent OHLC values")
    for key in ("last_price", "high_24h", "low_24h"):
        if key not in data["spot"]:
            raise RuntimeError(f"spot missing {key}")
    if float(data["spot"]["high_24h"]) < float(data["spot"]["low_24h"]):
        raise RuntimeError("24h high is below 24h low")
    return data, age_minutes

def build_prompt(data, age_minutes):
    return """你是独立、审慎的BTC量化交易分析员。只根据输入的同一份Binance BTCUSDT现货与USDⓈ-M合约数据判断，不得编造指标、数据或新闻。
数据包含日K、4小时K、15分钟K、24小时统计、合约标记价、资金费率、OI、多空账户比、主动买卖量。必须先检查数据时间和完整性。
重要规则：
1. 严格区分已收盘K线和未收盘K线；字段 candle_status 已说明未完成K线，未收盘K线只作实时观察，不能当作收盘确认。
2. 日K确定大结构，4小时K判断波段，15分钟K寻找触发；不能让低周期信号推翻高周期结构而不解释。
3. 重点分析“价格变化 × OI变化 × 主动买卖量”的组合：上涨/下跌时OI增加或减少分别意味着什么；区分新开仓、平仓/挤仓的可能性，不得把相关性写成确定因果。
4. 结合资金费率与多空账户比判断拥挤程度；账户比例不等于真实仓位规模，不得单独据此做反向交易。
5. 如果缺少某项数据，明确写缺失；不得自行补齐。数据源是单一交易所，说明其局限。
6. 必须输出未来4小时、24小时、3天三个独立预测，包含方向、目标区间、支撑/压力、入场触发、失效条件、止损逻辑和信心。信心不是胜率。
7. 只有在风险收益比合理且触发条件清晰时才给做多/做空，否则观望。不得为了给信号而强行交易。
8. 所有价位使用USDT。给出预测基准价和数据快照时间。不能使用当前输入快照之后的信息。
9. 最后列出最可能推翻当前判断的两项证据。
只返回严格JSON，结构：
{"snapshot_time_utc":"...","current_price":0,"overall_decision":"做多|做空|观望","overall_bias":"偏多|偏空|震荡","confidence_pct":0,"market_regime":"趋势上涨|趋势下跌|区间震荡|突破观察|高波动反转|无法判断","horizons":{"4h":{"direction":"偏多|偏空|震荡","target_low":0,"target_high":0,"support":[0],"resistance":[0],"entry_trigger":"...","stop_logic":"...","invalidation":"...","confidence_pct":0},"24h":{...},"3d":{...}},"entry_plan":"...","stop_plan":"...","reasoning":["..."],"flow_read":["..."],"risk_notes":["..."],"thesis_breakers":["..."]}
snapshot_time_utc必须与输入 generated_at_utc 完全一致；current_price必须与 spot.last_price一致；三个期限必须齐全。数据新鲜度（分钟）：""" + f"{age_minutes:.1f}" + """
数据快照：""" + json.dumps(data, ensure_ascii=False)

def validate_report(report, data):
    if report.get("snapshot_time_utc") != data["generated_at_utc"]:
        raise RuntimeError("DeepSeek snapshot_time_utc does not match input snapshot")
    if abs(float(report.get("current_price", 0)) - float(data["spot"]["last_price"])) > max(1.0, float(data["spot"]["last_price"]) * 0.0001):
        raise RuntimeError("DeepSeek current_price does not match snapshot spot price")
    if report.get("overall_decision") not in ("做多", "做空", "观望"):
        raise RuntimeError("Invalid overall_decision")
    horizons = report.get("horizons")
    if not isinstance(horizons, dict):
        raise RuntimeError("Missing horizons object")
    for horizon in ("4h", "24h", "3d"):
        item = horizons.get(horizon)
        if not isinstance(item, dict):
            raise RuntimeError(f"Missing {horizon} forecast")
        low, high = float(item.get("target_low", 0)), float(item.get("target_high", 0))
        if low <= 0 or high <= 0 or low > high:
            raise RuntimeError(f"Invalid target range for {horizon}")
        confidence = float(item.get("confidence_pct", -1))
        if not 0 <= confidence <= 100:
            raise RuntimeError(f"Invalid confidence for {horizon}")
        if item.get("direction") not in ("偏多", "偏空", "震荡"):
            raise RuntimeError(f"Invalid direction for {horizon}")

def append_locked_forecast(report, data):
    ledger = []
    if LEDGER.exists():
        ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if not isinstance(ledger, list):
        raise RuntimeError("Forecast ledger must be a JSON array")
    snapshot_time = data["generated_at_utc"]
    if any(x.get("snapshot_time_utc") == snapshot_time for x in ledger):
        raise RuntimeError(f"Forecast already locked for snapshot {snapshot_time}; refusing overwrite")
    ledger.append({
        "snapshot_time_utc": snapshot_time,
        "locked_at_utc": utc_now().isoformat(),
        "symbol": "BTCUSDT",
        "reference_price": float(data["spot"]["last_price"]),
        "model": MODEL,
        "overall_decision": report["overall_decision"],
        "overall_bias": report.get("overall_bias"),
        "confidence_pct": report.get("confidence_pct"),
        "market_regime": report.get("market_regime"),
        "horizons": report["horizons"]
    })
    LEDGER.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def main():
    client = None
    try:
        data, age_minutes = load_and_validate_snapshot()
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=180, max_retries=1)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是严谨、独立的BTC量化分析员。严格遵守输入数据边界，只返回有效JSON。"},
                {"role": "user", "content": build_prompt(data, age_minutes)}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=7000
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek returned empty content")
        report = json.loads(content)
        validate_report(report, data)
        report["model"] = MODEL
        report["snapshot"] = data
        report["snapshot_age_minutes_at_analysis"] = round(age_minutes, 1)
        report["report_generated_at_utc"] = utc_now().isoformat()
        report["data_quality"] = {
            "timestamp_valid": True,
            "snapshot_age_minutes": round(age_minutes, 1),
            "partial_candles_excluded_from_close_confirmation": True,
            "ohlc_consistency_checked": True
        }
        append_locked_forecast(report, data)
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# BTC｜DeepSeek 独立观察报告", "",
            f"- 数据快照：{data['generated_at_utc']}",
            f"- 分析时数据年龄：{age_minutes:.1f} 分钟",
            f"- 报告生成：{report['report_generated_at_utc']}",
            f"- 参考现价：{report['current_price']}",
            f"- 总结：{report['overall_decision']} / {report.get('overall_bias')} / 信心 {report.get('confidence_pct')}%",
            f"- 市场状态：{report.get('market_regime')}",
            "- 说明：信心百分比不是统计胜率；未收盘K线不作为收盘确认。", "",
            "## 三个预测期限", ""
        ]
        for horizon in ("4h", "24h", "3d"):
            x = report["horizons"][horizon]
            lines += [
                f"### {horizon}｜{x['direction']}",
                f"- 目标区间：{x['target_low']} – {x['target_high']}",
                f"- 支撑：{x.get('support')}",
                f"- 压力：{x.get('resistance')}",
                f"- 入场触发：{x.get('entry_trigger')}",
                f"- 止损逻辑：{x.get('stop_logic')}",
                f"- 失效条件：{x.get('invalidation')}",
                f"- 信心：{x['confidence_pct']}%", ""
            ]
        for title, key in (("判断依据", "reasoning"), ("资金行为解读", "flow_read"),
                           ("风险", "risk_notes"), ("推翻当前判断的证据", "thesis_breakers")):
            lines += [f"## {title}", ""] + [f"- {x}" for x in report.get(key, [])] + [""]
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({
            "success": True, "model": MODEL, "decision": report["overall_decision"],
            "bias": report.get("overall_bias"), "confidence_pct": report.get("confidence_pct"),
            "snapshot_time_utc": data["generated_at_utc"], "snapshot_age_minutes": round(age_minutes, 1),
            "ledger_entries": len(json.loads(LEDGER.read_text(encoding="utf-8")))
        }, ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    main()
