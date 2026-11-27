#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

ROOT = Path(__file__).resolve().parent
INPUT_JSON = ROOT / "binance_snapshot_input.json"
OUT_JSON = ROOT / "binance_snapshot_report.json"
OUT_MD = ROOT / "binance_snapshot_report.md"
MODEL = "deepseek-flash"

def utc_now():
    return datetime.now(timezone.utc).isoformat()

def load_snapshot():
    if not INPUT_JSON.exists():
        raise RuntimeError(f"Missing Binance connector snapshot: {INPUT_JSON}")
    data = json.loads(INPUT_JSON.read_text(encoding="utf-8"))
    if not isinstance(data.get("symbols"), dict) or not data["symbols"]:
        raise RuntimeError("Snapshot must contain a non-empty symbols object")
    if not data.get("generated_at_utc") or not data.get("source"):
        raise RuntimeError("Snapshot is missing timestamp or source provenance")
    return data

def make_prompt(data):
    return """你是审慎的美股短线量化分析员。请只根据输入的 Binance TradFi perpetual 代理合约K线快照分析，不得将其称为美股交易所真实报价。
本批数据包含日K、4小时K、15分钟K；数据源是 Binance ChatGPT connector，属于 USDⓈ-M TradFi perpetual proxy。它只能用于观察方向和结构，可能与 Tiger/NYSE/Nasdaq 的真实ETF/股票价格存在偏差。报告中必须明确写出该限制，价格位仅适用于该代理报价尺度，不能直接当作真实证券下单价位。
请逐个标的给出：偏多/偏空/观望、结构判断、关键支撑和压力、可执行的触发条件、失效条件、短线及下周情景。结合日K定方向、4小时K找结构、15分钟K判断短线触发。不要把缺失的成交量、订单簿、OI、新闻或资金流编造出来；如果输入没有足够OHLC字段，应明确降低信心并以可验证的价格点位为主。
只做分析信号，不下单。confidence_pct仅代表判断信心，不是胜率。没有优势时必须观望。
只返回严格JSON，结构：
{"market_overview":"...","results":[{"symbol":"QQQUSDT","bias":"BULLISH|BEARISH|NEUTRAL","decision":"偏多|偏空|观望","confidence_pct":0,"reference_price":0,"support_levels":[],"resistance_levels":[],"trigger_condition":"...","invalidation":"...","reasoning":"...","risk_notes":"...","next_week_scenario":"..."}],"overall_risk_notes":[]}
results必须覆盖输入中的每个symbol且仅出现一次。所有价格必须使用输入代理合约的报价尺度。
数据快照生成时间（UTC）：""" + str(data["generated_at_utc"]) + """
数据来源与限制：""" + str(data["source"]) + """
行情快照JSON：
""" + json.dumps(data, ensure_ascii=False)

def main():
    client = None
    try:
        data = load_snapshot()
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY GitHub Actions secret")
        expected = set(data["symbols"].keys())
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=180, max_retries=1)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "你是严格审慎的市场分析员，只返回有效JSON，不执行交易。"},
                {"role": "user", "content": make_prompt(data)}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=8000
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek returned empty content")
        report = json.loads(content)
        results = report.get("results")
        got = [x.get("symbol") for x in results] if isinstance(results, list) else []
        if len(got) != len(expected) or set(got) != expected:
            raise RuntimeError(f"DeepSeek result symbols mismatch; expected={sorted(expected)}, got={got}")
        report["market_data"] = data
        report["model"] = MODEL
        report["report_generated_at_utc"] = utc_now()
        report["data_source_warning"] = "Binance TradFi perpetual proxy; not official US exchange quotes."
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        lines = [
            "# Binance TradFi代理行情｜DeepSeek分析报告", "",
            f"- 数据快照时间（UTC）：{data['generated_at_utc']}",
            f"- 报告生成时间（UTC）：{report['report_generated_at_utc']}",
            f"- 数据源：{data['source']}",
            f"- 模型：{MODEL}",
            "- 重要限制：Binance TradFi perpetual代理合约，不是美股交易所真实报价；价位仅适用于该代理报价尺度。",
            "- 用途：仅市场分析与信号参考，不执行自动下单。", "",
            "## 市场总览", "", str(report.get("market_overview", "")), "",
            "## 标的结论", ""
        ]
        for x in results:
            lines.extend([
                f"### {x['symbol']}｜{x.get('decision')}｜{x.get('bias')}｜信心 {x.get('confidence_pct')}%",
                "", f"- 参考价：{x.get('reference_price')}",
                f"- 支撑：{x.get('support_levels')}",
                f"- 压力：{x.get('resistance_levels')}",
                f"- 触发条件：{x.get('trigger_condition')}",
                f"- 失效条件：{x.get('invalidation')}",
                f"- 分析：{x.get('reasoning')}",
                f"- 下周情景：{x.get('next_week_scenario')}",
                f"- 风险：{x.get('risk_notes')}", ""
            ])
        lines.extend(["## 总体风险", ""] + [f"- {v}" for v in report.get("overall_risk_notes", [])])
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({
            "success": True, "snapshot_time_utc": data["generated_at_utc"],
            "model": MODEL, "symbols": sorted(expected),
            "decisions": [{"symbol": x["symbol"], "decision": x.get("decision"),
                           "bias": x.get("bias"), "confidence_pct": x.get("confidence_pct")}
                          for x in results]
        }, ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
    finally:
        if client is not None:
            client.close()

if __name__ == "__main__":
    main()
