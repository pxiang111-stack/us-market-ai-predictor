#!/usr/bin/env python3
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import requests
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "binance_spot_config.json"
OUT_JSON = ROOT / "binance_snapshot_report.json"
OUT_MD = ROOT / "binance_snapshot_report.md"
MODEL = "deepseek-flash"
VALID_INTERVALS = {"1m", "5m", "15m"}

def utc_now():
    return datetime.now(timezone.utc)

def iso_ms(value):
    return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()

def api_get(session, base, path, params, timeout):
    url = base.rstrip("/") + path
    try:
        response = session.get(url, params=params, timeout=timeout)
    except requests.RequestException as exc:
        raise RuntimeError(f"Binance network request failed: {url}: {exc}") from exc
    if response.status_code != 200:
        body = response.text[:500].replace("\n", " ")
        raise RuntimeError(f"Binance API error HTTP {response.status_code} at {response.url}: {body}")
    try:
        return response.json()
    except ValueError as exc:
        raise RuntimeError(f"Binance returned non-JSON response at {response.url}") from exc

def summarize_klines(rows, interval, now_ms):
    if not isinstance(rows, list) or not rows:
        raise RuntimeError(f"Binance returned empty {interval} klines")
    bars = []
    for row in rows:
        if not isinstance(row, list) or len(row) < 11:
            raise RuntimeError(f"Malformed {interval} kline row")
        bars.append({
            "open_time": iso_ms(int(row[0])),
            "close_time": iso_ms(int(row[6])),
            "open": float(row[1]), "high": float(row[2]),
            "low": float(row[3]), "close": float(row[4]),
            "base_volume": float(row[5]), "quote_volume": float(row[7]),
            "trade_count": int(row[8]), "taker_buy_base_volume": float(row[9]),
            "taker_buy_quote_volume": float(row[10]),
            "is_closed": int(row[6]) < now_ms
        })
    first = bars[0]
    last = bars[-1]
    change_pct = (last["close"] / first["open"] - 1) * 100 if first["open"] else 0
    high = max(x["high"] for x in bars)
    low = min(x["low"] for x in bars)
    range_pct = (high / low - 1) * 100 if low else 0
    total_quote = sum(x["quote_volume"] for x in bars)
    total_base = sum(x["base_volume"] for x in bars)
    taker_quote = sum(x["taker_buy_quote_volume"] for x in bars)
    buy_ratio = taker_quote / total_quote if total_quote else None
    return {
        "interval": interval, "bars": bars,
        "bar_count": len(bars), "first_open_time": first["open_time"],
        "last_close_time": last["close_time"], "latest_bar_closed": last["is_closed"],
        "period_change_pct": round(change_pct, 4),
        "period_high": high, "period_low": low,
        "period_range_pct": round(range_pct, 4),
        "quote_volume_sum": total_quote, "base_volume_sum": total_base,
        "taker_buy_quote_ratio": round(buy_ratio, 4) if buy_ratio is not None else None
    }

def fetch_market(config):
    base = config.get("api_base", "https://api.binance.com").strip()
    symbols = config.get("symbols")
    intervals = config.get("intervals", ["1m", "5m", "15m"])
    limit = int(config.get("kline_limit", 60))
    timeout = int(config.get("request_timeout_seconds", 15))
    if not isinstance(symbols, list) or not symbols or any(not isinstance(s, str) or not s.strip() for s in symbols):
        raise RuntimeError("binance_spot_config.json: symbols must be a non-empty list of trading-pair strings")
    if not isinstance(intervals, list) or not intervals or set(intervals) - VALID_INTERVALS:
        raise RuntimeError(f"Only these intervals are supported: {sorted(VALID_INTERVALS)}")
    if not 1 <= limit <= 1000:
        raise RuntimeError("kline_limit must be between 1 and 1000")

    fetched_at = utc_now()
    now_ms = int(fetched_at.timestamp() * 1000)
    result = {
        "fetched_at_utc": fetched_at.isoformat(),
        "source": "Binance public Spot REST API (api.binance.com), unauthenticated",
        "symbols": {}
    }
    with requests.Session() as session:
        session.headers.update({"User-Agent": "market-snapshot-deepseek/1.0"})
        for raw_symbol in symbols:
            symbol = raw_symbol.strip().upper()
            price = api_get(session, base, "/api/v3/ticker/price", {"symbol": symbol}, timeout)
            stats = api_get(session, base, "/api/v3/ticker/24hr", {"symbol": symbol}, timeout)
            if not isinstance(price, dict) or "price" not in price:
                raise RuntimeError(f"{symbol}: malformed ticker/price response")
            if not isinstance(stats, dict) or "lastPrice" not in stats:
                raise RuntimeError(f"{symbol}: malformed ticker/24hr response")
            item = {
                "symbol": symbol,
                "latest_price": float(price["price"]),
                "price_time_utc": fetched_at.isoformat(),
                "price_change_24h_pct": float(stats["priceChangePercent"]),
                "price_change_24h": float(stats["priceChange"]),
                "open_24h": float(stats["openPrice"]),
                "high_24h": float(stats["highPrice"]),
                "low_24h": float(stats["lowPrice"]),
                "volume_24h_base": float(stats["volume"]),
                "volume_24h_quote": float(stats["quoteVolume"]),
                "trade_count_24h": int(stats["count"]),
                "klines": {}
            }
            for interval in intervals:
                rows = api_get(session, base, "/api/v3/klines",
                               {"symbol": symbol, "interval": interval, "limit": limit}, timeout)
                item["klines"][interval] = summarize_klines(rows, interval, now_ms)
            result["symbols"][symbol] = item
    if len(result["symbols"]) != len(set(s.strip().upper() for s in symbols)):
        raise RuntimeError("Not all configured symbols returned market data")
    return result

def make_prompt(data):
    return """你是审慎的短线市场分析员。以下数据来自 Binance 官方公开现货 REST 行情接口，不含 API Key。
请逐个交易对综合：最新成交价、24小时涨跌幅/成交额、1m/5m/15m K线区间、区间涨跌、主动买入成交额占比、成交量变化和K线是否收盘。
输出明确结论：偏多、偏空或观望；说明短线结构、关键参考位（支撑/压力）、触发条件、失效条件与风险提示。
不能把指标推导成确定性预测；不能编造订单簿/资金流/新闻等输入中没有的数据；没有优势时必须观望。
confidence_pct仅代表判断信心，不是胜率。只做分析信号，不下单、不调用交易接口。
只返回严格JSON，结构：
{"market_overview":"...","results":[{"symbol":"BTCUSDT","bias":"BULLISH|BEARISH|NEUTRAL","decision":"偏多|偏空|观望","confidence_pct":0,"reference_price":0,"support_levels":[],"resistance_levels":[],"trigger_condition":"...","invalidation":"...","reasoning":"...","risk_notes":"..."}],"overall_risk_notes":[]}
results必须覆盖输入中的每个symbol且仅出现一次。所有价格必须基于输入数据报价尺度。
抓取时间：""" + data["fetched_at_utc"] + "\n行情数据：\n" + json.dumps(data, ensure_ascii=False)

def main():
    try:
        if not CONFIG_PATH.exists():
            raise RuntimeError(f"Missing config file: {CONFIG_PATH}")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        data = fetch_market(config)
        key = os.getenv("DEEPSEEK_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY GitHub Actions secret")
        client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=180, max_retries=1)
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "你是严格审慎的行情分析员，只返回有效JSON，不执行交易。"},
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
        finally:
            client.close()

        expected = {s.strip().upper() for s in config["symbols"]}
        results = report.get("results")
        got = [x.get("symbol") for x in results] if isinstance(results, list) else []
        if len(got) != len(expected) or set(got) != expected:
            raise RuntimeError(f"DeepSeek result symbols mismatch; expected={sorted(expected)}, got={got}")
        report["market_data"] = data
        report["model"] = MODEL
        report["report_generated_at_utc"] = utc_now().isoformat()
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

        lines = [
            "# Binance实时现货行情｜DeepSeek分析报告", "",
            f"- 数据抓取时间（UTC）：{data['fetched_at_utc']}",
            f"- 报告生成时间（UTC）：{report['report_generated_at_utc']}",
            f"- 数据源：{data['source']}",
            f"- 模型：{MODEL}",
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
                f"- 风险：{x.get('risk_notes')}", ""
            ])
        lines.extend(["## 总体风险", ""] + [f"- {v}" for v in report.get("overall_risk_notes", [])])
        OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(json.dumps({
            "success": True, "fetched_at_utc": data["fetched_at_utc"],
            "report_json": str(OUT_JSON), "report_markdown": str(OUT_MD),
            "decisions": [{"symbol": x["symbol"], "decision": x.get("decision"),
                           "bias": x.get("bias"), "confidence_pct": x.get("confidence_pct")}
                          for x in results]
        }, ensure_ascii=False))
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
