#!/usr/bin/env python3
import json, os
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "btc_snapshot_input.json").read_text(encoding="utf-8"))
OUT_JSON = ROOT / "btc_deepseek_report.json"
OUT_MD = ROOT / "btc_deepseek_report.md"

def main():
    key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY")
    prompt = """你是独立的BTC短线交易分析员。只根据给定的同一份Binance BTCUSDT现货与USDⓈ-M合约数据做判断，不得编造指标或新闻。
数据包含日K、4小时K、15分钟K、24小时统计、合约标记价、资金费率、OI、多空账户比、主动买卖量。
严格区分未收盘K线与已收盘K线；最近日K/4小时/15分钟K均有未完成数据，不能当成收盘确认。
请给出三个预测期限：未来4小时、未来24小时、未来3天。每个期限分别给出方向（偏多/偏空/震荡）、参考价格、目标区间、关键支撑/压力、触发条件、失效条件、信心百分比。信心不是胜率。另给出现在的操作结论（做多/做空/观望），入场触发条件、止损价位/条件。若无明显优势必须观望。结合价格结构、成交量变化、OI变化、资金费率、多空账户拥挤和taker buy/sell；说明这些数据可能存在的局限。
只返回严格JSON，格式：
{"snapshot_time_utc":"...","current_price":0,"overall_decision":"做多|做空|观望","overall_bias":"偏多|偏空|震荡","confidence_pct":0,"horizons":{"4h":{"direction":"...","target_low":0,"target_high":0,"support":[0],"resistance":[0],"trigger":"...","invalidation":"...","confidence_pct":0},"24h":{...},"3d":{...}},"entry_plan":"...","stop_plan":"...","reasoning":["..."],"risk_notes":["..."]}
所有价格必须在USDT尺度。不得声称自动下单。不得把信心百分比称为胜率。
数据快照：""" + json.dumps(DATA, ensure_ascii=False)
    client = OpenAI(api_key=key, base_url="https://api.deepseek.com", timeout=180, max_retries=1)
    try:
        resp = client.chat.completions.create(
            model="deepseek-flash",
            messages=[
                {"role":"system","content":"你是严谨、独立的BTC市场分析员，只返回有效JSON。"},
                {"role":"user","content":prompt}
            ],
            response_format={"type":"json_object"},
            temperature=0.1,
            max_tokens=6000
        )
        content = resp.choices[0].message.content
        if not content:
            raise RuntimeError("DeepSeek returned empty content")
        report = json.loads(content)
        report["model"] = "deepseek-flash"
        report["snapshot"] = DATA
        report["report_generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
        lines = [
            "# BTC｜DeepSeek 独立观察报告", "",
            f"- 数据快照：{DATA['generated_at_utc']}",
            f"- 报告生成：{report['report_generated_at_utc']}",
            f"- 参考现价：{report.get('current_price')}",
            f"- 总结：{report.get('overall_decision')} / {report.get('overall_bias')} / 信心 {report.get('confidence_pct')}%",
            "", "## 三个预测期限", ""
        ]
        for horizon in ("4h","24h","3d"):
            x = report.get("horizons",{}).get(horizon,{})
            lines += [f"### {horizon}", f"- 方向：{x.get('direction')}",
                      f"- 目标区间：{x.get('target_low')} – {x.get('target_high')}",
                      f"- 支撑：{x.get('support')}", f"- 压力：{x.get('resistance')}",
                      f"- 触发：{x.get('trigger')}", f"- 失效：{x.get('invalidation')}",
                      f"- 信心：{x.get('confidence_pct')}%", ""]
        lines += ["## 入场与止损", "", str(report.get("entry_plan")), "", str(report.get("stop_plan")),
                  "", "## 判断依据", ""] + [f"- {x}" for x in report.get("reasoning",[])]
        lines += ["", "## 风险", ""] + [f"- {x}" for x in report.get("risk_notes",[])]
        OUT_MD.write_text("\\n".join(lines) + "\\n", encoding="utf-8")
        print(json.dumps({"success":True,"model":report["model"],"decision":report.get("overall_decision"),
                          "bias":report.get("overall_bias"),"confidence_pct":report.get("confidence_pct"),
                          "snapshot_time_utc":DATA["generated_at_utc"]},ensure_ascii=False))
    finally:
        client.close()

if __name__ == "__main__":
    main()
