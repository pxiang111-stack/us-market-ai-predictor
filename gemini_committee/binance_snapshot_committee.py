#!/usr/bin/env python3
import json, os
from datetime import datetime, timezone
from pathlib import Path
from openai import OpenAI

ROOT=Path(__file__).resolve().parent
INPUT=ROOT/"binance_snapshot_input.json"
OUT_JSON=ROOT/"binance_snapshot_report.json"
OUT_MD=ROOT/"binance_snapshot_report.md"
MODEL="deepseek-flash"
WATCH=["QQQUSDT","NVDAUSDT","AMDUSDT","SOXLUSDT"]

def main():
    key=os.getenv("DEEPSEEK_API_KEY","").strip()
    if not key: raise RuntimeError("Missing DEEPSEEK_API_KEY")
    data=json.loads(INPUT.read_text(encoding="utf-8"))
    if set(data.get("symbols",{})) != set(WATCH):
        raise RuntimeError("Binance snapshot ticker set incomplete")
    for ticker in WATCH:
        s=data["symbols"][ticker]
        for tf,n in (("daily",10),("four_hour",10),("fifteen_minute",10)):
            if len(s.get(tf,[]))<n: raise RuntimeError(f"{ticker}: insufficient {tf} bars")
    prompt=f"""
你是独立的美股短线量化交易委员会。请只根据输入的Binance TradFi永续合约代理行情，对QQQ、NVDA、AMD、SOXL做独立判断。
数据包含日K、4小时K、15分钟K及成交量、成交额、主动买入量。日K决定主趋势，4小时K判断波段结构，15分钟K只用于入场确认。
请识别趋势、关键成交量变化、放量下跌/吸筹可能性、支撑阻力、周期共振或冲突。不要凭空引用新闻或宏观事件。
每个标的必须给出：bias(BULLISH/BEARISH/NEUTRAL)、decision(LONG/SHORT/WAIT)、confidence_pct(0-100)、entry_low、entry_high、stop_loss、target_1、target_2、support_levels、resistance_levels、bull_case、bear_case、judge_reason、invalidation。
如果数据不支持明确交易，必须WAIT；不要为了给信号而给信号。若建议做空，也必须说明是否仅为方向判断，不代表该合约可在用户券商交易。
同时给出market_summary：整体偏向、主要判断、四标的强弱排序、下周主要情景和风险。
重要限制：这些是Binance TradFi永续代理行情，不是美股交易所官方现货/ETF报价；SOXL为杠杆ETF代理标的，波动放大。不可将代理报价写成Tiger等券商可直接下单的现价。最新K线可能是未完成K线，需指出时间和完整性。不得声称实时可成交。
只返回严格JSON，格式：
{"generated_at_utc":"...","source_note":"...","market_summary":{"bias":"NEUTRAL","confidence_pct":50,"summary":"...","strength_rank":["..."],"next_week_base_case":"...","risks":["..."]},"results":[{"ticker":"QQQUSDT","bias":"NEUTRAL","decision":"WAIT","confidence_pct":50,"reference_price":0,"entry_low":null,"entry_high":null,"stop_loss":null,"target_1":null,"target_2":null,"support_levels":["..."],"resistance_levels":["..."],"bull_case":"...","bear_case":"...","judge_reason":"...","invalidation":"..."}]}
results必须恰好包含四个标的且各一次。价格数值必须与输入代理行情同一报价尺度。禁止编造胜率；confidence_pct是判断置信度，不是胜率。
输入行情：
{json.dumps(data,ensure_ascii=False)}
"""
    client=OpenAI(api_key=key,base_url="https://api.deepseek.com",timeout=240,max_retries=1)
    try:
        r=client.chat.completions.create(model=MODEL,messages=[
            {"role":"system","content":"你是严格、审慎的交易委员会裁判。只返回有效JSON。"},
            {"role":"user","content":prompt}],
            response_format={"type":"json_object"},temperature=0.1,max_tokens=12000)
        content=r.choices[0].message.content
        if not content: raise RuntimeError("DeepSeek returned empty content")
        report=json.loads(content)
    finally:
        client.close()
    got=[x.get("ticker") for x in report.get("results",[])]
    if len(got)!=4 or set(got)!=set(WATCH): raise RuntimeError(f"Invalid result ticker set: {got}")
    report["input_snapshot_generated_at_utc"]=data.get("generated_at_utc")
    report["model"]=MODEL
    report["data_source"]=data.get("source")
    report["report_generated_at_utc"]=datetime.now(timezone.utc).isoformat()
    OUT_JSON.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    lines=["# Binance多周期行情｜DeepSeek独立委员会判断","",
      f"- 模型：{MODEL}",f"- 输入行情时间：{data.get('generated_at_utc')}",
      f"- 报告时间：{report['report_generated_at_utc']}",
      f"- 数据源：{data.get('source')}",
      "- 注意：Binance TradFi永续代理行情，不是美股交易所官方现货/ETF报价；不可直接作为券商下单价格。","",
      "## 市场总判断","",str(report.get("market_summary",{})),"","## 个股裁判结果",""]
    for x in report["results"]:
        lines += [f"### {x['ticker']}｜{x.get('decision')}｜{x.get('bias')}｜置信度{x.get('confidence_pct')}%",
          "",f"- 参考价：{x.get('reference_price')}",
          f"- 入场：{x.get('entry_low')}–{x.get('entry_high')}",
          f"- 止损：{x.get('stop_loss')}；目标：{x.get('target_1')} / {x.get('target_2')}",
          f"- 支撑：{x.get('support_levels')}",f"- 压力：{x.get('resistance_levels')}",
          f"- 多头逻辑：{x.get('bull_case')}",f"- 空头逻辑：{x.get('bear_case')}",
          f"- 裁判结论：{x.get('judge_reason')}",f"- 失效条件：{x.get('invalidation')}",""]
    OUT_MD.write_text("\n".join(lines),encoding="utf-8")
    print(json.dumps({"success":True,"report":str(OUT_MD),"decisions":[{"ticker":x["ticker"],"decision":x.get("decision"),"bias":x.get("bias"),"confidence_pct":x.get("confidence_pct")} for x in report["results"]]},ensure_ascii=False))

if __name__=="__main__": main()
