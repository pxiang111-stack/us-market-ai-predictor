#!/usr/bin/env python3
"""DeepSeek盘前量化委员会独立程序：8只标的一次批量推理。"""
import json, math, os, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import quote
import pandas as pd
import requests
import yfinance as yf
from openai import OpenAI
from pydantic import BaseModel, Field

WATCHLIST=["NVDA","MSFT","AAPL","META","AMZN","TSLA","QQQ","SPY"]
ACCOUNT_SIZE=10000.0
MAX_RISK_PCT=0.01
DEEPSEEK_MODEL="deepseek-flash"
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY","").strip()
BARK_KEY=os.getenv("BARK_KEY","").strip()
OUT=Path(__file__).resolve().parent

class CommitteeDecision(BaseModel):
    ticker: str
    bull_case: str=Field(description="多头分析师逻辑，1至2句")
    bear_case_risks: str=Field(description="空头风控官反驳，检查放量破位、高位派发、增发、债务、指引下调")
    verdict: Literal["GO","GO SMALLER","NO-GO"]
    entry_low: Optional[float]=None
    entry_high: Optional[float]=None
    stop_price: Optional[float]=None
    target_price: Optional[float]=None
    judge_reason: str
    pine_script: str=Field(description="Pine Script v5，最多5行，使用hline画入场、止损、目标")

class CommitteeBatch(BaseModel):
    results: list[CommitteeDecision]

def get_news(symbol):
    try:
        out=[]
        for row in (yf.Ticker(symbol).news or []):
            item=row.get("content",row); title=item.get("title") or row.get("title")
            if title and title not in out: out.append(str(title).strip())
            if len(out)>=4: break
        return out
    except Exception as e: return [f"新闻获取失败：{type(e).__name__}: {e}"]

def get_market(symbol):
    df=yf.download(symbol,period="3mo",interval="1d",auto_adjust=True,progress=False,threads=False)
    if df is None or df.empty: raise RuntimeError(f"{symbol}: 近3个月日线数据为空")
    if isinstance(df.columns,pd.MultiIndex): df.columns=[str(c[0]) for c in df.columns]
    df=df.dropna(subset=["Close"])
    close=pd.to_numeric(df["Close"],errors="coerce").dropna()
    high=pd.to_numeric(df["High"],errors="coerce").reindex(close.index)
    low=pd.to_numeric(df["Low"],errors="coerce").reindex(close.index)
    vol=pd.to_numeric(df["Volume"],errors="coerce").reindex(close.index)
    if len(close)<25: raise RuntimeError(f"{symbol}: 有效日线不足25根")
    daily_close=float(close.iloc[-1])
    p=daily_close
    intraday_timestamp=None
    intraday_source="Yahoo Finance daily close (非实时)"
    intraday_change_pct=None
    try:
        intra=yf.download(symbol,period="1d",interval="1m",prepost=True,auto_adjust=True,progress=False,threads=False)
        if intra is not None and not intra.empty:
            if isinstance(intra.columns,pd.MultiIndex): intra.columns=[str(x[0]) for x in intra.columns]
            ic=pd.to_numeric(intra["Close"],errors="coerce").dropna()
            if not ic.empty:
                ts=ic.index[-1]
                ts_utc=pd.Timestamp(ts)
                if ts_utc.tzinfo is None: ts_utc=ts_utc.tz_localize("America/New_York")
                age=(pd.Timestamp.now(tz="UTC")-ts_utc.tz_convert("UTC")).total_seconds()/60
                if 0<=age<=20:
                    p=float(ic.iloc[-1])
                    intraday_timestamp=ts_utc.isoformat()
                    intraday_source="Yahoo Finance 1-minute bars, extended hours (非保证实时)"
                    intraday_change_pct=(p/daily_close-1)*100 if daily_close else None
    except Exception as exc:
        print(f"{symbol}: 分钟行情不可用，回退日线收盘价：{type(exc).__name__}: {exc}")
    ma20=float(close.tail(20).mean())
    ma50=float(close.tail(50).mean()) if len(close)>=50 else None
    prior=vol.iloc[-21:-1].dropna() if len(vol)>=21 else vol.iloc[:-1].dropna()
    avg=float(prior.mean()) if not prior.empty else 0.0
    last=float(vol.iloc[-1]) if pd.notna(vol.iloc[-1]) else 0.0
    old5=float(close.iloc[-6]) if len(close)>=6 else None
    return {"ticker":symbol,"as_of":str(close.index[-1].date()),"current_price":p,
      "daily_close":daily_close,"intraday_timestamp":intraday_timestamp,
      "quote_source":intraday_source,"intraday_change_vs_daily_close_pct":intraday_change_pct,
      "ma20":ma20,"ma20_deviation_pct":(p/ma20-1)*100 if ma20 else None,
      "ma50":ma50,"ma50_deviation_pct":(p/ma50-1)*100 if ma50 else None,
      "return_5d_pct":(p/old5-1)*100 if old5 else None,
      "volume_ratio_20d":last/avg if avg>0 else None,
      "volume_ratio_definition":"最新日成交量/前20个交易日平均成交量（不含当天）",
      "high_20d":float(high.tail(20).max()),"low_20d":float(low.tail(20).min()),
      "latest_news_titles":get_news(symbol),"daily_bars":len(close)}

def run_committee(data):
    if not DEEPSEEK_API_KEY:
        raise RuntimeError("缺少DEEPSEEK_API_KEY；请在GitHub Actions Secrets中配置")
    prompt=f"""
你是严格的美股盘前量化委员会。以下包含8只标的的完整行情、技术面和新闻数据。
必须在这一次回答中逐一完成全部8只股票的分析，不得遗漏、合并或编造数据。
对每只股票分别按三个角色执行：1多头分析师写1至2句逻辑；2空头风控官反驳，检查放量破位、高位派发、新闻中的增发、债务、指引下调（新闻标题只是线索）；3裁判权衡后给唯一裁决GO、GO SMALLER、NO-GO。
证据不足选NO-GO。仅考虑做多；入场区间、止损、第一目标必须合理，止损低于现价、目标高于现价；无法给出合理价格时NO-GO且价格字段为null。
禁止计算股数、仓位、风险预算，全部由Python硬风控计算。
pine_script必须为TradingView Pine Script v5，最多5行，使用hline画入场、止损、目标，价格与字段一致，不要代码围栏。
必须只输出一个JSON对象，结构严格符合：{{"results":[{{"ticker":"NVDA","bull_case":"...","bear_case_risks":"...","verdict":"NO-GO","entry_low":null,"entry_high":null,"stop_price":null,"target_price":null,"judge_reason":"...","pine_script":"..."}}]}}。
results必须恰好包含以下8个ticker且每个一次：NVDA, MSFT, AAPL, META, AMZN, TSLA, QQQ, SPY。
数据：{json.dumps(data,ensure_ascii=False,allow_nan=False)}
"""
    client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url="https://api.deepseek.com",timeout=180.0,max_retries=0)
    content=None
    try:
        for attempt in range(3):
            try:
                response=client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role":"system","content":"你是严格遵循输出格式的美股量化委员会。只返回有效JSON，不要Markdown代码围栏。"},
                        {"role":"user","content":prompt},
                    ],
                    max_tokens=8192,
                    temperature=0.1,
                    response_format={"type":"json_object"},
                    stream=False,
                    extra_body={"thinking":{"type":"disabled"}},
                )
                if not response.choices:
                    raise RuntimeError("DeepSeek API未返回任何choices")
                choice=response.choices[0]
                if choice.finish_reason=="length":
                    raise RuntimeError("DeepSeek输出达到8192 token上限并被截断，拒绝生成不完整报告")
                content=choice.message.content
                if not content or not content.strip():
                    raise RuntimeError(f"DeepSeek API返回空文本，finish_reason={choice.finish_reason}")
                break
            except Exception as exc:
                status=getattr(exc,"status_code",None)
                message=str(exc)
                lower=message.lower()
                retryable=(status in (408,409,429,500,502,503,504) or any(
                    token in lower for token in ("408","409","429","500","502","503","504","timeout","temporarily","connection reset","service unavailable")
                ))
                if not retryable or attempt==2:
                    raise RuntimeError(f"DeepSeek API调用失败（模型={DEEPSEEK_MODEL}）：{type(exc).__name__}: {message[:2500]}") from exc
                import time
                delay=10*(attempt+1)
                print(f"DeepSeek API暂时故障，{delay}秒后重试第{attempt+2}次……")
                time.sleep(delay)
    finally:
        client.close()
    try:
        batch=CommitteeBatch.model_validate_json(content)
    except Exception as exc:
        raise RuntimeError(f"DeepSeek返回的委员会JSON无效：{exc}; 内容片段={content[:1500]}") from exc
    got=[x.ticker for x in batch.results]
    if len(got)!=len(WATCHLIST) or set(got)!=set(WATCHLIST) or len(set(got))!=len(got):
        raise RuntimeError(f"委员会结果不完整或ticker重复：{got}")
    return {x.ticker:x for x in batch.results}

def pine(d):
    if d.verdict=="NO-GO" or any(x is None for x in [d.entry_low,d.entry_high,d.stop_price,d.target_price]):
        return '//@version=5\nindicator("Committee NO-GO", overlay=true)\n// No valid levels'
    candidate=(d.pine_script or "").strip()
    if candidate.startswith("//@version=5") and "hline(" in candidate and len(candidate.splitlines())<=5:
        return candidate
    mid=(float(d.entry_low)+float(d.entry_high))/2
    return "\n".join(["//@version=5",'indicator("DeepSeek Committee Levels", overlay=true)',
      f'hline({mid:.4f}, "Entry", color=color.blue)',
      f'hline({float(d.stop_price):.4f}, "Stop", color=color.red)',
      f'hline({float(d.target_price):.4f}, "Target", color=color.green)'])

def hard_risk(m,d):
    p=float(m["current_price"])
    valid=(d.entry_low is not None and d.entry_high is not None and d.stop_price is not None and d.target_price is not None
      and 0<d.entry_low<=d.entry_high and 0<d.stop_price<p<d.target_price)
    verdict=d.verdict
    if verdict!="NO-GO" and not valid:
        verdict="NO-GO"; d.judge_reason+="；Python风控否决：价格计划缺失或不合理。"
    budget=ACCOUNT_SIZE*MAX_RISK_PCT; shares=0
    if verdict!="NO-GO":
        per_share=abs(p-float(d.stop_price))
        shares=math.floor(budget/per_share) if per_share>0 else 0
        if verdict=="GO SMALLER": shares=math.floor(shares/2)
        shares=min(shares,math.floor(ACCOUNT_SIZE/p))
    d.verdict=verdict
    return {"verdict":verdict,"risk_budget_usd":round(budget,2),
      "risk_per_share_usd":round(abs(p-float(d.stop_price)),4) if d.stop_price is not None else None,
      "max_shares":max(0,shares),"max_position_value_usd":round(max(0,shares)*p,2),
      "pine_script":pine(d)}

def write_reports(rows):
    stamp=datetime.now(timezone.utc).isoformat()
    report={"generated_at_utc":stamp,"watchlist":WATCHLIST,"account_size_usd":ACCOUNT_SIZE,
      "max_risk_pct":MAX_RISK_PCT,"ai_provider":"DeepSeek API","model":DEEPSEEK_MODEL,"api_calls":1,"results":rows}
    (OUT/"daily_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=str)+"\n",encoding="utf-8")
    lines=["# DeepSeek盘前量化委员会","",f"- UTC生成时间：{stamp}",f"- 股票池：{', '.join(WATCHLIST)}",
      "- DeepSeek API批量调用次数：1","- 账户资金：10000 USD","- 单笔风险上限：1%（Python硬风控）",f"- DeepSeek模型：{DEEPSEEK_MODEL}",""]
    for x in rows:
        m,c,r=x["market"],x["committee"],x["risk"]
        lines += [f"## {x['ticker']}","",f"- 裁决：**{r['verdict']}**",f"- 现价：{m['current_price']:.4f}（日线日期 {m['as_of']}）",
          f"- 行情来源：{m.get('quote_source','未知')}；分钟数据时间：{m.get('intraday_timestamp') or '无，使用日线收盘价'}",
          f"- 最近日线收盘：{m.get('daily_close',m['current_price']):.4f}；盘中相对日线收盘：{m.get('intraday_change_vs_daily_close_pct')}",
          f"- MA20偏离：{m['ma20_deviation_pct']:.2f}%；MA50偏离：{m['ma50_deviation_pct']}",
          f"- 5日涨跌幅：{m['return_5d_pct']:.2f}%；20日量比：{m['volume_ratio_20d']}",
          f"- 20日高/低：{m['high_20d']:.4f}/{m['low_20d']:.4f}",
          f"- 多头分析师：{c['bull_case']}",f"- 空头风控官：{c['bear_case_risks']}",f"- 裁判理由：{c['judge_reason']}",
          f"- 入场：{c['entry_low']}–{c['entry_high']}；止损：{c['stop_price']}；目标：{c['target_price']}",
          f"- 风控预算：{r['risk_budget_usd']} USD；每股风险：{r['risk_per_share_usd']}；最大股数：**{r['max_shares']}**",
          "- 最新新闻："]+[f"  - {n}" for n in m["latest_news_titles"]]
        lines += ["","Pine Script v5：","\x60\x60\x60pine",r["pine_script"],"\x60\x60\x60",""]
    (OUT/"daily_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def bark(rows):
    if not BARK_KEY: print("BARK_KEY未配置，跳过推送"); return
    picks=[x for x in rows if x.get("risk",{}).get("verdict") in ("GO","GO SMALLER")]
    if not picks: print("今日无GO/GO SMALLER标的"); return
    body="\n".join(f"{x['ticker']} {x['risk']['verdict']} | {x['market']['current_price']:.2f} | 最多{x['risk']['max_shares']}股" for x in picks)
    url=f"https://api.day.app/{quote(BARK_KEY,safe='')}/{quote('DeepSeek Quant Committee',safe='')}/{quote(body,safe='')}"
    response=requests.get(url,timeout=20); response.raise_for_status()

def main():
    if not DEEPSEEK_API_KEY: raise RuntimeError("GitHub Actions未提供DEEPSEEK_API_KEY，请检查Actions Secrets配置")
    print("正在获取8只标的行情、技术面与新闻数据……")
    market={s:get_market(s) for s in WATCHLIST}
    print("行情数据齐备，向DeepSeek API发送一次批量委员会请求……")
    decisions=run_committee(list(market.values()))
    rows=[]
    for symbol in WATCHLIST:
        m=market[symbol]; d=decisions[symbol]; r=hard_risk(m,d)
        rows.append({"ticker":symbol,"market":m,"committee":d.model_dump(),"risk":r})
        print(f"{symbol}: {r['verdict']}")
    # 只有8只股票全部取得有效委员会结论后才写报告，避免空报告/假成功。
    if len(rows)!=len(WATCHLIST) or any("committee" not in x for x in rows):
        raise RuntimeError("未取得完整的8只股票委员会报告，拒绝生成报告")
    write_reports(rows)
    try: bark(rows)
    except Exception as e: print(f"Bark失败（报告已生成）：{type(e).__name__}: {e}")
    print("8只股票DeepSeek批量委员会报告已生成")

if __name__=="__main__":
    try:
        main()
    except Exception as exc:
        print(f"FATAL：{type(exc).__name__}: {exc}",file=sys.stderr)
        sys.exit(1)
