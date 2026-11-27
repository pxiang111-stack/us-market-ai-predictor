#!/usr/bin/env python3
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

WATCHLIST=["SPY","QQQ","IWM","SMH","NVDA","AAPL","MSFT","AMZN","META","GOOGL","TSLA","AMD","AVGO","MU","ARM","TSM","PLTR","SMCI","NFLX","CRM","ORCL","APP","UBER","COIN","MSTR","HOOD","GLD","SLV","TLT","XLE"]
ACCOUNT_SIZE=10000.0
MAX_RISK_PCT=0.01
DEEPSEEK_MODEL="deepseek-flash"
DEEPSEEK_API_KEY=os.getenv("DEEPSEEK_API_KEY","").strip()
BARK_KEY=os.getenv("BARK_KEY","").strip()
OUT=Path(__file__).resolve().parent

class CommitteeDecision(BaseModel):
    ticker:str
    bull_case:str=Field(description="多头分析师逻辑")
    bear_case_risks:str=Field(description="空头风控官反驳")
    verdict:Literal["GO","GO SMALLER","NO-GO"]
    entry_low:Optional[float]=None
    entry_high:Optional[float]=None
    stop_price:Optional[float]=None
    target_price:Optional[float]=None
    judge_reason:str
    pine_script:str=Field(description="Pine Script v5，最多5行")

class CommitteeBatch(BaseModel):
    results:list[CommitteeDecision]

def ticker_frame(frame,symbol):
    if frame is None or frame.empty: return pd.DataFrame()
    if not isinstance(frame.columns,pd.MultiIndex): return frame.copy()
    if symbol in frame.columns.get_level_values(0): out=frame[symbol].copy()
    elif symbol in frame.columns.get_level_values(-1): out=frame.xs(symbol,axis=1,level=-1).copy()
    else: return pd.DataFrame()
    if isinstance(out.columns,pd.MultiIndex): out.columns=[str(c[0]) for c in out.columns]
    return out

def get_news(symbol):
    try:
        out=[]
        for row in (yf.Ticker(symbol).news or []):
            item=row.get("content",row); title=item.get("title") or row.get("title")
            if title and title not in out: out.append(str(title).strip())
            if len(out)>=4: break
        return out
    except Exception as e: return [f"新闻获取失败：{type(e).__name__}: {e}"]

def et_session(now=None):
    from zoneinfo import ZoneInfo
    now=now or datetime.now(ZoneInfo("America/New_York"))
    minute=now.hour*60+now.minute
    if now.weekday()>=5: return "CLOSED"
    if 240<=minute<570: return "PRE"
    if 570<=minute<960: return "REGULAR"
    if 960<=minute<1200: return "POST"
    return "CLOSED"

BINANCE_BASE="https://fapi.binance.com"
BINANCE_CONTRACTS={s:s+"USDT" for s in WATCHLIST if s not in {"GLD","SLV","TLT"}}
BINANCE_UNAVAILABLE={"GLD","SLV","TLT"}

def binance_klines(symbol,interval,limit):
    r=requests.get(f"{BINANCE_BASE}/fapi/v1/continuousKlines",
      params={"pair":symbol,"contractType":"TRADIFI_PERPETUAL","interval":interval,"limit":limit},timeout=25)
    r.raise_for_status()
    data=r.json()
    if not isinstance(data,list) or not data: raise RuntimeError(f"Binance无K线：{symbol} {interval}")
    return data

def get_market(symbol):
    from zoneinfo import ZoneInfo
    now_et=datetime.now(ZoneInfo("America/New_York"))
    if symbol in BINANCE_UNAVAILABLE:
        return {"ticker":symbol,"current_price":None,"daily_close":None,"intraday_timestamp":None,
          "quote_source":"Binance无对应TradFi永续合约","quote_verified_realtime":False,
          "quote_fallback_kind":"unsupported_symbol","quote_age_minutes":None,
          "market_session":et_session(now_et),"market_session_time_et":now_et.isoformat(),
          "ma20":None,"ma20_deviation_pct":None,"ma50":None,"ma50_deviation_pct":None,
          "return_5d_pct":None,"volume_ratio_20d":None,"high_20d":None,"low_20d":None,
          "atr14":None,"latest_news_titles":[],"daily_bars":0,"data_status":"UNAVAILABLE"}
    pair=BINANCE_CONTRACTS[symbol]
    daily=binance_klines(pair,"1d",100)
    intraday=binance_klines(pair,"1m",100)
    now_ms=int(datetime.now(timezone.utc).timestamp()*1000)
    valid=[k for k in intraday if int(k[6])<=now_ms+5000 and float(k[4])>0 and float(k[5])>0]
    if not valid: raise RuntimeError(f"{symbol}: Binance无有效1分钟成交K线")
    latest=valid[-1]
    qtime=datetime.fromtimestamp(int(latest[0])/1000,tz=timezone.utc)
    age=max(0,(now_ms-int(latest[0]))/60000)
    if age>3: raise RuntimeError(f"{symbol}: Binance 1分钟K线过期，{age:.1f}分钟")
    price=float(latest[4])
    d=pd.DataFrame(daily,columns=["open_time","Open","High","Low","Close","Volume","close_time","quote_volume","trades","taker_buy_base","taker_buy_quote","ignore"])
    for col in ["Open","High","Low","Close","Volume"]: d[col]=pd.to_numeric(d[col],errors="coerce")
    d=d.dropna(subset=["High","Low","Close"]).sort_values("open_time")
    close,high,low,vol=d["Close"],d["High"],d["Low"],d["Volume"]
    if len(d)<15: raise RuntimeError(f"{symbol}: Binance日线不足15根")
    prev=close.shift(1)
    tr=pd.concat([high-low,(high-prev).abs(),(low-prev).abs()],axis=1).max(axis=1)
    ma20=float(close.tail(20).mean()); ma50=float(close.tail(50).mean()) if len(close)>=50 else None
    avg=float(vol.iloc[-21:-1].mean()) if len(vol)>=21 else float(vol.iloc[:-1].mean())
    c5=float(close.iloc[-6]) if len(close)>=6 else None
    return {"ticker":symbol,"current_price":price,"daily_close":float(close.iloc[-1]),
      "intraday_timestamp":qtime.isoformat(),"quote_source":"Binance USDⓈ-M TradFi perpetual 1分钟K线",
      "quote_verified_realtime":True,"quote_fallback_kind":None,"quote_age_minutes":round(age,2),
      "market_session":et_session(now_et),"market_session_time_et":now_et.isoformat(),
      "ma20":ma20,"ma20_deviation_pct":(price/ma20-1)*100 if ma20 else None,
      "ma50":ma50,"ma50_deviation_pct":(price/ma50-1)*100 if ma50 else None,
      "return_5d_pct":(price/c5-1)*100 if c5 else None,
      "volume_ratio_20d":float(vol.iloc[-1])/avg if avg>0 else None,
      "high_20d":float(high.tail(20).max()),"low_20d":float(low.tail(20).min()),
      "atr14":float(tr.tail(14).mean()),"latest_news_titles":get_news(symbol),
      "daily_bars":len(d),"data_status":"OK","binance_symbol":pair}


def run_committee(data):
    if not DEEPSEEK_API_KEY: raise RuntimeError("缺少DEEPSEEK_API_KEY")
    prompt=f"""
你是严格的美股盘前/盘中量化委员会。逐一分析全部30只股票，不得遗漏。
每只股票分别给出多头分析师逻辑、空头风控官反驳（检查放量破位、高位派发、增发、债务、指引下调）、裁判结论。
行情源为Binance USDⓈ-M TradFi perpetual合约代理行情，不是美股交易所现货成交价。只有data_status=OK且quote_verified_realtime=true、报价时间戳不超过3分钟的币安1分钟K线，才可用于GO/GO SMALLER。data_status=UNAVAILABLE必须NO-GO。GO/GO SMALLER入场区间必须覆盖current_price；跳空突破或跌破原区间必须按现价重算，否则NO-GO。止损必须低于现价，目标必须高于现价。
仅考虑做多。证据不足NO-GO。不要计算股数/仓位，交给Python硬风控。
pine_script使用TradingView Pine Script v5，最多5行，hline画入场/止损/目标，不要代码围栏。
只输出JSON：{{"results":[{{"ticker":"SPY","bull_case":"...","bear_case_risks":"...","verdict":"NO-GO","entry_low":null,"entry_high":null,"stop_price":null,"target_price":null,"judge_reason":"...","pine_script":"..."}}]}}
results必须恰好包含以下30只且每只一次：{", ".join(WATCHLIST)}
行情数据（Binance TradFi永续代理行情）：{json.dumps(data,ensure_ascii=False,allow_nan=False)}
"""
    client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url="https://api.deepseek.com",timeout=240.0,max_retries=0)
    content=None
    try:
        for attempt in range(3):
            try:
                r=client.chat.completions.create(model=DEEPSEEK_MODEL,messages=[
                  {"role":"system","content":"只返回符合要求的有效JSON，不要Markdown代码围栏。"},
                  {"role":"user","content":prompt}],max_tokens=16000,temperature=0.1,
                  response_format={"type":"json_object"},stream=False,
                  extra_body={"thinking":{"type":"disabled"}})
                if not r.choices: raise RuntimeError("DeepSeek无choices")
                ch=r.choices[0]
                if ch.finish_reason=="length": raise RuntimeError("DeepSeek输出达到token上限并被截断")
                content=ch.message.content
                if not content or not content.strip(): raise RuntimeError("DeepSeek返回空文本")
                break
            except Exception as e:
                status=getattr(e,"status_code",None); msg=str(e).lower()
                retry=status in (408,409,429,500,502,503,504) or any(x in msg for x in ("timeout","429","500","502","503","504","connection reset"))
                if not retry or attempt==2: raise
                import time; time.sleep(10*(attempt+1))
    finally: client.close()
    batch=CommitteeBatch.model_validate_json(content)
    got=[x.ticker for x in batch.results]
    if len(got)!=30 or set(got)!=set(WATCHLIST) or len(set(got))!=30: raise RuntimeError(f"委员会结果不完整或ticker重复：{got}")
    return {x.ticker:x for x in batch.results}

def pine(d):
    if d.verdict=="NO-GO" or any(x is None for x in [d.entry_low,d.entry_high,d.stop_price,d.target_price]):
        return '//@version=5\nindicator("Committee NO-GO", overlay=true)\n// No valid levels'
    s=(d.pine_script or "").strip()
    if s.startswith("//@version=5") and "hline(" in s and len(s.splitlines())<=5: return s
    mid=(float(d.entry_low)+float(d.entry_high))/2
    return "\n".join(["//@version=5",'indicator("DeepSeek Committee Levels", overlay=true)',
      f'hline({mid:.4f}, "Entry", color=color.blue)',f'hline({float(d.stop_price):.4f}, "Stop", color=color.red)',
      f'hline({float(d.target_price):.4f}, "Target", color=color.green)'])

def hard_risk(m,d):
    p=float(m["current_price"]) if m.get("current_price") is not None else 0.0; verdict=d.verdict
    valid=(d.entry_low is not None and d.entry_high is not None and d.stop_price is not None and d.target_price is not None
      and 0<d.entry_low<=d.entry_high and 0<d.stop_price<p<d.target_price)
    if p<=0 or m.get("data_status")!="OK":
        verdict="NO-GO"; d.judge_reason+="；Python硬风控否决：币安无该标的合约行情。"
    if verdict!="NO-GO" and not m.get("quote_verified_realtime",False):
        verdict="NO-GO"; d.judge_reason+="；Python硬风控否决：没有带时间戳且3分钟内的1分钟实时K线。"
    if verdict!="NO-GO" and not valid:
        verdict="NO-GO"; d.judge_reason+="；Python硬风控否决：价格计划缺失或不合理。"
    if verdict!="NO-GO" and not (float(d.entry_low)<=p<=float(d.entry_high)):
        verdict="NO-GO"; d.judge_reason+=f"；Python硬风控否决：现价{p:.4f}不在入场区间{d.entry_low:.4f}-{d.entry_high:.4f}内。"
    budget=ACCOUNT_SIZE*MAX_RISK_PCT; shares=0
    if verdict!="NO-GO":
        shares=math.floor(budget/abs(p-float(d.stop_price)))
        if verdict=="GO SMALLER": shares=math.floor(shares/2)
        shares=min(shares,math.floor(ACCOUNT_SIZE/p))
    d.verdict=verdict
    return {"verdict":verdict,"risk_budget_usd":round(budget,2),
      "risk_per_share_usd":round(abs(p-float(d.stop_price)),4) if d.stop_price is not None else None,
      "max_shares":max(0,shares),"max_position_value_usd":round(max(0,shares)*p,2),"pine_script":pine(d)}

def write_reports(rows):
    stamp=datetime.now(timezone.utc).isoformat()
    rows=sorted(rows,key=lambda x:(x["risk"]["verdict"]=="NO-GO",WATCHLIST.index(x["ticker"])))
    report={"generated_at_utc":stamp,"watchlist":WATCHLIST,"account_size_usd":ACCOUNT_SIZE,"max_risk_pct":MAX_RISK_PCT,
      "ai_provider":"DeepSeek API","model":DEEPSEEK_MODEL,"api_calls":1,"market_data_provider":"Binance USDⓈ-M TradFi perpetual (public market data)","results":rows}
    (OUT/"daily_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=str)+"\n",encoding="utf-8")
    first=rows[0]["market"] if rows else {}
    session=first.get("market_session","UNKNOWN")
    session_label={"PRE":"盘前 PRE","REGULAR":"盘中 REGULAR","POST":"盘后 POST","CLOSED":"休市 CLOSED"}.get(session,session)
    latest=max((x["market"].get("intraday_timestamp") or "" for x in rows),default="")
    lines=["# DeepSeek量化委员会｜30只标的盘前/盘中报告","",f"- 当前交易时段：**{session_label}**（美东）",f"- 最新可验证1分钟行情时间：**{latest or '无可验证报价'}**",f"- UTC生成时间：{stamp}",
      "- 日线：Yahoo Finance 3mo/1d；盘前盘中盘后：Yahoo Finance 2d/1m，prepost=True（目标覆盖美东04:00-20:00）",
      "- 只有时间戳在3分钟内的1分钟K线才可产生GO；fast_info无时间戳及日线收盘价仅作参考，不视为实时行情",
      "- 账户资金：10,000 USD；单笔风险预算：100 USD（1%）",""]
    picks=[x for x in rows if x["risk"]["verdict"] in ("GO","GO SMALLER")]
    rejects=[x for x in rows if x["risk"]["verdict"]=="NO-GO"]
    lines += ["## 一、通过标的（GO / GO SMALLER）",""]
    if not picks: lines += ["本次没有通过标的。",""]
    for x in picks:
        m,c,r=x["market"],x["committee"],x["risk"]
        lines += [f"### {x['ticker']}｜{r['verdict']}","",f"- 当前参考现价：**{m['current_price']:.4f}**",
          f"- 行情时间：{m['intraday_timestamp'] or '无时间戳'}；来源：{m['quote_source']}；实时验证：{'是' if m['quote_verified_realtime'] else '否'}；数据年龄：{m['quote_age_minutes']}分钟",
          f"- 入场区间：**{c['entry_low']} – {c['entry_high']}**；止损：**{c['stop_price']}**；目标：**{c['target_price']}**",
          f"- 最大股数：**{r['max_shares']}股**；每股风险：{r['risk_per_share_usd']} USD；风险预算：{r['risk_budget_usd']} USD",
          f"- 多头分析师：{c['bull_case']}",f"- 空头风控官：{c['bear_case_risks']}",f"- 裁判结论：{c['judge_reason']}",
          f"- 技术参考：MA20 {m['ma20']:.2f}；MA50 {m['ma50']}；20日支撑/阻力 {m['low_20d']:.2f}/{m['high_20d']:.2f}；ATR14 {m['atr14']:.2f}",
          f"- 新闻线索：{'；'.join(m['latest_news_titles']) if m['latest_news_titles'] else '暂无'}","",
          "Pine Script v5：","\x60\x60\x60pine",r["pine_script"],"\x60\x60\x60",""]
    lines += ["## 二、NO-GO淘汰汇总","","| 代码 | 当前参考现价 | 一句话淘汰原因 |","|---|---:|---|"]
    for x in rejects:
        reason=(x["committee"].get("judge_reason") or x["committee"].get("bear_case_risks") or "未通过风控").replace("|","/").replace("\n"," ")
        price=x["market"].get("current_price")
        price_text=f"{price:.4f}" if price is not None else "N/A（币安无合约）"
        lines.append(f"| {x['ticker']} | {price_text} | {reason} |")
    (OUT/"daily_report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")

def bark(rows):
    if not BARK_KEY: print("BARK_KEY未配置，跳过推送"); return
    picks=[x for x in rows if x["risk"]["verdict"] in ("GO","GO SMALLER")]
    if not picks: print("今日无GO/GO SMALLER标的"); return
    body="\n".join(f"{x['ticker']} {x['risk']['verdict']} | {x['market']['current_price']:.2f} | 最多{x['risk']['max_shares']}股" for x in picks)
    requests.get(f"https://api.day.app/{quote(BARK_KEY,safe='')}/{quote('DeepSeek Quant Committee',safe='')}/{quote(body,safe='')}",timeout=20).raise_for_status()

def main():
    if not DEEPSEEK_API_KEY: raise RuntimeError("GitHub Actions未提供DEEPSEEK_API_KEY")
    print("从Binance公开接口读取TradFi永续合约日线及1分钟行情；无需API Key。")
    market={}; failures={}
    for s in WATCHLIST:
        try:
            market[s]=get_market(s); m=market[s]
            px=f"{m['current_price']:.4f}" if m["current_price"] is not None else "N/A"
            print(f"{s}: {px} | {m['intraday_timestamp'] or '无报价'} | {m['quote_source']}")
        except Exception as e: failures[s]=f"{type(e).__name__}: {e}"
    if failures: raise RuntimeError("Binance行情读取失败："+json.dumps(failures,ensure_ascii=False))
    print(f"Binance新鲜1分钟行情：{sum(1 for m in market.values() if m['quote_verified_realtime'])}/30；无合约：3/30")
    decisions=run_committee(list(market.values()))
    rows=[]
    for s in WATCHLIST:
        m,d=market[s],decisions[s]; r=hard_risk(m,d)
        rows.append({"ticker":s,"market":m,"committee":d.model_dump(),"risk":r})
        print(f"{s}: {r['verdict']}")
    if len(rows)!=30: raise RuntimeError("委员会报告不完整")
    write_reports(rows)
    try: bark(rows)
    except Exception as e: print(f"Bark失败：{type(e).__name__}: {e}")
    print("30只标的币安代理行情委员会报告已生成")


if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"FATAL：{type(e).__name__}: {e}",file=sys.stderr); sys.exit(1)
