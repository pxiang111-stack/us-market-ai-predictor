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

WATCHLIST=["SPY","QQQ","IWM","SMH","NVDA","AAPL","MSFT","AMZN","META","GOOGL","TSLA","AMD","AVGO","MU","ARM","TSM","PLTR","SMCI","NFLX","CRM","ORCL","APP","UBER","COIN","MSTR","HOOD","GLD","SLV","TLT","XLE","SOXL"]
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
    weekly_bias:Literal["BULLISH","NEUTRAL","BEARISH"]
    weekly_confidence_pct:int=Field(ge=0,le=100)
    weekly_range_low:float
    weekly_range_high:float
    weekly_support:float
    weekly_resistance:float
    weekly_scenario:str
    weekly_trigger:str
    weekly_risk:str

class WeeklyMarketSummary(BaseModel):
    market_bias:Literal["BULLISH","NEUTRAL","BEARISH"]
    confidence_pct:int=Field(ge=0,le=100)
    base_case:str
    bull_case:str
    bear_case:str
    key_catalysts:list[str]
    index_levels:str
    risk_management:str

class CommitteeBatch(BaseModel):
    market_summary:WeeklyMarketSummary
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

def _clean_frame(raw,symbol):
    df=ticker_frame(raw,symbol)
    if df.empty:return df
    df=df.copy(); df.columns=[str(c).title() for c in df.columns]
    for c in ["Open","High","Low","Close","Volume"]:
        if c not in df.columns:return pd.DataFrame()
        df[c]=pd.to_numeric(df[c],errors="coerce")
    df=df.dropna(subset=["Open","High","Low","Close"]).sort_index()
    idx=pd.DatetimeIndex(df.index)
    idx=idx.tz_localize("America/New_York") if idx.tz is None else idx.tz_convert("America/New_York")
    df.index=idx
    return df

def _indicators(df):
    if df is None or len(df)<5:return None
    c,h,l,v=df["Close"],df["High"],df["Low"],df["Volume"]
    prev=c.shift(1); tr=pd.concat([h-l,(h-prev).abs(),(l-prev).abs()],axis=1).max(axis=1)
    avg=float(v.tail(20).mean()) if len(v)>=20 else float(v.mean())
    return {"bars":int(len(df)),"last_timestamp":df.index[-1].isoformat(),"open":float(df.Open.iloc[-1]),
      "high":float(h.iloc[-1]),"low":float(l.iloc[-1]),"close":float(c.iloc[-1]),
      "volume":float(v.iloc[-1]) if pd.notna(v.iloc[-1]) else None,
      "volume_ratio_20":float(v.iloc[-1])/avg if avg>0 and pd.notna(v.iloc[-1]) else None,
      "ma20":float(c.tail(20).mean()) if len(c)>=20 else None,
      "ma50":float(c.tail(50).mean()) if len(c)>=50 else None,
      "return_5_bars_pct":float((c.iloc[-1]/c.iloc[-6]-1)*100) if len(c)>=6 and c.iloc[-6]!=0 else None,
      "high_20":float(h.tail(20).max()),"low_20":float(l.tail(20).min()),
      "atr14":float(tr.tail(14).mean()) if len(tr)>=14 else float(tr.mean())}

def _build_4h(intra,now_et):
    rows=[]
    for day,g in intra.groupby(intra.index.date):
        for start,end,expected in [("09:30","13:30",16),("13:30","16:00",10)]:
            z=g.between_time(start,end,inclusive="left")
            if pd.Timestamp(f"{day} {end}",tz="America/New_York")>now_et or len(z)<expected:continue
            rows.append({"Timestamp":z.index[0],"Open":float(z.Open.iloc[0]),"High":float(z.High.max()),
              "Low":float(z.Low.min()),"Close":float(z.Close.iloc[-1]),"Volume":float(z.Volume.sum())})
    return pd.DataFrame(rows).set_index("Timestamp").sort_index() if rows else pd.DataFrame()

def get_market(symbol):
    from zoneinfo import ZoneInfo
    now=datetime.now(ZoneInfo("America/New_York"))
    daily=_clean_frame(yf.download(symbol,period="1y",interval="1d",auto_adjust=True,progress=False,threads=False),symbol)
    intra=_clean_frame(yf.download(symbol,period="60d",interval="15m",auto_adjust=True,progress=False,threads=False,prepost=False),symbol)
    if daily.empty or intra.empty:raise RuntimeError(f"{symbol}: 日K或15分钟K为空")
    if now.hour<16:daily=daily[daily.index.date<now.date()]
    intra=intra[(intra.index.dayofweek<5)].between_time("09:30","15:45")
    intra=intra.loc[intra.index+pd.Timedelta(minutes=15)<=now]
    h4=_build_4h(intra,now)
    di,hi,mi=_indicators(daily),_indicators(h4),_indicators(intra)
    if not di or not hi or not mi:raise RuntimeError(f"{symbol}: 日K/4小时K/15分钟K数据不足")
    latest=pd.Timestamp(mi["last_timestamp"]); age=max(0,(now-latest).total_seconds()/60)
    p=mi["close"]
    return {"ticker":symbol,"current_price":p,"daily_close":di["close"],"intraday_timestamp":latest.isoformat(),
      "quote_source":"Yahoo Finance historical OHLC (15m; unofficial/delayed)","quote_verified_realtime":False,
      "quote_fallback_kind":"historical_ohlc","quote_age_minutes":round(age,1),"market_session":et_session(now),
      "market_session_time_et":now.isoformat(),"daily":di,"four_hour":hi,"fifteen_minute":mi,
      "ma20":di.get("ma20"),"ma20_deviation_pct":(p/di["ma20"]-1)*100 if di.get("ma20") else None,
      "ma50":di.get("ma50"),"ma50_deviation_pct":(p/di["ma50"]-1)*100 if di.get("ma50") else None,
      "return_5d_pct":di.get("return_5_bars_pct"),"volume_ratio_20d":di.get("volume_ratio_20"),
      "high_20d":di.get("high_20"),"low_20d":di.get("low_20"),"atr14":di.get("atr14"),
      "latest_news_titles":get_news(symbol),"daily_bars":di["bars"],"data_status":"OK",
      "signal_type":"MULTI_TIMEFRAME_SETUP_NOT_REALTIME_EXECUTION"}

def run_committee(data):
    if not DEEPSEEK_API_KEY: raise RuntimeError("缺少DEEPSEEK_API_KEY")
    prompt=f"""
你是严格的美股周度策略委员会。预测下一完整交易周（2026-11-30至2026-12-04）的美股走势；逐一分析以下31只标的，不得遗漏。
先给出全市场指数风险判断，再给每只标的未来一周方向、预期波动区间、关键支撑压力、触发条件和主要风险。预测是概率情景，不是保证。
每只股票分别给出多头分析师逻辑、空头风控官反驳（检查放量破位、高位派发、增发、债务、指引下调）、裁判结论。
输入包含日K、4小时K、15分钟K，均为Yahoo Finance历史OHLC而非交易所实时行情。日K判断主趋势，4小时K判断波段结构，15分钟K寻找入场形态。日K方向与4小时结构优先一致；15分钟只负责入场确认，不能单独反转大周期方向。若周期冲突、K线不足、成交量缺失或结构不清晰，NO-GO。GO/GO SMALLER代表条件式交易计划候选，不代表实时可成交；下单前必须用券商现价复核，价格离开入场区间则不追单。止损必须低于参考价，目标必须高于参考价。
对单只标的交易计划仍仅考虑做多；证据不足NO-GO。周度方向允许看多/中性/看空。weekly_range_low必须小于weekly_range_high；weekly_support和weekly_resistance必须符合数值逻辑。请特别分析QQQ、SPY、SMH、NVDA、AMD、SOXL之间的联动。不要把历史行情说成实时行情。
pine_script使用TradingView Pine Script v5，最多5行，hline画入场/止损/目标，不要代码围栏。
只输出JSON：{{"market_summary":{{"market_bias":"NEUTRAL","confidence_pct":55,"base_case":"...","bull_case":"...","bear_case":"...","key_catalysts":["..."],"index_levels":"...","risk_management":"..."}},
"results":[{{"ticker":"SPY","bull_case":"...","bear_case_risks":"...","verdict":"NO-GO","entry_low":null,"entry_high":null,"stop_price":null,"target_price":null,"judge_reason":"...","pine_script":"...","weekly_bias":"NEUTRAL","weekly_confidence_pct":55,"weekly_range_low":700,"weekly_range_high":740,"weekly_support":700,"weekly_resistance":740,"weekly_scenario":"...","weekly_trigger":"...","weekly_risk":"..."}}]}}
market_summary必须包含全市场下周研判；results必须恰好包含以下31只且每只一次：{", ".join(WATCHLIST)}
多周期行情数据（日K/4小时K/15分钟K）：{json.dumps(data,ensure_ascii=False,allow_nan=False)}
"""
    client=OpenAI(api_key=DEEPSEEK_API_KEY,base_url="https://api.deepseek.com",timeout=240.0,max_retries=0)
    content=None
    try:
        for attempt in range(3):
            try:
                r=client.chat.completions.create(model=DEEPSEEK_MODEL,messages=[
                  {"role":"system","content":"只返回符合要求的有效JSON，不要Markdown代码围栏。"},
                  {"role":"user","content":prompt}],max_tokens=22000,temperature=0.1,
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
    if len(got)!=len(WATCHLIST) or set(got)!=set(WATCHLIST) or len(set(got))!=len(WATCHLIST): raise RuntimeError(f"委员会结果不完整或ticker重复：{got}")
    return batch

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
    p=float(m.get("current_price") or 0); verdict=d.verdict
    valid=(d.entry_low is not None and d.entry_high is not None and d.stop_price is not None and d.target_price is not None
      and 0<d.entry_low<=d.entry_high and 0<d.stop_price<p<d.target_price)
    if p<=0 or m.get("data_status")!="OK" or not all(m.get(k) for k in ("daily","four_hour","fifteen_minute")):
        verdict="NO-GO";d.judge_reason+="；Python硬风控否决：多周期K线数据不完整。"
    if verdict!="NO-GO" and not valid:
        verdict="NO-GO";d.judge_reason+="；Python硬风控否决：价格计划缺失或不合理。"
    if verdict!="NO-GO" and not (float(d.entry_low)<=p<=float(d.entry_high)):
        verdict="NO-GO";d.judge_reason+=f"；Python硬风控否决：参考价{p:.4f}不在入场区间{d.entry_low:.4f}-{d.entry_high:.4f}内。"
    budget=ACCOUNT_SIZE*MAX_RISK_PCT;shares=0
    if verdict!="NO-GO":
        shares=math.floor(budget/abs(p-float(d.stop_price)))
        if verdict=="GO SMALLER":shares=math.floor(shares/2)
        shares=min(shares,math.floor(ACCOUNT_SIZE/p))
    d.verdict=verdict
    return {"verdict":verdict,"risk_budget_usd":round(budget,2),"risk_per_share_usd":round(abs(p-float(d.stop_price)),4) if d.stop_price is not None else None,
      "max_shares":max(0,shares),"max_position_value_usd":round(max(0,shares)*p,2),
      "execution_note":"历史K线条件式计划；下单前必须用券商实时价格复核","pine_script":pine(d)}

def write_reports(rows):
    stamp=datetime.now(timezone.utc).isoformat()
    rows=sorted(rows,key=lambda x:(x["risk"]["verdict"]=="NO-GO",WATCHLIST.index(x["ticker"])))
    report={"generated_at_utc":stamp,"watchlist":WATCHLIST,"account_size_usd":ACCOUNT_SIZE,"max_risk_pct":MAX_RISK_PCT,
      "ai_provider":"DeepSeek API","model":DEEPSEEK_MODEL,"api_calls":1,"market_data_provider":"Yahoo Finance historical OHLC (Daily + 4H + 15m)","results":rows}
    (OUT/"daily_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=str)+"\n",encoding="utf-8")
    first=rows[0]["market"] if rows else {}
    session=first.get("market_session","UNKNOWN")
    session_label={"PRE":"盘前 PRE","REGULAR":"盘中 REGULAR","POST":"盘后 POST","CLOSED":"休市 CLOSED"}.get(session,session)
    latest=max((x["market"].get("intraday_timestamp") or "" for x in rows),default="")
    lines=["# DeepSeek量化委员会｜30只标的盘前/盘中报告","",f"- 当前交易时段：**{session_label}**（美东）",f"- 最新15分钟K线时间：**{latest or '无可验证报价'}**",f"- UTC生成时间：{stamp}",
      "- 行情源：Yahoo Finance 日K + 4小时K（由15分钟聚合）+ 15分钟K历史OHLC（非实时）",
      "- GO/GO SMALLER仅代表条件式交易计划候选，不是实时成交信号；实际下单前必须用券商现价复核",
      "- 账户资金：10,000 USD；单笔风险预算：100 USD（1%）",""]
    picks=[x for x in rows if x["risk"]["verdict"] in ("GO","GO SMALLER")]
    rejects=[x for x in rows if x["risk"]["verdict"]=="NO-GO"]
    lines += ["## 一、通过标的（GO / GO SMALLER）",""]
    if not picks: lines += ["本次没有通过标的。",""]
    for x in picks:
        m,c,r=x["market"],x["committee"],x["risk"]
        lines += [f"### {x['ticker']}｜{r['verdict']}","",f"- 当前参考现价：**{m['current_price']:.4f}**",
          f"- 最新15分钟K线：{m['intraday_timestamp'] or '无时间戳'}；来源：{m['quote_source']}；非实时历史数据；距今：{m['quote_age_minutes']}分钟",
          f"- 日K：收盘 {m['daily']['close']:.2f}，MA20 {m['daily'].get('ma20')}，MA50 {m['daily'].get('ma50')}；4小时：收盘 {m['four_hour']['close']:.2f}，MA20 {m['four_hour'].get('ma20')}；15分钟：收盘 {m['fifteen_minute']['close']:.2f}，MA20 {m['fifteen_minute'].get('ma20')}",
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

def write_weekly_report(rows,summary):
    from zoneinfo import ZoneInfo
    stamp=datetime.now(timezone.utc).isoformat()
    generated=datetime.now(ZoneInfo("America/New_York")).isoformat()
    rows=sorted(rows,key=lambda x:(-x["committee"]["weekly_confidence_pct"],x["ticker"]))
    report={"generated_at_utc":stamp,"forecast_period":"2026-11-30 to 2026-12-04",
      "data_cutoff_note":"Yahoo Finance historical OHLC; last completed bars only; not exchange-grade real-time data",
      "market_summary":summary.model_dump(),"results":[{"ticker":x["ticker"],"reference_price":x["market"]["current_price"],
      "latest_15m_bar":x["market"]["intraday_timestamp"],"weekly":{k:x["committee"][k] for k in
      ["weekly_bias","weekly_confidence_pct","weekly_range_low","weekly_range_high","weekly_support","weekly_resistance","weekly_scenario","weekly_trigger","weekly_risk"]},
      "daily":x["market"]["daily"],"four_hour":x["market"]["four_hour"]} for x in rows]}
    (OUT/"weekly_outlook.json").write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False,default=str)+"\\n",encoding="utf-8")
    label={"BULLISH":"偏多","NEUTRAL":"震荡/中性","BEARISH":"偏空"}.get(summary.market_bias,summary.market_bias)
    lines=["# 美股下周展望｜2026-11-30 至 2026-12-04","",
      f"- 生成时间（美东）：{generated}",f"- 模型：DeepSeek {DEEPSEEK_MODEL}",
      "- 数据：Yahoo Finance历史日K、4小时K、15分钟K；不是实时行情。预测依据为截至最近已收盘K线及可获取新闻。",
      "- 注意：周度区间是模型情景估计，不是保证价格；开盘跳空后须重新评估。","",
      "## 一、全市场判断","",f"**总体方向：{label}｜置信度 {summary.confidence_pct}%**","",
      f"- 基准情景：{summary.base_case}",f"- 偏多情景：{summary.bull_case}",f"- 偏空情景：{summary.bear_case}",
      f"- 指数关键位：{summary.index_levels}","- 主要催化剂："]
    lines += [f"  - {x}" for x in summary.key_catalysts]
    lines += ["",f"- 风险控制：{summary.risk_management}","","## 二、重点标的下周预测","",
      "| 标的 | 方向 | 置信度 | 预期区间 | 支撑 | 压力 |","|---|---|---:|---:|---:|---:|"]
    for x in rows:
        c=x["committee"]; bias={"BULLISH":"偏多","NEUTRAL":"中性","BEARISH":"偏空"}[c["weekly_bias"]]
        lines.append(f"| {x['ticker']} | {bias} | {c['weekly_confidence_pct']}% | {c['weekly_range_low']:.2f}-{c['weekly_range_high']:.2f} | {c['weekly_support']:.2f} | {c['weekly_resistance']:.2f} |")
    for x in rows:
        c=x["committee"]; bias={"BULLISH":"偏多","NEUTRAL":"中性","BEARISH":"偏空"}[c["weekly_bias"]]
        lines += ["",f"### {x['ticker']}｜{bias}",
          f"- 参考价（非实时）：{x['market']['current_price']:.2f}；最新15分钟K：{x['market']['intraday_timestamp']}",
          f"- 下周预期波动区间：**{c['weekly_range_low']:.2f} – {c['weekly_range_high']:.2f}**",
          f"- 支撑：{c['weekly_support']:.2f}；压力：{c['weekly_resistance']:.2f}；置信度：{c['weekly_confidence_pct']}%",
          f"- 基本路径：{c['weekly_scenario']}",f"- 关键触发：{c['weekly_trigger']}",f"- 主要风险：{c['weekly_risk']}"]
    (OUT/"weekly_outlook.md").write_text("\\n".join(lines)+"\\n",encoding="utf-8")

def bark(rows):
    if not BARK_KEY: print("BARK_KEY未配置，跳过推送"); return
    picks=[x for x in rows if x["risk"]["verdict"] in ("GO","GO SMALLER")]
    if not picks: print("今日无GO/GO SMALLER标的"); return
    body="\n".join(f"{x['ticker']} {x['risk']['verdict']} | {x['market']['current_price']:.2f} | 最多{x['risk']['max_shares']}股" for x in picks)
    requests.get(f"https://api.day.app/{quote(BARK_KEY,safe='')}/{quote('DeepSeek Quant Committee',safe='')}/{quote(body,safe='')}",timeout=20).raise_for_status()

def main():
    if not DEEPSEEK_API_KEY:raise RuntimeError("GitHub Actions未提供DEEPSEEK_API_KEY")
    print(f"获取{len(WATCHLIST)}只标的日K、4小时K和15分钟K（历史OHLC，不宣称实时行情）。")
    market={};failures={}
    for sym in WATCHLIST:
        try:
            market[sym]=get_market(sym)
            print(f"{sym}: 日K {market[sym]['daily']['bars']}根 | 4H {market[sym]['four_hour']['bars']}根 | 15m {market[sym]['fifteen_minute']['bars']}根 | 参考价 {market[sym]['current_price']:.4f}")
        except Exception as e:failures[sym]=f"{type(e).__name__}: {e}"
    if failures:raise RuntimeError("多周期行情不完整，停止委员会分析："+json.dumps(failures,ensure_ascii=False))
    print(f"多周期数据完整：{len(market)}/{len(WATCHLIST)}；所有计划均需券商现价复核。")
    batch=run_committee(list(market.values()));decisions={x.ticker:x for x in batch.results};rows=[]
    for sym in WATCHLIST:
        m,d=market[sym],decisions[sym];r=hard_risk(m,d)
        rows.append({"ticker":sym,"market":m,"committee":d.model_dump(),"risk":r});print(f"{sym}: {r['verdict']}")
    if len(rows)!=len(WATCHLIST):raise RuntimeError("委员会报告不完整")
    write_reports(rows)
    write_weekly_report(rows,batch.market_summary)
    try:bark(rows)
    except Exception as e:print(f"Bark失败：{type(e).__name__}: {e}")
    print(f"{len(WATCHLIST)}只标的多周期委员会及下周展望报告生成完成；交易计划不是实时成交信号。")

if __name__=="__main__":
    try: main()
    except Exception as e:
        print(f"FATAL：{type(e).__name__}: {e}",file=sys.stderr); sys.exit(1)
