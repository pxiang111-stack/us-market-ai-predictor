#!/usr/bin/env python3
"""Free Alpaca Basic real-time IEX quotes for US equities/ETFs."""
import os
from datetime import datetime, timezone
import requests

API_URL="https://data.alpaca.markets/v2/stocks/snapshots"
MAX_QUOTE_AGE_MINUTES=10
WATCHLIST_LIMIT=30

def _dt(value):
    if not value: return None
    try: return datetime.fromisoformat(str(value).replace("Z","+00:00")).astimezone(timezone.utc)
    except (TypeError,ValueError): return None

def _num(value):
    try:
        n=float(value)
        return n if n>0 else None
    except (TypeError,ValueError): return None

def get_free_quote_snapshot(symbols):
    key=os.getenv("APCA_API_KEY_ID","").strip()
    secret=os.getenv("APCA_API_SECRET_KEY","").strip()
    if not key or not secret:
        raise RuntimeError("缺少Alpaca免费行情API凭证；请配置GitHub Secrets: APCA_API_KEY_ID、APCA_API_SECRET_KEY")
    if len(symbols)>WATCHLIST_LIMIT:
        raise RuntimeError(f"批量查询超过{WATCHLIST_LIMIT}只标的限制")
    response=requests.get(API_URL,params={"symbols":",".join(symbols),"feed":"iex"},
        headers={"APCA-API-KEY-ID":key,"APCA-API-SECRET-KEY":secret},timeout=25)
    if response.status_code in (401,403):
        raise RuntimeError("Alpaca API认证失败或账户无行情权限；请检查免费账户API Key/Secret")
    response.raise_for_status()
    payload=response.json()
    snapshots=payload.get("snapshots",payload)
    now=datetime.now(timezone.utc)
    quotes={}
    for symbol in symbols:
        snap=snapshots.get(symbol) or {}
        trade=snap.get("latestTrade") or {}
        q=snap.get("latestQuote") or {}
        trade_price,trade_ts=_num(trade.get("p")),_dt(trade.get("t"))
        bid,ask=_num(q.get("bp")),_num(q.get("ap"))
        quote_ts=_dt(q.get("t"))
        if trade_price is not None and trade_ts is not None:
            price,ts,basis=trade_price,trade_ts,"IEX latest trade"
        elif bid is not None and ask is not None and ask>=bid and quote_ts is not None:
            price,ts,basis=(bid+ask)/2,quote_ts,"IEX bid/ask midpoint (no fresh trade)"
        else:
            quotes[symbol]={"fresh":False,"error":"Alpaca IEX未返回有效成交或报价"}
            continue
        age=(now-ts).total_seconds()/60
        fresh=(-1.0<=age<=MAX_QUOTE_AGE_MINUTES)
        daily=snap.get("dailyBar") or {}
        prev=snap.get("prevDailyBar") or {}
        quotes[symbol]={"price":price,"timestamp_utc":ts.isoformat(),"age_minutes":round(age,2),
            "fresh":fresh,"session":"IEX feed","source":"Alpaca Basic free IEX","price_basis":basis,
            "bid":bid,"bid_size":q.get("bs"),"ask":ask,"ask_size":q.get("as"),
            "volume":daily.get("v"),"amount":None,"open":_num(daily.get("o")),
            "high":_num(daily.get("h")),"low":_num(daily.get("l")),
            "previous_close":_num(prev.get("c")),"minute_bar":snap.get("minuteBar")}
    return {"checked_at_utc":now.isoformat(),"source":"Alpaca Basic free IEX","feed":"iex",
        "quotes":quotes,"fresh_quote_count":sum(1 for x in quotes.values() if x.get("fresh")),
        "coverage_note":"IEX单一交易场所数据，不是全美股市场SIP合并行情"}
