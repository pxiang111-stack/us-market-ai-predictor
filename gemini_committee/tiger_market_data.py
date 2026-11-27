#!/usr/bin/env python3
"""Read-only Tiger OpenAPI US equity/ETF quotes; prefer real-time, label delayed fallback."""
import os
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient

REALTIME_MAX_AGE_MIN=10
DELAYED_MAX_AGE_MIN=20

def _number(value):
    try:
        if value is None or pd.isna(value): return None
        n=float(value)
        return n if n>0 else None
    except (TypeError,ValueError): return None

def _timestamp_ms(value):
    try:
        n=int(float(value))
        if n<=0: return None
        return datetime.fromtimestamp(n/1000,tz=timezone.utc)
    except (TypeError,ValueError,OverflowError,OSError): return None

def _records(frame):
    if frame is None or getattr(frame,"empty",True): return {}
    result={}
    for _,row in frame.iterrows():
        symbol=str(row.get("symbol","")).upper()
        if symbol: result[symbol]=row
    return result

def get_tiger_quote_snapshot(symbols):
    config_dir=os.getenv("TIGER_CONFIG_DIR","").strip()
    if not config_dir: raise RuntimeError("缺少TIGER_CONFIG_DIR；未检测到Tiger OpenAPI配置")
    config_file=Path(config_dir)/"tiger_openapi_config.properties"
    if not config_file.is_file(): raise RuntimeError("找不到Tiger配置文件；请确认GitHub Secret TIGER_OPENAPI_CONFIG")
    config=TigerOpenClientConfig(props_path=config_dir)
    client=QuoteClient(config)
    try:
        realtime_error=None
        try:
            live=_records(client.get_stock_briefs(symbols=list(symbols),include_hour_trading=True,lang="en_US"))
        except Exception as exc:
            live={}
            realtime_error=f"{type(exc).__name__}: {str(exc)[:300]}"
        delayed={}
        if not live:
            try:
                delayed=_records(client.get_stock_delay_briefs(symbols=list(symbols),lang="en_US"))
            except Exception as exc:
                raise RuntimeError(f"Tiger实时行情请求失败：{realtime_error}; 延迟行情请求也失败：{type(exc).__name__}: {str(exc)[:300]}") from exc
        now=datetime.now(timezone.utc)
        quotes={}
        for symbol in symbols:
            row=live.get(symbol)
            is_delayed=False
            if row is None:
                row=delayed.get(symbol)
                is_delayed=True
            if row is None:
                quotes[symbol]={"fresh":False,"error":"Tiger未返回该标的行情"}
                continue
            if is_delayed:
                price=_number(row.get("close"))
                ts=_timestamp_ms(row.get("time"))
                basis="Tiger delayed quote"
                session="Delayed (约15分钟)"
                max_age=DELAYED_MAX_AGE_MIN
                bid=ask=None
                volume=_number(row.get("volume"))
                open_price=_number(row.get("open"))
                high=_number(row.get("high"))
                low=_number(row.get("low"))
                prev=_number(row.get("pre_close"))
                amount=None
            else:
                regular_price=_number(row.get("latest_price"))
                regular_ts=_timestamp_ms(row.get("latest_time"))
                ext_price=_number(row.get("hour_trading_latest_price"))
                ext_ts=_timestamp_ms(row.get("hour_trading_timestamp"))
                ext_tag=row.get("hour_trading_tag")
                use_ext=ext_price is not None and ext_ts is not None and bool(str(ext_tag or "").strip())
                price=ext_price if use_ext else regular_price
                ts=ext_ts if use_ext else regular_ts
                basis="Tiger OpenAPI real-time"
                session=str(ext_tag) if use_ext else "Regular/Last trade"
                max_age=REALTIME_MAX_AGE_MIN
                bid=_number(row.get("bid_price"))
                ask=_number(row.get("ask_price"))
                volume=_number(row.get("volume"))
                open_price=_number(row.get("open"))
                high=_number(row.get("high"))
                low=_number(row.get("low"))
                prev=_number(row.get("pre_close"))
                amount=_number(row.get("amount"))
            if price is None or ts is None:
                quotes[symbol]={"fresh":False,"error":"Tiger行情缺少有效价格或时间戳"}
                continue
            age=(now-ts).total_seconds()/60
            fresh=(-1.0<=age<=max_age)
            quotes[symbol]={"price":price,"timestamp_utc":ts.isoformat(),"age_minutes":round(age,2),
                "fresh":fresh,"session":session,"source":basis,"price_basis":basis,
                "bid":bid,"ask":ask,"volume":volume,"amount":amount,
                "open":open_price,"high":high,"low":low,"previous_close":prev,
                "feed_type":"delayed" if is_delayed else "realtime"}
        return {"checked_at_utc":now.isoformat(),"source":"Tiger OpenAPI",
            "feed_type":"delayed" if delayed and not live else "realtime",
            "realtime_error":realtime_error,"quotes":quotes,
            "fresh_quote_count":sum(1 for q in quotes.values() if q.get("fresh"))}
    finally:
        close=getattr(client,"close",None)
        if callable(close): close()
