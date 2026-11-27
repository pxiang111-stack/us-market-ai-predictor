#!/usr/bin/env python3
"""Read-only Tiger OpenAPI US equity quotes. No trading client is created."""
import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tigeropen.tiger_open_config import TigerOpenClientConfig
from tigeropen.quote.quote_client import QuoteClient

REQUIRED_PERMISSION={"usQuoteBasic","usStockQuoteLv2Totalview"}
MAX_QUOTE_AGE_MINUTES=10

def _number(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError,ValueError):
        return None

def _timestamp_ms(value):
    try:
        n=int(value)
        if n<=0:
            return None
        return datetime.fromtimestamp(n/1000,tz=timezone.utc)
    except (TypeError,ValueError,OverflowError,OSError):
        return None

def get_tiger_quote_snapshot(symbols):
    config_dir=os.getenv("TIGER_CONFIG_DIR","").strip()
    if not config_dir:
        raise RuntimeError("缺少TIGER_CONFIG_DIR；GitHub Actions需要先写入TIGER_OPENAPI_CONFIG")
    config_file=Path(config_dir)/"tiger_openapi_config.properties"
    if not config_file.is_file():
        raise RuntimeError("找不到Tiger OpenAPI配置文件；请检查GitHub Secret TIGER_OPENAPI_CONFIG")
    config=TigerOpenClientConfig(props_path=config_dir)
    client=QuoteClient(config)
    try:
        permissions=client.get_quote_permission() or []
        permission_names={str(x.get("name","")) for x in permissions if isinstance(x,dict)}
        active=permission_names & REQUIRED_PERMISSION
        if not active:
            raise RuntimeError("Tiger OpenAPI未检测到美股实时行情权限（usQuoteBasic）；请在Tiger Trade App的行情商店购买OpenAPI美股L1")
        frame=client.get_stock_briefs(list(symbols),include_hour_trading=True,lang="en_US")
        if frame is None or frame.empty:
            raise RuntimeError("Tiger OpenAPI未返回美股实时行情")
        now=datetime.now(timezone.utc)
        quotes={}
        for _,row in frame.iterrows():
            symbol=str(row.get("symbol","")).upper()
            if not symbol:
                continue
            regular_price=_number(row.get("latest_price"))
            regular_ts=_timestamp_ms(row.get("latest_time"))
            ext_price=_number(row.get("hour_trading_latest_price"))
            ext_ts=_timestamp_ms(row.get("hour_trading_timestamp"))
            ext_tag=row.get("hour_trading_tag")
            use_ext=ext_price is not None and ext_price>0 and ext_ts is not None and bool(str(ext_tag or "").strip())
            price=ext_price if use_ext else regular_price
            quote_ts=ext_ts if use_ext else regular_ts
            if price is None or price<=0 or quote_ts is None:
                quotes[symbol]={"fresh":False,"error":"缺少有效价格或行情时间戳"}
                continue
            age=(now-quote_ts).total_seconds()/60
            fresh=(-1.0<=age<=MAX_QUOTE_AGE_MINUTES)
            quotes[symbol]={
                "price":price,
                "timestamp_utc":quote_ts.isoformat(),
                "age_minutes":round(age,2),
                "fresh":fresh,
                "session":str(ext_tag) if use_ext else "Regular/Last trade",
                "source":"Tiger OpenAPI US real-time quote",
                "bid":_number(row.get("bid_price")),
                "bid_size":_number(row.get("bid_size")),
                "ask":_number(row.get("ask_price")),
                "ask_size":_number(row.get("ask_size")),
                "volume":_number(row.get("volume")),
                "amount":_number(row.get("amount")),
                "open":_number(row.get("open")),
                "high":_number(row.get("high")),
                "low":_number(row.get("low")),
                "previous_close":_number(row.get("pre_close")),
            }
        return {
            "checked_at_utc":now.isoformat(),
            "source":"Tiger OpenAPI",
            "permissions":sorted(active),
            "quotes":quotes,
            "fresh_quote_count":sum(1 for q in quotes.values() if q.get("fresh")),
        }
    finally:
        close=getattr(client,"close",None)
        if callable(close):
            close()
