#!/usr/bin/env python3
"""Incremental public Binance USD-M TradFi flow collector.

Stores completed taker buy/sell volume, open-interest history and OHLCV klines.
It does NOT represent the complete US equity market order flow.
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

BASE = "https://fapi.binance.com"
SYMBOLS = [
    "QQQUSDT", "NVDAUSDT", "SMHUSDT", "SOXLUSDT", "SPYUSDT",
    "AMDUSDT", "AVGOUSDT", "TSMUSDT", "MUUSDT",
]
PERIODS = ["15m", "1h"]
INTERVAL_MS = {"15m": 15 * 60 * 1000, "1h": 60 * 60 * 1000}
LIMIT = 500
ROOT = Path("data/flowfirst")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "FlowFirstResearch/0.2 (+public market data collector)"})


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RegionalAccessError(RuntimeError):
    """The provider explicitly blocks the runner region; retries cannot fix this."""


def get_json(path: str, params: dict) -> list:
    url = BASE + path
    last_error = None
    for attempt in range(4):
        try:
            response = SESSION.get(url, params=params, timeout=25)
            if response.status_code == 451:
                raise RegionalAccessError(
                    "Binance returned HTTP 451 for this GitHub Actions runner region. "
                    "Stop collection here; do not retry or route around the regional restriction."
                )
            if response.status_code in (418, 429):
                time.sleep(3 * (attempt + 1))
            response.raise_for_status()
            payload = response.json()
            if isinstance(payload, dict) and payload.get("code") is not None:
                raise RuntimeError(f"Binance API error: {payload}")
            if not isinstance(payload, list):
                raise RuntimeError(f"Unexpected response for {path}: {payload}")
            return payload
        except RegionalAccessError:
            raise
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Request failed {path} {params}: {last_error}")


def completed_period_start(period: str, now_ms: int | None = None) -> int:
    """Return the start timestamp of the currently forming interval."""
    now_ms = now_ms if now_ms is not None else int(time.time() * 1000)
    width = INTERVAL_MS[period]
    return now_ms - (now_ms % width)


def append_unique(kind: str, symbol: str, period: str, rows: list[dict]) -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    added = 0
    by_date: dict[str, list[dict]] = {}
    for row in rows:
        stamp = int(row["timestamp"])
        day = datetime.fromtimestamp(stamp / 1000, timezone.utc).strftime("%Y-%m-%d")
        by_date.setdefault(day, []).append(row)

    for day, day_rows in by_date.items():
        path = ROOT / f"{day}_{kind}_{period}.jsonl"
        # Deduplicate against all files of this kind/period, including
        # connector-seeded history files and the regular daily append files.
        seen = set()
        for existing in ROOT.glob(f"*_{kind}_{period}.jsonl"):
            if not existing.exists():
                continue
            with existing.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        old = json.loads(line)
                        seen.add((old["symbol"], old["period"], int(old["timestamp"])))
                    except Exception:
                        continue
        with path.open("a", encoding="utf-8") as f:
            for row in day_rows:
                key = (row["symbol"], row["period"], int(row["timestamp"]))
                if key in seen:
                    continue
                f.write(json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n")
                seen.add(key)
                added += 1
    return added


def collect_taker(symbol: str, period: str) -> int:
    payload = get_json("/futures/data/takerlongshortRatio",
                       {"symbol": symbol, "period": period, "limit": LIMIT})
    fetched = utc_now()
    cutoff = completed_period_start(period)
    rows = [{
        "symbol": symbol, "period": period,
        "timestamp": int(x["timestamp"]),
        "buyVol": str(x["buyVol"]), "sellVol": str(x["sellVol"]),
        "buySellRatio": str(x["buySellRatio"]), "fetched_at": fetched,
        "source": "Binance USD-M futures takerlongshortRatio",
    } for x in payload if int(x["timestamp"]) < cutoff]
    return append_unique("taker", symbol, period, rows)


def collect_long_short(symbol: str, period: str) -> int:
    payload = get_json("/futures/data/globalLongShortAccountRatio",
                       {"symbol": symbol, "period": period, "limit": LIMIT})
    fetched = utc_now()
    cutoff = completed_period_start(period)
    rows = [{
        "symbol": symbol, "period": period,
        "timestamp": int(x["timestamp"]),
        "longAccount": str(x["longAccount"]),
        "shortAccount": str(x["shortAccount"]),
        "longShortRatio": str(x["longShortRatio"]),
        "fetched_at": fetched,
        "source": "Binance USD-M futures globalLongShortAccountRatio",
    } for x in payload if int(x["timestamp"]) < cutoff]
    return append_unique("longshort", symbol, period, rows)


def collect_oi(symbol: str, period: str) -> int:
    payload = get_json("/futures/data/openInterestHist",
                       {"symbol": symbol, "period": period, "limit": LIMIT})
    fetched = utc_now()
    cutoff = completed_period_start(period)
    rows = [{
        "symbol": symbol, "period": period,
        "timestamp": int(x["timestamp"]),
        "sumOpenInterest": str(x["sumOpenInterest"]),
        "sumOpenInterestValue": str(x["sumOpenInterestValue"]),
        "fetched_at": fetched,
        "source": "Binance USD-M futures openInterestHist",
    } for x in payload if int(x["timestamp"]) < cutoff]
    return append_unique("oi", symbol, period, rows)


def collect_klines(symbol: str, period: str) -> int:
    payload = get_json("/fapi/v1/klines",
                       {"symbol": symbol, "interval": period, "limit": LIMIT})
    fetched = utc_now()
    now_ms = int(time.time() * 1000)
    rows = [{
        "symbol": symbol, "period": period,
        "timestamp": int(x[0]), "open": str(x[1]), "high": str(x[2]),
        "low": str(x[3]), "close": str(x[4]), "volume": str(x[5]),
        "close_time": int(x[6]), "quote_volume": str(x[7]),
        "trade_count": int(x[8]), "taker_buy_base": str(x[9]),
        "taker_buy_quote": str(x[10]), "fetched_at": fetched,
        "source": "Binance USD-M futures klines",
    } for x in payload if int(x[6]) < now_ms]
    return append_unique("klines", symbol, period, rows)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    totals = {"taker": 0, "oi": 0, "longshort": 0, "klines": 0, "errors": 0}
    for symbol in SYMBOLS:
        for period in PERIODS:
            for kind, fn in (
                ("taker", collect_taker),
                ("oi", collect_oi),
                ("longshort", collect_long_short),
                ("klines", collect_klines),
            ):
                try:
                    count = fn(symbol, period)
                    totals[kind] += count
                    print(f"OK {kind:7s} {symbol:10s} {period:3s}: +{count}")
                except RegionalAccessError as exc:
                    print(f"FATAL regional access restriction: {exc}")
                    raise SystemExit(2) from exc
                except Exception as exc:
                    totals["errors"] += 1
                    print(f"ERROR {kind} {symbol} {period}: {exc}")
                time.sleep(0.12)
    summary = {"run_at_utc": utc_now(), "symbols": SYMBOLS,
               "periods": PERIODS, "new_records": totals,
               "note": "Binance TradFi perpetual data; not consolidated US equity tape."}
    (ROOT / "latest_run.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if totals["errors"] == len(SYMBOLS) * len(PERIODS) * 4:
        raise SystemExit("All collection requests failed; no data was collected.")


if __name__ == "__main__":
    main()
