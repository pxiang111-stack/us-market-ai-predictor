"""Daily OHLCV market-structure scan.

Yahoo Finance daily bars are a research feed, not intraday tape, true bid/ask
delta, or a dependable source of futures open interest.
"""
from pathlib import Path
import json
import time

import pandas as pd
import yfinance as yf

from market_structure_v5 import build_market_structure

OUT = Path("output")
OUT.mkdir(exist_ok=True)
SYMBOLS = ["NQ=F", "^NDX", "ES=F", "^GSPC", "QQQ", "SPY", "SOXL", "SMH", "NVDA"]
CASH_INDICES = {"^NDX", "^GSPC"}


def fetch_ohlcv(symbol: str, period: str = "2y") -> pd.DataFrame:
    raw = yf.download(
        symbol, period=period, interval="1d", auto_adjust=False,
        progress=False, threads=False, timeout=25,
    )
    if raw is None or raw.empty:
        raise ValueError(f"No OHLCV data returned for {symbol}")
    if isinstance(raw.columns, pd.MultiIndex):
        if symbol in raw.columns.get_level_values(-1):
            raw = raw.xs(symbol, axis=1, level=-1)
        elif symbol in raw.columns.get_level_values(0):
            raw = raw.xs(symbol, axis=1, level=0)
        else:
            raw.columns = raw.columns.get_level_values(0)
    required = ["Open", "High", "Low", "Close", "Volume"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise ValueError(f"{symbol} missing columns: {missing}")
    bars = raw[required].dropna(subset=["Close"]).sort_index()
    if symbol not in CASH_INDICES and (bars["Volume"].dropna() <= 0).all():
        raise ValueError(f"{symbol} returned no positive volume observations")
    return bars


def main():
    feature_frames = []
    latest_rows = []
    failures = []
    for symbol in SYMBOLS:
        try:
            bars = fetch_ohlcv(symbol)
            volume_valid = symbol not in CASH_INDICES
            features = build_market_structure(
                bars, open_interest=None, volume_is_valid=volume_valid
            )
            features.insert(0, "ticker", symbol)
            features.insert(1, "bar_date", features.index.astype(str))
            feature_frames.append(features.reset_index(drop=True))
            last = features.iloc[-1]
            latest_rows.append({
                "ticker": symbol,
                "bar_date": str(features.index[-1]),
                "close": float(bars["Close"].iloc[-1]),
                "relative_volume": None if pd.isna(last["relative_volume"]) else float(last["relative_volume"]),
                "volume_zscore": None if pd.isna(last["volume_zscore"]) else float(last["volume_zscore"]),
                "close_location": None if pd.isna(last["close_location"]) else float(last["close_location"]),
                "signed_volume_proxy_ratio": None if pd.isna(last["signed_volume_proxy_ratio"]) else float(last["signed_volume_proxy_ratio"]),
                "volume_status": str(last["volume_data_status"]),
                "oi_status": "unavailable",
                "feed": "Yahoo Finance daily OHLCV",
            })
        except Exception as exc:
            failures.append({"ticker": symbol, "error": str(exc)})
        time.sleep(0.15)

    if feature_frames:
        pd.concat(feature_frames, ignore_index=True).to_csv(
            OUT / "market_structure_features_v5.csv", index=False
        )
    pd.DataFrame(latest_rows).to_csv(OUT / "market_structure_latest_v5.csv", index=False)
    (OUT / "market_structure_summary_v5.json").write_text(
        json.dumps({
            "symbols_requested": SYMBOLS,
            "symbols_succeeded": [r["ticker"] for r in latest_rows],
            "symbols_failed": failures,
            "cash_index_volume_excluded": sorted(CASH_INDICES),
            "oi_connected": False,
            "intraday_session_segmentation": False,
            "warning": "Daily OHLCV is not intraday tape. Candle-direction volume is not bid/ask delta. OI unavailable is explicit. Cash-index vendor volume is excluded.",
        }, indent=2),
        encoding="utf-8",
    )
    if not latest_rows:
        raise RuntimeError("No symbols produced market-structure features")
    print(f"[OK] Market structure rows: {len(latest_rows)}; failures: {len(failures)}")


if __name__ == "__main__":
    main()
