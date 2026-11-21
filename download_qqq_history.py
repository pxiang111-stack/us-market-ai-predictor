"""Download split/dividend-adjusted QQQ daily OHLCV from Yahoo Finance via yfinance.

Example:
    python download_qqq_history.py --start 2010-01-01 --output data/QQQ_daily_adjusted.csv

This is an external data dependency. Keep the downloaded CSV with the run artifacts
and record its retrieval date; provider history can be revised.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download(start: str, output: str) -> Path:
    frame = yf.download(
        "QQQ", start=start, auto_adjust=True, actions=False,
        interval="1d", progress=False, threads=False,
    )
    if frame is None or frame.empty:
        raise RuntimeError("Yahoo Finance returned no QQQ daily data")
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    frame = frame.rename(columns={c: str(c).lower().replace(" ", "_") for c in frame.columns})
    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise RuntimeError(f"Downloaded data is missing columns: {missing}")
    frame = frame[required].dropna().copy()
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    if len(frame) < 1_000:
        raise RuntimeError(f"Only {len(frame)} rows downloaded; refusing to call this a 5-year sample")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download adjusted QQQ daily OHLCV")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--output", default="data/QQQ_daily_adjusted.csv")
    args = parser.parse_args()
    path = download(args.start, args.output)
    print(f"Saved {path} ({sum(1 for _ in path.open(encoding='utf-8')) - 1} daily rows)")


if __name__ == "__main__":
    main()
