"""Download QQQ execution OHLC with explicit corporate-action fields.

Yahoo Finance OHLC is split-adjusted but dividend-unadjusted when auto_adjust=False.
Dividends and split ratios are exported separately so the simulator can account for
cash distributions and share-count changes rather than pretending adjusted OHLC
were literal executable prices.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yfinance as yf


def download(start: str, output: str) -> Path:
    frame = yf.download(
        "QQQ", start=start, auto_adjust=False, actions=True,
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
    for action_col in ("dividends", "stock_splits"):
        if action_col not in frame.columns:
            frame[action_col] = 0.0
    frame = frame[required + ["dividends", "stock_splits"]].dropna(subset=required).copy()
    frame.index.name = "date"
    frame = frame.reset_index()
    frame["date"] = pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d")
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    if len(frame) < 1_000:
        raise RuntimeError(f"Only {len(frame)} rows downloaded; refusing to call this a 5-year sample")
    split_events = int((frame["stock_splits"] > 0).sum())
    dividend_events = int((frame["dividends"] > 0).sum())
    print(f"Saved {destination} ({len(frame)} rows; {dividend_events} dividend events; {split_events} split events)")
    print("Price basis: split-adjusted OHLC, dividend-unadjusted; actions are separate columns.")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Download QQQ split-adjusted execution OHLC and corporate actions")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--output", default="data/QQQ_daily_execution.csv")
    args = parser.parse_args()
    download(args.start, args.output)


if __name__ == "__main__":
    main()
