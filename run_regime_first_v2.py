"""Daily Regime-First v2 market data adapter.

Uses Yahoo Finance for NQ/NDX/BTC/ETH/VIX and public FRED CSV observations
for ICE BofA credit spreads. The market-wide rank_model_score is deliberately
not fabricated: provide output/rank_model_score.csv with date,rank_model_score.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from regime_first_v2 import evaluate_history
from regime_event_study import build_event_study

OUT = Path("output")
FRED_SERIES = {
    "CCC_spread": "BAMLH0A3HYC",
    "BBB_spread": "BAMLC0A4CBBB",
    "HY_spread": "BAMLH0A0HYM2",
}
YAHOO = {
    "NQ": "NQ=F",
    "NDX": "^NDX",
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "VIX": "^VIX",
}


def fetch_yahoo_close(symbol: str, period: str) -> pd.DataFrame:
    frame = yf.Ticker(symbol).history(period=period, auto_adjust=True)
    if frame is None or frame.empty or "Close" not in frame:
        raise RuntimeError(f"No Yahoo Finance history for {symbol}")
    frame = frame.copy()
    idx = pd.to_datetime(frame.index)
    if getattr(idx, "tz", None) is not None:
        target_tz = "UTC" if symbol in {"BTC-USD", "ETH-USD"} else "America/New_York"
        idx = idx.tz_convert(target_tz).tz_localize(None)
    frame.index = idx.normalize()
    frame = frame[~frame.index.duplicated(keep="last")].sort_index()
    cols = [c for c in ("Close", "High", "Low") if c in frame.columns]
    return frame[cols].apply(pd.to_numeric, errors="coerce")


def fetch_fred_series(series_id: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    response = requests.get(url, headers={"User-Agent": "RegimeFirstV2Research/1.0"},
                            timeout=30)
    response.raise_for_status()
    from io import StringIO
    frame = pd.read_csv(StringIO(response.text))
    if len(frame.columns) < 2:
        raise RuntimeError(f"Unexpected FRED CSV format for {series_id}")
    frame = frame.iloc[:, :2].copy()
    frame.columns = ["date", "value"]
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["date"]).set_index("date").sort_index()
    return frame["value"].rename(series_id)


def load_rank_score(path: Path, dates: pd.DatetimeIndex) -> pd.Series:
    if not path.exists():
        print(f"[DATA GAP] rank score file not found: {path}", flush=True)
        return pd.Series(np.nan, index=dates, name="rank_model_score")
    frame = pd.read_csv(path)
    if not {"date", "rank_model_score"}.issubset(frame.columns):
        raise ValueError("rank score CSV requires date,rank_model_score columns")
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce").dt.normalize()
    frame["rank_model_score"] = pd.to_numeric(frame["rank_model_score"], errors="coerce")
    frame = frame.dropna(subset=["date"]).drop_duplicates("date", keep="last")
    return frame.set_index("date")["rank_model_score"].reindex(dates).rename("rank_model_score")


def align_crypto_to_nq_close(crypto: pd.DataFrame,
                             nq_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Align UTC daily crypto bars to the NQ session close without look-ahead.

    Yahoo's crypto daily index labels the UTC calendar day whose bar closes at
    the *next* 00:00 UTC. That close is not yet known at the NQ cash-session
    close on the same calendar date. Therefore each completed UTC bar is made
    available to the following NQ session date. Prior-20-day bands are
    calculated before shifting the availability date, so the current aligned
    crypto close is compared with the 20 completed UTC sessions before it.
    """
    if crypto is None or crypto.empty or "Close" not in crypto:
        return pd.DataFrame(index=nq_dates,
                            columns=["Close", "prior_20d_high", "prior_20d_low"],
                            dtype=float)
    daily = crypto.copy().sort_index()
    idx = pd.DatetimeIndex(pd.to_datetime(daily.index))
    if idx.tz is not None:
        idx = idx.tz_convert("UTC").tz_localize(None)
    daily.index = idx.normalize()
    daily = daily[~daily.index.duplicated(keep="last")]
    close = pd.to_numeric(daily["Close"], errors="coerce")
    prior_high = close.shift(1).rolling(20, min_periods=20).max()
    prior_low = close.shift(1).rolling(20, min_periods=20).min()
    available = pd.DataFrame({
        "Close": close,
        "prior_20d_high": prior_high,
        "prior_20d_low": prior_low,
    })
    available.index = available.index + pd.Timedelta(days=1)
    return available.reindex(nq_dates).ffill()


def build_market_frame(period: str, rank_score_file: Path) -> pd.DataFrame:
    prices = {}
    errors = {}
    for key, symbol in YAHOO.items():
        try:
            prices[key] = fetch_yahoo_close(symbol, period)
            print(f"[DATA] {key} {symbol}: {len(prices[key])} rows", flush=True)
        except Exception as exc:
            errors[key] = str(exc)
            print(f"[DATA GAP] {key} {symbol}: {exc}", flush=True)

    if "NQ" not in prices:
        raise RuntimeError("NQ futures history unavailable; cannot establish session calendar")
    nq = prices["NQ"].copy()
    if not {"Close", "High", "Low"}.issubset(nq.columns):
        raise RuntimeError("NQ feed missing Close/High/Low")
    z = pd.DataFrame(index=nq.index)
    z["date"] = z.index
    z["NQ_close"] = nq["Close"]
    z["NQ_high"] = nq["High"]
    z["NQ_low"] = nq["Low"]
    z["NQ_200ma"] = nq["Close"].rolling(200, min_periods=200).mean()
    z["NQ_20d_high"] = nq["High"].shift(1).rolling(20, min_periods=20).max()
    z["NQ_20d_low"] = nq["Low"].shift(1).rolling(20, min_periods=20).min()
    z["NQ_60d_high"] = nq["High"].shift(1).rolling(60, min_periods=60).max()
    z["NQ_60d_low"] = nq["Low"].shift(1).rolling(60, min_periods=60).min()

    for key, field in (("NDX", "NDX_close"), ("BTC", "BTC_close"),
                       ("ETH", "ETH_close"), ("VIX", "VIX_close")):
        if key in prices:
            z[field] = prices[key]["Close"].reindex(z.index).ffill()
        else:
            z[field] = np.nan

    for field, series_id in FRED_SERIES.items():
        try:
            series = fetch_fred_series(series_id)
            z[field] = series.reindex(z.index).ffill()
            print(f"[DATA] {field} ({series_id}): {series.notna().sum()} observations",
                  flush=True)
        except Exception as exc:
            z[field] = np.nan
            errors[field] = str(exc)
            print(f"[DATA GAP] {field} ({series_id}): {exc}", flush=True)

    for asset in ("BTC", "ETH"):
        if asset in prices:
            # UTC crypto daily closes are only available at the next UTC
            # midnight; align both close and prior-only bands to the next NQ
            # session date to prevent same-day look-ahead.
            aligned = align_crypto_to_nq_close(prices[asset], z.index)
            z[f"{asset}_close"] = aligned["Close"]
            z[f"{asset}_20d_high"] = aligned["prior_20d_high"]
            z[f"{asset}_20d_low"] = aligned["prior_20d_low"]
        else:
            z[f"{asset}_close"] = np.nan
            z[f"{asset}_20d_high"] = np.nan
            z[f"{asset}_20d_low"] = np.nan

    z["rank_model_score"] = load_rank_score(rank_score_file, z.index)
    z = z.reset_index(drop=True)
    z["date"] = pd.to_datetime(z["date"]).dt.normalize()
    return z


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", default="3y")
    parser.add_argument("--rank-score-file", default="output/rank_model_score.csv")
    parser.add_argument("--initial-state", default="NEUTRAL",
                        choices=["RISK_OFF", "NEUTRAL", "RISK_ON"])
    args = parser.parse_args()

    OUT.mkdir(exist_ok=True)
    market = build_market_frame(args.period, Path(args.rank_score_file))
    result = evaluate_history(market, initial_state=args.initial_state)
    result["date"] = pd.to_datetime(result["date"]).dt.normalize()
    combined = market.merge(result, on="date", how="left", validate="one_to_one")
    combined.to_csv(OUT / "regime_first_v2_daily.csv", index=False)
    event_study = build_event_study(combined)
    event_study.to_csv(OUT / "regime_first_v2_event_study.csv", index=False)

    latest = combined.iloc[-1]
    lines = [
        "# Regime-First v2 daily report",
        "",
        f"- Date: {latest['date']}",
        f"- State: {latest['state']}",
        f"- Max risk budget: {latest['max_risk_pct']:.3f}% of account equity",
        f"- Volatility multiplier: {latest['volatility_multiplier']}",
        f"- Missing-data multiplier: {latest['data_gap_multiplier']}",
        f"- Monitor veto multiplier: {latest['monitor_veto_multiplier']}",
        f"- Flip conditions: {latest['flip_conditions']} ({latest['flip_count']}/4)",
        f"- Hard risk-off trigger: {latest['hard_risk_off']}",
        f"- Allowed action: {latest['allowed_action']}",
        f"- Data gaps: {latest['data_gaps']}",
        f"- Notes: {latest['notes']}",
        "",
        "Forward 5d/20d state study is descriptive; n<30 rows are RECORD_ONLY.",
        "This is not a fully sized portfolio backtest or evidence of a trading edge.",
        "Research output only; missing inputs are not imputed into bullish signals.",
        "No active short signal is generated by this engine.",
    ]
    (OUT / "regime_first_v2_latest.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[FATAL] Regime-First v2 daily run failed: {exc}", file=sys.stderr)
        raise
