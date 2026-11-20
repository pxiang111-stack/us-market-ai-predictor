"""V8 QQQ trend/breakout strategy research.

Signals are computed using information available at close t and applied to
close(t)->close(t+1) returns. This is a transparent, non-ML baseline intended
to test whether regime/trend participation adds value before more complexity.
No parameter optimization is performed in this experiment.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

OUT = Path("output/qqq_v8_trend_breakout")
OUT.mkdir(parents=True, exist_ok=True)
INITIAL_EQUITY = 100_000.0
COST_BPS_PER_UNIT_TURNOVER = 5.0
BREAKOUT_ENTRY = 55
BREAKOUT_EXIT = 20
ANNUAL_VOL_TARGET = 0.15
MAX_ABS_EXPOSURE = 1.0


def download_ohlcv() -> pd.DataFrame:
    raw = yf.download("QQQ", period="max", interval="1d", auto_adjust=True,
                      progress=False, threads=False, timeout=30)
    if raw is None or raw.empty:
        raise RuntimeError("QQQ daily OHLCV unavailable")
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    required = ["High", "Low", "Close"]
    missing = [c for c in required if c not in raw.columns]
    if missing:
        raise RuntimeError(f"QQQ data missing columns: {missing}")
    raw = raw[required].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    idx = pd.to_datetime(raw.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    raw.index = idx.normalize()
    return raw[~raw.index.duplicated(keep="last")].sort_index()


def build_signals(bars: pd.DataFrame) -> pd.DataFrame:
    close, high, low = bars["Close"], bars["High"], bars["Low"]
    f = pd.DataFrame(index=bars.index)
    f["Close"] = close
    f["ret_next"] = close.shift(-1) / close - 1.0
    f["sma50"] = close.rolling(50).mean()
    f["sma200"] = close.rolling(200).mean()
    f["prior_high_55"] = high.rolling(BREAKOUT_ENTRY).max().shift(1)
    f["prior_low_55"] = low.rolling(BREAKOUT_ENTRY).min().shift(1)
    f["prior_low_20"] = low.rolling(BREAKOUT_EXIT).min().shift(1)
    f["prior_high_20"] = high.rolling(BREAKOUT_EXIT).max().shift(1)
    f["vol20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
    f["trend_up"] = (close > f["sma200"]) & (f["sma50"] > f["sma200"])
    f["trend_down"] = (close < f["sma200"]) & (f["sma50"] < f["sma200"])
    f["long_breakout"] = close > f["prior_high_55"]
    f["short_breakdown"] = close < f["prior_low_55"]
    return f


def state_machine(f: pd.DataFrame, mode: str) -> pd.Series:
    """Close-time state; output at t is applied to the next session return."""
    pos = 0.0
    out = []
    for row in f.itertuples():
        if not np.isfinite(row.sma200) or not np.isfinite(row.prior_high_55):
            out.append(0.0)
            continue
        if mode == "trend":
            if row.trend_up:
                pos = 1.0
            elif row.trend_down:
                pos = -1.0
            else:
                pos = 0.0
        elif mode in ("breakout_ls", "breakout_long"):
            if pos > 0 and (row.Close < row.prior_low_20 or row.Close < row.sma50):
                pos = 0.0
            elif pos < 0 and (row.Close > row.prior_high_20 or row.Close > row.sma50):
                pos = 0.0
            if pos == 0:
                if row.trend_up and row.long_breakout:
                    pos = 1.0
                elif mode == "breakout_ls" and row.trend_down and row.short_breakdown:
                    pos = -1.0
        else:
            raise ValueError(f"unknown mode: {mode}")
        out.append(pos)
    return pd.Series(out, index=f.index, dtype=float)


def evaluate(name: str, signal: pd.Series, f: pd.DataFrame, use_vol_target: bool) -> tuple[dict, pd.DataFrame]:
    valid = f["ret_next"].notna() & signal.notna()
    idx = f.index[valid]
    raw = signal.loc[idx].astype(float)
    vol = f.loc[idx, "vol20"].replace(0, np.nan).ffill()
    if use_vol_target:
        scale = (ANNUAL_VOL_TARGET / vol).clip(upper=MAX_ABS_EXPOSURE).fillna(0.0)
        exposure = raw * scale
    else:
        exposure = raw
    # Limit exposure to 1x in either direction; no leverage.
    exposure = exposure.clip(-MAX_ABS_EXPOSURE, MAX_ABS_EXPOSURE)
    turnover = exposure.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(exposure.iloc[0])
    cost = turnover.fillna(0.0) * COST_BPS_PER_UNIT_TURNOVER / 10_000.0
    gross = exposure * f.loc[idx, "ret_next"].astype(float)
    net = gross - cost
    equity = INITIAL_EQUITY * (1.0 + net).cumprod()
    benchmark_ret = f.loc[idx, "ret_next"].astype(float)
    benchmark_equity = INITIAL_EQUITY * (1.0 + benchmark_ret).cumprod()
    years = len(net) / 252.0
    peak = equity.cummax()
    total = float(equity.iloc[-1] / INITIAL_EQUITY - 1.0)
    ann = float((equity.iloc[-1] / INITIAL_EQUITY) ** (1 / years) - 1) if years and equity.iloc[-1] > 0 else None
    vol_ann = float(net.std(ddof=1) * np.sqrt(252)) if len(net) > 1 else None
    sharpe = float(net.mean() / net.std(ddof=1) * np.sqrt(252)) if len(net) > 1 and net.std(ddof=1) > 0 else None
    stats = {
        "strategy": name, "observations": int(len(net)),
        "start_date": str(idx[0].date()), "end_date": str(idx[-1].date()),
        "total_return_pct": total * 100, "cagr_pct": ann * 100 if ann is not None else None,
        "max_drawdown_pct": float((equity / peak - 1).min() * 100),
        "annualized_volatility_pct": vol_ann * 100 if vol_ann is not None else None,
        "sharpe_rf0": sharpe, "exposure_pct_time": float((exposure.abs() > 0).mean() * 100),
        "average_signed_exposure_pct": float(exposure.mean() * 100),
        "average_abs_exposure_pct": float(exposure.abs().mean() * 100),
        "turnover_units": float(turnover.sum()), "cost_bps": COST_BPS_PER_UNIT_TURNOVER,
        "same_window_buy_hold_return_pct": float((benchmark_equity.iloc[-1] / INITIAL_EQUITY - 1) * 100),
        "same_window_buy_hold_mdd_pct": float((benchmark_equity / benchmark_equity.cummax() - 1).min() * 100),
    }
    curve = pd.DataFrame({
        "date": idx, "strategy": name, "signal_at_prior_close": raw.values,
        "exposure": exposure.values, "gross_return": gross.values,
        "turnover": turnover.fillna(0).values, "cost_return": cost.values,
        "net_return": net.values, "equity": equity.values,
        "buy_hold_equity_same_window": benchmark_equity.values,
    })
    return stats, curve


def main() -> None:
    bars = download_ohlcv()
    f = build_signals(bars)
    f = f.dropna(subset=["ret_next"])
    strategies = [
        ("trend_long_short_1x", "trend", False),
        ("trend_long_short_vol_target", "trend", True),
        ("breakout_long_only_1x", "breakout_long", False),
        ("breakout_long_short_1x", "breakout_ls", False),
        ("breakout_long_short_vol_target", "breakout_ls", True),
    ]
    summaries, curves = [], []
    for name, mode, vol_target in strategies:
        signal = state_machine(f, mode)
        summary, curve = evaluate(name, signal, f, vol_target)
        summaries.append(summary)
        curves.append(curve)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT / "strategy_summary.csv", index=False)
    pd.concat(curves, ignore_index=True).to_csv(OUT / "equity_curves.csv", index=False)
    metadata = {
        "status": "completed", "ticker": "QQQ",
        "signal_timing": "signal computed after close t; position earns close(t)-close(t+1) return",
        "benchmark": "buy-and-hold QQQ on exact same dates and initial equity",
        "strategies": [x[0] for x in strategies],
        "entry_lookback_sessions": BREAKOUT_ENTRY, "exit_lookback_sessions": BREAKOUT_EXIT,
        "trend_filter": "SMA50>SMA200 and close>SMA200 for long; inverse for short",
        "vol_target": ANNUAL_VOL_TARGET, "max_abs_exposure": MAX_ABS_EXPOSURE,
        "transaction_cost_bps_per_unit_exposure_turnover": COST_BPS_PER_UNIT_TURNOVER,
        "no_parameter_optimization": True,
        "limitations": [
            "Daily OHLCV only; intraday execution, gaps and market impact are not modeled.",
            "Short borrow, financing, dividends/tax and ETF tracking effects are not modeled.",
            "No parameter search or walk-forward parameter selection is performed.",
            "Full-period results are research diagnostics, not untouched final OOS evidence.",
            "Volatility target uses trailing realized volatility and may lag abrupt regime changes.",
        ],
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(summary_df.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
