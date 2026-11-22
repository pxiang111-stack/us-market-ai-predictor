"""Chronological walk-forward diagnostics for the fixed QQQ V1 rules.

Each fold uses all earlier bars only as indicator warmup, resets capital at the
fold start, and stops at that fold's final date. This is descriptive stability
testing, not parameter optimization.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from simple_nq_backtest import (
    DEFAULT_COMMISSION_BPS,
    DEFAULT_SLIPPAGE_BPS,
    load_bars,
    run_buy_hold,
    run_sma_baseline,
    run_strategy,
)

FOLDS = [
    ("2010-2014", "2010-01-01", "2014-12-31"),
    ("2015-2018", "2015-01-01", "2018-12-31"),
    ("2019-2022", "2019-01-01", "2022-12-31"),
    ("2023-2026", "2023-01-01", "2026-12-31"),
]


def run_walkforward(
    csv_path: str | Path,
    output_dir: str | Path = "output",
    initial_equity: float = 10_000.0,
    commission_bps: float = DEFAULT_COMMISSION_BPS,
    slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
) -> dict[str, Any]:
    bars = load_bars(csv_path)
    dates = [b["date"] for b in bars]
    results = []
    all_trades = []

    for label, requested_start, requested_end in FOLDS:
        first = next((i for i, d in enumerate(dates) if d >= requested_start), None)
        last = next((i for i in range(len(dates) - 1, -1, -1) if dates[i] <= requested_end), None)
        if first is None or last is None or last <= first:
            continue
        start = max(205, first)
        if start >= last:
            continue
        fold_bars = bars[: last + 1]
        strategy, trades, _ = run_strategy(
            fold_bars, initial_equity, commission_bps, slippage_bps,
            start_index=start, include_warmup=False,
        )
        buy_hold = run_buy_hold(
            fold_bars, initial_equity, commission_bps, slippage_bps,
            start_index=start,
        )
        sma = run_sma_baseline(
            fold_bars, initial_equity, commission_bps, slippage_bps,
            start_index=start,
        )
        cash = {
            "strategy": "CASH_NO_TRADE",
            "initial_equity": initial_equity,
            "final_equity": initial_equity,
            "total_return_pct": 0.0,
            "cagr_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "trades": 0,
        }
        strategy["fold"] = label
        buy_hold["fold"] = label
        sma["fold"] = label
        cash["fold"] = label
        results.append({
            "fold": label,
            "requested_start": requested_start,
            "requested_end": requested_end,
            "actual_start": dates[start],
            "actual_end": dates[last],
            "warmup_bars": start,
            "bars_in_fold": last - start + 1,
            "strategies": [strategy, buy_hold, sma, cash],
        })
        for trade in trades:
            all_trades.append({"fold": label, **trade})

    output = {
        "data_source": "Yahoo Finance via yfinance; split-adjusted, dividend-unadjusted OHLC with explicit cash dividends and split ratios",
        "first_date": dates[0],
        "last_date": dates[-1],
        "bars": len(bars),
        "initial_equity_each_fold": initial_equity,
        "commission_bps_each_side": commission_bps,
        "slippage_bps_each_side": slippage_bps,
        "dividend_policy": "Cash dividends credited; no dividend reinvestment",
        "method": "Fixed chronological folds; prior data used only for warmup; fresh capital at each fold start; no parameter fitting",
        "folds": results,
        "warning": "The 2023-2026 fold overlaps the previously inspected holdout and is descriptive only, not an untouched test.",
    }
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "simple_nq_walkforward_summary.json").write_text(
        json.dumps(output, indent=2), encoding="utf-8"
    )
    fields = [
        "fold", "signal_date", "entry_date", "exit_date", "shares",
        "shares_at_exit", "entry_price", "stop", "target", "exit_price",
        "exit_reason", "dividend_cash", "commission_total", "net_pnl",
        "net_return_pct",
    ]
    import csv
    with (out / "simple_nq_walkforward_trades.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_trades)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed chronological QQQ V1 walk-forward diagnostics")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--commission-bps", type=float, default=DEFAULT_COMMISSION_BPS)
    parser.add_argument("--slippage-bps", type=float, default=DEFAULT_SLIPPAGE_BPS)
    args = parser.parse_args()
    result = run_walkforward(
        args.csv, args.output_dir, args.capital,
        args.commission_bps, args.slippage_bps,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
