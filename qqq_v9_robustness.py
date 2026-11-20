"""V9 robustness audit: subperiods, execution costs and short-borrow sensitivity.

Imports the frozen V8 signal definitions; no parameters are tuned here.
Signals at close t earn only the next session return. All comparisons use the
same dates and same initial equity. Short borrow is charged only on negative
exposure days. Results are sensitivity analysis, not live performance.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd

from qqq_v8_trend_breakout import (
    download_ohlcv, build_signals, state_machine,
    INITIAL_EQUITY, ANNUAL_VOL_TARGET, MAX_ABS_EXPOSURE,
)

OUT = Path("output/qqq_v9_robustness")
OUT.mkdir(parents=True, exist_ok=True)
COST_BPS = (0, 5, 10, 20)
SHORT_BORROW_BPS = (0, 25, 50, 100)
PERIODS = {
    "1999_2009": ("1999-01-01", "2009-12-31"),
    "2010_2019": ("2010-01-01", "2019-12-31"),
    "2020_2026": ("2020-01-01", "2026-12-31"),
    "2022_2026": ("2022-01-01", "2026-12-31"),
}
STRATEGIES = [
    ("trend_long_short_1x", "trend", False),
    ("trend_long_short_vol_target", "trend", True),
    ("breakout_long_only_1x", "breakout_long", False),
    ("breakout_long_short_1x", "breakout_ls", False),
    ("breakout_long_short_vol_target", "breakout_ls", True),
]


def make_exposure(signal: pd.Series, f: pd.DataFrame, vol_target: bool) -> pd.Series:
    raw = signal.astype(float)
    if vol_target:
        vol = f["vol20"].replace(0, np.nan).ffill()
        scale = (ANNUAL_VOL_TARGET / vol).clip(upper=MAX_ABS_EXPOSURE).fillna(0.0)
        raw = raw * scale
    return raw.clip(-MAX_ABS_EXPOSURE, MAX_ABS_EXPOSURE)


def stats_for(name, exposure, asset_returns, cost_bps, borrow_bps):
    turnover = exposure.diff().abs()
    if len(turnover):
        turnover.iloc[0] = abs(exposure.iloc[0])
    trading_cost = turnover.fillna(0.0) * cost_bps / 10000.0
    borrow_cost = exposure.clip(upper=0).abs() * borrow_bps / 10000.0 / 252.0
    net = exposure * asset_returns - trading_cost - borrow_cost
    equity = INITIAL_EQUITY * (1.0 + net).cumprod()
    if len(net) == 0:
        return None
    years = len(net) / 252.0
    std = net.std(ddof=1)
    peak = equity.cummax()
    return {
        "strategy": name, "cost_bps_turnover": cost_bps,
        "short_borrow_bps_annual": borrow_bps,
        "observations": int(len(net)),
        "start_date": str(net.index[0].date()), "end_date": str(net.index[-1].date()),
        "total_return_pct": float((equity.iloc[-1] / INITIAL_EQUITY - 1) * 100),
        "cagr_pct": float(((equity.iloc[-1] / INITIAL_EQUITY) ** (1 / years) - 1) * 100) if years and equity.iloc[-1] > 0 else None,
        "max_drawdown_pct": float(((equity / peak) - 1).min() * 100),
        "sharpe_rf0": float(net.mean() / std * np.sqrt(252)) if std and np.isfinite(std) else None,
        "exposure_time_pct": float((exposure.abs() > 0).mean() * 100),
        "average_abs_exposure_pct": float(exposure.abs().mean() * 100),
        "short_exposure_time_pct": float((exposure < 0).mean() * 100),
        "turnover_units": float(turnover.sum()),
    }


def main():
    bars = download_ohlcv()
    f = build_signals(bars)
    f = f.dropna(subset=["ret_next"])
    asset_returns = f["ret_next"].astype(float)
    rows = []
    for strategy_name, mode, vol_target in STRATEGIES:
        signal = state_machine(f, mode)
        exposure = make_exposure(signal.loc[asset_returns.index], f.loc[asset_returns.index], vol_target)
        for period, (start, end) in PERIODS.items():
            mask = (asset_returns.index >= start) & (asset_returns.index <= end)
            r, e = asset_returns.loc[mask], exposure.loc[mask]
            if len(r) < 100:
                continue
            # Buy-and-hold benchmark uses exactly the same period and dates.
            bh = stats_for("QQQ_buy_hold", pd.Series(1.0, index=r.index), r, 0, 0)
            bh.update({"period": period, "cost_bps_turnover": 0, "short_borrow_bps_annual": 0})
            rows.append(bh)
            for cost in COST_BPS:
                borrow_cases = SHORT_BORROW_BPS if mode in ("trend", "breakout_ls") else (0,)
                for borrow in borrow_cases:
                    x = stats_for(strategy_name, e, r, cost, borrow)
                    x["period"] = period
                    rows.append(x)
    results = pd.DataFrame(rows)
    results.to_csv(OUT / "subperiod_cost_borrow_sensitivity.csv", index=False)

    # Full-period explicit cost grid, useful for seeing the effect of execution friction.
    full_rows = []
    for strategy_name, mode, vol_target in STRATEGIES:
        signal = state_machine(f, mode)
        exposure = make_exposure(signal.loc[asset_returns.index], f.loc[asset_returns.index], vol_target)
        for cost in COST_BPS:
            borrow_cases = SHORT_BORROW_BPS if mode in ("trend", "breakout_ls") else (0,)
            for borrow in borrow_cases:
                x = stats_for(strategy_name, exposure, asset_returns, cost, borrow)
                full_rows.append(x)
    full = pd.DataFrame(full_rows)
    full.to_csv(OUT / "full_period_cost_borrow_sensitivity.csv", index=False)

    metadata = {
        "status": "completed", "ticker": "QQQ",
        "signal_source": "frozen V8 strategy definitions; no parameter tuning",
        "signal_timing": "close t signal applied to close(t)-close(t+1) return",
        "periods": PERIODS, "transaction_cost_bps_per_unit_turnover": list(COST_BPS),
        "short_borrow_bps_annual_on_short_exposure": list(SHORT_BORROW_BPS),
        "short_borrow_method": "annual fee / 252 * absolute negative exposure each session",
        "vol_target": ANNUAL_VOL_TARGET, "max_abs_exposure": MAX_ABS_EXPOSURE,
        "limitations": [
            "Short borrow fee is a sensitivity assumption, not historical locate-specific data.",
            "No intraday execution, bid-ask spread, market impact, financing or taxes.",
            "QQQ adjusted prices; short dividend obligations are not separately modeled.",
            "Subperiods are descriptive and are not independent samples.",
            "No strategy or threshold is selected from these results.",
        ],
    }
    (OUT / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("FULL PERIOD COST/BORROW SENSITIVITY")
    print(full.to_string(index=False), flush=True)
    print("SUBPERIOD RESULTS")
    print(results.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
