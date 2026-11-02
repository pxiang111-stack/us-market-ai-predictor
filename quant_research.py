"""Cross-sectional factor research and a conservative long-only research backtest.

This module uses only information available at each signal date. Forward returns
are used exclusively for evaluation, never to construct factor scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

FACTOR_COLUMNS = [
    "mom_21", "mom_63", "mom_126", "mom_252",
    "relative_mom_21", "ma_gap_21", "ma_gap_63",
    "vol_21", "vol_63", "drawdown_126", "rsi_14",
]

def _rank(frame: pd.DataFrame, column: str, ascending: bool = True) -> pd.Series:
    return frame.groupby(level=0)[column].rank(pct=True, ascending=ascending)

def add_factor_scores(data: pd.DataFrame) -> pd.DataFrame:
    """Create date-wise percentile ranks and a transparent composite alpha score."""
    out = data.copy()
    required = [c for c in FACTOR_COLUMNS if c in out.columns]
    if len(required) < 8:
        raise ValueError(f"Too few factor inputs available: {required}")
    # All ranks are cross-sectional within a date; higher is better.
    ranks = pd.DataFrame(index=out.index)
    ranks["momentum"] = sum(
        _rank(out, c) * w for c, w in
        [("mom_21", .15), ("mom_63", .25), ("mom_126", .35), ("mom_252", .25)]
        if c in out
    )
    ranks["relative_strength"] = _rank(out, "relative_mom_21")
    ranks["trend"] = (_rank(out, "ma_gap_21") + _rank(out, "ma_gap_63")) / 2
    ranks["low_volatility"] = (_rank(out, "vol_21", ascending=False) +
                               _rank(out, "vol_63", ascending=False)) / 2
    ranks["drawdown_control"] = _rank(out, "drawdown_126")
    # RSI is treated as a broad middle-range preference, not a buy/sell trigger.
    ranks["rsi_balance"] = 1.0 - (_rank(out, "rsi_14") - .5).abs()
    out["factor_momentum"] = ranks["momentum"]
    out["factor_relative_strength"] = ranks["relative_strength"]
    out["factor_trend"] = ranks["trend"]
    out["factor_low_volatility"] = ranks["low_volatility"]
    out["factor_drawdown_control"] = ranks["drawdown_control"]
    out["factor_score"] = (
        .30 * ranks["momentum"] +
        .20 * ranks["relative_strength"] +
        .18 * ranks["trend"] +
        .14 * ranks["low_volatility"] +
        .12 * ranks["drawdown_control"] +
        .06 * ranks["rsi_balance"]
    )
    return out

def _max_drawdown(returns: list[float]) -> float:
    if not returns:
        return float("nan")
    curve = (1 + pd.Series(returns, dtype=float)).cumprod()
    dd = curve / curve.cummax() - 1
    return float(dd.min())

def evaluate_factor_portfolio(
    data: pd.DataFrame,
    horizon: str,
    etfs: set[str] | None = None,
    top_fraction: float = .10,
    transaction_cost_bps: float = 10.0,
) -> tuple[dict, pd.DataFrame]:
    """Evaluate periodic top-decile stock baskets on a final chronological holdout.

    Rebalance interval equals the forecast horizon to avoid overlapping holding
    periods. Cost is a transparent turnover-based estimate, not broker-specific.
    """
    days = {"1m": 21, "3m": 63, "6m": 126}[horizon]
    ret_col = f"target_ret_{horizon}"
    z = data.dropna(subset=["factor_score", ret_col]).copy()
    z = z[~z["ticker"].isin(etfs or set())]
    dates = np.array(sorted(z.index.unique()))
    if len(dates) < 300:
        return {"horizon": horizon, "status": "insufficient_history",
                "n_dates": len(dates)}, pd.DataFrame()
    start = int(len(dates) * .70)
    test_dates = dates[start::days]
    if len(test_dates) < 3:
        return {"horizon": horizon, "status": "insufficient_test_rebalances",
                "n_rebalances": len(test_dates)}, pd.DataFrame()
    rows, prev = [], set()
    for dt in test_dates:
        cross = z.loc[z.index == dt].dropna(subset=["factor_score", ret_col])
        if len(cross) < 30:
            continue
        n = max(1, int(np.ceil(len(cross) * top_fraction)))
        basket = cross.nlargest(n, "factor_score")
        names = set(basket["ticker"].astype(str))
        turnover = 1.0 if not prev else 1.0 - len(names & prev) / max(len(names), 1)
        cost = (transaction_cost_bps / 10000.0) * min(2.0 * turnover, 2.0)
        gross = float(basket[ret_col].mean())
        # SPY benchmark is read from the original cross-section, not the stock-only basket.
        original = data.loc[data.index == dt]
        spy = original.loc[original["ticker"].eq("SPY"), ret_col]
        spy_ret = float(spy.iloc[0]) if len(spy) else float("nan")
        rows.append({
            "date": str(pd.Timestamp(dt).date()), "horizon": horizon,
            "n_stocks": int(len(basket)), "gross_return": gross,
            "estimated_cost": float(cost), "net_return": gross - cost,
            "spy_return": spy_ret, "excess_vs_spy": gross - cost - spy_ret,
            "hit_rate": float((basket[ret_col] > 0).mean()),
            "turnover_proxy": float(turnover),
            "selected_tickers": ",".join(sorted(names)),
        })
        prev = names
    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"horizon": horizon, "status": "no_valid_rebalances"}, detail
    net = detail["net_return"].astype(float).tolist()
    benchmark = detail["spy_return"].dropna().astype(float).tolist()
    summary = {
        "horizon": horizon, "status": "ok", "test_start": detail.date.iloc[0],
        "test_end": detail.date.iloc[-1], "n_rebalances": int(len(detail)),
        "avg_gross_return": float(detail.gross_return.mean()),
        "avg_net_return": float(detail.net_return.mean()),
        "avg_spy_return": float(detail.spy_return.mean()),
        "avg_excess_vs_spy": float(detail.excess_vs_spy.mean()),
        "hit_rate": float(detail.hit_rate.mean()),
        "max_drawdown_proxy": _max_drawdown(net),
        "avg_turnover_proxy": float(detail.turnover_proxy.mean()),
        "transaction_cost_bps_per_turnover": float(transaction_cost_bps),
        "note": "Periodic non-overlapping horizon baskets; final 30% dates; estimated costs; research proxy, not live performance."
    }
    return summary, detail

def factor_information_coefficient(data: pd.DataFrame, horizon: str,
                                   etfs: set[str] | None = None) -> dict:
    """Mean daily Spearman rank IC and IC information ratio on the final 30%."""
    ret_col = f"target_ret_{horizon}"
    z = data.dropna(subset=["factor_score", ret_col]).copy()
    z = z[~z["ticker"].isin(etfs or set())]
    dates = sorted(z.index.unique())
    if len(dates) < 300:
        return {"horizon": horizon, "ic_status": "insufficient_history"}
    values = []
    for dt in dates[int(len(dates) * .70):]:
        cross = z.loc[z.index == dt, ["factor_score", ret_col]].dropna()
        if len(cross) >= 30:
            ic = cross["factor_score"].corr(cross[ret_col], method="spearman")
            if np.isfinite(ic):
                values.append(float(ic))
    if len(values) < 3:
        return {"horizon": horizon, "ic_status": "insufficient_test_dates",
                "ic_observations": len(values)}
    sd = float(np.std(values, ddof=1))
    return {"horizon": horizon, "ic_status": "ok",
            "mean_rank_ic": float(np.mean(values)),
            "rank_ic_std": sd,
            "rank_ic_ir": float(np.mean(values) / sd) if sd > 0 else float("nan"),
            "ic_observations": len(values)}
