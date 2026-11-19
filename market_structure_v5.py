"""Market-structure features from OHLCV and optional open-interest data.

These are observable proxies, not direct identification of institutions or aggressor flow.
Open interest is a two-sided contract count; price/OI combinations are regime labels only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


_REQUIRED = {"Open", "High", "Low", "Close", "Volume"}


def build_market_structure(
    ohlcv: pd.DataFrame,
    open_interest: pd.Series | None = None,
    *,
    volume_window: int = 20,
    oi_change_window: int = 1,
) -> pd.DataFrame:
    """Return causal OHLCV/volume/OI features aligned to the input bar index.

    Parameters
    ----------
    ohlcv:
        DataFrame with Open, High, Low, Close, Volume columns.
    open_interest:
        Optional time-indexed OI series. Values are forward-filled only after
        their timestamp; missing OI remains missing.
    volume_window:
        Rolling baseline window for relative volume.
    oi_change_window:
        Number of bars over which to calculate OI percentage change.
    """
    missing = _REQUIRED.difference(ohlcv.columns)
    if missing:
        raise ValueError(f"ohlcv missing required columns: {sorted(missing)}")
    if volume_window < 2 or oi_change_window < 1:
        raise ValueError("volume_window must be >=2 and oi_change_window >=1")

    d = ohlcv.loc[:, ["Open", "High", "Low", "Close", "Volume"]].copy()
    d = d.sort_index()
    for col in d.columns:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    if d.index.has_duplicates:
        d = d[~d.index.duplicated(keep="last")]

    out = pd.DataFrame(index=d.index)
    close = d["Close"]
    candle_range = (d["High"] - d["Low"]).replace(0, np.nan)
    volume_base = d["Volume"].rolling(volume_window, min_periods=volume_window).mean()

    out["return_1bar"] = close.pct_change()
    out["relative_volume"] = d["Volume"] / volume_base.replace(0, np.nan)
    out["close_location"] = ((close - d["Low"]) / candle_range).clip(0, 1)
    out["body_fraction"] = ((d["Close"] - d["Open"]) / candle_range).clip(-1, 1)
    out["dollar_volume"] = close * d["Volume"]
    out["volume_zscore"] = (
        (d["Volume"] - d["Volume"].rolling(volume_window, min_periods=volume_window).mean())
        / d["Volume"].rolling(volume_window, min_periods=volume_window).std(ddof=0).replace(0, np.nan)
    )

    # Candle-direction volume is only a rough proxy. It is NOT true bid/ask delta.
    out["signed_volume_proxy"] = np.sign(d["Close"] - d["Open"]) * d["Volume"]
    out["signed_volume_proxy_ratio"] = (
        out["signed_volume_proxy"] / d["Volume"].replace(0, np.nan)
    )

    if open_interest is None:
        out["open_interest"] = np.nan
        out["oi_change"] = np.nan
        out["price_oi_regime"] = "oi_unavailable"
    else:
        oi = pd.to_numeric(open_interest, errors="coerce").sort_index()
        oi = oi[~oi.index.duplicated(keep="last")].reindex(out.index).ffill()
        out["open_interest"] = oi
        out["oi_change"] = oi.pct_change(oi_change_window)
        price_change = out["return_1bar"]
        oi_change = out["oi_change"]

        conditions = [
            (price_change > 0) & (oi_change > 0),
            (price_change < 0) & (oi_change > 0),
            (price_change > 0) & (oi_change < 0),
            (price_change < 0) & (oi_change < 0),
        ]
        labels = [
            "price_up_oi_up_building_exposure",
            "price_down_oi_up_building_exposure",
            "price_up_oi_down_short_cover_or_deleveraging",
            "price_down_oi_down_long_liquidation_or_deleveraging",
        ]
        out["price_oi_regime"] = np.select(
            conditions, labels, default="mixed_or_flat"
        )
        out.loc[oi.isna() | oi_change.isna(), "price_oi_regime"] = "oi_unavailable"

    return out.replace([np.inf, -np.inf], np.nan)
