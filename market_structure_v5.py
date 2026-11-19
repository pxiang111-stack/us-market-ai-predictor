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
    volume_is_valid: bool = True,
) -> pd.DataFrame:
    """Return causal OHLCV/volume/OI features aligned to the input bar index.

    volume_is_valid must be False for cash-index symbols whose vendor Volume
    field is not an executable consolidated trading-volume measure.
    RVOL and z-score compare the current completed bar with prior bars only.
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
    out["return_1bar"] = close.pct_change()
    out["close_location"] = ((close - d["Low"]) / candle_range).clip(0, 1)
    out["body_fraction"] = ((d["Close"] - d["Open"]) / candle_range).clip(-1, 1)

    # Use only bars strictly before the current completed bar as the baseline.
    prior_volume = d["Volume"].shift(1)
    volume_mean = prior_volume.rolling(volume_window, min_periods=volume_window).mean()
    volume_std = prior_volume.rolling(volume_window, min_periods=volume_window).std(ddof=0)
    if volume_is_valid:
        out["relative_volume"] = d["Volume"] / volume_mean.replace(0, np.nan)
        out["volume_zscore"] = (
            (d["Volume"] - volume_mean) / volume_std.replace(0, np.nan)
        )
        out["dollar_volume"] = close * d["Volume"]
        # Candle-direction volume is a coarse bar proxy, never bid/ask delta.
        out["signed_volume_proxy"] = np.sign(d["Close"] - d["Open"]) * d["Volume"]
        out["signed_volume_proxy_ratio"] = (
            out["signed_volume_proxy"] / d["Volume"].replace(0, np.nan)
        )
        out["volume_data_status"] = "vendor_volume_available_proxy_only"
    else:
        for col in (
            "relative_volume", "volume_zscore", "dollar_volume",
            "signed_volume_proxy", "signed_volume_proxy_ratio",
        ):
            out[col] = np.nan
        out["volume_data_status"] = "invalid_for_cash_index"

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
        out["price_oi_regime"] = np.select(conditions, labels, default="mixed_or_flat")
        out.loc[oi.isna() | oi_change.isna(), "price_oi_regime"] = "oi_unavailable"

    return out.replace([np.inf, -np.inf], np.nan)
