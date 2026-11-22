# FlowFirst data collection (v0.2)

## What is collected

The scheduled GitHub Actions job queries Binance USD-M Futures public market-data endpoints for these TradFi perpetual symbols:

- QQQUSDT, NVDAUSDT, SMHUSDT, SOXLUSDT, SPYUSDT
- AMDUSDT, AVGOUSDT, TSMUSDT, MUUSDT

For each symbol it stores 15-minute and 1-hour records for:

1. Taker buy/sell volume (/futures/data/takerlongshortRatio)
2. Open-interest history (/futures/data/openInterestHist)
3. OHLCV klines (/fapi/v1/klines), including Binance taker-buy base/quote volume fields

Raw JSONL records are appended under data/flowfirst/, split by UTC date and data type. Duplicate symbol/period/timestamp records are skipped. The collector backfills the most recent 500 records returned by each endpoint, helping recover from missed scheduled runs.

## Schedule

GitHub Actions is scheduled every 15 minutes on weekdays from 12:00 through 22:59 UTC, and can also be run manually from the Actions tab. GitHub scheduled jobs can be delayed or skipped; this is not a real-time feed and does not guarantee every 15-minute bar is captured.

## Important interpretation limits

- Binance TradFi perpetual data reflects activity on Binance's own derivatives venue. It is not the consolidated US stock/ETF tape, not the full Nasdaq/NYSE order flow, and not proof of institutional buying or selling.
- Taker buy/sell volume is venue-specific aggressive trading volume. It is not the same as total equity-market volume.
- Open interest is a contract-position measure. It does not identify whether institutions, retail traders, longs, or shorts created the positions.
- OHLCV and taker-buy volume are stored separately so the research layer can align timestamps and avoid look-ahead bias.
- The collector does not place trades and does not connect to any brokerage account.
- Do not interpret an empty data file or a failed request as zero volume. Check workflow logs and the errors count.

## First validation checklist

Before treating any FlowFirst result as a signal:

1. Confirm data collection runs successfully over multiple US sessions.
2. Check timestamp alignment between taker volume, OI, and klines.
3. Verify whether each Binance symbol was listed and trading throughout the sample.
4. Use only completed bars; enter no earlier than the next bar.
5. Compare price-only, flow-only, flow-plus-price, and crowding-filter variants.
6. Report sample count, date range, trading costs, slippage, drawdown, and out-of-sample performance.

This is a data-collection foundation, not a validated trading strategy.

## FlowFirst signal layer (v0.1)

The branch now includes `flowfirst_signal.py`. After collection and raw-data validation, the workflow generates:

- `reports/flowfirst_latest.json`
- `reports/flowfirst_latest.md`

The initial signal layer is deliberately conservative:

1. Requires at least 60 completed 15m price bars and 55 completed 1h price bars.
2. Uses 15m and 1h EMA20/EMA50 trend alignment.
3. Uses the average taker buy/sell imbalance of the latest four completed 15m bars and its rolling z-score against the preceding 48 bars.
4. Requires both price trend and flow to confirm before issuing LONG or SHORT; otherwise it returns WAIT.
5. Uses 1.5 ATR as the reference stop distance and 3 ATR as the reference target distance (2R). These are research references, not guaranteed fills.
6. Rejects stale price data. It does not produce a signal when history or timestamps are insufficient.

The market regime is a simple breadth read across QQQ, SPY and SMH 1h trends. OI change is reported as participation context only; OI expansion alone is never treated as a bullish or bearish signal.

The report's entry reference is the signal bar close for analysis only. Backtests must execute at the next bar open and include fees/slippage. The script does not output a win probability because the strategy has not yet been validated.

## CI behavior

The pull-request workflow path compiles the Python modules, collects public data, validates the raw records and generates a report. On pull requests it does not push commits; it uploads the report as a workflow artifact. Scheduled/manual runs can append collected data and reports to the branch where the workflow is enabled.

