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

## First backtest harness

The branch also includes `flowfirst_backtest.py`, which compares four variants per symbol:

- `price_only`: 15m and 1h EMA trend alignment
- `flow_only`: 4-bar taker imbalance plus rolling z-score
- `combo`: price trend and flow confirmation together
- `combo_oi`: combo signal plus OI value expansion as a participation filter

Backtest safeguards and assumptions:

- Signal is evaluated only after a completed 15m bar; entry is the next 15m bar open.
- Uses the most recently completed 1h bar only.
- Stop = 1.5 ATR; target = 3 ATR; maximum hold = 16 bars.
- If stop and target are both touched in one bar, stop is assumed first.
- Starts each symbol/variant with $10,000, risks 0.5% of current equity per trade and caps gross exposure at 1.0x equity.
- Applies an explicit 7 bps per-side cost assumption (5 bps fee + 2 bps slippage); actual TradFi costs may differ.
- Reports trade count, win rate, net P&L, return, profit factor, expectancy, maximum drawdown and average R, with buy-and-hold return for comparison.

This is a first diagnostic harness. Results with small samples are not evidence of an edge. It does not yet include walk-forward parameter selection, a portfolio-level capital allocator, financing/funding costs, or a statistically robust out-of-sample validation. Those must be added before any strategy is considered deployable.

## Seeded research dataset and runner restriction

An initial research snapshot was imported through the connected Binance market-data interface into:

- `data/flowfirst/seed_klines_15m.jsonl`
- `data/flowfirst/seed_klines_1h.jsonl`
- `data/flowfirst/seed_taker_15m.jsonl`
- `data/flowfirst/seed_oi_15m.jsonl`

The seed contains the five core symbols and up to 500 completed records per symbol for each dataset. The collector's deduplication now checks all files for a kind/period so later daily files do not duplicate the seed.

The first GitHub-hosted collection attempt compiled successfully but Binance returned HTTP 451 from the hosted runner region (Azure westus3). The collector now stops immediately on this explicit regional restriction instead of retrying. Scheduled collection has been removed until a permitted execution region is available. Pull-request checks still compile, run unit tests, validate the seeded dataset, and generate the research/backtest reports without calling Binance from the runner.

Do not treat this one-time seed as an ongoing real-time feed. It is an initial sample for validating the analysis pipeline. Ongoing data refresh requires a Binance-permitted execution environment; do not use proxies or routing to bypass regional restrictions.

## Chronological holdout and excursion diagnostics

The backtest now adds a 70% chronological training-window / 30% later holdout split for each symbol and strategy. The holdout uses the same fixed rules and parameters; no tuning is performed on the holdout. The full-period result remains for comparison, but the holdout result is the more relevant early robustness check.

Each trade also records maximum adverse excursion (MAE) and maximum favorable excursion (MFE), expressed in R units. These diagnostics help determine whether stops are too tight or targets are unrealistic.

This is still only a short initial sample (the seed snapshot is approximately five days of 15m flow/price history). A 30% holdout of such a short sample is not statistically sufficient to establish a durable edge. Continue accumulating permitted data before making deployment decisions.

