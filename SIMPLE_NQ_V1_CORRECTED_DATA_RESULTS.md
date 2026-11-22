# Simple NQ / QQQ Trader V1 — Corrected Corporate-Action Data Audit

Run date: 2026-11-22  
Branch: `simple-nq-trader-v1`  
Data: Yahoo Finance via yfinance, `auto_adjust=False`  
Sample: 2010-01-04 to 2026-11-20, 4,248 daily bars  
Starting equity: USD 10,000  
Costs: 1 bp commission and 5 bp slippage per side

## 1. Data and execution correction

The earlier backtest used dividend-adjusted OHLC as if those adjusted prices were literal historical executable quotes. This rerun replaces that with:
- Split-adjusted, dividend-unadjusted OHLC for signal levels, entries, stops, targets, and fills.
- Separate dividend cash events, credited only when a position was held through the prior close.
- Separate split ratios, adjusting held share quantities and stop/target prices. There were no QQQ split events in this 2010–2026 sample, so split handling is covered by a deterministic synthetic unit test rather than a live historical split in this sample.
- Buy-and-hold and SMA baselines also receive dividend cash and split share adjustments.
- The loader now requires explicit `dividends` and `stock_splits` columns so an old adjusted-only CSV cannot silently be treated as an execution dataset.

The downloaded sample contains **70 dividend event dates and 0 split event dates**. The 70 dividend events total $26.431 per share over the sample. Dividends are held as cash and are **not reinvested**; this same no-reinvestment cash accounting is applied across strategies and baselines. Therefore these benchmark figures are not the same as a conventional dividend-reinvestment total-return index.

## 2. Corrected-data backtest results

### Full sample

| Strategy | Final equity | Total return | CAGR | Max drawdown | Trades | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support bounce V1, corporate actions | $6,369.37 | -36.31% | -2.64% | -36.80% | 288 | 19.44% | 0.537 |
| QQQ buy-and-hold, dividends held as cash | $136,454.01 | +1,264.54% | 16.77% | -33.90% | 1 | — | — |
| SMA50/200 long/flat, dividends held as cash | $29,551.93 | +195.52% | 6.64% | -24.29% | 123 | 28.46% | 1.820 |
| Cash | $10,000.00 | 0.00% | 0.00% | 0.00% | 0 | — | — |

Trade ledger reconciliation: 288 trades; net trade P&L totals -$3,630.6299; $10,000 + P&L = $6,369.37 after rounding. Dividends credited to the strategy total $76.08 across five trades.

Exit reasons: STOP 186; GAP_STOP 29; STOP_FIRST_AMBIGUOUS 17; TARGET 45; TARGET_GAP 11. Stop-related exits were 232/288 (80.56%).

### Previously inspected final-20% period

Period: 2023-07-06 to 2026-11-20, 850 sessions; 3,398 prior sessions available for indicator warmup; each strategy starts with fresh $10,000 at the split.

| Strategy | Final equity | Total return | CAGR | Max drawdown | Trades | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support bounce V1, corporate actions | $8,943.59 | -10.56% | -3.26% | -11.58% | 55 | 23.64% | 0.501 |
| QQQ buy-and-hold, dividends held as cash | $16,773.06 | +67.73% | 16.59% | -22.42% | 1 | — | — |
| SMA50/200 long/flat, dividends held as cash | $12,721.97 | +27.22% | 7.41% | -13.02% | 33 | 27.27% | 1.634 |
| Cash | $10,000.00 | 0.00% | 0.00% | 0.00% | 0 | — | — |

This period has already been inspected in the earlier V1 analysis and is **not a pristine holdout for future tuning**.

## 3. Decision

The data correction does not rescue V1. The support-bounce strategy still loses in the full sample and in the previously inspected final-20% period, and it underperforms QQQ buy-and-hold, the SMA baseline, and cash in that period.

**V1 remains rejected for live trading.** Do not deploy it or tune its parameters against this already-inspected holdout.

## 4. What changed in code

- Downloader now uses `auto_adjust=False`, exports `dividends` and `stock_splits` separately.
- Backtest applies ex-date dividend cash and split share/price adjustments to open strategy trades.
- QQQ buy-and-hold and SMA baselines apply the same cash-dividend and split accounting.
- CSV loader requires explicit corporate-action fields and rejects non-finite action values.
- Added deterministic tests for split adjustment, dividend eligibility on entry/ex-date, and dividend accounting in buy-and-hold.
- Historical workflow now saves `QQQ_daily_execution.csv` with the exact JSON summary and trade ledger.

GitHub Actions on commit `ce80cdcfbe03ad84a281d44d73448d53af7af313` completed successfully for both the focused unit tests and the historical backtest. The corporate-action backtest artifact is retained for reproducibility.

## 5. Remaining limitations and next gate

1. Yahoo Finance is a third-party historical data source and may revise its history.
2. Daily OHLC cannot reconstruct intraday event ordering; ambiguous same-day stop/target bars still assume stop first.
3. Slippage and commissions are fixed assumptions, not observed broker fills.
4. Risk sizing is based on entry-to-stop distance; 0.5% is planned price risk, not a guaranteed maximum loss. Gaps can exceed it.
5. Cash dividends are not reinvested; report comparisons should keep that convention explicit.
6. No multi-fold walk-forward test has yet been completed. The previously inspected holdout must not be used to tune the next version.

Next: preserve this corrected-data V1 as the benchmark; then test one V2 entry/exit hypothesis on development periods and evaluate it with chronological walk-forward folds. No more complexity until the corrected baseline is stable.
