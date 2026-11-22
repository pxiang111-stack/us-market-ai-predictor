# Simple NQ / QQQ Trader V1 — Chronological Walk-Forward Diagnostics

Run date: 2026-11-22  
Data: QQQ split-adjusted, dividend-unadjusted OHLC with dividends as cash (not reinvested)  
Starting paper equity: $10,000 at the start of each fold  
Costs: 1 bp commission and 5 bp slippage per side

## Method

Four chronological, non-overlapping evaluation windows. All data before each window is available only as indicator warmup; the strategy and benchmarks start with fresh $10,000 at the window start. Each run stops at that fold's final date. No parameters are fitted or optimized in these folds.

The first fold starts on 2010-10-26 because 205 daily bars are required for indicator warmup. Later folds start at the first trading session of their requested periods.

## Results

| Fold | V1 final equity | V1 return | V1 CAGR | V1 max DD | V1 trades | V1 win rate | V1 PF |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2010–2014 | $9,111.87 | -8.88% | -2.20% | -9.59% | 67 | 17.91% | 0.538 |
| 2015–2018 | $8,526.28 | -14.74% | -3.92% | -17.20% | 84 | 15.48% | 0.436 |
| 2019–2022 | $8,908.16 | -10.92% | -2.85% | -13.07% | 70 | 18.57% | 0.543 |
| 2023–2026 | $9,185.61 | -8.14% | -2.17% | -11.71% | 67 | 26.87% | 0.674 |

### Benchmark final equity by fold

| Fold | V1 | QQQ buy-and-hold, dividends cash | SMA50/200, dividends cash | Cash |
|---|---:|---:|---:|---:|
| 2010–2014 | $9,111.87 | $20,676.00 | $13,679.68 | $10,000 |
| 2015–2018 | $8,526.28 | $15,319.28 | $10,427.44 | $10,000 |
| 2019–2022 | $8,908.16 | $18,038.91 | $12,850.34 | $10,000 |
| 2023–2026 | $9,185.61 | $22,940.70 | $15,751.29 | $10,000 |

## Interpretation

- V1 lost money in **all four** chronological windows.
- V1 underperformed QQQ buy-and-hold, SMA50/200, and cash in every window.
- Profit factor remained below 1 in every window (0.436–0.674), so the negative result is not isolated to one market regime.
- The 2023–2026 window overlaps the previously inspected holdout and is descriptive only. It is not a fresh untouched test.
- These folds are a stability diagnostic for fixed V1 rules, not evidence that the strategy is profitable and not a parameter-search process.

**Decision: keep V1 rejected. Do not spend time optimizing its thresholds.** The next version should change the trade thesis itself, not just tweak numbers: for example, require a market regime and momentum confirmation before attempting a support bounce, then validate the single change on development windows without touching the previously inspected 2023–2026 period.
