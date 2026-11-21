# Simple NQ / QQQ Trader V1 — First Historical Backtest Findings

Run date: 2026-11-22  
Data: Yahoo Finance through yfinance, QQQ daily bars with `auto_adjust=True`  
Sample: 2010-01-04 through 2026-11-20, 4,248 sessions  
Starting capital: USD 10,000  
Costs: 1 bp commission and 5 bp slippage per side  
Execution: signal after completed close; entry at next session open; daily OHLC stop/target ambiguity resolved stop-first; stop gaps fill at the open.

## 1. Full-sample result

| Strategy | Final equity | Total return | CAGR | Max drawdown | Trades | Win rate | Profit factor |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support bounce V1 | $6,205.16 | -37.95% | -2.79% | -38.71% | 287 | 18.47% | 0.511 |
| QQQ buy and hold | $151,415.61 | 1,414.16% | 17.50% | -35.12% | 1 | 100%* | inf* |
| SMA50/200 long/flat | $28,720.95 | 187.21% | 6.46% | -21.22% | 122 | 31.97% | 1.869 |
| Cash, no trade | $10,000.00 | 0.00% | 0.00% | 0.00% | 0 | — | — |

*The buy-and-hold trade statistics are not comparable to strategy trade statistics; it is one continuous holding period.

## 2. What went wrong

Of 287 V1 trades:
- 176 exited at an ordinary stop.
- 38 exited after a gap below the stop.
- 20 had both stop and target touched in one daily bar and were conservatively recorded stop-first.
- 44 reached the target.
- 9 gapped through the target.

Thus, 234 of 287 trades (81.5%) exited through a stop variant. Median holding time was one session; mean holding time was 1.92 sessions.

By entry regime:
- RANGE_TRANSITION: 153 trades; 16.34% win rate.
- UPTREND: 134 trades; 20.90% win rate.

Neither regime subset showed a positive trade expectancy in this full-sample diagnostic.

The signal requires a daily support test and close above the zone midpoint, but the current stop is only 0.25 ATR below the support-zone low. The observed quick stop-outs are consistent with a stop that may be too close to ordinary daily noise, but this is a hypothesis to test, not a proven explanation. The nearest-resistance target can also be very far away; the planned reward/risk distribution is highly skewed, and very high theoretical R values did not translate into target hits.

## 3. Decision

**V1 is rejected as a usable trading strategy.** It loses money after the stated costs, has a worse drawdown than both QQQ buy-and-hold and SMA50/200, and has no demonstrated predictive edge. Do not use it for live trading or as a basis for real-money position sizing.

Do not optimize V1 thresholds against this same full sample and then report the improved in-sample result as proof.

## 4. Next validation gate

Before changing the rule, preserve this V1 result as the benchmark. The 80/20 date split for this dataset places the final 20% at **2023-07-06 through 2026-11-20** (850 sessions). That period should remain untouched until a revised rule is specified and frozen.

Next work should:
1. Add a genuine point-in-time holdout runner that warms indicators using pre-holdout bars but starts with fresh $10,000 cash at the holdout boundary.
2. Report V1 holdout results against buy-and-hold, SMA50/200 and cash.
3. Only then test one clearly motivated change at a time, such as stop placement below the actual signal-bar low with a volatility buffer, or skipping next-session entries that gap too far from the signal close.
4. Keep every revision as a separately named version and report both development-period and untouched holdout results.

## 5. Limitations

- This is a full-sample diagnostic, not a final walk-forward validation.
- Yahoo Finance adjusted history may be revised. The exact downloaded CSV is included in the workflow artifact for reproducibility.
- Adjusted OHLC is suitable for a first total-return-style research comparison, but adjusted historical prices are not literal executable historical quotes. Corporate-action treatment should be independently reviewed.
- The simulator uses daily OHLC; intraday order cannot be reconstructed. Same-day stop/target ambiguity is deliberately resolved against the strategy.
- Commissions and slippage are assumptions, not broker-specific observed fills.
- No interest is credited to idle cash. Taxes, market impact, and borrow costs are not modelled.
- Results do not imply future performance.
