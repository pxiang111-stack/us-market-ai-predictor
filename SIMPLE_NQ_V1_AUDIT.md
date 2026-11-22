# Simple NQ / QQQ Trader V1 — Independent Code and Artifact Audit

Audit date: 2026-11-22  
Branch: `simple-nq-trader-v1`  
Scope: static review of the V1 backtest, downloader, benchmark timing, and independent arithmetic checks against the retained workflow artifact. No strategy parameters were changed.

## Executive conclusion

V1 remains rejected as a live strategy. The retained trade ledger independently reconciles to a final strategy equity of **$6,205.12** from $10,000 (net trade P&L -$3,794.88). The earlier findings document says $6,204.63, a **$0.49 reporting mismatch**; the retained JSON summary and trade ledger agree with $6,205.12. The findings document should be corrected before treating it as the canonical report.

The most important unresolved research issue is the use of dividend-adjusted OHLC prices as if they were literal historical executable quotes. This can distort stop/target and entry/exit simulation around ex-dividend dates. V2 must not reuse this data/execution assumption.

## Checks completed

1. **Trade-ledger arithmetic:** 287 trades; sum of `net_pnl` = -$3,794.8846; $10,000 + trade P&L = $6,205.1154, which rounds to the artifact's $6,205.12.
2. **Exit reason counts:** STOP 176; GAP_STOP 38; STOP_FIRST_AMBIGUOUS 20; TARGET 46; TARGET_GAP 7. Stop-related exits = 234/287 (81.5%).
3. **Holding time:** median 1 trading session; mean 1.91 sessions, measured from signal date to exit date using the downloaded trading-session calendar.
4. **Planned reward/risk:** median 3.11R; mean 8.34R; maximum 228.87R. The mean is distorted by very distant resistance targets and should not be interpreted as a typical trade.
5. **Timing review:** the signal uses the completed bar at index i; entry is at the next bar's open. The SMA baseline computes its signal using bars before the execution bar. The holdout SMA baseline begins signal generation at the holdout split and therefore starts with its own $10,000 rather than carrying pre-split positions.
6. **Stop handling:** a gap below the stop is filled at the open; same-day stop/target ambiguity is resolved stop-first. This is conservative but cannot reconstruct intraday order from daily bars.
7. **No parameter changes:** this audit does not optimize V1 or use the inspected 2023–2026 holdout to tune a replacement.

## Issues to resolve before V2

### A. Price basis is not suitable for literal stop/limit execution

The downloader currently requests `auto_adjust=True`. The resulting OHLC series is dividend-adjusted. It is a useful total-return-style research series, but it is not the nominal market price history at which historical stop and limit orders would have executed.

**Required V2 data design:** use split-adjusted, dividend-unadjusted OHLC for entries, stops, targets, and fills; separately download corporate actions and credit cash dividends to the simulated account when eligible. Preserve the exact raw-price/action files with every run. Add tests around an ex-dividend date and a split so price continuity, share counts, and cash distributions are explicit.

### B. The 0.5% risk budget is a price-distance estimate, not a guaranteed all-in loss cap

Current share sizing divides the 0.5% cash risk budget by entry-to-stop price distance. It does not include both-side commissions and slippage in that sizing calculation, and gap risk can exceed the planned stop distance. V2 should report planned stop risk, estimated cost-inclusive risk, and realized gap loss separately. It must never describe 0.5% as a guaranteed maximum loss.

### C. Data validation can be stricter

The CSV loader checks required columns, positive OHLC, high/low consistency, and duplicate dates. Before V2, add explicit finite-number checks and chronological/date sanity checks, plus a clear failure if corporate-action fields are missing from the execution dataset.

### D. Reporting reproducibility

The retained artifact's JSON summary and trade ledger reconcile to $6,205.12, while the findings Markdown says $6,204.63. The retained artifact is the auditable source for this run. Regenerate the Markdown from the exact JSON/CSV outputs in future runs rather than hand-copying rounded values.

## V2 gate — no coding around the unresolved data basis

1. Build a corporate-action-aware QQQ dataset (split-adjusted execution OHLC + dividend cash events).
2. Add deterministic unit tests for ex-dividend, split, gap-stop, same-day ambiguity, fees, and risk sizing.
3. Re-run V1 unchanged on the corrected execution basis and label it **V1 corrected-data audit**, not a strategy improvement.
4. Only then define one V2 hypothesis and test it on development periods with chronological walk-forward folds. The previously inspected 2023-07-06 to 2026-11-20 holdout is not pristine and must not be used to select V2 parameters.

No live orders, broker integration, options, leverage, or automated trading are part of this project.
