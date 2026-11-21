# Simple NQ / QQQ Trader V1 — Frozen Specification

Status: design baseline only. No live trading, no scheduled workflow, no automated orders.
Date: 2026-11-22

## 1. Purpose

Build one small, explainable trading assistant that answers:
1. What is the NDX/NQ market regime?
2. Where are the nearest meaningful support and resistance zones?
3. Is there a confirmed setup at one of those zones?
4. What are entry, invalidation/stop, target(s), and position size?
5. Why is the correct action LONG, SHORT-SETUP, or WAIT?

The first release is a decision-support prototype, not a profit claim.

## 2. Instruments and execution

- Market context: NDX index and NQ futures.
- Initial simulated execution instrument: QQQ, using daily bars and next-session execution.
- Starting paper capital: USD 10,000.
- No leverage, no margin, no live orders.
- First backtest is long/flat. A bearish signal means reduce/avoid long exposure; short setups are recorded as research-only until a separate, correctly costed short/option execution model is validated.
- Whole QQQ shares for the cash-account simulation; unused cash remains cash.

## 3. Minimal inputs

Required:
- Daily OHLCV for QQQ and NDX (or a consistently adjusted equivalent).
- NQ daily data for futures context, with explicit contract-roll handling if used.
- ATR(14), SMA(20), SMA(50), SMA(200).
- VIX daily close.
- A market-breadth proxy: percentage of NDX constituents above their 50-day average, only if a reliable point-in-time source is available.

Optional later:
- 4-hour / 1-hour bars for entry refinement.
- News/event flags. News must not change direction unless a rule is explicit and historically testable.

Missing required data must produce WAIT / DATA INCOMPLETE, never a fabricated signal.

## 4. Market regime

Use a small deterministic state machine:

- UPTREND: close > SMA(50), SMA(50) > SMA(200), and SMA(20) rising over 5 sessions.
- DOWNTREND: close < SMA(50), SMA(50) < SMA(200), and SMA(20) falling over 5 sessions.
- RANGE / TRANSITION: everything else.

Use the prior completed session only. No intraday incomplete candle may enter a daily decision.

## 5. Support and resistance zones

Do not use one exact price as a level. Build zones from:
- Confirmed swing highs and swing lows (3 bars on each side); a pivot becomes usable only after the three right-side bars have closed.
- Prior 20-session and 60-session highs/lows.
- Repeated reaction areas clustered within 0.5 x ATR(14).
- Previous day high/low and prior week high/low as secondary references.

Zone rules:
- Merge nearby levels within 0.5 x ATR(14).
- Rank zones by recency, number of independent touches, and reaction magnitude.
- Require at least two separate reactions before labelling a zone "major".
- Expire or downgrade a zone after a decisive close through it; a broken support may become resistance only after a retest/rejection.
- Report source and timestamp for every level. Never draw a historical level using future bars.

## 6. Setups

Only three initial setups:

A. Support bounce (LONG candidate)
- Regime is UPTREND or RANGE.
- Price tests a ranked support zone.
- A completed daily candle closes back above the zone midpoint.
- Stop is below the zone by a volatility buffer (initially 0.25 x ATR).
- Target is the next ranked resistance zone.

B. Resistance rejection (risk-off / SHORT-SETUP observation)
- Price tests a ranked resistance zone and closes back below its midpoint.
- In DOWNTREND, mark SHORT-SETUP; in other regimes, WAIT for confirmation.
- V1 does not open short positions.

C. Breakout and retest (LONG candidate)
- Daily close exceeds a ranked resistance zone by at least 0.25 x ATR.
- Do not chase the breakout close.
- Wait for a retest within the next 5 sessions; the retest must hold and close back above the zone.
- Stop below the retest low / zone with the same volatility buffer.
- Target is the next ranked resistance zone or a fixed 2R fallback if no reliable zone exists.

No setup means WAIT. Do not force a trade every day.

## 7. Risk and position sizing

- Risk per trade: 0.5% of current paper equity.
- Maximum total invested notional: 100% of equity.
- Shares = floor((equity x 0.005) / (entry - stop)).
- If one share exceeds the risk budget, do not trade.
- Skip if target distance / stop distance < 1.5.
- Skip if spread/slippage assumptions make the expected reward-to-risk < 1.3.
- No averaging down; no adding to a losing position.
- Stop after a 3% daily simulated equity loss; resume only on the next session.

## 8. Execution and anti-look-ahead rules

- Signal calculated after the daily close using only data then available.
- Entry no earlier than the next session; use next-session open plus adverse slippage.
- Stop/target fills must model gaps conservatively: if the open gaps through a stop, fill at the open, not at the stop.
- If stop and target are both touched in the same daily bar and intraday ordering is unknown, assume the adverse outcome (stop first).
- Include commissions, spread and slippage; report each separately.
- Corporate actions and adjusted/unadjusted price handling must be consistent.
- All indicator and pivot calculations must be point-in-time; no centered rolling windows that leak future values.

## 9. Validation plan

1. Unit tests for pivot confirmation delay, zone clustering, broken-level state, stop sizing, gap-through-stop, and same-bar stop/target ambiguity.
2. Walk-forward test on at least 5 years of data, not just the latest year.
3. Show separate results for trend, range, and selloff periods.
4. Compare against:
   - QQQ buy-and-hold.
   - A simple SMA(50)/SMA(200) long/flat baseline.
   - A no-trade cash baseline.
5. Report net CAGR, total return, max drawdown, Sharpe, profit factor, win rate, average win/loss, turnover, exposure, and number of trades.
6. Keep the final 20% of dates untouched until rules are frozen.
7. Do not optimize thresholds on the test period. Any change creates a new version and requires a fresh holdout.
8. If performance does not beat the baselines on net return or meaningfully improve drawdown at comparable return, state that no edge has been demonstrated.

## 10. Daily output

Every report must show:
- Data timestamp and source.
- Regime: UPTREND / RANGE / DOWNTREND.
- Nearest support zones and resistance zones, with strength and reason.
- Setup: SUPPORT BOUNCE / RESISTANCE REJECTION / BREAKOUT RETEST / NONE.
- Action: LONG / WAIT / SHORT-SETUP (research only).
- Entry range, stop, target 1, target 2, risk/reward, shares, dollar risk.
- Invalidation condition.
- Missing/stale data warnings.

## 11. Explicit exclusions from V1

- No deep learning, LLM-generated buy/sell decisions, or opaque composite score.
- No broad-universe stock ranking.
- No options pricing or options execution.
- No intraday scalping.
- No automatic live trading.
- No scheduled scans or workflow automation until the user explicitly asks to enable them.

## Acceptance criterion

V1 is complete only when the code, tests, reproducible historical data, and a walk-forward report all exist and can be independently checked. A green CI badge alone is not proof of a trading edge.
