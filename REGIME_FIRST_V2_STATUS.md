# Regime-First v2 implementation status

Date: 2026-11-22

## Implemented and committed

- `regime_first_v2.py`: deterministic three-state engine, hysteresis, four flip gates, state risk caps, VIX multiplier, missing-critical-data haircut, optional hard monitor veto.
- Independent BTC, ETH, NQ/VIX and credit stress triggers; a missing feed does not hide another confirmed trigger.
- Credit trigger follows the written rule: CCC OAS >14% plus five consecutive widening sessions in both BBB and broad HY OAS (CCC itself need not widen).
- `run_regime_first_v2.py`: daily data adapter for NQ futures, NDX, BTC, ETH and VIX from Yahoo Finance; CCC/BBB/HY OAS from FRED CSV; prior-only NQ and crypto rolling bands; daily CSV and latest Markdown report.
- `regime_event_study.py`: 5-day/20-day forward NDX returns by daily state and state-entry events; n<30 is marked RECORD_ONLY, with no strategy-performance claim.
- The adapter refuses to fabricate `rank_model_score`. It reads `output/rank_model_score.csv` with columns `date,rank_model_score`; if unavailable, it records the critical data gap and halves the risk budget.
- GitHub Actions now runs the Regime-First adapter after the existing V4 research pipeline and uploads both reports under `output/`.
- Unit tests cover state transitions, missing feeds, hard veto, flip gates, credit widening, and event-study sample gating/coverage.

## Critical unresolved data interface

The market-wide `rank_model_score` used by v1.6-A is not yet mapped from the legacy source. The existing V4 stock-ranking score is a cross-sectional stock opportunity score and must not be substituted for a market direction score. Until a validated adapter exists, the daily engine will show this as a data gap and cannot authorize RISK_ON.

## Other incomplete layers

- Historical point-in-time FRED vintages and survivorship-bias-free equity universe are not yet integrated for the 2022-2026 validation.
- Eventized break/key-level/time/confirmation clusters are not yet connected to the daily output.
- GEX, DIX, SPY/QQQ flows and COT are not yet sourced or mapped to explicit hard-veto rules.
- ATR/stop-distance sizing, option spread/open-interest/volume checks, and portfolio-level risk aggregation remain to be implemented.
- The source note reports aggregate results for 19 calls and 117 days, but does not contain the complete dated row-level dataset needed to reproduce them; those samples have not yet been reconstructed and tested.
- A successful GitHub Actions run has not yet been confirmed.

## Frozen interpretation notes

- Flip score recovery is currently implemented as start <= -2, two consecutive rises of at least 1 point, ending >= 0. This is an explicit operational interpretation, not calibrated.
- The top-level rule prohibiting active short direction takes precedence; no active short signal is generated.
- Hard monitor veto requires an explicit `monitor_hard_veto` boolean input. GEX/DIX warnings are not treated as hard vetoes by default.
- The 27,480 NQ level remains frozen as written.

## Validation status

The rules engine and daily adapter are committed but not production-validated. No live orders should be authorized from these outputs. Daily data quality, rank-score mapping, historical validation and the first successful CI run remain gating items.
