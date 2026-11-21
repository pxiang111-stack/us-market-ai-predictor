# Regime-First v2 implementation status

Date: 2026-11-22

## Implemented in this repository

- `regime_first_v2.py`: deterministic three-state gate (`RISK_OFF`, `NEUTRAL`, `RISK_ON`).
- RISK_OFF triggers are evaluated independently, so one missing feed cannot hide another confirmed trigger.
- RISK_OFF recovery requires two consecutive valid recovery closes before moving to NEUTRAL.
- RISK_ON requires the base regime conditions plus all four flip gates.
- Risk budget applies the state cap, VIX multiplier, one-time missing-critical-data haircut, and optional hard-monitor-veto multiplier.
- Missing critical columns are recorded as data gaps rather than crashing the evaluator.
- The engine never emits an active short direction; RISK_OFF permits only reduce/hedge.
- Unit tests were added in `tests/test_regime_first_v2.py`; the Actions workflow now compiles the module and runs the test suite when the module, tests, or workflow changes.

## Not yet complete

This is currently a standalone rules engine. It is **not yet connected** to the daily market-data pipeline and does not yet write a live daily Regime-First report.

The following inputs or modules still require a point-in-time source and validation:

1. A canonical daily `rank_model_score`. The existing stock-ranking output has per-stock opportunity/factor scores; that is not automatically equivalent to the market-wide direction score used by the v1.6-A framework.
2. CCC, BBB and HY spread history with publication-date handling.
3. GEX, DIX, SPY/QQQ flows and institution-order veto mapping. A hard veto must be explicitly defined; warnings must not silently become zero-risk vetoes.
4. Eventized signal clusters, swing-point detection, ATR/stop-distance sizing, option liquidity checks and portfolio-level risk aggregation.
5. Walk-forward and point-in-time historical validation, including the 19 dated calls, 117-day sample and 2022-2026 window.

## Frozen interpretation notes requiring owner confirmation before live activation

- Rank-score recovery is currently coded as: the score starts at or below -2, rises by at least 1 point on each of two consecutive sessions, and finishes at or above 0. This is a literal operational interpretation of the written rule, not a calibrated parameter.
- The top-level principle that the short leg only closes/reduces risk takes precedence over examples that describe opening a QQQ put on a failed bounce. No active short signal is generated until that conflict is explicitly resolved.
- For the 2026-11-20 example, a missing critical NQ 20-day low halves the risk budget once. With RISK_OFF cap 1% and VIX 39.47 multiplier 0.5, the resulting maximum is 0.25% before any hard veto. The missing R5 statistic is a signal-data gap, not a second critical-contract haircut.
- GEX/DIX readings are not hard vetoes by themselves in this implementation. They require explicit, independently validated veto conditions.

## Validation status

Code and tests have been committed. A successful GitHub Actions run has not yet been confirmed; do not treat the module as production-validated or use it to authorize live orders.
